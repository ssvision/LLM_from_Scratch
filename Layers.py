import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import tiktoken



class CreateGPTDatasetV1(Dataset):
    def __init__(self, txt: str, tokenizer: tiktoken.Encoding, context_length: int, stride: int):
        
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})


        # Split into input and target using a sliding window approach
        for i in range(0, len(token_ids)-context_length, stride):

            input_chunk = token_ids[i:i+context_length]
            output_chunk = token_ids[i+1:i+1+context_length]

            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(output_chunk))

    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]
    
    @staticmethod
    def create_dataloader_v1(txt, batch_size, max_length, 
                         stride=1, shuffle=True, drop_last=True,
                         num_workers=0):

        # Initialize the tokenizer
        tokenizer = tiktoken.get_encoding("gpt2")

        # Create dataset
        dataset = CreateGPTDatasetV1(txt, tokenizer, max_length, stride)

        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers
        )

        return dataloader


class MultiHeadCausalAttention(nn.Module):

    def __init__(self, d_in, d_out, context_length, num_heads, dropout, qkv_bias=False):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.context_length = context_length
        self.num_heads = num_heads
        self.droput = dropout
        self.qkv_bias = qkv_bias
        self.droput = nn.Dropout(p=dropout)

        assert d_out%num_heads == 0
        self.d_k = d_out//num_heads #this is the head dimension, d_k as per paper

        self.Wq = nn.Linear(d_in, d_out, qkv_bias)
        self.Wk = nn.Linear(d_in, d_out, qkv_bias)
        self.Wv = nn.Linear(d_in, d_out, qkv_bias)

        self.out_proj =nn.Linear(d_out, d_out)
        
        self.register_buffer("mask", torch.triu(torch.ones(context_length,context_length),diagonal=1))


    def forward(self, X):
        '''extract the batch_size, num of tokens and the input shape'''
        batch_size, num_tokens, d_in = X.shape # note d_in = d_model in the paper

        '''compute the query, key and value matrices'''
        # they are of shape (batch_size, num_tokens, d_out)
        query = self.Wq(X)
        key = self.Wq(X)
        value = self.Wq(X)

        '''the last dimension of query,key,value matrices is d_out = num_heads*d_k'''
        """ Reshape the matrices from (batch_size, num_tokens, d_out) -> (batch_size, num_tokens, num_heads, d_k)"""
        query = query.view(batch_size, num_tokens, self.num_heads, self.d_k)
        key = key.view(batch_size, num_tokens, self.num_heads, self.d_k)
        value = value.view(batch_size, num_tokens, self.num_heads, self.d_k)

        '''the resulting matrices are grouped by num_tokens but'''
        '''in order to compute attention they must be grouped by no of heads'''
        """so the 2nd and 3rd dimesnions need to be interchanged"""
        query = query.transpose(1,2)
        key = key.transpose(1,2)
        value = value.transpose(1,2)

        '''Resulting dimension is (batch_size, num_heads, num_tokens, d_k)'''
        '''attn = query @ value.T [(batch_size, num_heads, num_tokens, d_k) * (batch_size, num_heads, d_k, num_tokens)]'''
        """Compute attention scores"""
        attn_scores = query @ key.transpose(2,3)

        '''Resulting dimesnion is (batch_size, num_heads, num_tokens, num_tokens)'''
        """Compute attention weights"""
        attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)  # `:num_tokens` to account for cases where the number of tokens in the batch is smaller than the supported context_size
        attn_weights = torch.softmax(attn_scores / key.shape[-1]**0.5, dim=-1)
        attn_weights = self.droput(attn_weights)

        '''Resulting dimension is (batch_size, num_heads, num_tokens, num_tokens)'''
        '''context vector = attn_weights * value'''
        '''(batch_size, num_heads, num_tokens, num_tokens) * (batch_size, num_heads, num_tokens, d_k) '''
        context_vec = attn_weights @ value

        '''Resulting dimesnion is (batch_size, num_heads, num_tokens, d_k)'''
        context_vec = context_vec.transpose(1,2)
        '''combine them such as d_out = num_heads * d_k'''
        # context_vec = context_vec.contigious().view(batch_size, num_tokens, self.d_out)
        context_vec = context_vec.reshape(batch_size, num_tokens, self.d_out)


        '''Resulting dimesnion is (batch_size, num_tokens, d_out)'''

        '''projection for output layer'''
        '''output is (batch_size, num_tokens, d_out)'''
        output = self.out_proj(context_vec)

        return output 

        
'''
Layer norm implemented from scratch. 
However Pytorch's nn.LayerNorm can be used
'''
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift
    

'''
This is the Feed Forward Block in the Transformer block, which is an expansion contraction layer
'''

class FeedForwardBlock(nn.Module):
    def __init__(self, cfg: dict, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
        self.layers = nn.Sequential(
                        nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]), #expansion
                        nn.GELU(), #activation
                        nn.Linear(4*cfg["emb_dim"], cfg["emb_dim"]) #contraction
                        )
        
    def forward(self, X):
        return self.layers(X)


'''
This is the Tramsformer block implemented as a class
'''

class TransformerBlock(nn.Module):

    def __init__(self,cfg, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.causal_multi_head_attention = MultiHeadCausalAttention(d_in = cfg["emb_dim"],
                                                                    d_out=cfg["emb_dim"],
                                                                    context_length=cfg["context_length"],
                                                                    num_heads=cfg["n_heads"],
                                                                    dropout=cfg["drop_rate"],
                                                                    qkv_bias=cfg["qkv_bias"])
        self.feed_forward_block = FeedForwardBlock(cfg)
        self.dropout_layer = nn.Dropout(cfg["drop_rate"])
        self.layer_norm = nn.LayerNorm(cfg["emb_dim"])


    def forward(self, X):
        out = X

        X = self.layer_norm(X)
        X = self.causal_multi_head_attention(X)
        X = self.dropout_layer(X) 
        X = X + out

        out = X
        X = self.layer_norm(X)
        X = self.feed_forward_block(X)
        X = self.dropout_layer(X)
        X = X + out

        return X
    
'''
This is the GPT Block implemented as a class
'''

class GPTBlock(nn.Module):

    def __init__(self,cfg, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.token_embeddings = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_embeddings = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.emb_dropout = nn.Dropout(cfg["drop_rate"])
        self.transformer_blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.final_norm = nn.LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, X):

        '''
        X is a minibatch of size (batch_size, seq_len)
        '''
        batch_size, seq_len = X.shape
        token_embed = self.token_embeddings(X)
        pos_embed = self.pos_embeddings(torch.arange(seq_len, device=X.device))
        out = token_embed + pos_embed
        out = self.emb_dropout(out)
        out = self.transformer_blocks(out)
        out = self.final_norm(out)
        logits = self.out_head(out)

        return logits

