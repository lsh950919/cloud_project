import math
import torch
from torch import nn
from typing import Literal

class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim, hidden_dim = 64, num_heads = 4):
        super(MultiHeadAttention, self).__init__()
        assert self.hidden_dim % num_heads == 0, 'Projection dimension must be divisible by the number of heads'
        # self.input_dim = actor_emb_dim + critic_emb_dim
        self.model_dim = model_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = self.hidden_dim // self.num_heads 

        self.Wv = nn.Linear(self.model_dim, self.hidden_dim)
        self.Wk = nn.Linear(self.model_dim, self.hidden_dim)
        self.Wq = nn.Linear(self.model_dim, self.hidden_dim)
        self.Wo = nn.Linear(self.hidden_dim, self.hidden_dim)
        
        self.softmax = nn.Softmax(dim = -1) # source에 대한 softmax

        self._initialize_parameters()

    def _initialize_parameters(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
    
    def dot_product_attention(self, Q, K, V = None, mask = None):
        attn_score = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_score = self.softmax(attn_score)
        if V is not None:
            attn_score = torch.matmul(attn_score, V)
        if mask is not None:
            attn_score = attn_score.masked_fill(mask, -1e9)
        return attn_score
    
    def split_heads(self, x):
        batch_size, num_grids, _ = x.size()
        return x.view(batch_size, num_grids, self.num_heads, self.head_dim).transpose(1, 2)
    
    def combine_heads(self, x):
        batch_size, _, num_grids, _ = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, num_grids, self.hidden_dim)
    
    def forward(self, input, cross_input = None, mask = None):
        Q = self.split_heads(self.Wq(input))
        
        if cross_input is not None:
            K = self.split_heads(self.Wk(cross_input))
            V = self.split_heads(self.Wv(cross_input))
        else:
            K = self.split_heads(self.Wk(input))
            V = self.split_heads(self.Wv(input))

        attn_score = self.dot_product_attention(Q, K, V, mask)
        attn_output = self.Wo(self.combine_heads(attn_score))

        return attn_output


class PositionWiseFeedForward(nn.Module):
    def __init__(self, model_dim, hidden_dim):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(model_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, model_dim)
    
    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))
    

class PositionalEncoding(nn.Module):
    def __init__(self, seq_len, model_dim):
        super(PositionalEncoding, self).__init__()
        
        encoding = torch.zeros(seq_len, model_dim)
        position = torch.arange(0, seq_len, dtype = torch.float).unsqueeze(1) # seq_len, 1
        
        # div_term = torch.arange(0, model_dim, 2) * 1/10000**model_dim # x = pos/10000^(2i/model_dim), sin(x) for even, cos(x) for odd
        # # another example
        div_term = torch.exp(torch.arange(0, model_dim, 2, dtype = torch.float) * -(math.log(10000) / model_dim))
        
        encoding[:, ::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('encoding', encoding.unsqueeze(0))
    
    def forward(self, x):
        return x + self.encoding[:, :x.size(1)]


class EncoderLayer(nn.Module):
    def __init__(self, model_dim, num_heads, hidden_dim, dropout_rate):
        super(EncoderLayer, self).__init__()
        self.attn = MultiHeadAttention(model_dim, hidden_dim, num_heads)
        self.ff = PositionWiseFeedForward(model_dim, hidden_dim)
        self.layer_norm1 = nn.LayerNorm(model_dim)
        self.layer_norm2 = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, mask):
        attn_output = self.attn(x, mask = mask)
        x = self.layer_norm1(x + self.dropout(attn_output)) # residual + dropout
        ff_output = self.ff(attn_output)
        x = self.layer_norm2(x + self.dropout(ff_output))
        return x
    

class DecoderLayer(nn.Module):
    def __init__(self, model_dim, num_heads, hidden_dim, dropout_rate):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(model_dim, hidden_dim, num_heads)
        self.cross_attn = MultiHeadAttention(model_dim, hidden_dim, num_heads)
        self.ff = PositionWiseFeedForward(model_dim, hidden_dim)
        self.layer_norm1 = nn.LayerNorm(model_dim)
        self.layer_norm2 = nn.LayerNorm(model_dim)
        self.layer_norm3 = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, enc_output, src_mask, tgt_mask):
        attn_output = self.self_attn(x, mask = tgt_mask)
        x = self.layer_norm1(x + self.dropout(attn_output)) # residual + dropout
        attn_output = self.cross_attn(x, enc_output, src_mask)
        x = self.layer_norm2(x + self.dropout(attn_output))
        ff_output = self.ff(x)
        x = self.layer_norm3(x + self.dropout(ff_output))
        return x


PossibleMode = Literal['encoder', 'decoder', 'both']
# TODO: add masking for decoder
class TimeSeriesTransformer(nn.Module):
    def __init__(self, id_unique, sub_unique, dep_unique, dec_input, hidden_dim, input_len = 12, n_heads = 4, n_layers = 8, dropout_rate = 0.15):
        super().__init__()
        # embedding
        self.id_emb = nn.Embedding(id_unique, hidden_dim)
        self.sub_emb = nn.Embedding(sub_unique, hidden_dim)
        self.dep_emb = nn.Embedding(dep_unique, hidden_dim)
        self.cat_emb = nn.Embedding(3, hidden_dim)
        self.core_emb = nn.Embedding(5, hidden_dim)
        self.ram_emb = nn.Embedding(14, hidden_dim)

        self.encoder_input = nn.Linear(hidden_dim, hidden_dim)
        self.decoder_input = nn.Linear(dec_input, hidden_dim)
        self.positional_encoding_enc = PositionalEncoding(hidden_dim, hidden_dim)
        self.positional_encoding_dec = PositionalEncoding(dec_input, hidden_dim)

        self.encoder_layers = nn.ModuleList([EncoderLayer(hidden_dim, n_heads, hidden_dim, dropout_rate) for _ in range(n_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(hidden_dim, n_heads, hidden_dim, dropout_rate) for _ in range(n_layers)])

        self.cpu_layer = nn.ModuleList([
                                        nn.Linear(hidden_dim, hidden_dim),
                                        nn.Dropout(0.1),
                                        nn.Linear(hidden_dim, 4),
                                        ])
        self.stability_layer = nn.ModuleList([
                                            nn.Linear(hidden_dim, hidden_dim),
                                            nn.Dropout(0.1),
                                            nn.Linear(hidden_dim, 2)
                                            ])
        
        self.forecast_layer = nn.ModuleList([
                                            nn.Linear(hidden_dim, hidden_dim),
                                            nn.Dropout(0.1),
                                            nn.Linear(hidden_dim, 1)
                                            ])
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, vm_vec, cpu_vec):
        assert len(vm_vec) == 6
        vm_id, sub_id, dep_id, vm_cat, num_cores, ram = vm_vec
        vm_vec = torch.cat([self.id_emb(vm_id), self.sub_emb(sub_id), self.dep_emb(dep_id),
                            vm_cat, num_cores, ram], dim = 0)
        
        enc_vec = self.dropout(self.positional_encoding_enc(self.encoder_input(vm_vec)))
        dec_vec = self.dropout(self.positional_encoding_dec(self.decoder_input(cpu_vec)))
    
        for enc_layer in self.encoder_layers:
            enc_vec = enc_layer(enc_vec)
        
        cpu_output = self.cpu_layer(enc_vec)
        stability_output = self.stability_layer(enc_vec)
        
        for dec_layer in self.decoder_layers:
            dec_vec = dec_layer(dec_vec, enc_vec)

        dec_output = self.forecast_layer(dec_vec)

        return cpu_output, stability_output, dec_output

class VMEncoder(nn.Module):
    def __init__(self, id_unique, sub_unique, dep_unique, hidden_dim, n_heads = 4, n_layers = 8, dropout_rate = 0.15):
        super().__init__()
        # embedding
        self.id_emb = nn.Embedding(id_unique, hidden_dim)
        self.sub_emb = nn.Embedding(sub_unique, hidden_dim)
        self.dep_emb = nn.Embedding(dep_unique, hidden_dim)
        self.cat_emb = nn.Embedding(3, hidden_dim)
        self.core_emb = nn.Embedding(5, hidden_dim)
        self.ram_emb = nn.Embedding(14, hidden_dim)

        self.encoder_input = nn.Linear(hidden_dim, hidden_dim)
        self.positional_encoding_enc = PositionalEncoding(hidden_dim, hidden_dim)

        self.encoder_layers = nn.ModuleList([EncoderLayer(hidden_dim, n_heads, hidden_dim, dropout_rate) for _ in range(n_layers)])

        self.cpu_layer = nn.ModuleList([
                                        nn.Linear(hidden_dim, hidden_dim),
                                        nn.Dropout(0.1),
                                        nn.Linear(hidden_dim, 4),
                                        ])
        self.stability_layer = nn.ModuleList([
                                            nn.Linear(hidden_dim, hidden_dim),
                                            nn.Dropout(0.1),
                                            nn.Linear(hidden_dim, 2)
                                            ])
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, vm_vec):
        assert len(vm_vec) == 6
        vm_id, sub_id, dep_id, vm_cat, num_cores, ram = vm_vec
        vm_vec = torch.cat([self.id_emb(vm_id), self.sub_emb(sub_id), self.dep_emb(dep_id),
                            vm_cat, num_cores, ram], dim = 0)
        
        enc_vec = self.dropout(self.positional_encoding_enc(self.encoder_input(vm_vec)))
    
        for enc_layer in self.encoder_layers:
            enc_vec = enc_layer(enc_vec)
        
        cpu_output = self.cpu_layer(enc_vec)
        stability_output = self.stability_layer(enc_vec)

        return cpu_output, stability_output
    

class ForecastDecoder(nn.Module):
    def __init__(self, dec_input, hidden_dim, input_len = 12, n_heads = 4, n_layers = 8, dropout_rate = 0.15):
        super().__init__()
        # embedding
        self.decoder_input = nn.Linear(dec_input, hidden_dim)
        self.positional_encoding_dec = PositionalEncoding(dec_input, hidden_dim)

        self.decoder_layers = nn.ModuleList([DecoderLayer(hidden_dim, n_heads, hidden_dim, dropout_rate) for _ in range(n_layers)])
        
        self.forecast_layer = nn.ModuleList([
                                            nn.Linear(hidden_dim, hidden_dim),
                                            nn.Dropout(0.1),
                                            nn.Linear(hidden_dim, 1)
                                            ])
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, cpu_vec, enc_vec = None):
        dec_vec = self.dropout(self.positional_encoding_dec(self.decoder_input(cpu_vec)))
        enc_vec = dec_vec.clone() if enc_vec is None else enc_vec
        
        for dec_layer in self.decoder_layers:
            dec_vec = dec_layer(dec_vec, enc_vec)

        dec_output = self.forecast_layer(dec_vec)

        return dec_output


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, model_dim, num_heads, hidden_dim, num_layers, seq_len, dropout_rate):
        super(Transformer, self).__init__()
        # embedding for words 
        self.encoder_embedding = nn.Embedding(src_vocab_size, model_dim)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, model_dim)
        self.positional_encoding = PositionalEncoding(model_dim, seq_len)

        self.encoder_layers = nn.ModuleList([EncoderLayer(model_dim, num_heads, hidden_dim, dropout_rate) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(model_dim, num_heads, hidden_dim, dropout_rate) for _ in range(num_layers)])

        self.fc = nn.Linear(model_dim, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout_rate)
    
    def generate_mask(self, src, tgt):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
        seq_length = tgt.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
        tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask

    def forward(self, src, tgt):
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src)))
        tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tgt)))

        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)

        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)

        output = self.fc(dec_output)
        return output


