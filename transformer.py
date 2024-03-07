import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import OpusTranslationDataset

class MLP(nn.Module):
    def __init__(self, size_in, size_hidden, size_out):
        super(MLP, self).__init__()
        self.linear_1 = nn.Linear(in_features=size_in, out_features=size_hidden)
        self.linear_2 = nn.Linear(in_features=size_hidden, out_features=size_out)
        self.activation = nn.ReLU()

    def forward(self, x):
        y = self.linear_1(x)
        y = self.activation(y)
        y = self.linear_2(y)
        return y


class AttentionHead(nn.Module):
    def __init__(self, d_model, h):
        super(AttentionHead, self).__init__()
        self.linear_queries = nn.Linear(in_features=d_model, out_features=int(d_model/h))
        self.linear_keys = nn.Linear(in_features=d_model, out_features=int(d_model/h))
        self.linear_values = nn.Linear(in_features=d_model, out_features=int(d_model/h))
        self.activation = nn.Softmax(dim=-1)
        # Q.size() = (T, D)
        # K.size() = (T', D)
        # V.size() = (T', D')
        # A.size() = (T, T')
        # Y.size() = (T, D')

    def forward(self, q, k, v):
        q = self.linear_queries(q)
        k = self.linear_keys(k)
        v = self.linear_values(v)
        a = self.activation(torch.matmul(q, k.transpose(-1, -2)).div(q.size(-1)**(0.5)))
        y = torch.matmul(a, v)
        return y
    

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, h):
        super(MultiHeadAttention, self).__init__()
        self.attention_heads = nn.ModuleList(
            [AttentionHead(d_model, h) for i in range(h)]
        )
        self.linear = nn.Linear(in_features=d_model, out_features=d_model)

    def forward(self, v, k, q):
        y = [head(q, k, v) for head in self.attention_heads]
        y = torch.concat(y, axis=-1)
        y = self.linear(y)
        return y
    

class MaskedAttentionHead(nn.Module):
    def __init__(self, d_model, h):
        super(MaskedAttentionHead, self).__init__()
        self.linear_queries = nn.Linear(in_features=d_model, out_features=int(d_model/h))
        self.linear_keys = nn.Linear(in_features=d_model, out_features=int(d_model/h))
        self.linear_values = nn.Linear(in_features=d_model, out_features=int(d_model/h))
        self.activation = nn.Softmax(dim=-1)
        
    def forward(self, q, k, v, m):
        q = self.linear_queries(q)
        k = self.linear_keys(k)
        v = self.linear_values(v)
        a = torch.matmul(q, k.transpose(-1, -2)).div(q.size(-1)**(0.5))
        a = a * m.unsqueeze(-1)
        a = self.activation(a)
        y = torch.matmul(a, v)
        return y
    

class MaskedMultiHeadAttention(nn.Module):
    def __init__(self, d_model, h):
        super(MaskedMultiHeadAttention, self).__init__()
        self.attention_heads = nn.ModuleList(
            [MaskedAttentionHead(d_model, h) for i in range(h)]
        )
        self.linear = nn.Linear(in_features=d_model, out_features=d_model)

    def forward(self, v, k, q, m):
        y = [head(q, k, v, m) for head in self.attention_heads]
        y = torch.concat(y, axis=-1)
        y = self.linear(y)
        return y
    

class EncoderLayer(nn.Module):
    def __init__(self, d_model, h):
        super(EncoderLayer, self).__init__()
        self.masked_multi_head_attention = MaskedMultiHeadAttention(d_model, h)
        self.feed_forward = MLP(d_model, int(d_model*4), d_model)
        self.dropout = nn.Dropout(p=0.1)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x, m):
        y0 = self.masked_multi_head_attention(x, x, x, m)
        y1 = self.dropout(y0) + x
        y2 = self.layer_norm(y1)
        y3 = self.feed_forward(y2)
        y4 = self.dropout(y3) + y2
        y5 = self.layer_norm(y4)
        return y5
    

class DecoderLayer(nn.Module):
    def __init__(self, d_model, h):
        super(DecoderLayer, self).__init__()
        self.masked_multi_head_attention = MaskedMultiHeadAttention(d_model, h)
        self.multi_head_attention = MultiHeadAttention(d_model, h)
        self.feed_forward = MLP(d_model, int(d_model*4), d_model)
        self.dropout = nn.Dropout(p=0.1)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x0, x1, m):
        y0 = self.masked_multi_head_attention(x0, x0, x0, m)
        y1 = self.dropout(y0) + x0
        y2 = self.layer_norm(y1)
        y3 = self.multi_head_attention(y2, x1, x1)
        y4 = self.dropout(y3) + y2
        y5 = self.layer_norm(y4)
        y6 = self.feed_forward(y5)
        y7 = self.dropout(y6) + y5
        y8 = self.layer_norm(y7)
        return y8
    

class EmbeddingLayer(nn.Module):
    def __init__(self, d_model, num_tokens):
        super(EmbeddingLayer, self).__init__()
        #PE(pos, 2i) = sin(pos/10000**(2i/d_model))
        #PE(pos, 2i+1) = cos(pos/10000**(2i/d_model))
        self.token_embeddings = nn.Parameter(torch.randn(num_tokens, d_model).mul(d_model**0.5), requires_grad=True)
        self.sine = lambda pos, j : torch.sin(pos.div(torch.tensor(10000).pow(j/d_model)))
        self.cosine = lambda pos, j : torch.cos(pos.div(torch.tensor(10000).pow(j/d_model)))
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        y1 = self.token_embeddings[x]
        y2 = torch.zeros_like(y1).to(dtype=y1.dtype)
        pos = torch.arange(y1.size(1))
        for i in range(int(y1.size(2)/2)):
            y2[:, :, 2*i] = self.sine(pos, 2*i)
            y2[:, :, 2*i+1] = self.cosine(pos, 2*i)
        y3 = self.dropout(y1.add(y2))
        return y3
    

class TransformerModel(nn.Module):
    def __init__(self, d_model, h, l, num_tokens):
        super(TransformerModel, self).__init__()
        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(d_model, h) for i in range(l)]
        )
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(d_model, h) for i in range(l)]
        )
        self.embedding_layer = EmbeddingLayer(d_model, num_tokens)
        self.linear = nn.Linear(in_features=d_model, out_features=num_tokens)
        self.activation = nn.Softmax(dim=-1)
    
    def forward(self, input_ids, input_mask):
        input_seq = self.embedding_layer(input_ids)
        output_ids = torch.zeros_like(input_ids).int()
        output_seq = torch.zeros_like(input_seq).to(dtype=input_seq.dtype)
        output_mask = torch.zeros_like(input_mask).to(dtype=input_seq.dtype)
        output_probs = torch.zeros((input_seq.size(0), input_seq.size(1), self.linear.out_features)).to(dtype=input_seq.dtype)
        encoder_hidden_states = []
        for encoder_layer in self.encoder_layers:
            input_seq = encoder_layer(input_seq, input_mask)
            encoder_hidden_states.append(input_seq)

        for i in range(output_ids.size(1)):
            for j, decoder_layer in enumerate(self.decoder_layers):
                output_seq = decoder_layer(
                    output_seq,
                    encoder_hidden_states[j],
                    output_mask
                )
            logits = self.linear(output_seq)
            probs = self.activation(logits)
            output_probs[:, i, :] = probs[:, i, :]
            output_ids[:, i] = probs[:, i, :].argmax(dim=-1)
            output_seq = self.embedding_layer(output_ids)
            output_mask[:, i] = 1

        return output_probs, output_ids

    
if __name__ == "__main__":

    dataset = OpusTranslationDataset(
        dataset_name="WikiMatrix",
        language_source="fr",
        language_target="it"
    )
    dataset.use_set("train")
    dataloader = DataLoader(dataset, batch_size=4)
    batch = next(iter(dataloader))
    net = TransformerModel(
        d_model=512, 
        h=8, 
        l=6, 
        num_tokens=dataset.vocab_size)
    
    #with torch.no_grad():
    #    probs, ids = net(batch[0], batch[1])

    #print(dataset.tokenizer.decode(ids[0].tolist()))
    
    parameters_count = 0
    for name, param in net.named_parameters():
        if param.requires_grad:
            parameters_count += param.numel()
    print(f"number of parameters in the model : {parameters_count}")


    layer = EmbeddingLayer(512, 30000)
    for name, param in layer.named_parameters():
        print(name)
        print(param.data.dtype)