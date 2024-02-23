import torch
import torch.nn as nn
import torch.nn.functional as F

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
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x, m):
        y = self.masked_multi_head_attention(x, x, x, m) + x
        y = self.layer_norm(y)
        y = self.feed_forward(y) + y
        y = self.layer_norm(y)
        return y
    
class DecoderLayer(nn.Module):
    def __init__(self, d_model, h):
        super(DecoderLayer, self).__init__()
        self.masked_multi_head_attention = MaskedMultiHeadAttention(d_model, h)
        self.multi_head_attention = MultiHeadAttention(d_model, h)
        self.feed_forward = MLP(d_model, int(d_model*4), d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x_1, x_2, m):
        y = self.masked_multi_head_attention(x_1, x_1, x_1, m) + x_1
        y = self.layer_norm(y)
        y = self.multi_head_attention(y, x_2, x_2) + y
        y = self.layer_norm(y)
        y = self.feed_forward(y) + y
        y = self.layer_norm(y)
        return y
    
# Batches of identical size only
class TransformerModel(nn.Module):
    def __init__(self, d_model, h, l, num_tokens):
        super(TransformerModel, self).__init__()
        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(d_model, h) for i in range(l)]
        )
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(d_model, h) for i in range(l)]
        )
        self.embeddings = torch.randn(num_tokens, d_model)
        self.linear = nn.Linear(in_features=d_model, out_features=num_tokens)
        self.activation = nn.Softmax(dim=-1)
    
    def forward(self, input_tokens, output_tokens=None):
        # embedding
        input_seq = self.embeddings[input_tokens]
        if output_tokens is not None:
            mask = torch.zeros_like(input_tokens)
            mask[:, :output_tokens.size()[-1]] = 1
            i = (1-mask).flatten().nonzero()[0]

            output_seq = self.embeddings[output_tokens]
            temp = torch.zeros_like(input_seq)
            temp[:output_seq.size()[0], :output_seq.size()[1]] = output_seq
            output_seq = temp

            temp = torch.zeros_like(input_tokens)
            temp[:output_tokens.size()[0], :output_tokens.size()[1]] = output_tokens
            output_tokens = temp

        else:
            output_seq = torch.randn_like(input_seq)
            output_tokens = torch.empty_like(input_tokens)

        """
        if output_tokens is not None:
            output_seq = self.embeddings[output_seq]
            mask = torch.zeros(input_seq.size()[:-1])
            mask[:, :output_seq.size()[-2]] = 1
            temp = torch.zeros_like(input_seq)
            temp[:output_seq.size()[0], :output_seq.size()[1]] = output_seq
            output_seq = temp
            i = (1-mask).flatten().nonzero()[0]
        else:
            output_seq = torch.randn_like(input_seq)"""

        print(input_seq.size())
        print(output_seq.size())
        print(output_seq[:, :, 0])
        print(output_tokens)
        print(mask)
        print(i)

        """
        input = input_seq
        output = output_seq
        
        if not output:
            output = torch.randn_like(input)
        if not mask:
            mask = torch.zeros(input.size()[:-1])
            i = 0
        else:
            i = mask.flatten().nonzero()[0]


        encoder_hidden_states = []
        for encoder_layer in self.encoder_layers:
            input = encoder_layer(input)
            encoder_hidden_states.append(input)

        for j, decoder_layer in enumerate(self.decoder_layers):
            print(output.size())
            output = decoder_layer(
                output,
                encoder_hidden_states[j],
                mask
            )
        output = self.linear(output)
        probabilities = self.activation(output)
        print(probabilities.size())
        return probabilities"""

    
if __name__ == "__main__":
    
    """a = torch.randn(4, 32, 512)
    enc = EncoderLayer(512, 8)
    m = torch.zeros(4, 32)
    m[:, :12] = 1
    b = enc(a, m)""" 
    """
    print(b.shape)
    dec = DecoderLayer(512, 8)
    mask = torch.zeros(a.size(0), a.size(1))
    mask[:, :10] = 1
    c = dec(a, b, mask)
    print(c.size())"""

    """#input = torch.randn(4, 32, 512)
    model = TransformerModel(512, 8, 6, 30000)
    #model(input)

    input_seq = torch.randint(10, (4, 32))
    output_seq = torch.randint(10, (4, 12))

    model(input_seq, output_seq)"""

