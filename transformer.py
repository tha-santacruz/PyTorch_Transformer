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
        self.linear_queries = nn.Linear(in_features=d_model, out_features=int(d_model/h), bias=False)
        self.linear_keys = nn.Linear(in_features=d_model, out_features=int(d_model/h), bias=False)
        self.linear_values = nn.Linear(in_features=d_model, out_features=int(d_model/h), bias=False)
        nn.init.xavier_normal_(self.linear_queries.weight)
        nn.init.xavier_normal_(self.linear_keys.weight)
        nn.init.xavier_normal_(self.linear_values.weight)
        self.activation = nn.Softmax(dim=-1)

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
        nn.init.xavier_normal_(self.linear.weight)

    def forward(self, v, k, q):
        y = [head(q, k, v) for head in self.attention_heads]
        y = torch.cat(y, axis=-1)
        y = self.linear(y)
        return y
    

class MaskedAttentionHead(nn.Module):
    def __init__(self, d_model, h):
        super(MaskedAttentionHead, self).__init__()
        self.linear_queries = nn.Linear(in_features=d_model, out_features=int(d_model/h), bias=False)
        self.linear_keys = nn.Linear(in_features=d_model, out_features=int(d_model/h), bias=False)
        self.linear_values = nn.Linear(in_features=d_model, out_features=int(d_model/h), bias=False)
        nn.init.xavier_normal_(self.linear_queries.weight)
        nn.init.xavier_normal_(self.linear_keys.weight)
        nn.init.xavier_normal_(self.linear_values.weight)
        self.activation = nn.Softmax(dim=-1)

        # smarter, but works for self attention only
        # qkv = nn.Linear(in, out*3, no bias)
        # q, k, v = qkv(x).chunk(chunks=3, dim=-1)
        
    def forward(self, q, k, v, m):
        q = self.linear_queries(q)
        k = self.linear_keys(k)
        v = self.linear_values(v)
        a = torch.matmul(q, k.transpose(-1, -2)).div(q.size(-1)**(0.5))
        a = torch.mul(a, torch.unsqueeze(m, -1))
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
        nn.init.xavier_normal_(self.linear.weight)

    def forward(self, v, k, q, m):
        y = [head(q, k, v, m) for head in self.attention_heads]
        y = torch.cat(y, axis=-1)
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
        y1 = self.dropout(y0) + x #.mul(m.unsqueeze(dim=-1).unsqueeze(dim=-1))
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
        y1 = self.dropout(y0) + x0#.mul(m.unsqueeze(dim=-1).unsqueeze(dim=-1))
        y2 = self.layer_norm(y1)
        y3 = self.multi_head_attention(y2, x1, x1)
        y4 = self.dropout(y3) + y2
        y5 = self.layer_norm(y4)
        y6 = self.feed_forward(y5)
        y7 = self.dropout(y6) + y5
        y8 = self.layer_norm(y7)
        return y8

class TransformerModel(nn.Module):
    def __init__(self, d_model, h, l, num_tokens):
        super(TransformerModel, self).__init__()
        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(d_model, h) for i in range(l)]
        )
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(d_model, h) for i in range(l)]
        )
        self.linear = nn.Linear(in_features=d_model, out_features=num_tokens)
        nn.init.xavier_normal_(self.linear.weight)

        self.token_embeddings_matrix = self.linear.weight #nn.Parameter(torch.randn(num_tokens, d_model), requires_grad=True)
        self.token_embeddings_factor = d_model**0.5
        self.sine = lambda pos, j : torch.sin(pos.div(torch.tensor(10000).pow(j/d_model)))
        self.cosine = lambda pos, j : torch.cos(pos.div(torch.tensor(10000).pow(j/d_model)))
        self.dropout = nn.Dropout(p=0.1)
        self.activation = nn.Softmax(dim=-1)

    def get_position_embeddings(self, batch_size, sequence_length):
        pos_emb = torch.zeros(batch_size, sequence_length, self.token_embeddings_matrix.size(1))
        pos_emb = pos_emb.to(dtype=self.token_embeddings_matrix.dtype)
        pos = torch.arange(pos_emb.size(1))
        for i in range(int(pos_emb.size(2)/2)):
            pos_emb[:, :, 2*i] = self.sine(pos, 2*i)
            pos_emb[:, :, 2*i+1] = self.cosine(pos, 2*i)
        
        return pos_emb

    def forward_train(self, input_ids, input_mask, target_ids, target_mask):
        """Teacher forcing for next token prediction"""

        token_embeddings = self.token_embeddings_matrix[input_ids].mul(self.token_embeddings_factor)
        positional_embeddings = self.get_position_embeddings(input_mask.size(0), input_mask.size(1))
        input_seq = self.dropout(token_embeddings + positional_embeddings)


        encoder_hidden_states = []
        for encoder_layer in self.encoder_layers:
            input_seq = encoder_layer(input_seq, input_mask)
            encoder_hidden_states.append(input_seq)

        output_ids = input_ids.clone().detach()
        output_probs = F.one_hot(output_ids, num_classes=self.linear.out_features).to(dtype=input_seq.dtype)
        output_ids[:, 1:] = 0
        output_probs[:, 1:, :] = 0

        output_masks = torch.triu(torch.ones(output_ids.size(1), output_ids.size(1)-1)).T
        output_masks = output_masks.unsqueeze(dim=1).repeat([1, output_ids.size(0), 1]).to(dtype=input_seq.dtype)

        update_masks = torch.triu(torch.triu(torch.ones(output_ids.size(1), output_ids.size(1)-1)).T).flip([0, 1])
        update_masks = update_masks.unsqueeze(dim=1).repeat([1, output_ids.size(0), 1]).to(dtype=input_seq.dtype)

        
        for i in range(1, output_ids.size(1)):

            #output_mask = torch.zeros_like(input_mask).to(dtype=input_seq.dtype)
            #output_mask[:, :i] = 1
            output_mask = output_masks[i-1]
            
            known_token_embeddings = self.token_embeddings_matrix[target_ids].mul(self.token_embeddings_factor)
            known_token_embeddings = known_token_embeddings*output_mask.to(dtype=target_ids.dtype).unsqueeze(dim=-1)

            output_seq = self.dropout(known_token_embeddings + positional_embeddings)

            for j, decoder_layer in enumerate(self.decoder_layers):
                output_seq = decoder_layer(
                    output_seq,
                    encoder_hidden_states[j],
                    output_mask
                )

            output_logits = self.linear(output_seq)
            
            #update_mask = torch.zeros_like(input_ids)
            #update_mask[:, i] = 1
            update_mask = update_masks[i-1]
            
            output_probs = output_probs + torch.mul(update_mask.unsqueeze(dim=-1), self.activation(output_logits))
            output_ids = output_ids + torch.mul(update_mask, output_probs.argmax(dim=-1))

        return output_probs, output_ids
    
    def forward_eval(self, input_ids, input_mask):
        """Autoregressive output generation"""
        
        token_embeddings = self.token_embeddings_matrix[input_ids].mul(self.token_embeddings_factor)
        positional_embeddings = self.get_position_embeddings(input_mask.size(0), input_mask.size(1))
        input_seq = self.dropout(token_embeddings + positional_embeddings)

        encoder_hidden_states = []
        for encoder_layer in self.encoder_layers:
            input_seq = encoder_layer(input_seq, input_mask)
            encoder_hidden_states.append(input_seq)

        output_ids = input_ids.clone().detach()
        output_probs = F.one_hot(output_ids, num_classes=self.linear.out_features).to(dtype=input_seq.dtype)
        output_ids[:, 1:] = 0
        output_probs[:, 1:, :] = 0
        
        for i in range(1, output_ids.size(1)):
            
            output_mask = torch.zeros_like(input_mask).to(dtype=input_seq.dtype)
            output_mask[:, :i] = 1

            known_token_embeddings = self.token_embeddings_matrix[output_ids].mul(self.token_embeddings_factor)
            known_token_embeddings = known_token_embeddings*output_mask.to(dtype=output_ids.dtype).unsqueeze(dim=-1)

            output_seq = self.dropout(known_token_embeddings + positional_embeddings)

            for j, decoder_layer in enumerate(self.decoder_layers):
                output_seq = decoder_layer(
                    output_seq,
                    encoder_hidden_states[j],
                    output_mask
                )

            output_logits = self.linear(output_seq)
            
            update_mask = torch.zeros_like(input_ids)
            update_mask[:, i] = 1
            
            output_probs = output_probs + torch.mul(update_mask.unsqueeze(dim=-1), self.activation(output_logits))
            output_ids = output_ids + torch.mul(update_mask, output_probs.argmax(dim=-1))
            
        return output_probs, output_ids
    
    def forward(self, input_ids, input_mask, target_ids=None, target_mask=None):
        if self.training:
            #return self.forward_eval(input_ids, input_mask)
            return self.forward_train(input_ids, input_mask, target_ids, target_mask)
        
        else:
            return self.forward_eval(input_ids, input_mask)

    
if __name__ == "__main__":
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Cudnn setup
    #torch.autograd.set_detect_anomaly(True)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    # Batch
    dataset = OpusTranslationDataset(
        dataset_name="WikiMatrix",
        language_source="fr",
        language_target="it",
        vocab_size=30000,
        sequence_length=128
    )
    dataset.use_set("train")
    dataloader = DataLoader(dataset, batch_size=2)
    batch = next(iter(dataloader))

    # Net
    net = TransformerModel(
        d_model=512, 
        h=8,
        l=6, 
        num_tokens=dataset.vocab_size).to(device=device)
    
    # Float Precision
    for param in net.parameters():
            param.data = param.data.to(dtype=torch.float32)
            if param.grad is not None:
                param.grad.data = param.grad.data.to(dtype=torch.float32)
    

    parameters_count = 0
    for name, param in net.named_parameters():
        if param.requires_grad:
            parameters_count += param.numel()
    print(f"number of parameters in the model : {parameters_count}")

    print(batch[1].sum())
    print(dataset.tokenizer.decode(batch[0][0].tolist()))
    print(batch[0][0])

    input_ids = batch[0].to(dtype=torch.int64, device=device)
    input_mask = batch[1].to(dtype=torch.float32, device=device)
    target_ids = batch[2].to(dtype=torch.int64, device=device)
    target_mask = batch[3].to(dtype=torch.float32, device=device)

    net.train()

    probs, ids = net(
        input_ids, 
        input_mask,
        target_ids,
        target_mask 
        )

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1).to(device=device)
    loss = criterion(probs.flatten(), torch.randn_like(probs.flatten()).to(device=device))
    loss.backward()