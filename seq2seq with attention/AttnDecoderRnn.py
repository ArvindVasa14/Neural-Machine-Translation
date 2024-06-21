import torch
from torch import nn

from NMT.BahdanauAttention import BahdanauAttention

SOS_token=1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AttnDecoderRnn(nn.Module):
    def __init__(self,  hidden_size, vocab_size):
        super(AttnDecoderRnn, self).__init__()
        # self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.attention= BahdanauAttention(hidden_size)
        self.gru = nn.GRU(2 * hidden_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.relu = nn.ReLU()
        self.dropout= nn.Dropout(0.2)
        self.linear= nn.Linear(hidden_size, vocab_size)

    def forward_step(self, input, hidden, encoder_outputs):
        embeddings= self.dropout(self.embedding(input)) # (10, 30) => (10, 30, 256)
        query= hidden.permute(1, 0, 2)  # (batch, 1, 512) => (1, batch, 512)
        context, attn_weights= self.attention(query, encoder_outputs) # (1, batch, 512), (30, batch, 512)
        # print("sizes", embeddings.size(), context.size())
        input_gru= torch.cat((embeddings, context), dim=2)
        # print(input_gru.size(), hidden.size())
        output, hidden= self.gru(input_gru, hidden)
        output= self.linear(output)

        return output, hidden, attn_weights


    def forward(self, encoder_outputs, encoder_hidden, target_tensor= None):
        batch_size= encoder_outputs.size(0)
        decoder_input= torch.empty(batch_size, 1, dtype=torch.long, device= device).fill_(SOS_token)
        decoder_hidden= encoder_hidden
        decoder_outputs= []
        attentions= []

        MAX_LEN= target_tensor.size(1) if target_tensor is not None else 25

        for i in range(MAX_LEN):
            decoder_output, decoder_hidden, attn_weights = self.forward_step(decoder_input, decoder_hidden, encoder_outputs)
            decoder_outputs.append(decoder_output)

            if target_tensor is not None:
                decoder_input = target_tensor[:, i].unsqueeze(1)
            else:
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        
        return decoder_outputs


