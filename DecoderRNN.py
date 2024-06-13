import torch
from torch import nn
import torch.nn.functional as F

SOS_token = 0
EOS_token = 1
MAX_LEN = 25

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size):
        super(DecoderRNN, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.gru = nn.GRU(embed_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.relu = nn.ReLU()

    def forward_step(self, input, hidden):
        output = self.embedding(input)
        output = self.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.linear(output)
        return output, hidden

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        batch_size = encoder_outputs.size(0)
        decoder_input = t1 = torch.empty(batch_size, 1, dtype=torch.long).fill_(SOS_token).to(device)
        decoder_hidden = encoder_hidden.to(device)
        decoder_outputs = []

        MAX_LEN = target_tensor.size(1) if target_tensor is not None else 25

        for i in range(MAX_LEN):
            decoder_output, decoder_hidden = self.forward_step(decoder_input, decoder_hidden)
            decoder_outputs.append(decoder_output)

            if target_tensor is not None:
                decoder_input = target_tensor[:, i].unsqueeze(1)
            else:
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        return decoder_outputs


if __name__ == "__main__":
    decoder = DecoderRNN(256, 512, 2048).to(device)
    e_o = torch.randint(0, 2048, (10, 30, 512)).to(device)
    e_h = torch.randint(0, 2048, (1, 10, 512)).to(device).to(torch.float)
    d_o = decoder(e_o, e_h)
    print(d_o.size())
    outputs = torch.argmax(d_o, dim=2)
    print(outputs.size())
