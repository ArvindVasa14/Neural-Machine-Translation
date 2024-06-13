import torch
from torch import nn

class EncoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, dropout_rate= 0.2):
        super(EncoderRNN, self).__init__()
        self.embed_size= embed_size
        self.hidden_size= hidden_size
        self.vocab_size= vocab_size
        self.dropout_rate= dropout_rate

        self.embedding= nn.Embedding(vocab_size, embed_size)
        self.gru= nn.GRU(embed_size, hidden_size, batch_first=True)
        self.dropout= nn.Dropout(dropout_rate)

    def forward(self, source_vector):
        embeddings= self.dropout(self.embedding(source_vector))
        outputs, hidden= self.gru(embeddings)

        return outputs, hidden

if __name__=="__main__":
    enc= EncoderRNN(256, 512, 2048)
    t= torch.randint(0, 2048, (10, 30))
    out, hid= enc(t)
    print(out.shape, hid.shape)