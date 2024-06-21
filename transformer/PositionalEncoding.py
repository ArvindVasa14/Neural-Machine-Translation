import torch
from torch import nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, dropout_p, max_len):
        super(PositionalEncoding, self).__init__()
        self.dropout= nn.Dropout(dropout_p)

        pos_encoding= torch.zeros(max_len, dim_model)
        positions_list= torch.arange(0, max_len, dtype=torch.float).view(-1, 1)
        division_term= torch.exp(torch.arange(0, dim_model, 2).float() * (-math.log(10000.0))/dim_model)

        pos_encoding[:, 0::2]= torch.sin(positions_list * division_term)

        pos_encoding[:, 1::2]= torch.cos(positions_list * division_term)

        pos_encoding= pos_encoding.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pos_encoding", pos_encoding)

    def forward(self, token_embedding: torch.tensor)-> torch.tensor:
        return self.dropout(token_embedding + self.pos_encoding[:token_embedding.size(0),:])


if __name__=="__main__":
    pos= PositionalEncoding(256, 0.2, 30)
    t= torch.randint(0, 2048, (10, 30, 256))
    print(pos(t).shape)