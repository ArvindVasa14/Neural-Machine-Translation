import torch
from torch import nn
import torch.nn.functional as F

class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.Wa= nn.Linear(hidden_size, hidden_size)
        self.Ua= nn.Linear(hidden_size, hidden_size)
        self.Va= nn.Linear(hidden_size, 1)

    def forward(self, query, keys):
        scores= self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
        # print(scores.size())
        scores= scores.squeeze(2).unsqueeze(1)
        # print(scores.size())
        weights= F.softmax(scores, dim=-1)
        context= torch.bmm(weights, keys)

        return context, weights


if __name__=="__main__":
    att= BahdanauAttention(512)
    query= torch.rand((10, 1, 512))
    keys= torch.rand((10, 30, 512))
    con, wei= att(query, keys)
    print(con.size(), wei.size())