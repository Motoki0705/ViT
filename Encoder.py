import torch
import torch.nn as nn 
from Attention import MHSA
from MLP import FFN

class Encoder(nn.Module):
    def __init__(self, dim, num_head, dropout_rate, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.MHSA = MHSA(dim, num_head, dropout_rate)
        self.FFN = FFN(dim, dropout_rate)
        self.layernorm_1 = nn.LayerNorm(dim)
        self.layernorm_2 = nn.LayerNorm(dim)

    def forward(self, x):
        residual = x
        x = self.layernorm_1(x)
        x = self.MHSA(x) + residual

        residual = x
        x = self.layernorm_2(x)
        x = self.FFN(x) + residual

        return x
    
if __name__ == '__main__':

    model = Encoder(128, 4, 0.3)
    x = torch.rand(32, 5, 128)

    print(model(x).shape)

    