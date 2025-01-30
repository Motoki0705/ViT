import torch
import torch.nn as nn 

class FFN(nn.Module):
    def __init__(self, dim, dropout_rate, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.fc1 = nn.Linear(dim, dim*4)
        self.fc2 = nn.Linear(dim*4, dim)

        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)

        return x
    