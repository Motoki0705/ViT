import torch
import torch.nn as nn

class MHSA(nn.Module):
    def __init__(self, dim, num_head, dropout_rate, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dim = dim
        self.num_head = num_head

        self.scale = 1 / ((dim // num_head) ** 0.5)
        self.to_qkv = nn.Linear(dim, dim*3)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(dim, dim)

    def forward(self, x:torch.Tensor):
        bs, patch, dim = x.shape

        #(bs, 5, dim) -> (bs, 5, dim*3)
        qkv: torch.Tensor = self.to_qkv(x)

        #(bs, 5, dim*3) -> (bs, 5, 3, dim) -> (bs, 5, 3, num_head, dim//num_head) -> (3, bs, num_head, 5, dim//num_head)
        qkv_head = qkv.view(bs, patch, 3, self.num_head, dim // self.num_head).permute(2, 0, 3, 1, 4)

        #(bs, num_head, 5, dim//num_head) * 3
        query, key, value = torch.unbind(qkv_head, dim=0)

        attn = torch.matmul(query * self.scale, key.transpose(-2, -1))

        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        value = torch.matmul(attn, value)
        value = value.permute(0, 2, 1, 3).flatten(2) #(bs, 5, dim)

        return self.fc(value) #(bs, 5, dim) -> (bs, 5, dim)


if __name__ == '__main__':

    model = MHSA(dim=128, num_head=4)

    x = torch.rand(32, 5, 128)

    print(model(x).shape)




       
