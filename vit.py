import torch
import torch.nn as nn
from Encoder import Encoder

class VisionTransformer(nn.Module):
    def __init__(self, input_shape, patch_size, dim, num_class, num_enc_layers, num_head, dropout_rate, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.channel, self.height, self.width = input_shape
        self.h_patches = self.height // patch_size
        self.w_patches = self.width // patch_size

        self.patch_size = patch_size
        patch_length = self.channel * (patch_size ** 2)
        self.fc1 = nn.Linear(patch_length, dim)

        self.class_token = nn.Parameter(torch.zeros(1, 1, dim))

        self.pos_enc = nn.Parameter(torch.zeros(1, self.h_patches * self.w_patches + 1, dim))

        self.encoder_layers = nn.ModuleList([Encoder(dim, num_head, dropout_rate) for _ in range(num_enc_layers)])

        self.layer_norm = nn.LayerNorm(dim)

        self.fc2 = nn.Linear(dim, num_class)

    def forward(self, x: torch.Tensor):
        bs, c, h, w = x.shape

        x = x.view(bs, c, (self.h_patches), self.patch_size, (self.w_patches), self.patch_size)
        x = x.permute(0, 2, 4, 1, 3, 5)
        x = x.reshape(bs, self.h_patches * self.w_patches, -1)

        x = self.fc1(x)
        class_token = self.class_token.expand(bs, -1, -1)
        x = torch.cat((class_token, x), dim=1)
        x += self.pos_enc

        for layer in self.encoder_layers:
            x = layer(x)

        x = self.layer_norm(x)
        x_class_token = x[:,0]
        x = self.fc2(x_class_token)

        return x
        
if __name__ == '__main__':
    bs, c, h, w = 32, 3, 28, 28
    input_shape = (c, h, w)
    patch_size = 14
    dim = 512
    num_calss = 10
    num_enc_layers = 3
    num_head = 4
    dropout_rate = 0.3

    ViT = VisionTransformer(input_shape, patch_size, dim, num_calss, num_enc_layers, num_head, dropout_rate)
    x = torch.rand(bs, *input_shape)
    print(ViT(x).shape)