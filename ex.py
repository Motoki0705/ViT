import ssl
import certifi

ssl._create_default_https_context = ssl._create_unverified_context

import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  # 32x32 の画像に 4ピクセルのパディングを加えてランダムクロップ
    transforms.RandomHorizontalFlip(p=0.5),  # 50% の確率で左右反転
    transforms.ToTensor(),  # Tensor に変換
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))  # 正規化
])

train_dataset = torchvision.datasets.CIFAR10('data', train=True, transform=transform_train, download=True)
eval_dataset = torchvision.datasets.CIFAR10('data', train=False, transform=transforms.ToTensor(), download=True)

train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
eval_dataloader = DataLoader(eval_dataset, batch_size=64, shuffle=False)

img, target = next(iter(train_dataloader))

out = torch.rand(64, 10)
criterion = nn.CrossEntropyLoss()
loss = criterion(out, target)

print(loss)