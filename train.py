import ssl
import certifi

ssl._create_default_https_context = ssl._create_unverified_context

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from vit import VisionTransformer

# データの前処理（トレーニング用）
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])

# データの前処理（テスト用）
transform_eval = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])

# CIFAR-10 データセット
train_dataset = torchvision.datasets.CIFAR10('data', train=True, transform=transform_train, download=True)
eval_dataset = torchvision.datasets.CIFAR10('data', train=False, transform=transform_eval, download=True)

# DataLoader
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
eval_dataloader = DataLoader(eval_dataset, batch_size=64, shuffle=False, num_workers=4)

# Vision Transformer モデル
ViT = VisionTransformer(
    input_shape=(3, 32, 32),
    patch_size=2,
    dim=512,
    num_class=10,
    num_enc_layers=3,
    num_head=4,
    dropout_rate=0.3
)

# デバイス設定 (CUDA or CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ViT.to(device)

# 損失関数 & 最適化手法
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(ViT.parameters(), lr=3e-4)  # 初期学習率

# 学習率スケジューラー (10エポックごとに学習率を 0.1倍)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# アーリーストッピングのパラメータ
early_stopping_patience = 5
best_acc = 0.0
epochs_no_improve = 0

# 訓練データの記録用
train_losses, eval_losses = [], []
train_accs, eval_accs = [], []

# 訓練 & 評価ループ
Epochs = 30
for epoch in range(Epochs):
    epoch_loss = 0
    correct, total = 0, 0

    ViT.train()
    for img, target in train_dataloader:
        img, target = img.to(device), target.to(device)

        # 順伝播
        out = ViT(img)
        train_loss = criterion(out, target)

        # 逆伝播
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        epoch_loss += train_loss.item()

        # 精度計算
        _, predicted = out.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

    train_acc = 100. * correct / total
    train_losses.append(epoch_loss / len(train_dataloader))
    train_accs.append(train_acc)

    print(f"[Epoch {epoch+1}/{Epochs}] Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_acc:.2f}%")

    # **評価（検証）**
    ViT.eval()
    eval_loss = 0
    correct, total = 0, 0

    with torch.no_grad():
        for img, target in eval_dataloader:
            img, target = img.to(device), target.to(device)

            out = ViT(img)
            loss = criterion(out, target)
            eval_loss += loss.item()

            # 精度計算
            _, predicted = out.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    eval_acc = 100. * correct / total
    eval_losses.append(eval_loss / len(eval_dataloader))
    eval_accs.append(eval_acc)

    print(f"[Epoch {epoch+1}/{Epochs}] Eval Loss: {eval_losses[-1]:.4f}, Eval Acc: {eval_acc:.2f}%")

    # **最高精度のモデルを保存**
    if eval_acc > best_acc:
        best_acc = eval_acc
        epochs_no_improve = 0
        torch.save(ViT.state_dict(), "best_model.pth")
        print(f"✅ Model saved with Accuracy: {best_acc:.2f}%")
    else:
        epochs_no_improve += 1
        print(f"⚠️ No improvement for {epochs_no_improve} epoch(s)")

    # **アーリーストッピング**
    if epochs_no_improve >= early_stopping_patience:
        print("⏹️ Early Stopping triggered! Stopping training.")
        break

    # 学習率を更新
    scheduler.step()

print("🎉 Training Complete!")

# **損失 & 精度の推移を保存**
import pandas as pd
history = pd.DataFrame({
    "train_loss": train_losses,
    "eval_loss": eval_losses,
    "train_acc": train_accs,
    "eval_acc": eval_accs
})
history.to_csv("training_history.csv", index=False)
print("📊 Training history saved to training_history.csv")

# **損失と精度の推移をプロット**
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label="Train Loss")
plt.plot(eval_losses, label="Eval Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss Curve")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accs, label="Train Acc")
plt.plot(eval_accs, label="Eval Acc")
plt.xlabel("Epochs")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy Curve")
plt.legend()

plt.savefig("training_curve.png")
plt.show()
