"""
CS 240 - Lab 8 : Part 2
CIFAR-10 CNN Experiments (PyTorch)
Run this on Google Colab with GPU runtime enabled.
"""

# ─────────────────────────────────────────────
# 0. Imports & device
# ─────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ─────────────────────────────────────────────
# 1. Data – CIFAR-10
# ─────────────────────────────────────────────
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),        # data augmentation
    transforms.RandomHorizontalFlip(),           # data augmentation
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])

train_set = torchvision.datasets.CIFAR10(root='./data', train=True,  download=True, transform=train_transform)
test_set  = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True,  num_workers=2)
test_loader  = torch.utils.data.DataLoader(test_set,  batch_size=128, shuffle=False, num_workers=2)

classes = ('airplane','car','bird','cat','deer','dog','frog','horse','ship','truck')
print(f"Train size: {len(train_set)} | Test size: {len(test_set)}")

# ─────────────────────────────────────────────
# 2. Model Definitions
# ─────────────────────────────────────────────

# ── Model 1: Baseline Shallow CNN ──
class ShallowCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),   # 32→16
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),  # 16→8
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 512), nn.ReLU(),
            nn.Linear(512, 10)
        )
    def forward(self, x):
        return self.classifier(self.features(x))


# ── Model 2: Shallow CNN + Batch Normalization ──
class ShallowCNN_BN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 512), nn.ReLU(),
            nn.Linear(512, 10)
        )
    def forward(self, x):
        return self.classifier(self.features(x))


# ── Model 3: Deeper CNN (VGG-style, no skip connections) ──
class DeepCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3,   64, 3, padding=1), nn.BatchNorm2d(64),  nn.ReLU(),
            nn.Conv2d(64,  64, 3, padding=1), nn.BatchNorm2d(64),  nn.ReLU(),
            nn.MaxPool2d(2, 2),   # 32→16
            # Block 2
            nn.Conv2d(64,  128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2, 2),   # 16→8
            # Block 3
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2, 2),   # 8→4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, 10)
        )
    def forward(self, x):
        return self.classifier(self.features(x))


# ── Model 4: ResNet-style (with skip connections) ──
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)

        # Shortcut: match dimensions if needed
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)   # <── skip connection
        out = self.relu(out)
        return out


class ResNetSmall(nn.Module):
    def __init__(self):
        super().__init__()
        self.prep = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU()
        )
        self.layer1 = ResidualBlock(64,  64)
        self.layer2 = ResidualBlock(64,  128, stride=2)
        self.layer3 = ResidualBlock(128, 256, stride=2)
        self.layer4 = ResidualBlock(256, 512, stride=2)
        self.pool   = nn.AdaptiveAvgPool2d(1)   # Global Average Pooling
        self.fc     = nn.Linear(512, 10)

    def forward(self, x):
        x = self.prep(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# ─────────────────────────────────────────────
# 3. Training & Evaluation helpers
# ─────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        correct    += (outputs.argmax(1) == labels).sum().item()
        total      += images.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)
        correct    += (outputs.argmax(1) == labels).sum().item()
        total      += images.size(0)
    return total_loss / total, correct / total


def train_model(model, num_epochs=20, lr=1e-3):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(1, num_epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        va_loss, va_acc = evaluate(model, test_loader, criterion)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(va_acc)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:2d}/{num_epochs} | "
                  f"Train Loss: {tr_loss:.4f}  Acc: {tr_acc*100:.2f}% | "
                  f"Val Loss: {va_loss:.4f}  Acc: {va_acc*100:.2f}%")

    return model, history


def plot_history(history, title):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, history["train_loss"], label="Train Loss")
    ax1.plot(epochs, history["val_loss"],   label="Val Loss")
    ax1.set_title(f"{title} — Loss"); ax1.set_xlabel("Epoch"); ax1.legend()

    ax2.plot(epochs, [a*100 for a in history["train_acc"]], label="Train Acc")
    ax2.plot(epochs, [a*100 for a in history["val_acc"]],   label="Val Acc")
    ax2.set_title(f"{title} — Accuracy"); ax2.set_xlabel("Epoch"); ax2.set_ylabel("%"); ax2.legend()

    plt.tight_layout(); plt.show()
    print(f"  → Final Val Accuracy: {history['val_acc'][-1]*100:.2f}%\n")


# ─────────────────────────────────────────────
# 4. Run all experiments
# ─────────────────────────────────────────────

EPOCHS = 20   # increase to 30-40 for better accuracy on Colab GPU

print("\n" + "="*60)
print("MODEL 1: Baseline Shallow CNN")
print("="*60)
m1, h1 = train_model(ShallowCNN(), num_epochs=EPOCHS)
plot_history(h1, "Model 1: Shallow CNN")

print("\n" + "="*60)
print("MODEL 2: Shallow CNN + Batch Normalization")
print("="*60)
m2, h2 = train_model(ShallowCNN_BN(), num_epochs=EPOCHS)
plot_history(h2, "Model 2: Shallow CNN + BN")

print("\n" + "="*60)
print("MODEL 3: Deep CNN (no skip connections)")
print("="*60)
m3, h3 = train_model(DeepCNN(), num_epochs=EPOCHS)
plot_history(h3, "Model 3: Deep CNN")

print("\n" + "="*60)
print("MODEL 4: ResNet-style (with skip connections)")
print("="*60)
m4, h4 = train_model(ResNetSmall(), num_epochs=EPOCHS)
plot_history(h4, "Model 4: ResNet (skip connections)")

# ─────────────────────────────────────────────
# 5. Summary comparison plot
# ─────────────────────────────────────────────
labels  = ["Shallow CNN", "Shallow+BN", "Deep CNN", "ResNet"]
val_acc = [h["val_acc"][-1]*100 for h in [h1, h2, h3, h4]]

plt.figure(figsize=(8, 5))
bars = plt.bar(labels, val_acc, color=["steelblue","orange","green","red"])
plt.bar_label(bars, fmt="%.1f%%", padding=3)
plt.ylim(0, 100); plt.ylabel("Test Accuracy (%)")
plt.title("Model Comparison — CIFAR-10 Test Accuracy")
plt.tight_layout(); plt.show()
