"""Entraîne un réseau entièrement connecté sur MNIST et sauvegarde le modèle."""

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class DigitMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        return self.network(x)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--output", type=Path, default=Path("mnist_mlp.pt"))
    args = parser.parse_args()

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_data = datasets.MNIST("data", train=True, download=True, transform=transform)
    test_data = datasets.MNIST("data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=1000)

    model = DigitMLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                logits = model(images.to(device))
                correct += (logits.argmax(1).cpu() == labels).sum().item()
                total += labels.size(0)
        print(f"Époque {epoch}/{args.epochs} — perte {total_loss / len(train_loader):.4f} — précision test {100 * correct / total:.2f}%")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.cpu().state_dict(), args.output)
    print(f"Modèle sauvegardé dans {args.output}")


if __name__ == "__main__":
    main()
