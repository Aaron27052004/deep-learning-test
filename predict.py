"""Prédit le chiffre d'une image PNG/JPG (un chiffre centré, fond uni)."""

import argparse

import torch
from PIL import Image, ImageOps
from torchvision import transforms

from train import DigitMLP


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", help="Chemin vers l'image du chiffre")
    parser.add_argument("--model", default="mnist_cnn.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DigitMLP().to(device)
    model.load_state_dict(torch.load(args.model, map_location=device, weights_only=True))
    model.eval()

    image = Image.open(args.image).convert("L")
    # MNIST présente des chiffres clairs sur fond noir.
    if sum(ImageOps.invert(image).convert("L").getdata()) < sum(image.getdata()):
        image = ImageOps.invert(image)
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    with torch.no_grad():
        probabilities = model(transform(image).unsqueeze(0).to(device)).softmax(1)[0]
    top = probabilities.topk(3)
    print(f"Chiffre prédit : {top.indices[0].item()}")
    print("Meilleures hypothèses : " + ", ".join(
        f"{digit.item()} ({prob.item() * 100:.1f}%)" for prob, digit in zip(top.values, top.indices)
    ))


if __name__ == "__main__":
    main()
