"""Prédit le chiffre d'une image PNG/JPG (un chiffre centré, fond uni)."""

import argparse

import torch
from PIL import Image, ImageOps
from torchvision import transforms

from train import DigitMLP, choose_device


def main():
    # L'image à analyser et le fichier de poids appris sont fournis au terminal.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", help="Chemin vers l'image du chiffre")
    parser.add_argument("--model", default="mnist_mlp.pt", help="Fichier de poids créé par train.py")
    args = parser.parse_args()

    # Utilise le même choix de matériel que pendant l'entraînement.
    device = choose_device()
    model = DigitMLP().to(device)
    # Reconstruit le réseau, puis remet dedans les poids sauvegardés après l'entraînement.
    model.load_state_dict(torch.load(args.model, map_location=device, weights_only=True))
    # Mode évaluation : on ne fait plus d'apprentissage.
    model.eval()

    # Ouvre l'image et la convertit en niveaux de gris (L = luminance).
    image = Image.open(args.image).convert("L")
    # MNIST présente des chiffres clairs sur fond sombre. Cette vérification
    # simple inverse l'image si elle semble plutôt avoir un fond clair.
    if sum(ImageOps.invert(image).convert("L").getdata()) < sum(image.getdata()):
        image = ImageOps.invert(image)

    # Applique le même prétraitement que pour les images MNIST d'entraînement.
    # Resize donne 28x28, ToTensor crée le tenseur, Normalize centre les pixels.
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    # unsqueeze ajoute la dimension du batch : une image seule a quand même
    # besoin d'une forme [1, 1, 28, 28] (batch, canal, hauteur, largeur).
    # no_grad désactive le calcul des gradients, inutile pour une prédiction.
    with torch.no_grad():
        scores = model(transform(image).unsqueeze(0).to(device))
        # Les sorties du réseau sont des scores; softmax les convertit en
        # probabilités dont la somme vaut 1. [0] enlève la dimension du batch.
        probabilities = scores.softmax(1)[0]

    # topk(3) garde les trois chiffres ayant la probabilité la plus élevée.
    top = probabilities.topk(3)
    # indices[0] est le chiffre ayant la probabilité la plus élevée.
    print(f"Chiffre prédit : {top.indices[0].item()}")
    print("Meilleures hypothèses : " + ", ".join(
        f"{digit.item()} ({prob.item() * 100:.1f}%)" for prob, digit in zip(top.values, top.indices)
    ))


if __name__ == "__main__":
    # Lance la prédiction seulement si ce fichier est exécuté directement.
    main()
