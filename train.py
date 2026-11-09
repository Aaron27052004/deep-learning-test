"""Entraîne un réseau entièrement connecté sur MNIST et sauvegarde le modèle."""

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class DigitMLP(nn.Module):
    """Réseau de neurones qui transforme 784 pixels en 10 scores de chiffres."""

    def __init__(self):
        # Initialise les fonctions de base de PyTorch pour un modèle.
        super().__init__()
        # Sequential enchaîne les opérations dans l'ordre, de haut en bas.
        self.network = nn.Sequential(
            # Une image [1, 28, 28] devient un vecteur de 784 nombres.
            nn.Flatten(),
            # Première couche : chaque neurone combine les 784 pixels.
            nn.Linear(28 * 28, 128),
            # ReLU ajoute une non-linéarité et aide le réseau à apprendre
            # des relations plus complexes qu'une simple transformation linéaire.
            nn.ReLU(),
            # Deuxième couche cachée : 128 valeurs deviennent 64.
            nn.Linear(128, 64),
            nn.ReLU(),
            # Couche de sortie : un score brut pour chacun des chiffres 0 à 9.
            nn.Linear(64, 10),
        )

    def forward(self, x):
        # PyTorch appelle forward quand on écrit model(images).
        return self.network(x)


def choose_device():
    """Choisit le matériel disponible : GPU NVIDIA, GPU Mac, ou processeur."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    # MPS permet d'utiliser les puces Apple Silicon avec PyTorch.
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    # Permet de régler quelques options depuis le terminal, sans modifier le code.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=5, help="Nombre de passages complets sur les images d'entraînement")
    parser.add_argument("--batch-size", type=int, default=128, help="Nombre d'images traitées avant une mise à jour des poids")
    parser.add_argument("--output", type=Path, default=Path("mnist_mlp.pt"), help="Fichier où enregistrer les poids appris")
    args = parser.parse_args()

    # Rend l'initialisation aléatoire du modèle reproductible.
    torch.manual_seed(42)
    device = choose_device()
    print(f"Matériel utilisé : {device}")

    # ToTensor convertit les pixels en nombres entre 0 et 1.
    # Normalize centre ensuite les valeurs autour de 0 avec l'écart-type MNIST;
    # c'est pourquoi les valeurs finales ne restent plus entre 0 et 1.
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    # Télécharge MNIST la première fois, puis le lit depuis le dossier data/.
    # train=True réserve les images à l'apprentissage; train=False aux évaluations.
    train_data = datasets.MNIST("data", train=True, download=True, transform=transform)
    test_data = datasets.MNIST("data", train=False, download=True, transform=transform)

    # Le DataLoader découpe le jeu en lots (batches).
    # shuffle mélange les images d'entraînement à chaque époque.
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=1000)

    # Crée le réseau et le place sur le matériel choisi.
    model = DigitMLP().to(device)
    # Adam mettra à jour les poids et biais; lr est la taille de ses ajustements.
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # Compare les scores bruts aux bonnes étiquettes (0 à 9).
    # CrossEntropyLoss intègre déjà le calcul de softmax : ne pas l'ajouter ici.
    loss_fn = nn.CrossEntropyLoss()

    # Une époque = un passage complet sur les images d'entraînement.
    for epoch in range(1, args.epochs + 1):
        # En mode entraînement, le modèle est prêt à apprendre.
        model.train()
        total_loss = 0.0

        # Chaque lot contient des images et leurs réponses correctes (labels).
        for images, labels in train_loader:
            # Les données et le modèle doivent être sur le même matériel.
            images, labels = images.to(device), labels.to(device)

            # 1. Efface les gradients du lot précédent (PyTorch les accumule sinon).
            optimizer.zero_grad()
            # 2. Forward : calcule les dix scores prédits par image.
            # 3. Loss : mesure l'écart entre ces scores et les bonnes réponses.
            loss = loss_fn(model(images), labels)
            # 4. Backward : calcule comment chaque poids a contribué à l'erreur.
            loss.backward()
            # 5. Step : Adam ajuste les poids et biais pour réduire l'erreur.
            optimizer.step()
            total_loss += loss.item()

        # Évaluation sur des images gardées à part, qui n'ont pas servi à apprendre.
        model.eval()
        correct = total = 0
        # Pas besoin de gradients pendant l'évaluation : cela économise mémoire
        # et calcul, car le modèle ne modifie pas ses poids dans ce bloc.
        with torch.no_grad():
            for images, labels in test_loader:
                logits = model(images.to(device))
                # argmax choisit le score le plus élevé; on compare à la réponse.
                correct += (logits.argmax(1).cpu() == labels).sum().item()
                total += labels.size(0)
        average_loss = total_loss / len(train_loader)
        accuracy = 100 * correct / total
        print(f"Époque {epoch}/{args.epochs} — perte {average_loss:.4f} — précision test {accuracy:.2f}%")

    # Crée le dossier de destination si le chemin fourni en demande un.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Enregistre les poids appris (pas le jeu de données ni le code du modèle).
    torch.save(model.cpu().state_dict(), args.output)
    print(f"Modèle sauvegardé dans {args.output}")


if __name__ == "__main__":
    # N'exécute l'entraînement que si ce fichier est lancé directement.
    main()
