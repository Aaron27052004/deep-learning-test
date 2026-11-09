# Reconnaissance de chiffres manuscrits

Un petit projet de deep learning qui apprend à reconnaître les chiffres de 0 à 9 avec [MNIST](https://yann.lecun.com/exdb/mnist/). Il contient un script d'entraînement et un script de prédiction pour tester une image.

## Prérequis

- Python 3.10 à 3.13
- pip

## Installation

```bash
git clone <URL_DU_DEPOT>
cd <NOM_DU_DEPOT>
python -m venv .venv
source .venv/bin/activate
```

Sur Windows, active l'environnement avec `.venv\Scripts\activate`.

Installe ensuite les dépendances :

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Entraîner le modèle

```bash
python train.py
```

MNIST est téléchargé automatiquement dans `data/`. Le script entraîne un petit réseau convolutif pendant cinq époques, affiche la perte et la précision sur les données de test, puis sauvegarde les poids dans `mnist_cnn.pt`.

Tu peux modifier le nombre d'époques et le chemin de sortie :

```bash
python train.py --epochs 3 --output modeles/mnist_cnn.pt
```

## Prédire un chiffre

```bash
python predict.py chemin/vers/mon_chiffre.png
```

Pour un modèle enregistré ailleurs :

```bash
python predict.py mon_chiffre.png --model modeles/mnist_cnn.pt
```

L'image doit contenir un seul chiffre, assez grand et centré. Le modèle a été entraîné sur des images MNIST en niveaux de gris de 28 × 28 pixels, avec le chiffre clair sur fond sombre. Le script tente d'inverser automatiquement une image au fond clair. La prédiction affiche le chiffre choisi et les trois hypothèses les plus probables.

## Architecture

Le modèle utilise deux couches convolutives avec ReLU et max pooling, puis deux couches entièrement connectées. Il est entraîné avec Adam et la perte `CrossEntropyLoss`. Le code choisit CUDA si disponible, sinon le processeur.

## Structure du projet

```text
├── train.py         # Entraînement et évaluation
├── predict.py       # Prédiction sur une image
├── requirements.txt # Dépendances Python
├── LICENSE          # Licence MIT
└── .gitignore       # Fichiers locaux exclus de Git
```

## Licence

Ce projet est distribué sous licence MIT. Voir [LICENSE](LICENSE).
