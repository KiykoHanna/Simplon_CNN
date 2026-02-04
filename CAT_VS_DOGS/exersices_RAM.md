 CAT Vs DOGS

#### Introduction

Si vous n'avez pas peur de commencer à traiter des images de tailles convenable **(224, 224)** on peut passer à la classification binaire de chien et de chat. Afin de rendre le problème utilisable sur vos machine sans exploser la RAM vous allez mettre en oeuvre une [lecture à la volée des images](#lecture-depuis-des-fichiers).

Le **dataset** (pré filtré) se trouve ici [cats-and-dogs-filtered](https://www.kaggle.com/datasets/birajsth/cats-and-dogs-filtered)

Le but est de comparé plusieurs modèles de classification, en partant d'un CNN simple

>**CONV2D > MAXPOOLING > DROPOUT > FLATTEN > DENSE > DENSE(1)**

à des modèles plus compliqués, avec plus de couches

>**CONV2D > MAXPOOLING > DROPOUT > CONV2D > MAXPOOLING > DROPOUT > FLATTEN > DENSE > DENSE(1)**

Le classifier peut aussi devenir plus complexe

>**FLATTEN > DENSE > DROPOUT > DENSE > DENSE(1)**

Pour finalement utiliser un modèle pré entraîné pour faire du [transfer learning](#transfer-learning).


# Lecture depuis des fichiers

Lundi matin, on a chargé des petits tableaux NumPy (CIFAR-10). Mais dans le monde réel, on travaille avec des dossiers d'images. 
`image_dataset_from_directory` crée un pipeline intelligent qui relie les fichiers disque directement à votre GPU sans jamais saturer la RAM.

Pour lier les images "à la volée" (sans tout charger en RAM), on utilise l'outil **`tf.keras.utils.image_dataset_from_directory`**.

#### La structure de dossier requise

Pour que TensorFlow comprenne tout seul les classes, les images doivent être rangées comme ceci :

```text
dataset/
├── train/
│   ├── chats/        <-- Classe 0
│   │   ├── img1.jpg
│   └── chiens/       <-- Classe 1
│       ├── img2.jpg
└── test/
    ├── chats/
    └── chiens/

```

#### Le code pour créer le flux (Pipeline)

Voici comment on configure le chargement automatique.

```python
import tensorflow as tf

# Paramètres
BATCH_SIZE = 32
IMG_SIZE = (224, 224) # Redimensionnement automatique 

# Création du dataset d'entraînement
train_ds = tf.keras.utils.image_dataset_from_directory(
    'dataset/train',
    shuffle=True,
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    label_mode='binary' # Ou 'categorical' si + de 2 classes
)

# Création du dataset de validation/test
val_ds = tf.keras.utils.image_dataset_from_directory(
    'dataset/test',
    shuffle=False, # Pas besoin de mélanger le test
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    label_mode='binary'
)

```
**Avec `binary`** : Les labels seront encodés sous forme d'un vecteur de **0** et de **1**. 
   * Dernière couche : `layers.Dense(1, activation='sigmoid')` (un seul neurone qui donne une probabilité entre 0 et 1).
   * Compilation : `loss='binary_crossentropy'`.

**Avec `categorical`** : Les labels seront encodés en **One-Hot**.
  * Dernière couche : `layers.Dense(N, activation='softmax')` 
  * Compilation : 
    * `int` : Les labels sont des entiers (0, 1, 2... 10). Utilise `sparse_categorical_crossentropy`.
    * `categorical` : Les labels sont des vecteurs binaires. Utilise `categorical_crossentropy`.
---

#### Avantages 

1. **Mémoire vive (RAM) :** On peut avoir 1 million d'images et le PC ne plantera jamais. Seules 32 images (le `batch_size`) sont en mémoire à un instant T.
2. **Preprocessing intégré :** La fonction redimensionne (`image_size`) les images automatiquement. Plus besoin de faire de `cv2.resize` à la main sur chaque fichier.
3. **Performance (Prefetching) :** On peut optimiser encore plus pour que le CPU prépare le batch suivant pendant que le GPU calcule le batch actuel :

```python
# Optionnel mais recommandé : optimise la vitesse
train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

```

#### Intégration avec le modèle

Une fois que l'on a les objets `train_ds` et `val_ds`, on passe directement au `fit` :

```python
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10
)

```
#### Automatiser la normalisation des images (pas des neurones)

On peut intégrer directement la normalisation des images entre **0** et **1** dans le réseau.

```python
model = tf.keras.Sequential([
    # Entrée fixe
    tf.keras.layers.Input(shape=(224, 224, 3)),
    
    # Normalisation : on passe de [0,255] à [0,1] ici
    tf.keras.layers.Rescaling(1./255),
    ...     
])
```
# Transfer Learning
#### Charger un modèle performant

```python
from tensorflow.keras.applications import VGG16, MobileNetV2, ResNet50
```
```python
# --- OPTION 1 : VGG16 (Le classique, un peu lourd) ---
base_model_vgg = VGG16(
    weights='imagenet',  # On charge les poids pré-entraînés
    include_top=False,   # On retire la "tête" (les dernières couches Dense)
    input_shape=(224, 224, 3)
)
```
```python
# --- OPTION 2 : MobileNetV2 (Ultra rapide, idéal pour CPU/Mobile) ---
base_model_mobile = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)
```
```python
# --- OPTION 3 : ResNet50 (Très performant, architecture résiduelle) ---
base_model_resnet = ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)
```

#### adapter la taille de l'input

**Important :** Le `input_shape` doit correspondre exactement aux dimensions de vos données (Largeur, Hauteur, Canaux), sinon les calculs matriciels du modèle échoueront immédiatement.

| Dataset | Taille Image | Canaux | `input_shape` | Classes | Méthode de chargement |
| --- | --- | --- | --- | --- | --- |
| **Fashion MNIST** | 28 x 28 | 1 (Gris) | `(28, 28, 1)` | 10 | `datasets.fashion_mnist` |
| **CIFAR-10** | 32 x 32 | 3 (RGB) | `(32, 32, 3)` | 10 | `datasets.cifar10` |
| **CIFAR-100** | 32 x 32 | 3 (RGB) | `(32, 32, 3)` | 100 | `datasets.cifar100` |
| **Cats & Dogs** | 224 x 224 | 3 (RGB) | `(224, 224, 3)` | 2 | `image_dataset_from_directory` |