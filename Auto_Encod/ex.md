# L'Art de l'Auto-encodeur

### Mécanique : Compresser et Reconstruire

Pour manipuler la taille des images au sein d'un Auto-encodeur, nous utilisons deux types de couches complémentaires :

1. **L'Encoder (Décimation) :** On utilise des couches `Conv2D` pour extraire les caractéristiques, suivies de `MaxPooling2D` pour réduire la résolution spatiale. Cela force le réseau à ne garder que l'essentiel (le "code").
2. **Le Decoder (Expansion) :** Pour regagner la taille d'origine, on utilise la **`Conv2DTranspose`**. Contrairement à une convolution classique, elle utilise des `strides` (pas) pour "étaler" les pixels et reconstruire une image plus grande.
3. **Passage 2D vers 3D (Colorisation) :** Pour passer d'une entrée Noir & Blanc (1 canal) à une sortie Couleur (3 canaux), il suffit de régler le nombre de filtres de la **dernière** couche `Conv2D` du décodeur sur **3** avec une activation `sigmoid` ou `tanh`.



```python
# Exemple de structure Encodeur -> Décodeur
model = tf.keras.Sequential([
    # ENCODER : Réduction (ex: 128x128 -> 64x64)
    layers.Conv2D(32, 3, activation='relu', padding='same', input_shape=(128,128,3)),
    layers.MaxPooling2D((2, 2)), 
    
    # DECODER : Expansion (ex: 64x64 -> 128x128)
    layers.Conv2DTranspose(32, 3, strides=2, activation='relu', padding='same'),
    # Sortie finale : 3 filtres pour le RGB
    layers.Conv2D(3, 3, activation='sigmoid', padding='same') 
])

```

---

### Note sur les Datasets

Selon la puissance de votre machine ou l'utilisation de Google Colab, vous pourrez choisir des datasets plus ou moins lourds. Les datasets **Flowers 102** ou **stl10** offrent un rendu "Waouh", mais demandent plus de mémoire que **CIFAR-10** ou **MNIST**.

[stl10](https://www.kaggle.com/datasets/jessicali9530/stl10)

---

## Exercice 1 : Le Débruitage (Denoising)

**Objectif :** Apprendre au réseau à ignorer les artefacts pour reconstruire une image propre.

* **Dataset :** MNIST (rapide), CIFAR-10 ou Oxford Pets.
* **Méthode :** Vous devez injecter du bruit aléatoire dans vos images d'entrée tout en gardant les images originales comme cibles.
* **Code pour bruiter :** `x_train_noisy = x_train + noise_factor * tf.random.normal(shape=x_train.shape)`

## Exercice 2 : La Colorisation Automatique

**Objectif :** Prédire les couleurs d'une scène à partir de sa version grise.

* **Dataset :** **Flowers 102** (recommandé pour les couleurs vives), sinon **STL10** ou **CIFAR-10**.
* **Méthode :** Convertissez vos images RGB en niveaux de gris pour l'entrée. Le réseau doit ressortir une image à 3 canaux.
* **Code Noir & Blanc :** `x_train_gray = tf.image.rgb_to_grayscale(x_train)`

## Exercice 3 : La Super-Résolution

**Objectif :** Reconstruire une image haute définition à partir d'une source dégradée.

* **Dataset :** **Flowers 102** ou **STL-10** (96x96), sinon **CIFAR-10**.
* **Seuils :** Entrée basse résolution et sortie haute résolution.
* **Méthode :** Réduisez la taille de l'image originale pour créer votre entrée.
* **Code pour Resize :** `x_train_low = tf.image.resize(x_train, [32, 32])`

## Exercice 4 : Détection d’Anomalies (Fraude Bancaire)

**Objectif :** Identifier des comportements suspects sans avoir de labels "fraude".

* **Dataset :** **Credit Card Fraud Detection**.
* **Volume :** Inutile d'utiliser les 284 000 lignes. Un échantillon de **50 000 transactions normales** suffit amplement pour entraîner l'AE.
* **Méthode :** Entraînez l'AE **uniquement** sur des transactions valides.
* **Mesure de l'anomalie :** Calculez la **MSE (Mean Squared Error)** entre l'entrée et la sortie. Si le MSE est "fort" alors la transaction est marquée comme une anomalie (car le réseau n'a jamais appris à reconstruire ce type de profil).

---
2_1_AE.md