
# nombre prédit

definir le nombre prédit pour la creation de modèle



``` python
number=2
```
Dans notre programme, on a besoin de quelques bibliothèques:
`<br>`{=html}1- Tensorflow: pour créer un modèle. `<br>`{=html}2-
Matplotlib : affiche les images sous forme graphique. `<br>`{=html}3-
random : choisir d\'une manière aleatoire un chifre. `<br>`{=html}4-
Numpy: pour l\'utilisation des matrices.

``` python
import tensorflow as tf
from tensorflow import keras
import matplotlib. pyplot as plt
import random as rnd
import numpy as np
```

# Importer les donneés

importer la base de donnee à partire de la bibliothèques
\"tensorflow.keras\"

``` python
(X_train,Y_train),(X_test,Y_test)=keras.datasets.mnist.load_data()
```

parcoure les listes (X_train et X_test) et verfier si y est égale au
nombre definit auparavant c\'est vrai s\'il n\'est pas égale c\' faut

``` python
for i in range(len(Y_train)):
    if Y_train[i]==number:
        Y_train[i]=True
    else:
        Y_train[i]=False
        
for i in range(len(Y_test)):
    if Y_test[i]==number:
        Y_test[i]=True
    else:
        Y_test[i]=False
```

affichage de la taille des tableaux et les dimensions des images

``` python
print("shape of x_train",X_train.shape)
print("shape of Y_train",Y_train.shape)
print("shape of X_test",X_test.shape)
print("shape of Y_test",Y_test.shape)
```


    shape of x_train (60000, 28, 28)
    shape of Y_train (60000,)
    shape of X_test (10000, 28, 28)
    shape of Y_test (10000,)

# Remodeler les donnees

toBainryMatrix(X) cette fanction permet d\'eleminer la couleur grise et
la ramplacer par le blanc si \>=127 si non par noirs, le noir est
représenter par 0 et le blanc par 1

``` python
def toBainryMatrix(X):
    new_X=[]
    for img in X:
        newImg=[]
        for line in img:
            newLine=[]
            for pixel in line:
                if pixel<127:
                    newLine.append(1)
                else:
                    newLine.append(0)
            newImg.append(newLine)
        new_X.append(newImg)
    return np.array(new_X)
```

reverseBlackWhite(X) cette fonction inverse le noir et le blanc

``` python
def reverseBlackWhite(X):
    new_X=[]
    for img in X:
        newImg=[]
        for line in img:
            newLine=[]
            for pixel in line:
                if pixel==1:
                    newLine.append(0)
                else:
                    newLine.append(1)
            newImg.append(newLine)
        new_X.append(newImg)
    return np.array(new_X)
```

extractSubMatrix(X) devise la matrice en 16 sous matrices (4\*4) chaque
sous matrice contient 7 lignes et 7 colonnes

``` python
def extractSubMatrix(X):
    new_X=[]

    for img in X:
        i=0
        step=len(img)/4
        newImage=[]
        while i<len(img):
            end_row=i+step
            j=0
            while j<len(img):
                end_column=j+step
                block=img[int(i):int(end_row),int(j):int(end_column)]
                j=j+step
                newImage.append(block)
            i=i+step
        newImage=np.array(newImage)
        new_X.append(newImage)
    new_X=np.array(new_X)
    
    return new_X
```

imageToVector(X) elle transrorme les image en vecteur. le traitment
appliqué sera :`<br>`{=html} remplacer chaque sous matrice (7\*7) par un
nombre entre \[0,1\] égale à (nombre de pixeles noirs/nombre de pixeles)

``` python
def imageToVector(X):
    new_X=[]
    for img in X:
        new_img=[]
        for block in img:
            blacks=0
            for line in block:
                for pixel in line:
                    if pixel==1:
                        blacks=blacks+1
            new_img.append(blacks/49)
        new_X.append(new_img)
    return np.array(new_X)
    
```

``` python
train=toBainryMatrix(X_train)
test=toBainryMatrix(X_test)
```

``` python
train=reverseBlackWhite(train)
test=reverseBlackWhite(test)
print(train[0])
```


    [[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 1 1 1 1 0 0 0 0]
     [0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0]
     [0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0 0 0 1 0 1 1 1 0 0 0 1 0 0 0 0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
     [0 0 0 0 0 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
     [0 0 0 0 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]]


``` python
train=extractSubMatrix(train)
test=extractSubMatrix(test)
train[0]
```

    array([[[0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0]],

           [[0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 1]],

           [[0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 0, 1],
            [1, 1, 1, 1, 1, 1, 1]],

           [[0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0]],

           [[0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0]],

           [[0, 1, 1, 1, 1, 1, 1],
            [0, 1, 1, 1, 1, 1, 1],
            [0, 0, 1, 0, 1, 1, 1],
            [0, 0, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 1, 1, 1],
            [0, 0, 0, 0, 0, 1, 1],
            [0, 0, 0, 0, 0, 0, 1]],

           [[1, 1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0]],

           [[0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0]],

           [[0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0]],

           [[0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 1, 1]],

           [[1, 1, 1, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 0, 0],
            [0, 0, 0, 1, 1, 1, 0],
            [0, 0, 0, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 0, 0, 0]],

           [[0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0]],

           [[0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 1],
            [0, 0, 0, 0, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0]],

           [[0, 0, 0, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0]],

           [[1, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0]],

           [[0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0]]])

``` python
new_X_train=imageToVector(train)
new_X_test=imageToVector(test)
Y_train=Y_train.reshape(len(Y_train),1)
Y_test=Y_test.reshape(len(Y_test),1)
```

Afficher les dimensions apré les traitements

``` python
print("shape of x_train",new_X_train.shape)
print("shape of Y_train",Y_train.shape)
print("shape of X_test",new_X_test.shape)
print("shape of Y_test",Y_test.shape)
```

    shape of x_train (60000, 16)
    shape of Y_train (60000, 1)
    shape of X_test (10000, 16)
    shape of Y_test (10000, 1)

``` python
new_X_train[0]
```


    array([0.        , 0.06122449, 0.20408163, 0.10204082, 0.        ,
           0.48979592, 0.2244898 , 0.        , 0.        , 0.06122449,
           0.57142857, 0.        , 0.10204082, 0.42857143, 0.04081633,
           0.        ])



``` python
new_X_test[0]
```


    array([0.        , 0.        , 0.        , 0.        , 0.02040816,
           0.28571429, 0.46938776, 0.02040816, 0.        , 0.02040816,
           0.32653061, 0.        , 0.        , 0.28571429, 0.02040816,
           0.        ])


# Modele
:::


Création du modele selon le tableau presanté ci-dessous qui contient 3
couches: `<br>`{=html}a- couche 1: designe les taille des vecteur
attendu `<br>`{=html}b- couche 2: contient 512 reasaux de neron pour le
traitement ,la fonction d\'activation=\"relu\" `<br>`{=html}c- couche 2:
contient 512 reasaux de neron pour le traitement ,la fonction
d\'activation=\"relu\" `<br>`{=html}d- couche 3: contient un seul car on
a une seule categorie ,la fonction d\'activation=\"sigmoid\"

``` python
model=keras.Sequential([
    keras.layers.Flatten(input_shape=(16,)),
    keras.layers.Dense(512,activation='relu'),
    keras.layers.Dense(250,activation='relu'),
    keras.layers.Dense(1,activation='sigmoid')
])
```

``` python
model.summary()
```


    Model: "sequential"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     flatten (Flatten)           (None, 16)                0         
                                                                     
     dense (Dense)               (None, 512)               8704      
                                                                     
     dense_1 (Dense)             (None, 250)               128250    
                                                                     
     dense_2 (Dense)             (None, 1)                 251       
                                                                     
    =================================================================
    Total params: 137,205
    Trainable params: 137,205
    Non-trainable params: 0
    _________________________________________________________________

compiler le modele choisir type de classificateur optimiseur adam et
afficher l\' exactitude du modele

``` python
model.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)
```

``` python
model.fit(new_X_train,Y_train,epochs=5)
```


    Epoch 1/5
    1875/1875 [==============================] - 3s 2ms/step - loss: 0.1057 - accuracy: 0.9652
    Epoch 2/5
    1875/1875 [==============================] - 3s 1ms/step - loss: 0.0716 - accuracy: 0.9757
    Epoch 3/5
    1875/1875 [==============================] - 3s 1ms/step - loss: 0.0653 - accuracy: 0.9782
    Epoch 4/5
    1875/1875 [==============================] - 3s 1ms/step - loss: 0.0624 - accuracy: 0.9792
    Epoch 5/5
    1875/1875 [==============================] - 3s 1ms/step - loss: 0.0609 - accuracy: 0.9789



    <keras.callbacks.History at 0x256b321b790>



evaluer le modele pour le pourcentage de reussite et de perte en
utilisant X et Y de test


``` python
model.evaluate(new_X_test,Y_test)
```


    313/313 [==============================] - 0s 892us/step - loss: 0.0655 - accuracy: 0.9778


    [0.0654883161187172, 0.9778000116348267]

# Prediction

choisir un chiffre aleatoirment, represanter sous forme graphique,
predecter le chiffre, si le resultat \>0.5 afficher\"it\'s a number\" ,
si non \"it\'s not a number\"


``` python
index=rnd.randint(0,len(new_X_test)-1)

#pour teste la preiction sur le nombre 5
#index=15

#pour teste la prediction sur le number 2
#index=1

plt.imshow(X_test[index])
plt.show()

predictions=model.predict(new_X_test)
y=predictions[index]
print(y)

if y>0.5:
    print("it's a number "+str(number))
else:
    print("it's not a number "+str(number))
```






    313/313 [==============================] - 0s 773us/step
    [0.04831472]
    it's not a number 2

