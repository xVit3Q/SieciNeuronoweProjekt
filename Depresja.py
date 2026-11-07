import pandas as pd
import numpy as np
import matplotlib
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('Student_Depression_Dataset.csv')
dataset.dropna(inplace=True)

#Zmiana kolumn z textem na naumeracje po kategoriach
for col in dataset.select_dtypes(include=["object", "str"]).columns:
  dataset[col] = dataset[col].astype("category").cat.codes

X = dataset.drop(columns=["Depression"])
Y = dataset["Depression"]

#Zmienic potem random_state= 42 na jakis randomowy numer, najlepiej losowany 
X_temp, X_test, Y_temp, Y_test = train_test_split(X, Y, test_size = 0.15, random_state=42)
X_train, X_val, Y_train, Y_val = train_test_split(X_temp, Y_temp, test_size=0.176, random_state=42)

train_min = X_train.min(axis=0)
train_max = X_train.max(axis=0)

denominator = train_max - train_min
#jesli w kolumnie jest wszedzie ta sama wartosc aby nie było dzielenia przez 0
denominator[denominator == 0] = 1
def NormalizacjaMinMax(data, min_val, max_val):
  return (data - min_val) / (max_val - min_val)

X_train_normal = NormalizacjaMinMax(X_train, train_min, train_max)
X_val_normal = NormalizacjaMinMax(X_val, train_min, train_max)
X_test_normal = NormalizacjaMinMax(X_test, train_min, train_max)


# Silnik do trenowania modelu

Neurons = 10
#random.rand losuje z układu z przedziału (0,1)
W1 = np.random.rand(X_train_normal.shape[1], Neurons) * 0.2 - 0.1
W2 = np.random.rand(Neurons, 1) * 0.2 - 0.1

#macierze zerowych predkosci
vW1 = np.zeros_like(W1)
vW2 = np.zeros_like(W2)

vb1 = np.zeros_like((1, Neurons))
vb2 = np.zeros_like((1, 1))



def aktywacja_unipolarna(u, beta):
  return 1 / (1+ np.exp(-beta * u))

def pochodna_unipolar(y, beta):
  return beta * y * (1-y)