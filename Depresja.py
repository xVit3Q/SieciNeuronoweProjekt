import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Wczytanie i przygotowanie danych
dataset = pd.read_csv('Student_Depression_Dataset.csv')
dataset.dropna(inplace=True)

# Oddzielenie zmiennej celu Y przed kodowaniem cech X
Y = (dataset["Depression"] == "Yes").astype(int)
X = dataset.drop(columns=["Depression"])

# Kodowanie kolumn binarnych (0/1) oraz wielokategorialnych (1 z N)
for col in X.select_dtypes(include=["object", "string"]).columns:
    if X[col].nunique() == 2:
        X[col] = X[col].astype("category").cat.codes

X = pd.get_dummies(X, drop_first=False, dtype=int)

# Podział na zbiory: uczący (~70%), walidacyjny (~15%), testowy (~15%)
X_temp, X_test, Y_temp, Y_test = train_test_split(X, Y, test_size=0.15, random_state=42)
X_train, X_val, Y_train, Y_val = train_test_split(X_temp, Y_temp, test_size=0.176, random_state=42)

# Normalizacja Min-Max na podstawie zbioru uczącego
train_min = X_train.min(axis=0)
train_max = X_train.max(axis=0)

denominator = train_max - train_min
denominator[denominator == 0] = 1  # zabezpieczenie przed dzieleniem przez 0

def NormalizacjaMinMax(data, min_val, denom_val):
    return (data - min_val) / denom_val

# Konwersja danych na macierze NumPy
X_train_normal = NormalizacjaMinMax(X_train, train_min, denominator).to_numpy()
X_val_normal = NormalizacjaMinMax(X_val, train_min, denominator).to_numpy()
X_test_normal = NormalizacjaMinMax(X_test, train_min, denominator).to_numpy()

Y_train = Y_train.to_numpy().reshape(-1, 1)
Y_val = Y_val.to_numpy().reshape(-1, 1)
Y_test = Y_test.to_numpy().reshape(-1, 1)


# Funkcja aktywacji i jej pochodna
def aktywacja_unipolarna(u, beta=1.0):
    return 1.0 / (1.0 + np.exp(-np.clip(beta * u, -500, 500)))

def pochodna_unipolar(y, beta=1.0):
    return beta * y * (1.0 - y)


# Parametry sieci i funkcja ucząca
beta = 1.0          # stromość funkcji aktywacji
lr_start = 0.1      # początkowy współczynnik uczenia
momentum = 0.9      # współczynnik bezwładności
epochs = 500        # maksymalna liczba epok
Neurons = 16        # liczba neuronów w warstwie ukrytej

def trenuj_model(batch_size):
    # Ustawienie ziarna losowości dla takich samych wag początkowych
    np.random.seed(42)
    input_dim = X_train_normal.shape[1]
    
    W1 = np.random.rand(input_dim, Neurons) * 0.2 - 0.1
    b1 = np.zeros((1, Neurons))
    
    W2 = np.random.rand(Neurons, 1) * 0.2 - 0.1
    b2 = np.zeros((1, 1))
    
    # Macierze poprzednich zmian wag dla metody z bezwładnością (momentum)
    vW1 = np.zeros_like(W1)
    vb1 = np.zeros_like(b1)
    vW2 = np.zeros_like(W2)
    vb2 = np.zeros_like(b2)
    
    # Parametry adaptacyjnego współczynnika uczenia i wcześniejszego kończenia
    lr = lr_start
    lr_min = 1e-4
    lr_max = 0.5
    
    target_mse = 0.005
    patience = max(1, int(0.05 * epochs))
    min_delta = 1e-5
    
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    best_W1, best_b1 = W1.copy(), b1.copy()
    best_W2, best_b2 = W2.copy(), b2.copy()
    
    # Listy do zapisywania historii uczenia
    batch_mse_out_hist = []
    batch_mse_hid_hist = []
    train_mse_out_hist = []
    train_mse_hid_hist = []
    val_mse_out_hist = []
    
    train_class_err_hist = []
    val_class_err_hist = []
    
    W1_hist = []
    W2_hist = []
    
    num_samples = X_train_normal.shape[0]
    stop_reason = f"Osiągnięto maksymalną liczbę epok ({epochs})"
    
    for epoch in range(epochs):
        # Tasowanie danych przed każdą epoką
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        
        X_shuffled = X_train_normal[indices]
        Y_shuffled = Y_train[indices]
        
        batch_errors_out = []
        batch_errors_hid = []
        
        # Pętla po paczkach danych
        for i in range(0, num_samples, batch_size):
            X_batch = X_shuffled[i:i + batch_size]
            Y_batch = Y_shuffled[i:i + batch_size]
            current_batch_size = X_batch.shape[0]
            
            # Obliczenie wyjść sieci
            u1 = np.dot(X_batch, W1) + b1
            y1 = aktywacja_unipolarna(u1, beta)
            
            u2 = np.dot(y1, W2) + b2
            y2 = aktywacja_unipolarna(u2, beta)
            
            # Błąd warstwy wyjściowej na paczce
            error_output = Y_batch - y2
            batch_errors_out.append(np.mean(0.5 * (error_output ** 2)))
            
            # Propagacja wsteczna błędu
            delta2 = error_output * pochodna_unipolar(y2, beta)
            error_hidden = np.dot(delta2, W2.T)
            batch_errors_hid.append(np.mean(0.5 * (error_hidden ** 2)))
            delta1 = error_hidden * pochodna_unipolar(y1, beta)
            
            # Obliczenie poprawek wag uśrednionych po paczce
            dW2 = np.dot(y1.T, delta2) / current_batch_size
            db2 = np.sum(delta2, axis=0, keepdims=True) / current_batch_size
            
            dW1 = np.dot(X_batch.T, delta1) / current_batch_size
            db1 = np.sum(delta1, axis=0, keepdims=True) / current_batch_size
            
            # Aktualizacja wag z uwzględnieniem bezwładności (momentum)
            vW2 = momentum * vW2 + lr * dW2
            vb2 = momentum * vb2 + lr * db2
            
            vW1 = momentum * vW1 + lr * dW1
            vb1 = momentum * vb1 + lr * db1
            
            W2 += vW2
            b2 += vb2
            W1 += vW1
            b1 += vb1
            
        # Zapis średniego błędu z paczek po epoce
        batch_mse_out_hist.append(np.mean(batch_errors_out))
        batch_mse_hid_hist.append(np.mean(batch_errors_hid))
        
        # Obliczenie błędów na całym zbiorze uczącym
        y1_train = aktywacja_unipolarna(np.dot(X_train_normal, W1) + b1, beta)
        y2_train = aktywacja_unipolarna(np.dot(y1_train, W2) + b2, beta)
        
        err_out_train = Y_train - y2_train
        train_loss = np.mean(0.5 * (err_out_train ** 2))
        train_mse_out_hist.append(train_loss)
        
        delta2_train = err_out_train * pochodna_unipolar(y2_train, beta)
        err_hid_train = np.dot(delta2_train, W2.T)
        train_mse_hid_hist.append(np.mean(0.5 * (err_hid_train ** 2)))
        
        preds_train = (y2_train >= 0.5).astype(int)
        train_class_err_hist.append(np.mean(preds_train != Y_train) * 100)
        
        # Obliczenie błędów na zbiorze walidacyjnym
        y1_val = aktywacja_unipolarna(np.dot(X_val_normal, W1) + b1, beta)
        y2_val = aktywacja_unipolarna(np.dot(y1_val, W2) + b2, beta)
        
        val_loss = np.mean(0.5 * (Y_val - y2_val) ** 2)
        val_mse_out_hist.append(val_loss)
        
        preds_val = (y2_val >= 0.5).astype(int)
        val_class_err_hist.append(np.mean(preds_val != Y_val) * 100)
        
        # Zapis wag obu warstw
        W1_hist.append(W1.copy())
        W2_hist.append(W2.copy())
        
        # Adaptacyjny współczynnik uczenia
        if epoch > 0:
            if train_mse_out_hist[-1] < train_mse_out_hist[-2]:
                lr = min(lr * 1.01, lr_max)
            else:
                lr = max(lr * 0.7, lr_min)
                
        # Sprawdzanie poprawy na zbiorze walidacyjnym
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            best_W1, best_b1 = W1.copy(), b1.copy()
            best_W2, best_b2 = W2.copy(), b2.copy()
        else:
            epochs_without_improvement += 1
            
        if epochs_without_improvement >= patience:
            stop_reason = f"Brak poprawy błędu walidacyjnego przez {patience} epok (epoka {epoch + 1})"
            W1, b1 = best_W1, best_b1
            W2, b2 = best_W2, best_b2
            break
            
        # Szybsze kończenie uczenia po osiągnięciu zadanego błędu
        if train_loss <= target_mse:
            stop_reason = f"Osiągnięto zadany błąd MSE w epoce {epoch + 1} ({train_loss:.5f} <= {target_mse})"
            break
            
    # Test końcowy na zbiorze testowym
    y1_test = aktywacja_unipolarna(np.dot(X_test_normal, W1) + b1, beta)
    y2_test = aktywacja_unipolarna(np.dot(y1_test, W2) + b2, beta)
    
    predictions = (y2_test >= 0.5).astype(int)
    test_accuracy = np.mean(predictions == Y_test) * 100
    test_class_err = 100.0 - test_accuracy
    
    return {
        "batch_size": batch_size,
        "epochs_run": len(train_mse_out_hist),
        "stop_reason": stop_reason,
        "final_train_mse": train_mse_out_hist[-1],
        "best_val_mse": best_val_loss,
        "test_accuracy": test_accuracy,
        "test_class_err": test_class_err,
        "batch_mse_out": batch_mse_out_hist,
        "batch_mse_hid": batch_mse_hid_hist,
        "train_mse_out": train_mse_out_hist,
        "train_mse_hid": train_mse_hid_hist,
        "val_mse_out": val_mse_out_hist,
        "train_class_err": train_class_err_hist,
        "val_class_err": val_class_err_hist,
        "W1_hist": np.array(W1_hist),
        "W2_hist": np.array(W2_hist)
    }


# Uruchomienie uczenia dla dwóch rozmiarów paczki i wyświetlenie wyników
wyniki_32 = trenuj_model(32)
wyniki_64 = trenuj_model(64)

for w in [wyniki_32, wyniki_64]:
    print(f"=== WYNIKI DLA PACZKI = {w['batch_size']} ===")
    print(f"Zakończenie:            {w['stop_reason']}")
    print(f"Końcowy MSE (uczący):   {w['final_train_mse']:.5f}")
    print(f"Najlepszy MSE (walid.): {w['best_val_mse']:.5f}")
    print(f"Skuteczność testowa:    {w['test_accuracy']:.2f}% (Błąd klasyfikacji: {w['test_class_err']:.2f}%)\n")


# Generowanie i zapis wykresów do osobnych plików
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=[
    '#0072B2', '#D55E00', '#009E73', '#CC79A7', 
    '#F0E442', '#56B4E9', '#E69F00', '#000000'
])

C_32 = '#0072B2'
C_64 = '#D55E00'

def zapisz_wykres(fig, nazwa_pliku):
    fig.tight_layout()
    fig.savefig(nazwa_pliku, dpi=300, bbox_inches='tight')
    plt.close(fig)

# Błąd MSE warstwy wyjściowej (cały zbiór vs przykłady uczące)
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(wyniki_32["train_mse_out"], label='Cały zbiór (paczka=32)', color=C_32)
ax.plot(wyniki_32["batch_mse_out"], label='Przykłady uczące (paczka=32)', color=C_32, alpha=0.35)
ax.plot(wyniki_64["train_mse_out"], label='Cały zbiór (paczka=64)', color=C_64)
ax.plot(wyniki_64["batch_mse_out"], label='Przykłady uczące (paczka=64)', color=C_64, alpha=0.35)
ax.set_title('MSE warstwy wyjściowej (cały zbiór vs przykłady uczące)')
ax.set_xlabel('Epoka')
ax.set_ylabel('MSE')
ax.legend()
ax.grid(True, alpha=0.3) # Dodanie przezroczytości siatki dla lepszej czytelności
zapisz_wykres(fig, 'wykres_1_mse_wyjscie.png')

# Błąd MSE warstwy ukrytej (cały zbiór vs przykłady uczące)
fig, ax1 = plt.subplots(figsize=(8, 5))
ax2 = ax1.twinx()

l1 = ax1.plot(wyniki_32["val_mse_out"], label='Walidacja (batch=32)', color=C_32, linestyle='--')
l2 = ax1.plot(wyniki_64["val_mse_out"], label='Walidacja (batch=64)', color=C_64, linestyle='--')
l3 = ax2.plot(wyniki_32["train_mse_hid"], label='Ukryta (batch=32)', color=C_32)
l4 = ax2.plot(wyniki_64["train_mse_hid"], label='Ukryta (batch=64)', color=C_64)

ax1.set_title('MSE: warstwa ukryta (prawa oś) vs walidacja (lewa oś)')
ax1.set_xlabel('Epoka')
ax1.set_ylabel('MSE - Walidacja')
ax2.set_ylabel('MSE - Warstwa ukryta')

# Wspólna legenda dla obu osi
linie = l1 + l2 + l3 + l4
etykiety = [l.get_label() for l in linie]
ax1.legend(linie, etykiety, loc='center right')
ax1.grid(True, alpha=0.3)

zapisz_wykres(fig, 'wykres_2_mse_ukryta_val.png')

# Błąd klasyfikacji 0-1
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(wyniki_32["train_class_err"], label='Zbiór uczący (paczka=32)', color=C_32)
ax.plot(wyniki_32["val_class_err"], label='Zbiór walidacyjny (paczka=32)', color=C_32, linestyle='--')
ax.plot(wyniki_64["train_class_err"], label='Zbiór uczący (paczka=64)', color=C_64)
ax.plot(wyniki_64["val_class_err"], label='Zbiór walidacyjny (paczka=64)', color=C_64, linestyle='--')
ax.set_title('Błąd klasyfikacji (0-1)')
ax.set_xlabel('Epoka')
ax.set_ylabel('Błąd [%]')
ax.legend()
ax.grid(True, alpha=0.3)
zapisz_wykres(fig, 'wykres_3_blad_klasyfikacji.png')

# Wagi w warstwie 1
fig, ax = plt.subplots(figsize=(8, 5))
w1_32 = wyniki_32["W1_hist"].reshape(wyniki_32["epochs_run"], -1)[:, :8]
w1_64 = wyniki_64["W1_hist"].reshape(wyniki_64["epochs_run"], -1)[:, :8]
ax.plot(w1_32, alpha=0.85)
ax.set_prop_cycle(None)
ax.plot(w1_64, linestyle='--', alpha=0.85)
ax.set_title('Wagi warstwy 1 (ciągła: paczka=32, przerywana: paczka=64)')
ax.set_xlabel('Epoka')
ax.set_ylabel('Wartość wagi')
ax.grid(True, alpha=0.3)
zapisz_wykres(fig, 'wykres_4_wagi_warstwa_1.png')

# Wagi w warstwie 2 (paczka=32)
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(wyniki_32["W2_hist"].reshape(wyniki_32["epochs_run"], -1))
ax.set_title('Wagi warstwy 2 (paczka=32)')
ax.set_xlabel('Epoka')
ax.set_ylabel('Wartość wagi')
ax.grid(True, alpha=0.3)
zapisz_wykres(fig, 'wykres_5_wagi_warstwa_2_b32.png')

# Wagi w warstwie 2 (paczka=64)
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(wyniki_64["W2_hist"].reshape(wyniki_64["epochs_run"], -1))
ax.set_title('Wagi warstwy 2 (paczka=64)')
ax.set_xlabel('Epoka')
ax.set_ylabel('Wartość wagi')
ax.grid(True, alpha=0.3)
zapisz_wykres(fig, 'wykres_6_wagi_warstwa_2_b64.png')