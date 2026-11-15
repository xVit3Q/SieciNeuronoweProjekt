import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# --- 1. WCZYTANIE I PRZYGOTOWANIE DANYCH ---
dataset = pd.read_csv('Student_Depression_Dataset.csv')
dataset.dropna(inplace=True)

# Oddzielenie zmiennej celu Y (Yes -> 1, No -> 0) PRZED kodowaniem cech X
Y = (dataset["Depression"] == "Yes").astype(int)
X = dataset.drop(columns=["Depression"])

# Kodowanie kolumn binarnych (2 wartości -> 0/1) oraz wielokategorialnych (metoda "1 z N")
for col in X.select_dtypes(include=["object", "string"]).columns:
    if X[col].nunique() == 2:
        X[col] = X[col].astype("category").cat.codes

X = pd.get_dummies(X, drop_first=False, dtype=int)

# Podział na zbiory: uczący (~70%), walidacyjny (~15%), testowy (~15%)
X_temp, X_test, Y_temp, Y_test = train_test_split(X, Y, test_size=0.15, random_state=42)
X_train, X_val, Y_train, Y_val = train_test_split(X_temp, Y_temp, test_size=0.176, random_state=42)

# Normalizacja Min-Max wyznaczona wyłącznie na zbiorze uczącym
train_min = X_train.min(axis=0)
train_max = X_train.max(axis=0)

denominator = train_max - train_min
denominator[denominator == 0] = 1  # zabezpieczenie przed dzieleniem przez 0

def NormalizacjaMinMax(data, min_val, denom_val):
    return (data - min_val) / denom_val

# Konwersja cech X na znormalizowane macierze NumPy
X_train_normal = NormalizacjaMinMax(X_train, train_min, denominator).to_numpy()
X_val_normal = NormalizacjaMinMax(X_val, train_min, denominator).to_numpy()
X_test_normal = NormalizacjaMinMax(X_test, train_min, denominator).to_numpy()

# Konwersja wyjść Y na macierze kolumnowe NumPy o wymiarze (N, 1)
Y_train = Y_train.to_numpy().reshape(-1, 1)
Y_val = Y_val.to_numpy().reshape(-1, 1)
Y_test = Y_test.to_numpy().reshape(-1, 1)


def aktywacja_unipolarna(u, beta=1.0):
    # np.clip chroni funkcję exp przed przepełnieniem (overflow)
    return 1.0 / (1.0 + np.exp(-np.clip(beta * u, -500, 500)))

def pochodna_unipolar(y, beta=1.0):
    return beta * y * (1.0 - y)


# --- 3. PĘTLA UCZENIA Z MINI-BATCHAMI I MOMENTUM ---
beta = 1.0          # stromość funkcji aktywacji
lr_start = 0.1         # współczynnik uczenia (Learning Rate)
momentum = 0.9      # współczynnik bezwładności (Momentum)
epochs = 500
batch_size = 64
Neurons = 16

train_losses = []
val_losses = []

num_samples = X_train_normal.shape[0]

# --- 3. FUNKCJA TRENUJĄCA MODEL DLA ZADANEGO BATCH_SIZE ---
def trenuj_model(batch_size):
    # Reset ziarna losowości, aby oba warianty startowały z identycznych wag początkowych
    input_dim = X_train_normal.shape[1]
    W1 = np.random.rand(input_dim, Neurons) * 0.2 - 0.1
    b1 = np.zeros((1, Neurons))
    
    W2 = np.random.rand(Neurons, 1) * 0.2 - 0.1
    b2 = np.zeros((1, 1))
    
    # Macierze zerowych prędkości dla Momentum
    vW1 = np.zeros_like(W1)
    vb1 = np.zeros_like(b1)
    vW2 = np.zeros_like(W2)
    vb2 = np.zeros_like(b2)
    
    # Parametry adaptacyjnego lr oraz Early Stopping
    lr = lr_start
    lr_min = 1e-4
    lr_max = 0.5
    
    target_mse = 0.005
    patience = max(1, int(0.05 * epochs))  # 5% liczby epok (dla 500 epok -> 25)
    min_delta = 1e-5
    
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    best_W1, best_b1 = W1.copy(), b1.copy()
    best_W2, best_b2 = W2.copy(), b2.copy()
    
    # Listy do przechowywania historii do wykresów
    batch_mse_out_hist = []
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
        
        batch_errors = []
        
        # Iteracja po mini-batchach
        for i in range(0, num_samples, batch_size):
            X_batch = X_shuffled[i:i + batch_size]
            Y_batch = Y_shuffled[i:i + batch_size]
            current_batch_size = X_batch.shape[0]
            
            # Propagacja w przód (Forward pass)
            u1 = np.dot(X_batch, W1) + b1
            y1 = aktywacja_unipolarna(u1, beta)
            
            u2 = np.dot(y1, W2) + b2
            y2 = aktywacja_unipolarna(u2, beta)
            
            # Zapis błędu MSE na bieżącej paczce przykładów
            error_output = Y_batch - y2
            batch_errors.append(np.mean(0.5 * (error_output ** 2)))
            
            # Propagacja wsteczna (Backward pass)
            delta2 = error_output * pochodna_unipolar(y2, beta)
            delta1 = np.dot(delta2, W2.T) * pochodna_unipolar(y1, beta)
            
            # Gradienty uśrednione po paczce
            dW2 = np.dot(y1.T, delta2) / current_batch_size
            db2 = np.sum(delta2, axis=0, keepdims=True) / current_batch_size
            
            dW1 = np.dot(X_batch.T, delta1) / current_batch_size
            db1 = np.sum(delta1, axis=0, keepdims=True) / current_batch_size
            
            # Aktualizacja prędkości i wag (Momentum)
            vW2 = momentum * vW2 + lr * dW2
            vb2 = momentum * vb2 + lr * db2
            
            vW1 = momentum * vW1 + lr * dW1
            vb1 = momentum * vb1 + lr * db1
            
            W2 += vW2
            b2 += vb2
            W1 += vW1
            b1 += vb1
            
        # --- EWALUACJA PO KAŻDEJ EPOCE ---
        batch_mse_out_hist.append(np.mean(batch_errors))
        
        # Zbiór uczący (obie warstwy + klasyfikacja 0-1)
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
        
        # Zbiór walidacyjny
        y1_val = aktywacja_unipolarna(np.dot(X_val_normal, W1) + b1, beta)
        y2_val = aktywacja_unipolarna(np.dot(y1_val, W2) + b2, beta)
        
        val_loss = np.mean(0.5 * (Y_val - y2_val) ** 2)
        val_mse_out_hist.append(val_loss)
        
        preds_val = (y2_val >= 0.5).astype(int)
        val_class_err_hist.append(np.mean(preds_val != Y_val) * 100)
        
        # Zapis wag obu warstw
        W1_hist.append(W1.copy())
        W2_hist.append(W2.copy())
        
        # Adaptacyjny współczynnik uczenia (Bold Driver)
        if epoch > 0:
            if train_mse_out_hist[-1] < train_mse_out_hist[-2]:
                lr = min(lr * 1.01, lr_max)
            else:
                lr = max(lr * 0.7, lr_min)
                
        # Early Stopping (Patience)
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            best_W1, best_b1 = W1.copy(), b1.copy()
            best_W2, best_b2 = W2.copy(), b2.copy()
        else:
            epochs_without_improvement += 1
            
        if epochs_without_improvement >= patience:
            stop_reason = f"Early Stopping w epoce {epoch + 1} (brak poprawy Val MSE przez {patience} epok)"
            W1, b1 = best_W1, best_b1
            W2, b2 = best_W2, best_b2
            break
            
        # Target MSE
        if train_loss <= target_mse:
            stop_reason = f"Target MSE w epoce {epoch + 1} (Train MSE = {train_loss:.5f} <= {target_mse})"
            break
            
    # --- TEST KOŃCOWY NA ZBIORZE TESTOWYM ---
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
        "train_mse_out": train_mse_out_hist,
        "train_mse_hid": train_mse_hid_hist,
        "val_mse_out": val_mse_out_hist,
        "train_class_err": train_class_err_hist,
        "val_class_err": val_class_err_hist,
        "W1_hist": np.array(W1_hist),
        "W2_hist": np.array(W2_hist)
    }


# --- 4. URUCHOMIENIE OBU WARIANTÓW I PORÓWNANIE WYNIKÓW ---
wyniki_32 = trenuj_model(32)
wyniki_64 = trenuj_model(64)

for w in [wyniki_32, wyniki_64]:
    print(f"=== WYNIKI DLA BATCH_SIZE = {w['batch_size']} ===")
    print(f"Zakończenie:            {w['stop_reason']}")
    print(f"Końcowy Train MSE:      {w['final_train_mse']:.5f}")
    print(f"Najlepszy Val MSE:      {w['best_val_mse']:.5f}")
    print(f"Skuteczność Test (Acc): {w['test_accuracy']:.2f}% (Błąd klasyfikacji: {w['test_class_err']:.2f}%)\n")


# --- 5. WYKRESY PORÓWNAWCZE (BATCH_SIZE 32 vs 64) ---
fig, axes = plt.subplots(3, 2, figsize=(15, 13))

# Wykres 1: Błąd MSE warstwy wyjściowej (cały zbiór uczący vs mini-batche)
axes[0, 0].plot(wyniki_32["train_mse_out"], label='Train MSE wyjście (batch=32)', color='tab:blue')
axes[0, 0].plot(wyniki_32["batch_mse_out"], label='Batch MSE wyjście (batch=32)', color='tab:blue', alpha=0.35)
axes[0, 0].plot(wyniki_64["train_mse_out"], label='Train MSE wyjście (batch=64)', color='tab:orange')
axes[0, 0].plot(wyniki_64["batch_mse_out"], label='Batch MSE wyjście (batch=64)', color='tab:orange', alpha=0.35)
axes[0, 0].set_title('Błąd MSE - warstwa wyjściowa (cały zbiór vs przykłady)')
axes[0, 0].set_xlabel('Epoka')
axes[0, 0].set_ylabel('MSE')
axes[0, 0].legend()
axes[0, 0].grid(True)

# Wykres 2: Błąd MSE warstwy ukrytej oraz zbioru walidacyjnego
axes[0, 1].plot(wyniki_32["train_mse_hid"], label='MSE warstwa ukryta (batch=32)', color='tab:blue')
axes[0, 1].plot(wyniki_64["train_mse_hid"], label='MSE warstwa ukryta (batch=64)', color='tab:orange')
axes[0, 1].plot(wyniki_32["val_mse_out"], label='Val MSE wyjście (batch=32)', color='tab:blue', linestyle='--')
axes[0, 1].plot(wyniki_64["val_mse_out"], label='Val MSE wyjście (batch=64)', color='tab:orange', linestyle='--')
axes[0, 1].set_title('Błąd MSE - warstwa ukryta oraz zbiór walidacyjny')
axes[0, 1].set_xlabel('Epoka')
axes[0, 1].set_ylabel('MSE')
axes[0, 1].legend()
axes[0, 1].grid(True)

# Wykres 3: Błąd klasyfikacji 0-1 (zbiór uczący i walidacyjny)
axes[1, 0].plot(wyniki_32["train_class_err"], label='Błąd klas. Train [%] (batch=32)', color='tab:blue')
axes[1, 0].plot(wyniki_32["val_class_err"], label='Błąd klas. Val [%] (batch=32)', color='tab:blue', linestyle='--')
axes[1, 0].plot(wyniki_64["train_class_err"], label='Błąd klas. Train [%] (batch=64)', color='tab:orange')
axes[1, 0].plot(wyniki_64["val_class_err"], label='Błąd klas. Val [%] (batch=64)', color='tab:orange', linestyle='--')
axes[1, 0].set_title('Błąd klasyfikacji (0-1) na przestrzeni epok')
axes[1, 0].set_xlabel('Epoka')
axes[1, 0].set_ylabel('Błąd klasyfikacji [%]')
axes[1, 0].legend()
axes[1, 0].grid(True)

# Wykres 4: Wagi w warstwie 1 (ukrytej) - porównanie dla pierwszych 8 wag
w1_32 = wyniki_32["W1_hist"].reshape(wyniki_32["epochs_run"], -1)[:, :8]
w1_64 = wyniki_64["W1_hist"].reshape(wyniki_64["epochs_run"], -1)[:, :8]
axes[1, 1].plot(w1_32, alpha=0.85)
axes[1, 1].set_prop_cycle(None)  # reset kolorów, aby te same wagi miały ten sam kolor
axes[1, 1].plot(w1_64, linestyle='--', alpha=0.85)
axes[1, 1].set_title('Wagi warstwy 1 (ukrytej) - ciągła: batch=32, przerywana: batch=64')
axes[1, 1].set_xlabel('Epoka')
axes[1, 1].set_ylabel('Wartość wagi')
axes[1, 1].grid(True)

# Wykres 5: Wagi w warstwie 2 (wyjściowej) dla batch_size = 32
axes[2, 0].plot(wyniki_32["W2_hist"].reshape(wyniki_32["epochs_run"], -1))
axes[2, 0].set_title('Wagi warstwy 2 (wyjściowej) - batch_size = 32')
axes[2, 0].set_xlabel('Epoka')
axes[2, 0].set_ylabel('Wartość wagi')
axes[2, 0].grid(True)

# Wykres 6: Wagi w warstwie 2 (wyjściowej) dla batch_size = 64
axes[2, 1].plot(wyniki_64["W2_hist"].reshape(wyniki_64["epochs_run"], -1))
axes[2, 1].set_title('Wagi warstwy 2 (wyjściowej) - batch_size = 64')
axes[2, 1].set_xlabel('Epoka')
axes[2, 1].set_ylabel('Wartość wagi')
axes[2, 1].grid(True)

plt.tight_layout()
plt.show()