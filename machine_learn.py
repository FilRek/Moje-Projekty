import numpy as np
import copy
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from PIL import ImageFile
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix

###########################################################
### 1. KONFIGURACJA POCZĄTKOWA I URZĄDZENIE OBLICZENIOWE ###
###########################################################

ImageFile.LOAD_TRUNCATED_IMAGES = True
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

##############################################tworz###
### 2. TRANSFORMACJE I PRZYGOTOWANIE DANYCH ###
#################################################

# Transformacje dla zbioru treningowego (z augmentacją)
train_transform = transforms.Compose([
    transforms.Resize((32, 32)), # Zmiana rozmiaru
    transforms.RandomHorizontalFlip(), # Losowe odbicie poziome
    transforms.RandomVerticalFlip(), # Losowe odbicie pionowe (jeśli ma sens dla obiektu)
    transforms.RandomRotation(20), # Losowy obrót o +/- 20 stopni
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), # Losowa zmiana kolorów
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5), # Losowa perspektywa
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.8, 1.2)), # Losowe przesunięcie i skalowanie
    transforms.ToTensor(),  # Konwersja do tensora i skalowanie do 0-1
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Normalizacja do -1-1
])

# Transformacje dla zbioru walidacyjnego i testowego (bez augmentacji)
val_test_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

full_train_data = ImageFolder('sciezka_do_pliku/TRAIN', transform=train_transform)
test_data = ImageFolder('sciezka_do_pliku/TEST', transform=val_test_transform)


# Podział zbioru treningowego na train/validation (80/20)
train_size = int(0.8 * len(full_train_data))
val_size = len(full_train_data) - train_size
train_data, val_data = random_split(full_train_data,[train_size, val_size])

# Tworzenie DataLoaderów
train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
val_loader = DataLoader(val_data, batch_size=8, shuffle=False)
test_loader = DataLoader(test_data, batch_size=8, shuffle=False)

class_names = full_train_data.classes

###################################
### 3. DEFINICJA MODELU CNN ###
###################################

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 4 * 4, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def get_features(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.flatten(x)
        features = self.fc1(x)
        return features

    def forward(self, x):
        features = self.get_features(x)
        x = F.relu(features)
        x = self.fc2(x)
        return x

###################################################
### 4. TRENING MODELU CNN Z EARLY STOPPING ###
###################################################

net = SimpleCNN(num_classes=len(class_names)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

best_val_loss = np.inf
patience = 3
counter = 0
best_model_weights = copy.deepcopy(net.state_dict())

print("\n--- Rozpoczęcie treningu CNN ---")
for epoch in range(20):
    net.train()
    train_loss = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    net.eval()
    val_loss = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    print(f"Epoch {epoch}: train_loss={train_loss}, val_loss={val_loss}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_weights = copy.deepcopy(net.state_dict())
        torch.save(net.state_dict(), 'best_model.pth')
        counter = 0
    else:
        counter += 1
        print(f"val loss nie polepszyl sie {counter} raz")
        if counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
print("--- Zakończono trening CNN ---\n")

# Wczytanie najlepszych wag do modelu
net.load_state_dict(torch.load('best_model.pth'))


#########################################
### 5. EKSTRAKCJA CECH Z NAUCZONEJ SIECI ###
#########################################

def extract_features(loader, model):
    X, y = [], []
    model.eval()
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            features = model.get_features(inputs)
            X.append(features.cpu().numpy())
            y.append(labels.numpy())
    return np.concatenate(X), np.concatenate(y)

X_train_features, y_train_labels = extract_features(train_loader, net)
X_test_features, y_test_labels = extract_features(test_loader, net)

###########################################
### 6. OCENA METODY 1: SAMODZIELNY CNN ###
###########################################

y_true_cnn, p_test_cnn = [], []
net.eval()
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        outputs = net(inputs)
        probs = F.softmax(outputs, dim=1).cpu().numpy()
        p_test_cnn.append(probs)
        y_true_cnn.append(labels.numpy())

y_true_cnn = np.concatenate(y_true_cnn)
p_test_cnn = np.concatenate(p_test_cnn)
cnn_predictions = np.argmax(p_test_cnn, axis=1)

ACC_cnn = np.mean(y_true_cnn == cnn_predictions)
AUC_cnn = roc_auc_score(y_true_cnn, p_test_cnn, multi_class='ovr')
cm_cnn = confusion_matrix(y_true_cnn, cnn_predictions)

#######################################################
### 7. OCENA METODY 2: CNN + REGRESJA LOGISTYCZNA ###
#######################################################

clf = LogisticRegression(max_iter=1000, random_state=42)
clf.fit(X_train_features, y_train_labels)

y_pred_clf = clf.predict(X_test_features)
y_pred_clf_proba = clf.predict_proba(X_test_features)

ACC_clf = np.mean(y_test_labels == y_pred_clf)
AUC_clf = roc_auc_score(y_test_labels, y_pred_clf_proba, multi_class='ovr')
cm_clf = confusion_matrix(y_test_labels, y_pred_clf)


#######################################################
### 8. OCENA METODY 3: CNN + KNN (K-NAJBLIŻSI SĄSIEDZI) ###
#######################################################

# Inicjalizacja i "trening" modelu KNN (k-NN po prostu zapamiętuje dane)
k = 5
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train_features, y_train_labels)

# Predykcja na zbiorze testowym
y_pred_knn = knn.predict(X_test_features)
y_pred_knn_proba = knn.predict_proba(X_test_features)

# Obliczenie metryk dla KNN
ACC_knn = np.mean(y_test_labels == y_pred_knn)
AUC_knn = roc_auc_score(y_test_labels, y_pred_knn_proba, multi_class='ovr')
cm_knn = confusion_matrix(y_test_labels, y_pred_knn)


###########################################
### 9. PODSUMOWANIE I WIZUALIZACJA WYNIKÓW ###
###########################################

print("--- Porównanie wyników klasyfikatorów ---")
print(f"Metoda 1 (Samodzielny CNN):         ACC = {ACC_cnn:.4f}, AUC = {AUC_cnn:.4f}")
print(f"Metoda 2 (CNN + Regresja Log.):     ACC = {ACC_clf:.4f}, AUC = {AUC_clf:.4f}")
print(f"Metoda 3 (CNN + KNN, k={k}):          ACC = {ACC_knn:.4f}, AUC = {AUC_knn:.4f}")
print("------------------------------------------\n")

# Tworzenie wykresów macierzy pomyłek
fig, axes = plt.subplots(1, 3, figsize=(24, 6))
fig.suptitle('Macierze Pomyłek dla Trzech Metod Klasyfikacji', fontsize=16)

# Macierz dla CNN
sns.heatmap(cm_cnn, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=axes[0])
axes[0].set_title('Samodzielny CNN')
axes[0].set_xlabel('Przewidziana klasa')
axes[0].set_ylabel('Prawdziwa klasa')

# Macierz dla Regresji Logistycznej
sns.heatmap(cm_clf, annot=True, fmt='d', cmap='Greens', xticklabels=class_names, yticklabels=class_names, ax=axes[1])
axes[1].set_title('CNN + Regresja Logistyczna')
axes[1].set_xlabel('Przewidziana klasa')
axes[1].set_ylabel('Prawdziwa klasa')

# Macierz dla KNN
sns.heatmap(cm_knn, annot=True, fmt='d', cmap='Oranges', xticklabels=class_names, yticklabels=class_names, ax=axes[2])
axes[2].set_title(f'CNN + KNN (k={k})')
axes[2].set_xlabel('Przewidziana klasa')
axes[2].set_ylabel('Prawdziwa klasa')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
