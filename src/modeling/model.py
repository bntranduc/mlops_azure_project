import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os

# Configuration
BATCH_SIZE = 128
LEARNING_RATE = 0.001
EPOCHS = 10
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = 'mnist_mlp_model.pth'

print(f"Utilisation du device: {DEVICE}")

# Préparation des données
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # Normalisation MNIST
])

# Chargement des datasets
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Dataset d'entraînement: {len(train_dataset)} échantillons")
print(f"Dataset de test: {len(test_dataset)} échantillons")

# Définition du modèle MLP
class MLP(nn.Module):
    def __init__(self, input_size=784, hidden_sizes=[512, 256], num_classes=10, dropout_rate=0.2):
        super(MLP, self).__init__()
        
        layers = []
        prev_size = input_size
        
        # Couches cachées
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
        
        # Couche de sortie
        layers.append(nn.Linear(prev_size, num_classes))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        # Aplatir l'image (28x28 -> 784)
        x = x.view(x.size(0), -1)
        return self.network(x)

# Initialisation du modèle
model = MLP().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print("Architecture du modèle:")
print(model)
print(f"Nombre de paramètres: {sum(p.numel() for p in model.parameters()):,}")

# Fonction d'entraînement
def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
        
        if batch_idx % 100 == 0:
            print(f'Batch {batch_idx}/{len(train_loader)} - Loss: {loss.item():.4f}')
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy

# Fonction de test
def test_model(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    
    avg_loss = test_loss / len(test_loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy

# Entraînement du modèle
print("\nDébut de l'entraînement...")
train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []

best_test_acc = 0
best_model_state = None

for epoch in range(EPOCHS):
    print(f'\nÉpoque {epoch+1}/{EPOCHS}')
    print('-' * 50)
    
    # Entraînement
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
    
    # Test
    test_loss, test_acc = test_model(model, test_loader, criterion, DEVICE)
    
    # Sauvegarde du meilleur modèle
    if test_acc > best_test_acc:
        best_test_acc = test_acc
        best_model_state = model.state_dict().copy()
    
    # Stockage des métriques
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)
    test_losses.append(test_loss)
    test_accuracies.append(test_acc)
    
    print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
    print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')

print(f'\nEntraînement terminé!')
print(f'Meilleure précision de test: {best_test_acc:.2f}%')

# Sauvegarde du modèle
print(f'\nSauvegarde du modèle dans {MODEL_PATH}...')

# Sauvegarde du meilleur modèle
model.load_state_dict(best_model_state)

# Sauvegarde complète avec métadonnées
torch.save({
    'model_state_dict': best_model_state,
    'model_architecture': {
        'input_size': 784,
        'hidden_sizes': [512, 256],
        'num_classes': 10,
        'dropout_rate': 0.2
    },
    'training_info': {
        'epochs': EPOCHS,
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'best_test_accuracy': best_test_acc,
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'test_losses': test_losses,
        'test_accuracies': test_accuracies
    }
}, MODEL_PATH)

print(f'Modèle sauvegardé avec succès!')

# Visualisation des courbes d'apprentissage
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.title('Évolution de la perte')
plt.xlabel('Époque')
plt.ylabel('Perte')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(test_accuracies, label='Test Accuracy')
plt.title('Évolution de la précision')
plt.xlabel('Époque')
plt.ylabel('Précision (%)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
plt.show()

# Test de chargement du modèle sauvegardé
print("\nTest de chargement du modèle sauvegardé...")
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

# Recréation du modèle
loaded_model = MLP(
    input_size=checkpoint['model_architecture']['input_size'],
    hidden_sizes=checkpoint['model_architecture']['hidden_sizes'],
    num_classes=checkpoint['model_architecture']['num_classes'],
    dropout_rate=checkpoint['model_architecture']['dropout_rate']
).to(DEVICE)

# Chargement des poids
loaded_model.load_state_dict(checkpoint['model_state_dict'])

# Vérification
test_loss, test_acc = test_model(loaded_model, test_loader, criterion, DEVICE)
print(f'Précision du modèle chargé: {test_acc:.2f}%')

print("\nScript terminé avec succès!")