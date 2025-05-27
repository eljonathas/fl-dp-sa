import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class FMNISTNet(nn.Module):
    """Rede Neural Convolucional para Fashion-MNIST"""
    
    def __init__(self, num_classes: int = 10):
        super(FMNISTNet, self).__init__()
        
        # Camadas convolucionais
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Camadas de pooling
        self.pool = nn.MaxPool2d(2, 2)
        
        # Dropout para regularização
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        
        # Camadas fully connected
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Primeira camada convolucional + ReLU + Pooling
        x = self.pool(F.relu(self.conv1(x)))
        
        # Segunda camada convolucional + ReLU + Pooling
        x = self.pool(F.relu(self.conv2(x)))
        
        # Terceira camada convolucional + ReLU + Pooling
        x = self.pool(F.relu(self.conv3(x)))
        
        # Flatten para as camadas fully connected
        x = x.view(-1, 128 * 3 * 3)
        
        # Primeira camada fully connected + ReLU + Dropout
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        
        # Segunda camada fully connected + ReLU + Dropout
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        
        # Camada de saída
        x = self.fc3(x)
        
        return x


def train_model(model: nn.Module, train_loader, val_loader, epochs: int = 5, 
                lr: float = 0.001, device: str = "cpu") -> Tuple[float, float]:
    """
    Treina o modelo e retorna a acurácia de treino e validação
    """
    model.to(device)
    model.train()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for _ in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for _, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    # Calcular acurácia de treino
    train_accuracy = 100 * correct / total
    
    # Calcular acurácia de validação
    val_accuracy = evaluate_model(model, val_loader, device)
    
    return train_accuracy, val_accuracy


def evaluate_model(model: nn.Module, data_loader, device: str = "cpu") -> float:
    """
    Avalia o modelo e retorna a acurácia
    """
    model.to(device)
    model.eval()
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy


def get_model_parameters(model: nn.Module):
    """Retorna os parâmetros do modelo como uma lista de numpy arrays"""
    return [param.data.cpu().numpy() for param in model.parameters()]


def set_model_parameters(model: nn.Module, parameters):
    """Define os parâmetros do modelo a partir de numpy arrays ou tensors"""
    for param, new_param in zip(model.parameters(), parameters):
        if isinstance(new_param, torch.Tensor):
            param.data = new_param.clone()
        else:
            # Converter numpy array para tensor PyTorch
            param.data = torch.tensor(new_param, dtype=param.dtype, device=param.device) 