"""fl_dp_sa: Flower Example using Differential Privacy and Secure Aggregation."""

import torch
import flwr as fl
from flwr.client import NumPyClient
from typing import Dict, List, Tuple

from .dataset import FMNISTNonIID
from .model import FMNISTNet, train_model, evaluate_model, get_model_parameters, set_model_parameters


class FMNISTClient(NumPyClient):
    """Cliente para treinamento federado com FMNIST"""
    
    def __init__(self, client_id: int, dataset: FMNISTNonIID):
        self.client_id = client_id
        self.dataset = dataset
        self.model = FMNISTNet()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Obter dados do cliente
        self.train_loader, self.val_loader = dataset.get_client_data(client_id, batch_size=32)
        
        print(f"Cliente {client_id} inicializado com {len(self.train_loader.dataset)} amostras de treino")
    
    def get_parameters(self, config: Dict[str, any]) -> List[any]:
        """Retorna os parâmetros atuais do modelo"""
        return get_model_parameters(self.model)
    
    def set_parameters(self, parameters: List[any]) -> None:
        """Define os parâmetros do modelo"""
        set_model_parameters(self.model, parameters)
    
    def fit(self, parameters: List[any], config: Dict[str, any]) -> Tuple[List[any], int, Dict[str, any]]:
        """Treina o modelo local"""
        # Definir parâmetros recebidos do servidor
        self.set_parameters(parameters)
        
        train_accuracy, val_accuracy = train_model(
            self.model, 
            self.train_loader, 
            self.val_loader,
            epochs=1, 
            lr=0.001,  
            device=self.device
        )
        
        # Calcular loss de validação
        val_loss = self._calculate_loss()
        
        # Retornar parâmetros atualizados e métricas
        return (
            get_model_parameters(self.model),
            len(self.train_loader.dataset),
            {
                "accuracy": val_accuracy,
                "loss": val_loss,
                "train_accuracy": train_accuracy
            }
        )
    
    def evaluate(self, parameters: List[any], config: Dict[str, any]) -> Tuple[float, int, Dict[str, any]]:
        """Avalia o modelo local"""
        # Definir parâmetros recebidos do servidor
        self.set_parameters(parameters)
        
        # Avaliar modelo
        accuracy = evaluate_model(self.model, self.val_loader, self.device)
        loss = self._calculate_loss()
        
        return (
            loss,
            len(self.val_loader.dataset),
            {"accuracy": accuracy}
        )
    
    def _calculate_loss(self) -> float:
        """Calcula a loss de validação"""
        self.model.eval()
        total_loss = 0.0
        criterion = torch.nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = criterion(output, target)
                total_loss += loss.item()
        
        return total_loss / len(self.val_loader)


def create_client_fn(dataset: FMNISTNonIID):
    """Cria uma função de cliente para a simulação"""
    
    def client_fn(cid: str) -> FMNISTClient:
        """Retorna um cliente para o ID especificado"""
        client_id = int(cid)
        return FMNISTClient(client_id, dataset)
    
    return client_fn


# Configuração global do dataset (será usado na simulação)
DATASET = None

def get_dataset():
    """Retorna o dataset global"""
    global DATASET
    if DATASET is None:
        DATASET = FMNISTNonIID(num_clients=50, alpha=0.5)
    return DATASET


# App do cliente para uso com Flower
app = fl.client.ClientApp(
    client_fn=lambda cid: FMNISTClient(int(cid), get_dataset())
)
