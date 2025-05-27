import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
from typing import List, Tuple, Dict
import random


class FMNISTNonIID:
    """Classe para gerenciar o dataset FMNIST com distribuição Non-IID"""
    
    def __init__(self, num_clients: int = 50, alpha: float = 0.5):
        """
        Inicializa o dataset FMNIST com distribuição Non-IID
        
        Args:
            num_clients: Número de clientes
            alpha: Parâmetro de concentração da distribuição Dirichlet (menor = mais Non-IID)
        """
        self.num_clients = num_clients
        self.alpha = alpha
        self.num_classes = 10
        
        # Transformações para o dataset
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))
        ])
        
        # Carregar datasets
        self.train_dataset = torchvision.datasets.FashionMNIST(
            root="./data", train=True, download=True, transform=self.transform
        )
        
        self.test_dataset = torchvision.datasets.FashionMNIST(
            root="./data", train=False, download=True, transform=self.transform
        )
        
        # Criar distribuição Non-IID
        self.client_indices = self._create_non_iid_distribution()
        
    def _create_non_iid_distribution(self) -> Dict[int, List[int]]:
        """
        Cria distribuição Non-IID usando distribuição Dirichlet
        
        Returns:
            Dicionário mapeando client_id para lista de índices
        """
        # Obter labels do dataset de treino
        labels = np.array([self.train_dataset[i][1] for i in range(len(self.train_dataset))])
        
        # Agrupar índices por classe
        class_indices = {}
        for class_id in range(self.num_classes):
            class_indices[class_id] = np.where(labels == class_id)[0].tolist()
        
        # Distribuir usando Dirichlet
        client_indices = {i: [] for i in range(self.num_clients)}
        
        for class_id in range(self.num_classes):
            # Gerar proporções usando distribuição Dirichlet
            proportions = np.random.dirichlet([self.alpha] * self.num_clients)
            
            # Distribuir amostras da classe atual
            class_samples = class_indices[class_id]
            np.random.shuffle(class_samples)
            
            start_idx = 0
            for client_id in range(self.num_clients):
                # Calcular número de amostras para este cliente
                num_samples = int(proportions[client_id] * len(class_samples))
                
                # Evitar que um cliente fique sem amostras
                if client_id == self.num_clients - 1:
                    end_idx = len(class_samples)
                else:
                    end_idx = start_idx + num_samples
                
                # Adicionar amostras ao cliente
                client_indices[client_id].extend(class_samples[start_idx:end_idx])
                start_idx = end_idx
        
        # Embaralhar índices de cada cliente
        for client_id in range(self.num_clients):
            random.shuffle(client_indices[client_id])
            
        return client_indices
    
    def get_client_data(self, client_id: int, batch_size: int = 32) -> Tuple[DataLoader, DataLoader]:
        """
        Retorna DataLoaders de treino e validação para um cliente específico
        
        Args:
            client_id: ID do cliente
            batch_size: Tamanho do batch
            
        Returns:
            Tuple com DataLoaders de treino e validação
        """
        if client_id not in self.client_indices:
            raise ValueError(f"Cliente {client_id} não existe")
        
        # Obter índices do cliente
        indices = self.client_indices[client_id]
        
        # Dividir em treino e validação (80/20)
        split_idx = int(0.8 * len(indices))
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]
        
        # Criar subsets
        train_subset = Subset(self.train_dataset, train_indices)
        val_subset = Subset(self.train_dataset, val_indices)
        
        # Criar DataLoaders
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader
    
    def get_test_data(self, batch_size: int = 32) -> DataLoader:
        """
        Retorna DataLoader para o conjunto de teste global
        
        Args:
            batch_size: Tamanho do batch
            
        Returns:
            DataLoader do conjunto de teste
        """
        return DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)
    
    def get_client_stats(self) -> Dict[str, any]:
        """
        Retorna estatísticas sobre a distribuição dos dados
        
        Returns:
            Dicionário com estatísticas
        """
        stats = {
            'num_clients': self.num_clients,
            'alpha': self.alpha,
            'samples_per_client': [],
            'class_distribution': {}
        }
        
        # Calcular amostras por cliente
        for client_id in range(self.num_clients):
            stats['samples_per_client'].append(len(self.client_indices[client_id]))
        
        # Calcular distribuição de classes por cliente
        for client_id in range(self.num_clients):
            indices = self.client_indices[client_id]
            labels = [self.train_dataset[i][1] for i in indices]
            class_counts = np.bincount(labels, minlength=self.num_classes)
            stats['class_distribution'][client_id] = class_counts.tolist()
        
        return stats 