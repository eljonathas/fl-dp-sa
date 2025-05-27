import random
import numpy as np
from typing import List, Tuple
import flwr as fl
from flwr.common import Parameters, FitRes
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg


class PowerOfChoiceSelection:
    """Implementação da estratégia Power of Choice"""
    
    def __init__(self, d: int = 2):
        """
        Inicializa Power of Choice
        
        Args:
            d: Número de clientes candidatos a serem considerados (power of choice)
        """
        self.d = d
        self.client_scores = {}  # Histórico de performance dos clientes
    
    def update_client_score(self, client_id: int, accuracy: float, loss: float):
        """Atualiza o score de um cliente baseado na performance"""
        # Score baseado na acurácia
        score = accuracy
        
        if client_id not in self.client_scores:
            self.client_scores[client_id] = []
        
        self.client_scores[client_id].append(score)
        
        # Manter apenas os últimos 5 scores para evitar bias histórico
        if len(self.client_scores[client_id]) > 5:
            self.client_scores[client_id] = self.client_scores[client_id][-5:]
    
    def get_client_score(self, client_id: int) -> float:
        """Retorna o score médio de um cliente"""
        if client_id not in self.client_scores or not self.client_scores[client_id]:
            return 0.0  # Score neutro para clientes novos
        
        return np.mean(self.client_scores[client_id])
    
    def select_clients(self, available_clients: List[int], num_clients: int) -> List[int]:
        """
        Seleciona clientes usando Power of Choice
        
        Para cada slot:
        1. Seleciona d clientes candidatos aleatoriamente
        2. Escolhe o melhor entre eles baseado no score
        """
        selected_clients = []
        remaining_clients = available_clients.copy()
        
        for _ in range(min(num_clients, len(available_clients))):
            if not remaining_clients:
                break
            
            # Selecionar d candidatos (ou todos se restarem menos que d)
            candidates = random.sample(
                remaining_clients, 
                min(self.d, len(remaining_clients))
            )
            
            # Escolher o melhor candidato baseado no score
            best_client = max(candidates, key=self.get_client_score)
            
            selected_clients.append(best_client)
            remaining_clients.remove(best_client)
        
        return selected_clients


class FedAvgStrategy(FedAvg):
    """Estratégia FedAvg tradicional com seleção aleatória"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.round_accuracies = []
        self.round_losses = []
    
    def aggregate_fit(self, server_round: int, results: List[Tuple[ClientProxy, FitRes]], failures):
        """Agrega os resultados do treinamento"""
        # Chamar agregação padrão do FedAvg
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )
        
        # Coletar métricas
        if results:
            accuracies = [res.metrics.get("accuracy", 0.0) for _, res in results]
            losses = [res.metrics.get("loss", 0.0) for _, res in results]
            
            avg_accuracy = np.mean(accuracies)
            avg_loss = np.mean(losses)
            
            self.round_accuracies.append(avg_accuracy)
            self.round_losses.append(avg_loss)
            
            print(f"Rodada {server_round} - FedAvg - Acurácia média: {avg_accuracy:.2f}%")
        
        return aggregated_parameters, aggregated_metrics


class PowerOfChoiceStrategy(FedAvg):
    """Estratégia Power of Choice"""
    
    def __init__(self, d: int = 2, **kwargs):
        super().__init__(**kwargs)
        self.selection_strategy = PowerOfChoiceSelection(d=d)
        self.round_accuracies = []
        self.round_losses = []
    
    def aggregate_fit(self, server_round: int, results: List[Tuple[ClientProxy, FitRes]], failures):
        """Agrega os resultados do treinamento"""
        # Atualizar scores dos clientes baseado na performance
        for client_proxy, fit_res in results:
            client_id = int(client_proxy.cid)
            accuracy = fit_res.metrics.get("accuracy", 0.0)
            loss = fit_res.metrics.get("loss", 0.0)
            
            self.selection_strategy.update_client_score(client_id, accuracy, loss)
        
        # Chamar agregação padrão do FedAvg
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )
        
        # Coletar métricas
        if results:
            accuracies = [res.metrics.get("accuracy", 0.0) for _, res in results]
            losses = [res.metrics.get("loss", 0.0) for _, res in results]
            
            avg_accuracy = np.mean(accuracies)
            avg_loss = np.mean(losses)
            
            self.round_accuracies.append(avg_accuracy)
            self.round_losses.append(avg_loss)
            
            print(f"Rodada {server_round} - Power of Choice - Acurácia média: {avg_accuracy:.2f}%")
        
        return aggregated_parameters, aggregated_metrics 