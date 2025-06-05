import random
import numpy as np
from typing import List, Tuple, Optional, Dict, Union
import flwr as fl
from flwr.common import Parameters, FitRes, FitIns, Scalar
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg


class PowerOfChoiceSelection:
    """Implementação da estratégia Power of Choice baseada no artigo original"""
    
    def __init__(self, d: int = 10):
        """
        Inicializa Power of Choice
        
        Args:
            d: Número de clientes candidatos a serem considerados (power of choice)
        """
        self.d = d
        self.client_losses = {}
        self.client_ratios = {}
        self.num_clients = 0
    
    def update_client_info(self, client_id: str, local_loss: float, data_ratio: float = None):
        """Atualiza informações do cliente"""
        self.client_losses[client_id] = local_loss
        if data_ratio is not None:
            self.client_ratios[client_id] = data_ratio
    
    def set_client_ratios(self, client_ids: List[str]):
        """Define ratios uniformes se não especificados"""
        if not self.client_ratios:
            ratio = 1.0 / len(client_ids)
            for client_id in client_ids:
                self.client_ratios[client_id] = ratio
        self.num_clients = len(client_ids)
    
    def select_clients_power_of_choice(self, available_clients: List[ClientProxy], num_clients: int) -> List[ClientProxy]:
        """
        Implementa o algoritmo Power of Choice original do artigo:
        
        1. Se primeira rodada (sem histórico), seleciona m clientes aleatoriamente  
        2. Caso contrário:
           - Seleciona d clientes com probabilidade proporcional ao tamanho do dataset (sem reposição)
           - Ordena por loss decrescente
           - Seleciona os top m clientes
        """
        client_ids = [self._get_client_id(client) for client in available_clients]
        self.set_client_ratios(client_ids)
        
        # Primeira rodada: seleção aleatória uniforme (como no artigo original)
        if not self.client_losses:
            selected_indices = np.random.choice(
                len(available_clients), 
                size=min(num_clients, len(available_clients)), 
                replace=False
            )
            return [available_clients[i] for i in selected_indices]
        
        # Power of Choice Algorithm (como implementado no artigo)
        # Passo 1: Selecionar d candidatos com probabilidade proporcional ao dataset size
        ratios = np.array([self.client_ratios.get(cid, 1.0/len(client_ids)) for cid in client_ids])
        ratios = ratios / ratios.sum()  # Normalizar para garantir soma = 1
        
        d_candidates = min(self.d, len(available_clients))
        
        # Seleção com reposição para permitir probabilidades proporcionais
        candidate_indices = np.random.choice(
            len(available_clients), 
            p=ratios, 
            size=d_candidates, 
            replace=False
        )
        
        # Passo 2: Ordenar candidatos selecionados por loss decrescente
        candidates_with_loss = []
        for idx in candidate_indices:
            client_id = client_ids[idx]
            loss = self.client_losses.get(client_id, 0.0)
            candidates_with_loss.append((loss, idx, available_clients[idx]))
        
        # Ordenar por loss decrescente (maior loss = maior prioridade)
        candidates_with_loss.sort(key=lambda x: x[0], reverse=True)
        
        # Passo 3: Selecionar os top m clientes da lista ordenada
        selected_clients = []
        for i in range(min(num_clients, len(candidates_with_loss))):
            selected_clients.append(candidates_with_loss[i][2])
        
        return selected_clients
    
    def _get_client_id(self, client: ClientProxy) -> str:
        """Extrai ID do cliente"""
        if hasattr(client, 'cid'):
            return str(client.cid)
        else:
            return str(client)


class FedAvgStrategy(FedAvg):
    """Estratégia FedAvg tradicional"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.round_accuracies = []
        self.round_losses = []
    
    def aggregate_fit(self, server_round: int, results: List[Tuple[ClientProxy, FitRes]], failures):
        """Agrega os resultados do treinamento"""
        print(f"FedAvg - Rodada {server_round}: Recebidos {len(results)} resultados")
        
        # Coletar métricas ANTES da agregação
        if results:
            accuracies = []
            losses = []
            
            for _, fit_res in results:
                accuracy = fit_res.metrics.get("accuracy", 0.0)
                loss = fit_res.metrics.get("loss", 0.0)
                accuracies.append(accuracy)
                losses.append(loss)
            
            avg_accuracy = np.mean(accuracies)
            avg_loss = np.mean(losses)
            
            self.round_accuracies.append(avg_accuracy)
            self.round_losses.append(avg_loss)
            
            print(f"Rodada {server_round} - FedAvg - Acurácia Média: {avg_accuracy:.2f}%, Loss Média: {avg_loss:.4f}")
        
        # Agregação padrão
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )
        
        return aggregated_parameters, aggregated_metrics


class PowerOfChoiceStrategy(FedAvg):
    """Estratégia Power of Choice implementada conforme o artigo original"""
    
    def __init__(self, d: int = 15, **kwargs):
        super().__init__(**kwargs)
        self.selection_strategy = PowerOfChoiceSelection(d=d)
        self.round_accuracies = []
        self.round_losses = []
        
        print(f"Power of Choice iniciado com d={d}")
    
    def configure_fit(self, server_round: int, parameters: Parameters, client_manager) -> List[Tuple[ClientProxy, FitIns]]:
        """Configura seleção de clientes usando Power of Choice"""
        available_clients = list(client_manager.all())
        
        if len(available_clients) == 0:
            return []
        
        num_clients = max(1, int(len(available_clients) * self.fraction_fit))
        num_clients = min(num_clients, len(available_clients))
        
        print(f"Rodada {server_round}: Selecionando {num_clients} de {len(available_clients)} clientes")
        
        # Aplicar Power of Choice
        selected_clients = self.selection_strategy.select_clients_power_of_choice(
            available_clients, num_clients
        )
        
        selected_ids = [self.selection_strategy._get_client_id(client) for client in selected_clients]
        print(f"  Clientes selecionados: {selected_ids}")
        
        # Debug: mostrar informações de seleção
        if self.selection_strategy.client_losses:
            print(f"  Histórico de losses disponível para {len(self.selection_strategy.client_losses)} clientes")
            selected_losses = []
            for client in selected_clients:
                client_id = self.selection_strategy._get_client_id(client)
                loss = self.selection_strategy.client_losses.get(client_id, 0.0)
                selected_losses.append(loss)
            print(f"  Losses dos selecionados: {[f'{l:.4f}' for l in selected_losses]}")
        
        fit_ins = FitIns(parameters, {})
        return [(client, fit_ins) for client in selected_clients]
    
    def aggregate_fit(self, server_round: int, results: List[Tuple[ClientProxy, FitRes]], failures):
        """Agrega resultados e atualiza histórico de losses"""
        
        # Debug: verificar se há resultados
        print(f"Power of Choice - Rodada {server_round}: Recebidos {len(results)} resultados")
        
        # Coletar métricas ANTES da agregação
        if results:
            accuracies = []
            losses = []
            
            for client_proxy, fit_res in results:
                client_id = self.selection_strategy._get_client_id(client_proxy)
                
                # Extrair métricas
                accuracy = fit_res.metrics.get("accuracy", 0.0)
                loss = fit_res.metrics.get("loss", 1.0)
                
                accuracies.append(accuracy)
                losses.append(loss)
                
                # Atualizar histórico de losses para próxima rodada  
                self.selection_strategy.update_client_info(client_id, loss)
                
                print(f"  Cliente {client_id}: acc={accuracy:.2f}%, loss={loss:.4f}")
            
            # Calcular médias
            avg_accuracy = np.mean(accuracies)
            avg_loss = np.mean(losses)
            
            # Armazenar métricas
            self.round_accuracies.append(avg_accuracy)
            self.round_losses.append(avg_loss)
            
            print(f"Rodada {server_round} - Power of Choice - Acurácia Média: {avg_accuracy:.2f}%, Loss Média: {avg_loss:.4f}")
            print(f"  Clientes com histórico: {len(self.selection_strategy.client_losses)}")
        
        # Agregação padrão (sempre deve ser chamada)
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )
        
        return aggregated_parameters, aggregated_metrics 