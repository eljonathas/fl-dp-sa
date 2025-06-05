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
        self.client_losses = {}  # Histórico de loss local dos clientes
        self.current_round_losses = {}  # Loss atual dos clientes para seleção
        self.round_count = 0
    
    def update_client_loss(self, client_id: str, local_loss: float):
        """Atualiza o histórico de loss local de um cliente"""
        if client_id not in self.client_losses:
            self.client_losses[client_id] = []
        
        self.client_losses[client_id].append(local_loss)
        
        # Manter apenas os últimos 5 valores para melhor histórico
        if len(self.client_losses[client_id]) > 5:
            self.client_losses[client_id] = self.client_losses[client_id][-5:]
    
    def get_client_loss_score(self, client_id: str) -> float:
        """
        Retorna o score de loss de um cliente
        Clientes com maior loss local têm prioridade (como no artigo original)
        """
        if client_id not in self.client_losses or not self.client_losses[client_id]:
            # Para clientes sem histórico, usar loss aleatória para dar chance igual
            return random.uniform(0.5, 2.0)
        
        # Usar a média das últimas losses como score, priorizando perdas maiores
        return np.mean(self.client_losses[client_id])
    
    def select_clients_power_of_choice(self, available_clients: List[ClientProxy], num_clients: int) -> List[ClientProxy]:
        """
        Implementa o algoritmo Power of Choice original:
        
        1. Para cada slot de cliente:
           - Seleciona d candidatos aleatoriamente
           - Escolhe o candidato com maior loss local (ou score aleatório se sem histórico)
           - Remove o selecionado da lista de disponíveis
        
        Args:
            available_clients: Lista de ClientProxy disponíveis
            num_clients: Número de clientes a selecionar
            
        Returns:
            Lista de ClientProxy selecionados
        """
        selected_clients = []
        remaining_clients = available_clients.copy()
        
        # Se não temos histórico suficiente, aumentar aleatoriedade
        has_history = len(self.client_losses) > 0
        
        for _ in range(min(num_clients, len(available_clients))):
            if not remaining_clients:
                break
            
            # Passo 1: Selecionar d candidatos aleatoriamente
            candidates = random.sample(
                remaining_clients, 
                min(self.d, len(remaining_clients))
            )
            
            # Passo 2: Escolher baseado em loss ou aleatoriedade inteligente
            if has_history and self.round_count > 2:
                # Usar histórico de losses para seleção inteligente
                best_client = max(candidates, key=lambda client: self.get_client_loss_score(self._get_client_id(client)))
            else:
                # Primeiras rodadas: seleção aleatória entre candidatos (ainda é power of choice)
                best_client = random.choice(candidates)
            
            selected_clients.append(best_client)
            remaining_clients.remove(best_client)
        
        return selected_clients
    
    def _get_client_id(self, client: ClientProxy) -> str:
        """Extrai ID do cliente de forma compatível"""
        if hasattr(client, 'cid'):
            return str(client.cid)
        else:
            return str(client)


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
            
            print(f"Rodada {server_round} - FedAvg - Acurácia média: {avg_accuracy:.2f}%, Loss média: {avg_loss:.4f}")
        
        return aggregated_parameters, aggregated_metrics


class PowerOfChoiceStrategy(FedAvg):
    """
    Implementação corrigida da estratégia Power of Choice conforme o artigo original.
    
    O algoritmo funciona da seguinte forma:
    1. Seleciona um conjunto candidato de d clientes aleatoriamente
    2. Escolhe entre esses candidatos baseado em critério de loss (ou aleatório inicialmente)
    3. Repete até selecionar m clientes
    """
    
    def __init__(self, d: int = 15, **kwargs):
        super().__init__(**kwargs)
        self.selection_strategy = PowerOfChoiceSelection(d=d)
        self.round_accuracies = []
        self.round_losses = []
        
        print(f"Inicializando Power of Choice com d={d} candidatos")
    
    def configure_fit(self, server_round: int, parameters: Parameters, client_manager) -> List[Tuple[ClientProxy, FitIns]]:
        """
        Configura a seleção de clientes usando Power of Choice
        """
        # Obter todos os clientes disponíveis
        available_clients = list(client_manager.all())
        
        if len(available_clients) == 0:
            return []
        
        # Calcular número de clientes para seleção
        num_clients = max(1, int(len(available_clients) * self.fraction_fit))
        num_clients = min(num_clients, len(available_clients))
        
        print(f"Rodada {server_round}: Power of Choice selecionando {num_clients} de {len(available_clients)} clientes")
        
        # Atualizar contador de rodadas
        self.selection_strategy.round_count = server_round
        
        # Aplicar Power of Choice - funciona desde a primeira rodada
        selected_clients = self.selection_strategy.select_clients_power_of_choice(
            available_clients, num_clients
        )
        
        # Debug: mostrar clientes selecionados
        selected_ids = [self.selection_strategy._get_client_id(client) for client in selected_clients]
        print(f"  Clientes selecionados: {selected_ids}")
        
        # Se temos histórico, mostrar scores
        if self.selection_strategy.client_losses:
            print(f"  Scores dos selecionados:")
            for client in selected_clients:
                client_id = self.selection_strategy._get_client_id(client)
                score = self.selection_strategy.get_client_loss_score(client_id)
                history = self.selection_strategy.client_losses.get(client_id, [])
                print(f"    Cliente {client_id}: score={score:.4f}, histórico={len(history)} valores")
        
        # Criar instruções de fit para os clientes selecionados
        fit_ins = FitIns(parameters, {})
        return [(client, fit_ins) for client in selected_clients]
    
    def aggregate_fit(self, server_round: int, results: List[Tuple[ClientProxy, FitRes]], failures):
        """Agrega os resultados do treinamento e atualiza histórico de losses"""
        
        # Atualizar scores dos clientes baseado na loss local (critério do artigo)
        for client_proxy, fit_res in results:
            # Extrair client ID
            client_id = self.selection_strategy._get_client_id(client_proxy)
            
            # No artigo original, o critério é a LOCAL LOSS, não a acurácia
            local_loss = fit_res.metrics.get("loss", 1.0)  # Default 1.0 se não encontrar
            
            # Atualizar histórico de losses
            self.selection_strategy.update_client_loss(client_id, local_loss)
        
        # Chamar agregação padrão do FedAvg
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )
        
        # Coletar métricas para análise
        if results:
            accuracies = [res.metrics.get("accuracy", 0.0) for _, res in results]
            losses = [res.metrics.get("loss", 0.0) for _, res in results]
            
            avg_accuracy = np.mean(accuracies)
            avg_loss = np.mean(losses)
            
            self.round_accuracies.append(avg_accuracy)
            self.round_losses.append(avg_loss)
            
            print(f"Rodada {server_round} - Power of Choice - Acurácia média: {avg_accuracy:.2f}%, Loss média: {avg_loss:.4f}")
            
            # Debug: mostrar estatísticas do histórico
            total_clients_with_history = len(self.selection_strategy.client_losses)
            print(f"  Histórico: {total_clients_with_history} clientes com dados de loss")
        
        return aggregated_parameters, aggregated_metrics 