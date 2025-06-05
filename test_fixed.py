#!/usr/bin/env python3
"""
Teste das correções implementadas
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import flwr as fl
from flwr.simulation import start_simulation

from fl_dp_sa.dataset import FMNISTNonIID
from fl_dp_sa.client_app import create_client_fn
from fl_dp_sa.server_app import create_server_app


def test_fixed_implementation():
    """Teste para verificar se as correções funcionaram"""
    print("Testando implementação corrigida...")
    print("="*50)
    
    # Teste com configuração pequena
    dataset = FMNISTNonIID(num_clients=10, alpha=0.1)
    client_fn = create_client_fn(dataset)
    
    # Testar FedAvg primeiro
    print("\n1. Testando FedAvg...")
    _, fedavg_strategy = create_server_app("fedavg")
    
    history_fedavg = start_simulation(
        client_fn=client_fn,
        num_clients=10,
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=fedavg_strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.0},
    )
    
    print(f"FedAvg - Métricas coletadas:")
    print(f"  round_accuracies: {fedavg_strategy.round_accuracies}")
    print(f"  round_losses: {fedavg_strategy.round_losses}")
    
    # Testar Power of Choice
    print("\n2. Testando Power of Choice...")
    _, poc_strategy = create_server_app("powerofchoice")
    
    history_poc = start_simulation(
        client_fn=client_fn,
        num_clients=10,
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=poc_strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.0},
    )
    
    print(f"Power of Choice - Métricas coletadas:")
    print(f"  round_accuracies: {poc_strategy.round_accuracies}")
    print(f"  round_losses: {poc_strategy.round_losses}")
    
    # Verificar se as correções funcionaram
    print("\n3. Verificação dos resultados:")
    
    fedavg_ok = len(fedavg_strategy.round_accuracies) == 3 and len(fedavg_strategy.round_losses) == 3
    poc_ok = len(poc_strategy.round_accuracies) == 3 and len(poc_strategy.round_losses) == 3
    
    print(f"  FedAvg métricas OK: {'✅' if fedavg_ok else '❌'}")
    print(f"  Power of Choice métricas OK: {'✅' if poc_ok else '❌'}")
    
    # Verificar se a acurácia global está evoluindo
    fedavg_global_acc = []
    poc_global_acc = []
    
    if hasattr(history_fedavg, 'metrics_centralized') and 'accuracy' in history_fedavg.metrics_centralized:
        fedavg_global_acc = [acc for _, acc in history_fedavg.metrics_centralized['accuracy']]
    
    if hasattr(history_poc, 'metrics_centralized') and 'accuracy' in history_poc.metrics_centralized:
        poc_global_acc = [acc for _, acc in history_poc.metrics_centralized['accuracy']]
    
    print(f"  FedAvg acurácia global: {fedavg_global_acc}")
    print(f"  Power of Choice acurácia global: {poc_global_acc}")
    
    # Verificar evolução
    fedavg_evolving = len(set(fedavg_global_acc)) > 1 if fedavg_global_acc else False
    poc_evolving = len(set(poc_global_acc)) > 1 if poc_global_acc else False
    
    print(f"  FedAvg acurácia evoluindo: {'✅' if fedavg_evolving else '❌'}")
    print(f"  Power of Choice acurácia evoluindo: {'✅' if poc_evolving else '❌'}")
    
    return {
        'fedavg_strategy': fedavg_strategy,
        'poc_strategy': poc_strategy,
        'fedavg_global_acc': fedavg_global_acc,
        'poc_global_acc': poc_global_acc
    }


def plot_test_results(results):
    """Plotar resultados do teste"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    fedavg = results['fedavg_strategy']
    poc = results['poc_strategy']
    
    # Acurácia dos clientes
    if fedavg.round_accuracies and poc.round_accuracies:
        rounds = range(1, len(fedavg.round_accuracies) + 1)
        axes[0,0].plot(rounds, fedavg.round_accuracies, 'b-o', label='FedAvg')
        axes[0,0].plot(rounds, poc.round_accuracies, 'r-s', label='Power of Choice')
        axes[0,0].set_title('Acurácia dos Clientes')
        axes[0,0].set_ylabel('Acurácia (%)')
        axes[0,0].legend()
        axes[0,0].grid(True)
    
    # Loss dos clientes
    if fedavg.round_losses and poc.round_losses:
        rounds = range(1, len(fedavg.round_losses) + 1)
        axes[0,1].plot(rounds, fedavg.round_losses, 'b-o', label='FedAvg')
        axes[0,1].plot(rounds, poc.round_losses, 'r-s', label='Power of Choice')
        axes[0,1].set_title('Loss dos Clientes')
        axes[0,1].set_ylabel('Loss')
        axes[0,1].legend()
        axes[0,1].grid(True)
    
    # Acurácia global
    if results['fedavg_global_acc'] and results['poc_global_acc']:
        rounds = range(len(results['fedavg_global_acc']))
        axes[1,0].plot(rounds, results['fedavg_global_acc'], 'b-o', label='FedAvg')
        axes[1,0].plot(rounds, results['poc_global_acc'], 'r-s', label='Power of Choice')
        axes[1,0].set_title('Acurácia Global')
        axes[1,0].set_ylabel('Acurácia (%)')
        axes[1,0].legend()
        axes[1,0].grid(True)
    
    # Comparação final
    final_fedavg = fedavg.round_accuracies[-1] if fedavg.round_accuracies else 0
    final_poc = poc.round_accuracies[-1] if poc.round_accuracies else 0
    
    axes[1,1].bar(['FedAvg', 'Power of Choice'], [final_fedavg, final_poc], 
                  color=['blue', 'red'], alpha=0.7)
    axes[1,1].set_title('Acurácia Final')
    axes[1,1].set_ylabel('Acurácia (%)')
    
    plt.tight_layout()
    plt.savefig('test_results_fixed.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("Gráfico salvo como test_results_fixed.png")


if __name__ == "__main__":
    try:
        results = test_fixed_implementation()
        plot_test_results(results)
        
        print("\n" + "="*50)
        print("TESTE DAS CORREÇÕES CONCLUÍDO!")
        print("="*50)
        
    except Exception as e:
        print(f"❌ ERRO no teste: {e}")
        import traceback
        traceback.print_exc() 