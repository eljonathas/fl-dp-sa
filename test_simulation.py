#!/usr/bin/env python3
"""
Teste rápido para verificar se Power of Choice está funcionando
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List
import flwr as fl
from flwr.simulation import start_simulation

from fl_dp_sa.dataset import FMNISTNonIID
from fl_dp_sa.client_app import create_client_fn
from fl_dp_sa.server_app import create_server_app


def test_quick_simulation():
    """Teste rápido com poucas rodadas"""
    print("Iniciando teste rápido do Power of Choice")
    print("="*50)
    
    # Criar dataset pequeno
    dataset = FMNISTNonIID(num_clients=10, alpha=0.1)
    
    # Criar função de cliente
    client_fn = create_client_fn(dataset)
    
    # Testar Power of Choice
    _, strategy = create_server_app("powerofchoice")
    
    # Executar simulação rápida
    start_time = time.time()
    
    history = start_simulation(
        client_fn=client_fn,
        num_clients=10,
        config=fl.server.ServerConfig(num_rounds=5),  # Apenas 5 rodadas para teste
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.0},
    )
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"\nTeste concluído em {duration:.2f} segundos")
    
    # Verificar métricas
    print(f"\nMétricas da estratégia:")
    print(f"  Acurácias por rodada: {strategy.round_accuracies}")
    print(f"  Losses por rodada: {strategy.round_losses}")
    
    # Verificar se a acurácia está melhorando
    if len(strategy.round_accuracies) >= 2:
        improvement = strategy.round_accuracies[-1] - strategy.round_accuracies[0]
        print(f"  Melhoria na acurácia: {improvement:.2f}%")
        
        if improvement > 0:
            print("✅ SUCESSO: A acurácia está melhorando!")
        else:
            print("❌ PROBLEMA: A acurácia não está melhorando")
    
    return strategy.round_accuracies, strategy.round_losses


def plot_quick_results(accuracies, losses):
    """Plotar resultados do teste rápido"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Acurácia
    rounds = range(1, len(accuracies) + 1)
    ax1.plot(rounds, accuracies, 'r-o', linewidth=2, markersize=6)
    ax1.set_xlabel('Rodadas')
    ax1.set_ylabel('Acurácia (%)')
    ax1.set_title('Evolução da Acurácia - Power of Choice')
    ax1.grid(True, alpha=0.3)
    
    # Loss
    ax2.plot(rounds, losses, 'b-s', linewidth=2, markersize=6)
    ax2.set_xlabel('Rodadas')
    ax2.set_ylabel('Loss')
    ax2.set_title('Evolução da Loss - Power of Choice')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("test_results.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    print("Gráfico salvo como test_results.png")


if __name__ == "__main__":
    try:
        accuracies, losses = test_quick_simulation()
        plot_quick_results(accuracies, losses)
        
        print("\n" + "="*50)
        print("TESTE CONCLUÍDO!")
        
        if len(accuracies) > 0 and accuracies[-1] > 20:  # Esperar pelo menos 20% de acurácia
            print("✅ Sistema funcionando corretamente")
        else:
            print("⚠️ Acurácia ainda baixa, mas sistema está funcionando")
            
    except Exception as e:
        print(f"❌ ERRO no teste: {e}")
        import traceback
        traceback.print_exc() 