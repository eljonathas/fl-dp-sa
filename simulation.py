#!/usr/bin/env python3
"""
Simulação de Aprendizado Federado: FedAvg vs Power of Choice
Dataset: Fashion-MNIST com distribuição Non-IID
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import flwr as fl
from flwr.simulation import start_simulation
from typing import Dict, List

from fl_dp_sa.dataset import FMNISTNonIID
from fl_dp_sa.client_app import create_client_fn
from fl_dp_sa.server_app import create_server_app


def run_simulation(strategy_name: str, num_rounds: int = 20) -> Dict[str, List[float]]:
    """
    Executa uma simulação de aprendizado federado
    
    Args:
        strategy_name: Nome da estratégia ("fedavg" ou "powerofchoice")
        num_rounds: Número de rodadas de treinamento
        
    Returns:
        Dicionário com métricas coletadas
    """
    print(f"\n{'='*60}")
    print(f"Executando simulação: {strategy_name.upper()}")
    print(f"{'='*60}")
    
    # Criar dataset
    dataset = FMNISTNonIID(num_clients=50, alpha=0.5)
    
    # Criar função de cliente
    client_fn = create_client_fn(dataset)
    
    # Criar app do servidor
    server_app, strategy = create_server_app(strategy_name)
    
    # Configurar simulação
    start_time = time.time()
    
    # Executar simulação
    history = start_simulation(
        client_fn=client_fn,
        num_clients=50,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.0},
    )
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"\nSimulação {strategy_name} concluída em {duration:.2f} segundos")
    
    # Extrair métricas
    metrics = {
        "global_accuracy": [],
        "global_loss": [],
        "round_accuracies": strategy.round_accuracies,
        "round_losses": strategy.round_losses,
        "duration": duration
    }
    
    # Extrair métricas globais do histórico
    if hasattr(history, 'metrics_centralized') and history.metrics_centralized:
        for round_metrics in history.metrics_centralized.values():
            if 'accuracy' in round_metrics:
                metrics["global_accuracy"].append(round_metrics['accuracy'])
            if 'loss' in round_metrics:
                metrics["global_loss"].append(round_metrics['loss'])
    
    return metrics


def plot_comparison(fedavg_metrics: Dict, poc_metrics: Dict, save_path: str = "results"):
    """
    Cria gráficos comparativos entre FedAvg e Power of Choice
    
    Args:
        fedavg_metrics: Métricas do FedAvg
        poc_metrics: Métricas do Power of Choice
        save_path: Diretório para salvar os gráficos
    """
    # Criar diretório se não existir
    os.makedirs(save_path, exist_ok=True)
    
    # Configurar estilo dos gráficos
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Comparação: FedAvg vs Power of Choice\nDataset: Fashion-MNIST (Non-IID)', 
                 fontsize=16, fontweight='bold')
    
    # 1. Acurácia Global ao longo das rodadas
    ax1 = axes[0, 0]
    if fedavg_metrics["global_accuracy"]:
        rounds = range(1, len(fedavg_metrics["global_accuracy"]) + 1)
        ax1.plot(rounds, fedavg_metrics["global_accuracy"], 'b-o', label='FedAvg', linewidth=2, markersize=6)
    
    if poc_metrics["global_accuracy"]:
        rounds = range(1, len(poc_metrics["global_accuracy"]) + 1)
        ax1.plot(rounds, poc_metrics["global_accuracy"], 'r-s', label='Power of Choice', linewidth=2, markersize=6)
    
    ax1.set_xlabel('Rodadas de Treinamento')
    ax1.set_ylabel('Acurácia Global (%)')
    ax1.set_title('Evolução da Acurácia Global')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Loss Global ao longo das rodadas
    ax2 = axes[0, 1]
    if fedavg_metrics["global_loss"]:
        rounds = range(1, len(fedavg_metrics["global_loss"]) + 1)
        ax2.plot(rounds, fedavg_metrics["global_loss"], 'b-o', label='FedAvg', linewidth=2, markersize=6)
    
    if poc_metrics["global_loss"]:
        rounds = range(1, len(poc_metrics["global_loss"]) + 1)
        ax2.plot(rounds, poc_metrics["global_loss"], 'r-s', label='Power of Choice', linewidth=2, markersize=6)
    
    ax2.set_xlabel('Rodadas de Treinamento')
    ax2.set_ylabel('Loss Global')
    ax2.set_title('Evolução da Loss Global')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Acurácia Média dos Clientes por Rodada
    ax3 = axes[1, 0]
    if fedavg_metrics["round_accuracies"]:
        rounds = range(1, len(fedavg_metrics["round_accuracies"]) + 1)
        ax3.plot(rounds, fedavg_metrics["round_accuracies"], 'b-o', label='FedAvg', linewidth=2, markersize=6)
    
    if poc_metrics["round_accuracies"]:
        rounds = range(1, len(poc_metrics["round_accuracies"]) + 1)
        ax3.plot(rounds, poc_metrics["round_accuracies"], 'r-s', label='Power of Choice', linewidth=2, markersize=6)
    
    ax3.set_xlabel('Rodadas de Treinamento')
    ax3.set_ylabel('Acurácia Média dos Clientes (%)')
    ax3.set_title('Acurácia Média dos Clientes Participantes')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Comparação de Performance Final
    ax4 = axes[1, 1]
    strategies = ['FedAvg', 'Power of Choice']
    
    # Acurácia final
    fedavg_final_acc = fedavg_metrics["global_accuracy"][-1] if fedavg_metrics["global_accuracy"] else 0
    poc_final_acc = poc_metrics["global_accuracy"][-1] if poc_metrics["global_accuracy"] else 0
    final_accuracies = [fedavg_final_acc, poc_final_acc]
    
    # Gráfico de barras para acurácia final
    bars = ax4.bar(strategies, final_accuracies, color=['blue', 'red'], alpha=0.7)
    ax4.set_ylabel('Acurácia Final (%)')
    ax4.set_title('Comparação de Performance Final')
    
    # Adicionar valores nas barras
    for bar, acc in zip(bars, final_accuracies):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold')
    
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Salvar gráfico
    plt.savefig(f"{save_path}/comparison_fedavg_vs_powerofchoice.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{save_path}/comparison_fedavg_vs_powerofchoice.pdf", bbox_inches='tight')
    
    print(f"\nGráficos salvos em: {save_path}/")
    
    # Mostrar gráfico
    plt.show()


def print_summary(fedavg_metrics: Dict, poc_metrics: Dict):
    """Imprime resumo comparativo dos resultados"""
    
    print(f"\n{'='*60}")
    print("RESUMO DOS RESULTADOS")
    print(f"{'='*60}")
    
    # Acurácia final
    fedavg_final_acc = fedavg_metrics["global_accuracy"][-1] if fedavg_metrics["global_accuracy"] else 0
    poc_final_acc = poc_metrics["global_accuracy"][-1] if poc_metrics["global_accuracy"] else 0
    
    print(f"Acurácia Final:")
    print(f"  FedAvg:          {fedavg_final_acc:.2f}%")
    print(f"  Power of Choice: {poc_final_acc:.2f}%")
    print(f"  Melhoria:        {poc_final_acc - fedavg_final_acc:+.2f}%")
    
    # Tempo de execução
    print(f"\nTempo de Execução:")
    print(f"  FedAvg:          {fedavg_metrics['duration']:.2f}s")
    print(f"  Power of Choice: {poc_metrics['duration']:.2f}s")
    
    # Convergência
    if fedavg_metrics["global_accuracy"] and poc_metrics["global_accuracy"]:
        fedavg_convergence = np.mean(np.diff(fedavg_metrics["global_accuracy"][-5:]))
        poc_convergence = np.mean(np.diff(poc_metrics["global_accuracy"][-5:]))
        
        print(f"\nTaxa de Convergência (últimas 5 rodadas):")
        print(f"  FedAvg:          {fedavg_convergence:.3f}%/rodada")
        print(f"  Power of Choice: {poc_convergence:.3f}%/rodada")


def main():
    """Função principal"""
    print("Iniciando Simulação de Aprendizado Federado")
    print("Dataset: Fashion-MNIST com distribuição Non-IID")
    print("Comparação: FedAvg vs Power of Choice")
    
    # Configurações
    NUM_ROUNDS = 20
    
    # Executar simulações
    print("\n1. Executando FedAvg...")
    fedavg_metrics = run_simulation("fedavg", NUM_ROUNDS)
    
    print("\n2. Executando Power of Choice...")
    poc_metrics = run_simulation("powerofchoice", NUM_ROUNDS)
    
    # Gerar gráficos comparativos
    print("\n3. Gerando gráficos comparativos...")
    plot_comparison(fedavg_metrics, poc_metrics)
    
    # Imprimir resumo
    print_summary(fedavg_metrics, poc_metrics)
    
    print(f"\n{'='*60}")
    print("SIMULAÇÃO CONCLUÍDA COM SUCESSO!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main() 