#!/usr/bin/env python3
"""
Simulação comparativa entre FedAvg e Power of Choice
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
    print(f"Rodadas: {num_rounds}")
    print(f"{'='*60}")
    
    # Criar dataset
    dataset = FMNISTNonIID(num_clients=20, alpha=0.1)  # Menos non-IID para melhor aprendizado
    
    # Criar função de cliente
    client_fn = create_client_fn(dataset)
    
    # Criar app do servidor
    _, strategy = create_server_app(strategy_name)
    
    # Configurar simulação
    start_time = time.time()
    
    # Executar simulação
    history = start_simulation(
        client_fn=client_fn,
        num_clients=20,  # Atualizado para 20 clientes
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
    if hasattr(history, 'losses_centralized') and history.losses_centralized:
        for round_num, loss in history.losses_centralized:
            metrics["global_loss"].append(loss)
    
    if hasattr(history, 'metrics_centralized') and history.metrics_centralized:
        if 'accuracy' in history.metrics_centralized:
            for round_num, accuracy in history.metrics_centralized['accuracy']:
                metrics["global_accuracy"].append(accuracy)
    
    # Debug: imprimir estatísticas das métricas
    print(f"\nEstatísticas das métricas para {strategy_name}:")
    print(f"  Acurácia global: {len(metrics['global_accuracy'])} valores")
    print(f"  Loss global: {len(metrics['global_loss'])} valores")
    print(f"  Acurácia de clientes: {len(metrics['round_accuracies'])} valores")
    print(f"  Loss de clientes: {len(metrics['round_losses'])} valores")
    
    if metrics['round_accuracies']:
        print(f"  Acurácia final (clientes): {metrics['round_accuracies'][-1]:.2f}%")
    if metrics['global_accuracy']:
        print(f"  Acurácia final (global): {metrics['global_accuracy'][-1]:.2f}%")
    
    return metrics


def plot_comparison(fedavg_metrics: Dict, poc_metrics: Dict, save_path: str = "results"):
    """
    Cria gráficos comparativos entre FedAvg e Power of Choice
    """
    # Criar diretório se não existir
    os.makedirs(save_path, exist_ok=True)
    
    # Debug: imprimir informações sobre as métricas
    print(f"\nDEBUG - Métricas FedAvg:")
    print(f"  global_accuracy: {len(fedavg_metrics['global_accuracy'])} valores")
    print(f"  global_loss: {len(fedavg_metrics['global_loss'])} valores")
    print(f"  round_accuracies: {len(fedavg_metrics['round_accuracies'])} valores")
    print(f"  round_losses: {len(fedavg_metrics['round_losses'])} valores")
    
    print(f"\nDEBUG - Métricas Power of Choice:")
    print(f"  global_accuracy: {len(poc_metrics['global_accuracy'])} valores")
    print(f"  global_loss: {len(poc_metrics['global_loss'])} valores")
    print(f"  round_accuracies: {len(poc_metrics['round_accuracies'])} valores")
    print(f"  round_losses: {len(poc_metrics['round_losses'])} valores")
    
    # Configurar estilo dos gráficos
    plt.style.use('default')  # Mudança para estilo mais compatível
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
    
    # Adicionar texto se não há dados
    if not fedavg_metrics["global_accuracy"] and not poc_metrics["global_accuracy"]:
        ax1.text(0.5, 0.5, 'Dados não disponíveis\n(Avaliação centralizada não configurada)', 
                ha='center', va='center', transform=ax1.transAxes, fontsize=12, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
    
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
    
    # Adicionar texto se não há dados
    if not fedavg_metrics["global_loss"] and not poc_metrics["global_loss"]:
        ax2.text(0.5, 0.5, 'Dados não disponíveis\n(Avaliação centralizada não configurada)', 
                ha='center', va='center', transform=ax2.transAxes, fontsize=12, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
    
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
    
    # 4. Loss dos Clientes
    ax4 = axes[1, 1]
    if fedavg_metrics["round_losses"]:
        rounds = range(1, len(fedavg_metrics["round_losses"]) + 1)
        ax4.plot(rounds, fedavg_metrics["round_losses"], 'b-o', label='FedAvg', linewidth=2, markersize=6)
    
    if poc_metrics["round_losses"]:
        rounds = range(1, len(poc_metrics["round_losses"]) + 1)
        ax4.plot(rounds, poc_metrics["round_losses"], 'r-s', label='Power of Choice', linewidth=2, markersize=6)
    
    ax4.set_xlabel('Rodadas de Treinamento')
    ax4.set_ylabel('Loss Média dos Clientes')
    ax4.set_title('Evolução da Loss dos Clientes')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Salvar gráfico
    plt.savefig(f"{save_path}/comparison_fedavg_vs_powerofchoice.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{save_path}/comparison_fedavg_vs_powerofchoice.pdf", bbox_inches='tight')
    
    print(f"\nGráficos salvos em: {save_path}/")
    
    # Mostrar gráfico
    plt.show()


def plot_performance_bars(fedavg_metrics: Dict, poc_metrics: Dict, save_path: str = "results"):
    """Gráfico de barras para comparação final"""
    
    # Usar acurácia global se disponível, senão usar acurácia dos clientes
    fedavg_final_acc = (fedavg_metrics["global_accuracy"][-1] if fedavg_metrics["global_accuracy"] 
                       else fedavg_metrics["round_accuracies"][-1] if fedavg_metrics["round_accuracies"] else 0)
    poc_final_acc = (poc_metrics["global_accuracy"][-1] if poc_metrics["global_accuracy"] 
                    else poc_metrics["round_accuracies"][-1] if poc_metrics["round_accuracies"] else 0)
    
    # Criar gráfico de barras
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Acurácia
    strategies = ['FedAvg', 'Power of Choice']
    final_accuracies = [fedavg_final_acc, poc_final_acc]
    
    bars1 = ax1.bar(strategies, final_accuracies, color=['blue', 'red'], alpha=0.7)
    ax1.set_ylabel('Acurácia Final (%)')
    ax1.set_title('Comparação de Acurácia Final')
    
    # Adicionar valores nas barras
    for bar, acc in zip(bars1, final_accuracies):
        height = bar.get_height()
        if height > 0:
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold')
    
    # Tempo de execução
    times = [fedavg_metrics["duration"], poc_metrics["duration"]]
    bars2 = ax2.bar(strategies, times, color=['blue', 'red'], alpha=0.7)
    ax2.set_ylabel('Tempo de Execução (segundos)')
    ax2.set_title('Comparação de Tempo de Execução')
    
    # Adicionar valores nas barras
    for bar, time_val in zip(bars2, times):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{time_val:.1f}s', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{save_path}/performance_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()


def print_summary(fedavg_metrics: Dict, poc_metrics: Dict):
    """Imprime resumo comparativo dos resultados"""
    
    print(f"\n{'='*60}")
    print("RESUMO DOS RESULTADOS")
    print(f"{'='*60}")
    
    # Usar acurácia global se disponível, senão usar acurácia dos clientes
    fedavg_final_acc = (fedavg_metrics["global_accuracy"][-1] if fedavg_metrics["global_accuracy"] 
                       else fedavg_metrics["round_accuracies"][-1] if fedavg_metrics["round_accuracies"] else 0)
    poc_final_acc = (poc_metrics["global_accuracy"][-1] if poc_metrics["global_accuracy"] 
                    else poc_metrics["round_accuracies"][-1] if poc_metrics["round_accuracies"] else 0)
    
    # Indicar fonte dos dados
    accuracy_source = "Global" if fedavg_metrics["global_accuracy"] or poc_metrics["global_accuracy"] else "Clientes"
    
    print(f"Acurácia Final ({accuracy_source}):")
    print(f"  FedAvg:          {fedavg_final_acc:.2f}%")
    print(f"  Power of Choice: {poc_final_acc:.2f}%")
    print(f"  Melhoria:        {poc_final_acc - fedavg_final_acc:+.2f}%")
    
    # Tempo de execução
    print(f"\nTempo de Execução:")
    print(f"  FedAvg:          {fedavg_metrics['duration']:.2f}s")
    print(f"  Power of Choice: {poc_metrics['duration']:.2f}s")
    
    # Convergência - usar dados disponíveis
    accuracy_data_fedavg = fedavg_metrics["global_accuracy"] if fedavg_metrics["global_accuracy"] else fedavg_metrics["round_accuracies"]
    accuracy_data_poc = poc_metrics["global_accuracy"] if poc_metrics["global_accuracy"] else poc_metrics["round_accuracies"]
    
    if accuracy_data_fedavg and accuracy_data_poc and len(accuracy_data_fedavg) >= 5 and len(accuracy_data_poc) >= 5:
        fedavg_convergence = np.mean(np.diff(accuracy_data_fedavg[-5:]))
        poc_convergence = np.mean(np.diff(accuracy_data_poc[-5:]))
        
        print(f"\nTaxa de Convergência (últimas 5 rodadas):")
        print(f"  FedAvg:          {fedavg_convergence:.3f}%/rodada")
        print(f"  Power of Choice: {poc_convergence:.3f}%/rodada")
    
    # Informações adicionais sobre dados disponíveis
    print(f"\nInformações sobre Dados:")
    print(f"  Avaliação Global: {'Disponível' if fedavg_metrics['global_accuracy'] or poc_metrics['global_accuracy'] else 'Não disponível'}")
    print(f"  Avaliação Clientes: {'Disponível' if fedavg_metrics['round_accuracies'] or poc_metrics['round_accuracies'] else 'Não disponível'}")


def main():
    """Função principal"""
    print("Iniciando Simulação de Aprendizado Federado")
    print("Dataset: Fashion-MNIST com distribuição Non-IID")
    print("Comparação: FedAvg vs Power of Choice")
    print("="*60)
    
    # Configurações
    NUM_ROUNDS = 20
    
    # Executar simulações
    print("\n1. Executando FedAvg...")
    fedavg_metrics = run_simulation("fedavg", NUM_ROUNDS)  # CORREÇÃO: usar NUM_ROUNDS ao invés de 1
    
    print("\n2. Executando Power of Choice...")
    poc_metrics = run_simulation("powerofchoice", NUM_ROUNDS)
    
    # Gerar gráficos comparativos
    print("\n3. Gerando gráficos comparativos...")
    plot_comparison(fedavg_metrics, poc_metrics)  # CORREÇÃO: remover lista desnecessária
    
    # Gráfico de barras de performance
    print("\n4. Gerando gráfico de barras...")
    plot_performance_bars(fedavg_metrics, poc_metrics)
    
    # Imprimir resumo
    print_summary(fedavg_metrics, poc_metrics)
    
    print(f"\n{'='*60}")
    print("SIMULAÇÃO CONCLUÍDA COM SUCESSO!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main() 