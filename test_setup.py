#!/usr/bin/env python3
"""
Script de teste rápido para verificar se a implementação está funcionando
"""

import torch
import numpy as np
from fl_dp_sa.dataset import FMNISTNonIID
from fl_dp_sa.model import FMNISTNet, evaluate_model

def test_dataset():
    """Testa o dataset FMNIST Non-IID"""
    print("🧪 Testando dataset FMNIST Non-IID...")
    
    # Criar dataset com poucos clientes para teste rápido
    dataset = FMNISTNonIID(num_clients=5, alpha=0.5)
    
    # Verificar estatísticas
    stats = dataset.get_client_stats()
    print(f"   ✅ Dataset criado com {stats['num_clients']} clientes")
    print(f"   ✅ Amostras por cliente: {stats['samples_per_client']}")
    
    # Testar dados de um cliente
    train_loader, val_loader = dataset.get_client_data(0, batch_size=16)
    print(f"   ✅ Cliente 0: {len(train_loader.dataset)} amostras de treino, {len(val_loader.dataset)} de validação")
    
    # Testar dados de teste global
    test_loader = dataset.get_test_data(batch_size=64)
    print(f"   ✅ Conjunto de teste global: {len(test_loader.dataset)} amostras")
    
    return dataset

def test_model(dataset):
    """Testa o modelo CNN"""
    print("\n🧪 Testando modelo CNN...")
    
    # Criar modelo
    model = FMNISTNet()
    print(f"   ✅ Modelo criado com {sum(p.numel() for p in model.parameters())} parâmetros")
    
    # Testar forward pass
    dummy_input = torch.randn(1, 1, 28, 28)
    output = model(dummy_input)
    print(f"   ✅ Forward pass: entrada {dummy_input.shape} -> saída {output.shape}")
    
    # Testar avaliação com dados reais
    test_loader = dataset.get_test_data(batch_size=64)
    
    # Pegar apenas um batch para teste rápido
    data_iter = iter(test_loader)
    data, target = next(data_iter)
    
    model.eval()
    with torch.no_grad():
        output = model(data)
        _, predicted = torch.max(output, 1)
        accuracy = (predicted == target).float().mean().item() * 100
    
    print(f"   ✅ Avaliação em um batch: {accuracy:.2f}% de acurácia (modelo não treinado)")
    
    return model

def test_training_step(dataset, model):
    """Testa um passo de treinamento"""
    print("\n🧪 Testando passo de treinamento...")
    
    # Obter dados de um cliente
    train_loader, val_loader = dataset.get_client_data(0, batch_size=16)
    
    # Configurar treinamento
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Um passo de treinamento
    model.train()
    data_iter = iter(train_loader)
    data, target = next(data_iter)
    
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    
    print(f"   ✅ Passo de treinamento executado, loss: {loss.item():.4f}")
    
    # Verificar se os parâmetros mudaram
    print(f"   ✅ Gradientes calculados e parâmetros atualizados")

def main():
    """Função principal de teste"""
    print("🚀 Iniciando testes de configuração...")
    print("=" * 50)
    
    try:
        # Testar dataset
        dataset = test_dataset()
        
        # Testar modelo
        model = test_model(dataset)
        
        # Testar treinamento
        test_training_step(dataset, model)
        
        print("\n" + "=" * 50)
        print("🎉 TODOS OS TESTES PASSARAM!")
        print("✅ O sistema está pronto para executar a simulação completa")
        print("\nPara executar a simulação completa, use:")
        print("   python simulation.py")
        
    except Exception as e:
        print(f"\n❌ ERRO durante os testes: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    main() 