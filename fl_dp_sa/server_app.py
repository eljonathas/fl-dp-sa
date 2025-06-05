"""fl_dp_sa: Flower Example using Differential Privacy and Secure Aggregation."""

import torch
import numpy as np
import flwr as fl
from flwr.server import ServerApp, ServerConfig
from flwr.common import Context
from typing import Dict, Tuple, Optional

from .dataset import FMNISTNonIID
from .model import FMNISTNet, evaluate_model, get_model_parameters
from .strategies import FedAvgStrategy, PowerOfChoiceStrategy


def get_evaluate_fn(dataset: FMNISTNonIID):
    """Retorna função de avaliação global"""
    
    def evaluate_fn(server_round: int, parameters, config: Dict[str, any]) -> Optional[Tuple[float, Dict[str, any]]]:
        """Avalia o modelo global no conjunto de teste"""
        # Criar modelo
        model = FMNISTNet()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        
        # Definir parâmetros do modelo - CORRIGIDO
        if parameters:
            # Converter parameters de ndarrays para tensors
            params_dict = {}
            param_names = list(model.state_dict().keys())
            
            for i, (param_name, param_value) in enumerate(zip(param_names, parameters)):
                if isinstance(param_value, np.ndarray):
                    params_dict[param_name] = torch.tensor(param_value, dtype=torch.float32)
                else:
                    params_dict[param_name] = torch.tensor(param_value, dtype=torch.float32)
            
            model.load_state_dict(params_dict, strict=True)
        
        # Obter dados de teste
        test_loader = dataset.get_test_data(batch_size=64)
        
        # Avaliar modelo
        accuracy = evaluate_model(model, test_loader, device)
        
        # Calcular loss
        model.eval()
        total_loss = 0.0
        criterion = torch.nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                total_loss += loss.item()
        
        avg_loss = total_loss / len(test_loader)
        
        print(f"Rodada {server_round} - Avaliação Global - Acurácia: {accuracy:.2f}%, Loss: {avg_loss:.4f}")
        
        return avg_loss, {"accuracy": accuracy}
    
    return evaluate_fn


def create_server_app(strategy_name: str = "fedavg"):
    """Cria o app do servidor com a estratégia especificada"""
    
    # Criar dataset (menos clientes para melhor distribuição de dados)
    dataset = FMNISTNonIID(num_clients=20, alpha=0.1)  # Menos non-IID e menos clientes
    
    # Criar modelo inicial
    model = FMNISTNet()
    initial_parameters = get_model_parameters(model)
    
    # Selecionar estratégia
    if strategy_name.lower() == "fedavg":
        strategy = FedAvgStrategy(
            fraction_fit=0.5,  # 50% dos clientes por rodada (10 de 20)
            fraction_evaluate=0.5,
            min_fit_clients=8,
            min_evaluate_clients=8,
            min_available_clients=20,
            evaluate_fn=get_evaluate_fn(dataset),
            initial_parameters=fl.common.ndarrays_to_parameters(initial_parameters),
        )
    elif strategy_name.lower() == "powerofchoice":
        strategy = PowerOfChoiceStrategy(
            fraction_fit=0.5,  # 50% dos clientes por rodada (10 de 20)
            fraction_evaluate=0.5,
            min_fit_clients=8,
            min_evaluate_clients=8,
            min_available_clients=20,
            evaluate_fn=get_evaluate_fn(dataset),
            initial_parameters=fl.common.ndarrays_to_parameters(initial_parameters),
            d=15,  # Power of choice com d=15 candidatos
        )
    else:
        raise ValueError(f"Estratégia desconhecida: {strategy_name}")
    
    # Configuração do servidor
    config = ServerConfig(num_rounds=20)  # 20 rodadas de treinamento
    
    def server_fn(context: Context):
        return fl.server.Server(strategy=strategy).to_server()
    
    return ServerApp(server_fn=server_fn), strategy


# App padrão (FedAvg)
app, _ = create_server_app("fedavg")
