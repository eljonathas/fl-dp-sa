"""fl_dp_sa: Flower Example using Differential Privacy and Secure Aggregation."""

import torch
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
        # Criar modelo e definir parâmetros
        model = FMNISTNet()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Definir parâmetros do modelo
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        model.load_state_dict(state_dict, strict=True)
        
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
    
    # Criar dataset
    dataset = FMNISTNonIID(num_clients=50, alpha=0.5)
    
    # Criar modelo inicial
    model = FMNISTNet()
    initial_parameters = get_model_parameters(model)
    
    # Selecionar estratégia
    if strategy_name.lower() == "fedavg":
        strategy = FedAvgStrategy(
            fraction_fit=0.2,  # 20% dos clientes por rodada (10 de 50)
            fraction_evaluate=0.2,
            min_fit_clients=10,
            min_evaluate_clients=10,
            min_available_clients=50,
            evaluate_fn=get_evaluate_fn(dataset),
            initial_parameters=fl.common.ndarrays_to_parameters(initial_parameters),
        )
    elif strategy_name.lower() == "powerofchoice":
        strategy = PowerOfChoiceStrategy(
            fraction_fit=0.2,  # 20% dos clientes por rodada (10 de 50)
            fraction_evaluate=0.2,
            min_fit_clients=10,
            min_evaluate_clients=10,
            min_available_clients=50,
            evaluate_fn=get_evaluate_fn(dataset),
            initial_parameters=fl.common.ndarrays_to_parameters(initial_parameters),
            d=20,  # Power of choice com d=20 (mais candidatos para melhor seleção)
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
