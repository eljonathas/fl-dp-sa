---
tags: [DP, SecAgg, vision, fds]
dataset: [MNIST]
framework: [torch, torchvision]
---

# Flower Example on MNIST with Differential Privacy and Secure Aggregation

This example demonstrates a federated learning setup using the Flower, incorporating central differential privacy (DP) with client-side fixed clipping and secure aggregation (SA). It is intended for a small number of rounds for demonstration purposes.

This example is similar to the [quickstart-pytorch example](https://github.com/adap/flower/tree/main/examples/quickstart-pytorch) and extends it by integrating central differential privacy and secure aggregation. For more details on differential privacy and secure aggregation in Flower, please refer to the documentation [here](https://flower.ai/docs/framework/how-to-use-differential-privacy.html) and [here](https://flower.ai/docs/framework/contributor-ref-secure-aggregation-protocols.html).

## Set up the project

### Clone the project

Start by cloning the example project:

```shell
git clone --depth=1 https://github.com/adap/flower.git && mv flower/examples/fl-dp-sa . && rm -rf flower && cd fl-dp-sa
```

This will create a new directory called `fl-dp-sa` containing the following files:

```shell
fl-dp-sa
├── fl_dp_sa
│   ├── client_app.py   # Defines your ClientApp
│   ├── server_app.py   # Defines your ServerApp
│   └── task.py         # Defines your model, training, and data loading
├── pyproject.toml      # Project metadata like dependencies and configs
└── README.md
```

### Install dependencies and project

Install the dependencies defined in `pyproject.toml` as well as the `fl_dp_sa` package.

```shell
# From a new python environment, run:
pip install -e .
```

## Run the project

You can run your Flower project in both _simulation_ and _deployment_ mode without making changes to the code. If you are starting with Flower, we recommend you using the _simulation_ mode as it requires fewer components to be launched manually. By default, `flwr run` will make use of the Simulation Engine.

### Run with the Simulation Engine

> \[!NOTE\]
> Check the [Simulation Engine documentation](https://flower.ai/docs/framework/how-to-run-simulations.html) to learn more about Flower simulations and how to optimize them.

```bash
flwr run .
```

You can also override some of the settings for your `ClientApp` and `ServerApp` defined in `pyproject.toml`. For example:

```bash
flwr run . --run-config "noise-multiplier=0.1 clipping-norm=5"
```

### Run with the Deployment Engine

Follow this [how-to guide](https://flower.ai/docs/framework/how-to-run-flower-with-deployment-engine.html) to run the same app in this example but with Flower's Deployment Engine. After that, you might be intersted in setting up [secure TLS-enabled communications](https://flower.ai/docs/framework/how-to-enable-tls-connections.html) and [SuperNode authentication](https://flower.ai/docs/framework/how-to-authenticate-supernodes.html) in your federation.

If you are already familiar with how the Deployment Engine works, you may want to learn how to run it using Docker. Check out the [Flower with Docker](https://flower.ai/docs/framework/docker/index.html) documentation.

# Simulação de Aprendizado Federado: FedAvg vs Power of Choice

Este projeto implementa uma simulação comparativa entre o algoritmo FedAvg tradicional e a estratégia Power of Choice para seleção de clientes em aprendizado federado, utilizando o dataset Fashion-MNIST com distribuição Non-IID.

## 📋 Características da Simulação

- **Dataset**: Fashion-MNIST com distribuição Non-IID (usando distribuição Dirichlet)
- **Modelo**: Rede Neural Convolucional (CNN) otimizada para Fashion-MNIST
- **Clientes**: 50 clientes com dados heterogêneos
- **Estratégias Comparadas**:
  - **FedAvg**: Seleção aleatória tradicional de clientes
  - **Power of Choice**: Seleção baseada em performance histórica dos clientes
- **Métrica Principal**: Acurácia global no conjunto de teste
- **Visualização**: Gráficos comparativos gerados automaticamente

## 🚀 Instalação e Configuração

### 1. Clonar o Repositório

```bash
git clone <repository-url>
cd fl-dp-sa
```

### 2. Criar Ambiente Virtual

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# ou
.venv\Scripts\activate     # Windows
```

### 3. Atualizar pip e Instalar Dependências

```bash
pip install --upgrade pip
pip install -e .
```

### 4. Verificar Instalação

```bash
python test_setup.py
```

## 🎯 Executando a Simulação

### Simulação Completa (Recomendado)

Execute o script principal que compara ambas as estratégias:

```bash
python simulation.py
```

Este comando irá:

1. Executar simulação com FedAvg (20 rodadas)
2. Executar simulação com Power of Choice (20 rodadas)
3. Gerar gráficos comparativos
4. Exibir resumo dos resultados

### Teste Rápido

Para verificar se tudo está funcionando:

```bash
python test_setup.py
```

## 📊 Resultados e Visualizações

A simulação gera automaticamente:

### Gráficos Comparativos

- **Evolução da Acurácia Global**: Comparação da acurácia ao longo das rodadas
- **Evolução da Loss Global**: Comparação da loss ao longo das rodadas
- **Acurácia Média dos Clientes**: Performance média dos clientes participantes
- **Comparação Final**: Acurácia final de ambas as estratégias

### Arquivos Gerados

- `results/comparison_fedavg_vs_powerofchoice.png`: Gráfico em alta resolução
- `results/comparison_fedavg_vs_powerofchoice.pdf`: Versão em PDF para relatórios

### Métricas Coletadas

- Acurácia global por rodada
- Loss global por rodada
- Acurácia média dos clientes participantes
- Tempo de execução de cada simulação
- Taxa de convergência

## ⚙️ Configurações Principais

### Parâmetros do Dataset

```python
num_clients = 50        # Número total de clientes
alpha = 0.5            # Concentração Dirichlet (menor = mais Non-IID)
```

### Parâmetros de Treinamento

```python
num_rounds = 20        # Rodadas de treinamento federado
clients_per_round = 10 # Clientes por rodada (20% de 50)
local_epochs = 1       # Épocas de treinamento local
batch_size = 32        # Tamanho do batch
learning_rate = 0.001  # Taxa de aprendizado
```

### Parâmetros Power of Choice

```python
d = 3                  # Número de candidatos considerados por slot
```

## 🏗️ Estrutura do Projeto

```
fl-dp-sa/
├── fl_dp_sa/
│   ├── __init__.py         # Inicialização do pacote
│   ├── dataset.py          # Gerenciamento do FMNIST Non-IID
│   ├── model.py            # Modelo CNN e funções de treinamento
│   ├── strategies.py       # Estratégias FedAvg e Power of Choice
│   ├── client_app.py       # Aplicação do cliente
│   └── server_app.py       # Aplicação do servidor
├── simulation.py           # Script principal de simulação
├── test_setup.py          # Testes de configuração
├── pyproject.toml         # Configuração do projeto
└── README.md              # Este arquivo
```

## 📈 Resultados Esperados

### Power of Choice vs FedAvg

- **Convergência**: Power of Choice tende a convergir mais rapidamente
- **Acurácia Final**: Melhoria esperada de 2-5% na acurácia final
- **Estabilidade**: Menor variância entre rodadas
- **Eficiência**: Melhor utilização de clientes com boa performance

### Fatores de Performance

- **Heterogeneidade dos Dados**: Maior benefício em cenários mais Non-IID
- **Número de Clientes**: Benefício aumenta com mais clientes disponíveis
- **Parâmetro d**: Valores entre 2-5 geralmente oferecem melhor trade-off

## 🔧 Otimizações Realizadas

### Código Limpo e Eficiente

- ✅ Removidos arquivos redundantes (`quick_test_simulation.py`, `simple_simulation.py`, `task.py`)
- ✅ Eliminados imports não utilizados
- ✅ Simplificadas classes abstratas desnecessárias
- ✅ Otimizadas dependências no `pyproject.toml`
- ✅ Código modular e bem estruturado

### Dependências Mínimas

```toml
dependencies = [
    "flwr[simulation]>=1.18.0",
    "torch==2.6.0",
    "torchvision==0.21.0",
    "matplotlib>=3.5.0",
    "numpy>=1.21.0",
]
```

## 🧪 Testes

O projeto inclui testes abrangentes:

```bash
python test_setup.py
```

Verifica:

- ✅ Criação e distribuição do dataset Non-IID
- ✅ Inicialização e forward pass do modelo
- ✅ Processo de treinamento
- ✅ Compatibilidade entre componentes

## 📝 Notas Técnicas

### Modelo CNN

- **Arquitetura**: 3 camadas convolucionais + 3 fully connected
- **Parâmetros**: 422,026 parâmetros treináveis
- **Regularização**: Dropout (0.25 e 0.5)
- **Otimizador**: Adam com lr=0.001

### Distribuição Non-IID

- **Método**: Distribuição Dirichlet com α=0.5
- **Heterogeneidade**: Cada cliente tem distribuição diferente de classes
- **Balanceamento**: Evita clientes sem dados

### Power of Choice

- **Algoritmo**: Para cada slot, considera d=3 candidatos e escolhe o melhor
- **Métrica**: Score baseado na acurácia histórica (últimas 5 rodadas)
- **Adaptação**: Score neutro para clientes novos

## 🤝 Contribuições

Para contribuir com o projeto:

1. Faça fork do repositório
2. Crie uma branch para sua feature
3. Execute os testes: `python test_setup.py`
4. Submeta um pull request

## 📄 Licença

Apache-2.0

---

**Nota**: Esta implementação foi otimizada para fins educacionais e de pesquisa. Para uso em produção, considere otimizações adicionais de performance e segurança.
