# Relatório do Sistema: Aprendizado Federado - FedAvg vs Power of Choice

## 📋 Visão Geral

Este projeto implementa um sistema completo de **Aprendizado Federado** usando o framework **Flower**, comparando duas estratégias de agregação:

- **FedAvg** (Federated Averaging) - estratégia tradicional
- **Power of Choice** - estratégia otimizada de seleção de clientes

O sistema utiliza o dataset **Fashion-MNIST** com distribuição **Non-IID** para simular cenários realistas de aprendizado federado.

## 🏗️ Arquitetura do Sistema

### Estrutura de Diretórios

```
fl-dp-sa/
├── fl_dp_sa/                 # Pacote principal
│   ├── __init__.py          # Inicialização do pacote
│   ├── model.py             # Arquitetura da rede neural
│   ├── dataset.py           # Gerenciamento do dataset Non-IID
│   ├── strategies.py        # Estratégias de agregação
│   ├── client_app.py        # Implementação do cliente
│   └── server_app.py        # Implementação do servidor
├── simulation.py            # Script principal de simulação
├── pyproject.toml          # Configurações e dependências
└── results/                # Diretório para resultados
```

## 🧠 Componentes Principais

### 1. Modelo Neural (`model.py`)

**Arquitetura: CNN para Fashion-MNIST**

```python
class FMNISTNet(nn.Module):
    - 3 Camadas Convolucionais (32, 64, 128 filtros)
    - MaxPooling 2x2 após cada convolução
    - 3 Camadas Fully Connected (256, 128, 10 neurônios)
    - Dropout (0.25, 0.5) para regularização
    - Ativação ReLU
```

**Características:**

- **Input**: Imagens 28x28x1 (Fashion-MNIST)
- **Output**: 10 classes (roupas/acessórios)
- **Parâmetros**: ~1.2M parâmetros treináveis
- **Otimizador**: Adam (lr=0.001)
- **Loss**: CrossEntropyLoss

### 2. Dataset Non-IID (`dataset.py`)

**Distribuição Dirichlet para Non-IID**

```python
class FMNISTNonIID:
    - num_clients: 50 clientes
    - alpha: 0.5 (controla heterogeneidade)
    - Distribuição Dirichlet por classe
    - Split 80/20 treino/validação por cliente
```

**Características da Distribuição:**

- **Alpha = 0.5**: Distribuição moderadamente heterogênea
- **Menor Alpha**: Mais Non-IID (clientes especializados)
- **Maior Alpha**: Mais IID (distribuição uniforme)
- **Classes**: 10 categorias do Fashion-MNIST

### 3. Estratégias de Agregação (`strategies.py`)

#### 3.1 FedAvg (Baseline)

```python
class FedAvgStrategy(FedAvg):
    - Seleção aleatória de clientes
    - Agregação por média ponderada
    - 20% dos clientes por rodada (10/50)
    - Coleta métricas de acurácia e loss
```

#### 3.2 Power of Choice (Otimizada)

```python
class PowerOfChoiceStrategy(FedAvg):
    - Seleção inteligente de clientes
    - Power of Choice com d=3 candidatos
    - Score baseado em performance histórica
    - Mantém histórico dos últimos 5 rounds
```

**Algoritmo Power of Choice:**

1. Para cada slot de cliente:
   - Seleciona `d=3` candidatos aleatórios
   - Escolhe o melhor baseado no score histórico
   - Remove o selecionado da lista de disponíveis
2. Score = média da acurácia dos últimos 5 rounds
3. Clientes novos recebem score neutro (0.0)

### 4. Cliente Federado (`client_app.py`)

**Implementação do Cliente**

```python
class FMNISTClient(NumPyClient):
    - Treinamento local: 3 épocas por round
    - Batch size: 32
    - Métricas: acurácia treino/validação, loss
    - Suporte CPU/GPU automático
```

**Fluxo do Cliente:**

1. **Recebe** parâmetros globais do servidor
2. **Treina** modelo local por 3 épocas
3. **Avalia** performance no conjunto de validação
4. **Envia** parâmetros atualizados + métricas

### 5. Servidor Federado (`server_app.py`)

**Configuração do Servidor**

```python
ServerConfig:
    - num_rounds: 20 rodadas de treinamento
    - fraction_fit: 0.2 (20% clientes por rodada)
    - min_fit_clients: 10 clientes mínimos
    - evaluate_fn: Avaliação global no teste
```

**Fluxo do Servidor:**

1. **Inicializa** modelo global
2. **Seleciona** clientes (estratégia específica)
3. **Distribui** parâmetros globais
4. **Agrega** atualizações dos clientes
5. **Avalia** modelo global (opcional)
6. **Repete** por 20 rodadas

## 🔄 Fluxo de Execução

### Simulação Completa (`simulation.py`)

```mermaid
[![](https://mermaid.ink/img/pako:eNpdkE1uwjAQha9izTog4iSEeFEJEoKyqSrUVRMWVmyI1TqOHIfSIg7TA_QUXKwmpD_UK79538w8zRFKxTgQ2GnaVOgxKWpk3zzP6vNnKdQGjUZ3aJHHWlCNEmpoyw26V_Uoy5LNFV70TJwvD7zsjMVSzub73eDGvZv8ug_qlWuktiiulCj5gCU9tsxXXFtmpc8fW1GqFsVKNlRTI_aqHdBlj6Z5JhstpNBozdtOqsFNe3eVp0Layt8MsZvjCVorRhltb0Netyc3wP-c4NgjCQbE6I47ILmW9CLheJlRgKm45AUQ-2VUPxdQ1Cfb09D6SSn53aZVt6uAbOlLa1XXMGp4Iqg9v_ypal4zrmPV1QYI9voZQI5wsMqNxrPID6Kp50c48B14A-J7YzfAXhBg7IVR6AYnB977nZPxbBKFFnRnUzcIMQ5PX7ytltg?type=png)](https://mermaid.live/edit#pako:eNpdkE1uwjAQha9izTog4iSEeFEJEoKyqSrUVRMWVmyI1TqOHIfSIg7TA_QUXKwmpD_UK79538w8zRFKxTgQ2GnaVOgxKWpk3zzP6vNnKdQGjUZ3aJHHWlCNEmpoyw26V_Uoy5LNFV70TJwvD7zsjMVSzub73eDGvZv8ug_qlWuktiiulCj5gCU9tsxXXFtmpc8fW1GqFsVKNlRTI_aqHdBlj6Z5JhstpNBozdtOqsFNe3eVp0Layt8MsZvjCVorRhltb0Netyc3wP-c4NgjCQbE6I47ILmW9CLheJlRgKm45AUQ-2VUPxdQ1Cfb09D6SSn53aZVt6uAbOlLa1XXMGp4Iqg9v_ypal4zrmPV1QYI9voZQI5wsMqNxrPID6Kp50c48B14A-J7YzfAXhBg7IVR6AYnB977nZPxbBKFFnRnUzcIMQ5PX7ytltg)
```

### Rodada de Treinamento

[![](https://mermaid.ink/img/pako:eNqNkkFOwzAQRa8SzZa2ipPWabyohAJiAxFSWKFsTDKkFoldHKcCql6GHefoxXDaFFJRKF7Z-m_8_4y9gkzlCAxqfG5QZngheKF5lUrHrgXXRmRiwaVxEofXToJ6KXKlf8oRafWoFCgNOuQI4PUB7wgQ94E4lTskGc5mCbPOJWZCSa73SP2tR4Q5l3IprHjL9ea9QqNV7VyV6oGLPuf9k4v_5nZkRDrrO42iDXatMl5WbbYO8DrPX4G4MzsG9Exs__0cZ87N5sNokfG6Z3QSik9AB_M-LzQWh_0f6ktetvPZzmQXGQZQaJEDM7rBAVSoK94eYdVWpmDmWGEKzG5zrp9SSOXa1tjHv1eq2pdp1RRzYI-8rO2pWeTc7D_lF4IyRx2pRhpg0-0NwFbwAowE7ohO_GAS0gkJfHdi1VdgQ88beWNCw8AnNHDpeOyvB_C2dSUj153SqR-SMHQpJdRbfwK5z__Z?type=png)](https://mermaid.live/edit#pako:eNqNkkFOwzAQRa8SzZa2ipPWabyohAJiAxFSWKFsTDKkFoldHKcCql6GHefoxXDaFFJRKF7Z-m_8_4y9gkzlCAxqfG5QZngheKF5lUrHrgXXRmRiwaVxEofXToJ6KXKlf8oRafWoFCgNOuQI4PUB7wgQ94E4lTskGc5mCbPOJWZCSa73SP2tR4Q5l3IprHjL9ea9QqNV7VyV6oGLPuf9k4v_5nZkRDrrO42iDXatMl5WbbYO8DrPX4G4MzsG9Exs__0cZ87N5sNokfG6Z3QSik9AB_M-LzQWh_0f6ktetvPZzmQXGQZQaJEDM7rBAVSoK94eYdVWpmDmWGEKzG5zrp9SSOXa1tjHv1eq2pdp1RRzYI-8rO2pWeTc7D_lF4IyRx2pRhpg0-0NwFbwAowE7ohO_GAS0gkJfHdi1VdgQ88beWNCw8AnNHDpeOyvB_C2dSUj153SqR-SMHQpJdRbfwK5z__Z)

## 📊 Métricas e Avaliação

### Métricas Coletadas

1. **Métricas dos Clientes:**

   - Acurácia de treino local
   - Acurácia de validação local
   - Loss de validação local

2. **Métricas Globais:**

   - Acurácia média dos clientes participantes
   - Loss média dos clientes participantes
   - Acurácia global (avaliação centralizada)
   - Loss global (avaliação centralizada)

3. **Métricas de Performance:**
   - Tempo de execução por estratégia
   - Taxa de convergência
   - Performance final

### Visualizações Geradas

O sistema gera automaticamente 4 gráficos comparativos:

1. **Evolução da Acurácia Global**

   - Acurácia ao longo das 20 rodadas
   - Comparação FedAvg vs Power of Choice

2. **Evolução da Loss Global**

   - Loss ao longo das 20 rodadas
   - Tendência de convergência

3. **Acurácia Média dos Clientes**

   - Performance dos clientes participantes
   - Sempre disponível (fallback)

4. **Comparação de Performance Final**
   - Gráfico de barras com acurácia final
   - Valores numéricos nas barras

## 🔧 Configurações Técnicas

### Dependências Principais

```toml
flwr[simulation] >= 1.18.0    # Framework de FL
torch == 2.6.0                # Deep Learning
torchvision == 0.21.0         # Datasets e transforms
matplotlib >= 3.5.0           # Visualizações
numpy >= 1.21.0               # Computação numérica
```

### Parâmetros de Configuração

**Dataset:**

- Clientes: 50
- Alpha (Dirichlet): 0.5
- Classes: 10 (Fashion-MNIST)
- Split: 80% treino, 20% validação

**Treinamento:**

- Rodadas globais: 20
- Épocas locais: 3
- Batch size: 32
- Learning rate: 0.001
- Clientes por rodada: 10 (20%)

**Power of Choice:**

- d = 3 candidatos
- Histórico: 5 rounds
- Score: média da acurácia

## 🎯 Objetivos e Hipóteses

### Objetivo Principal

Comparar a eficácia da estratégia **Power of Choice** contra **FedAvg** tradicional em cenários Non-IID.

### Hipóteses Testadas

1. **H1**: Power of Choice converge mais rapidamente
2. **H2**: Power of Choice atinge maior acurácia final
3. **H3**: Power of Choice é mais estável em ambientes Non-IID
4. **H4**: Seleção inteligente supera seleção aleatória

### Métricas de Sucesso

- **Acurácia Final**: Power of Choice > FedAvg
- **Convergência**: Menos rodadas para estabilizar
- **Estabilidade**: Menor variância entre rodadas
- **Robustez**: Melhor performance em Non-IID

## 🚀 Como Executar

### Execução Simples

```bash
python simulation.py
```

### Execução com Flower CLI

```bash
# Servidor
flwr run --run-config num-server-rounds=20

# Cliente (em outro terminal)
flwr run
```

### Personalização

```python
# Modificar parâmetros em simulation.py
NUM_ROUNDS = 50          # Mais rodadas
dataset = FMNISTNonIID(
    num_clients=100,     # Mais clientes
    alpha=0.1           # Mais Non-IID
)
```

## 📈 Resultados Esperados

### Cenário Típico (Non-IID moderado)

- **FedAvg**: ~85-88% acurácia final
- **Power of Choice**: ~87-91% acurácia final
- **Melhoria**: +2-3% em acurácia
- **Convergência**: 15-20% mais rápida

### Fatores de Influência

- **Alpha baixo**: Maior vantagem do Power of Choice
- **Mais clientes**: Melhor seleção disponível
- **Mais rodadas**: Convergência mais clara
- **Hardware**: GPU acelera significativamente

## 🔍 Análise Técnica

### Vantagens do Power of Choice

1. **Seleção Inteligente**: Escolhe clientes com melhor performance
2. **Adaptativo**: Aprende com histórico de performance
3. **Robusto**: Funciona bem em cenários heterogêneos
4. **Escalável**: Eficiência aumenta com mais clientes

### Limitações

1. **Cold Start**: Clientes novos têm score neutro
2. **Memória**: Mantém histórico de performance
3. **Complexidade**: Mais complexo que seleção aleatória
4. **Bias**: Pode favorecer clientes consistentemente bons

### Considerações de Implementação

1. **Eficiência**: O(d × log n) para seleção
2. **Memória**: O(clientes × histórico)
3. **Comunicação**: Mesma que FedAvg
4. **Privacidade**: Preserva privacidade dos dados

## 📝 Conclusões

Este sistema demonstra como estratégias inteligentes de seleção de clientes podem melhorar significativamente a performance do aprendizado federado em cenários realistas com dados heterogêneos (Non-IID). A implementação usando Flower fornece uma base sólida para pesquisa e desenvolvimento em aprendizado federado.

### Próximos Passos

1. **Implementar Differential Privacy**
2. **Adicionar Secure Aggregation**
3. **Testar com datasets maiores**
4. **Comparar com outras estratégias**
5. **Otimizar para produção**

---

**Desenvolvido com**: Python 3.8+, PyTorch 2.6, Flower 1.18+
**Dataset**: Fashion-MNIST (Non-IID)
**Estratégias**: FedAvg vs Power of Choice
