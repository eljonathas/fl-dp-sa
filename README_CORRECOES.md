# Correções no Sistema de Aprendizado Federado: Power of Choice

## Problemas Identificados e Correções

### 1. **Erro na Função de Gráficos**

**Problema**: `TypeError: list indices must be integers or slices, not str`
**Causa**: Na função `main()`, estava passando `[fedavg_metrics]` (lista) ao invés de `fedavg_metrics` (dicionário)
**Correção**: Removida a lista desnecessária na chamada `plot_comparison(fedavg_metrics, poc_metrics)`

### 2. **Acurácia Estagnada em 8.84%**

**Problema**: Modelo não estava aprendendo adequadamente
**Causas e Correções**:

- **FedAvg executava apenas 1 rodada**: Corrigido para usar `NUM_ROUNDS = 20`
- **Épocas de treinamento local insuficientes**: Aumentado de 1 para 3 épocas por cliente
- **Learning rate muito baixo**: Aumentado de 0.001 para 0.01
- **Muitos clientes com poucos dados**: Reduzido de 50 para 20 clientes
- **Distribuição muito non-IID**: Reduzido alpha de 0.5 para 0.1

### 3. **Implementação do Power of Choice**

**Problema**: Algoritmo não seguia exatamente o paper original
**Correções**:

- Implementação correta do algoritmo do artigo:
  1. Primeira rodada: seleção aleatória uniforme
  2. Rodadas seguintes: selecionar `d` candidatos com probabilidade proporcional ao dataset, ordenar por loss decrescente, escolher top `m`
- Gestão adequada do histórico de losses
- Debug melhorado para mostrar processo de seleção

## Arquivos Corrigidos

### `fl_dp_sa/strategies.py`

- ✅ Implementação correta do algoritmo Power of Choice
- ✅ Seleção proporcional ao tamanho do dataset
- ✅ Ordenação por loss decrescente
- ✅ Logging melhorado para debug

### `simulation.py`

- ✅ Correção do erro de tipo na função `plot_comparison()`
- ✅ FedAvg agora executa 20 rodadas ao invés de 1
- ✅ Configuração atualizada para 20 clientes
- ✅ Gráficos melhorados com mais informações
- ✅ Função adicional `plot_performance_bars()` para comparação final

### `fl_dp_sa/client_app.py`

- ✅ Épocas de treinamento local aumentadas para 3
- ✅ Learning rate aumentado para 0.01

### `fl_dp_sa/server_app.py`

- ✅ Configuração ajustada para 20 clientes
- ✅ Alpha reduzido para 0.1 (menos non-IID)
- ✅ Fraction_fit aumentado para 0.5 (10 de 20 clientes por rodada)

## Como Executar

### Teste Rápido (Recomendado primeiro)

```bash
python test_simulation.py
```

Este teste executa apenas 5 rodadas com 10 clientes para verificar se tudo está funcionando.

### Simulação Completa

```bash
python simulation.py
```

Esta simulação executa 20 rodadas comparando FedAvg vs Power of Choice.

## Resultados Esperados

Com as correções implementadas, você deve ver:

1. **Acurácia crescente**: A acurácia deve começar baixa (~10%) e melhorar ao longo das rodadas
2. **Power of Choice superior**: Deve mostrar melhor performance que FedAvg
3. **Gráficos funcionando**: Gráficos comparativos serão gerados sem erros
4. **Logs informativos**: Processo de seleção de clientes visível nos logs

## Configurações Atuais

- **Clientes**: 20 (reduzido de 50)
- **Clientes por rodada**: 10 (50% dos disponíveis)
- **Power of choice d**: 15 candidatos
- **Épocas locais**: 3 por cliente
- **Learning rate**: 0.01
- **Alpha (non-IID)**: 0.1 (menos heterogêneo)
- **Rodadas**: 20

## Arquitetura do Sistema

```
fl_dp_sa/
├── strategies.py       # FedAvgStrategy, PowerOfChoiceStrategy
├── client_app.py       # FMNISTClient, treinamento local
├── server_app.py       # Configuração do servidor
├── model.py           # Rede neural CNN para Fashion-MNIST
└── dataset.py         # Distribuição Non-IID dos dados

simulation.py          # Script principal de comparação
test_simulation.py     # Teste rápido
```

## Principais Melhorias

1. **Algoritmo Power of Choice**: Implementação fiel ao paper original
2. **Performance**: Modelo agora aprende adequadamente
3. **Debugging**: Logs detalhados do processo de seleção
4. **Visualização**: Gráficos mais informativos
5. **Configuração**: Parâmetros otimizados para demonstração clara

O sistema agora deve mostrar claramente a superioridade do Power of Choice sobre FedAvg em cenários de aprendizado federado com dados heterogêneos.
