# 📊 Preditor de População - IBGE com Machine Learning

Projeto de **Machine Learning** que utiliza **TensorFlow.js** para analisar dados populacionais brasileiros e **prever o crescimento futuro** usando regressão linear. Desenvolvido como parte da **Pós-Graduação em Engenharia de IA Aplicada**.

## 🎯 Sobre o Projeto

Este projeto demonstra o **ciclo completo de Machine Learning**, desde o carregamento de dados até a predição de valores futuros:

1. **Carregamento** de dados históricos do IBGE (2010-2025)
2. **Pré-processamento** e normalização de dados
3. **Treinamento** de modelo de regressão linear
4. **Predição** da população para 2030
5. **Análise estatística** com tensores

### ⚙️ Parametrização Flexível

**Analise qualquer estado brasileiro!** Basta alterar uma linha de código:

```javascript
const ESTADO_ALVO = 'SP'; // SP, MG, RJ, BA, PR, RS, PE, CE, ES
```

### 🔮 Exemplos de Predição

- **São Paulo (SP)**: ~46,5 milhões de habitantes em 2030 (+4,7%)
- **Minas Gerais (MG)**: ~21,4 milhões de habitantes em 2030 (+4,3%)
- **Rio de Janeiro (RJ)**: Predição disponível no sistema

## 🛠️ Tecnologias Utilizadas

- **Node.js** - Runtime JavaScript
- **TensorFlow.js** - Framework de Machine Learning para JavaScript
- **Regressão Linear** - Algoritmo de predição
- **File System (fs)** - Módulo nativo do Node.js para leitura de arquivos
- **Dados do IBGE** - Informações populacionais reais dos estados brasileiros

## 📋 Pré-requisitos

- Node.js (versão 14 ou superior)
- npm (gerenciador de pacotes)

## 🚀 Como Rodar o Projeto

### 1. Instalar as dependências

```bash
npm install
```

Este comando irá instalar todas as dependências necessárias listadas no `package.json`, incluindo o `@tensorflow/tfjs`.

### 2. (Opcional) Configurar o estado a ser analisado

Abra o arquivo `index.js` e altere a constante `ESTADO_ALVO` no topo do arquivo:

```javascript
// CONFIGURAÇÃO DO PROJETO
const ESTADO_ALVO = 'MG'; // Altere para: SP, MG, RJ, BA, PR, RS, PE, CE, ES
```

### 3. Executar o projeto

```bash
node index.js
```

## 📊 Exemplo de Resultado

Ao executar o projeto, você verá uma análise completa com predição usando IA:

```
=== 📊 Preditor de População - São Paulo (IBGE) ===

🔮 Análise Temporal e Predição com Machine Learning

============================================================

📅 Evolução populacional de São Paulo:

  2010: 41.262.199 habitantes
  2015: 44.396.484 habitantes
  2020: 46.289.333 habitantes
  2025: 44.411.238 habitantes

📊 População média no período: 44.089.816 habitantes

=== Normalização de Dados ===

Anos originais: [ 2010, 2015, 2020, 2025 ]
Anos normalizados: [ '0.00', '0.33', '0.67', '1.00' ]

=== Treinamento do Modelo de IA ===

📊 Arquitetura do Modelo:
_____________________________________________________________________________
Layer (type)                Input Shape               Output shape         Param #
=============================================================================
dense_Dense1 (Dense)        [[null,1]]                [null,1]             2
=============================================================================
Total params: 2
Trainable params: 2

🎯 Treinando modelo...
✅ Treinamento concluído!

=== Testando o Modelo com Dados Conhecidos ===

2010: Real = 41.262.199, Predito = 42.862.606
2015: Real = 44.396.484, Predito = 43.715.006
2020: Real = 46.289.333, Predito = 44.567.406
2025: Real = 44.411.238, Predito = 45.419.806

=== 🔮 Predição para 2030 ===

🎯 População prevista para São Paulo em 2030: 46.508.243 habitantes

📈 Variação esperada (2025 → 2030):
   Diferença: +2.097.005 habitantes
   Percentual: +4.72%

============================================================
📊 RESUMO DA ANÁLISE COM IA
============================================================

🔢 Dados analisados:
   • Estado: São Paulo (SP)
   • Período histórico: 2010 - 2025
   • Pontos de dados temporais: 4
   • População em 2025: 44.411.238 habitantes

🧠 Modelo de Machine Learning:
   • Tipo: Regressão Linear
   • Framework: TensorFlow.js
   • Parâmetros treináveis: 2 (peso + bias)
   • Épocas de treinamento: 100

🎯 Predição Gerada:
   • Ano alvo: 2030
   • População prevista: 46.508.243 habitantes
   • Crescimento esperado: 4.72% em 5 anos

💡 Insights:
   • São Paulo apresenta crescimento moderado
   • O modelo sugere aumento de 2.097.005 habitantes

============================================================
✨ Análise concluída com sucesso!
============================================================
```

## 📁 Estrutura do Projeto

```
Projeto-00/
├── dados-historicos.json   # Série temporal (2010-2025) de 9 estados para treino
├── index.js                # Script principal com modelo de IA parametrizável
├── package.json            # Configurações e dependências do projeto
└── README.md               # Documentação completa do projeto
```

## 🎓 Conceitos de IA Aplicados

### **Tensores**
Estruturas de dados multidimensionais que permitem:
- Operações matemáticas vetorizadas
- Processamento paralelo eficiente
- Base fundamental do TensorFlow

### **Normalização**
Técnica essencial de pré-processamento que:
- Coloca dados em escala comum (0-1)
- Acelera o treinamento do modelo
- Melhora a convergência do otimizador

### **Regressão Linear**
Algoritmo supervisionado que:
- Aprende relação entre X (tempo) e Y (população)
- Fórmula: `Y = peso × X + bias`
- Minimiza erro quadrático médio (MSE)

### **Gradient Descent**
Algoritmo de otimização que:
- Ajusta pesos iterativamente
- Minimiza a função de perda (loss)
- Descida gradual em direção ao mínimo global

### **Overfitting vs Generalização**
- Com poucos dados (4 pontos), o modelo é simples
- Regressão linear evita overfitting naturalmente
- Predições são baseadas em tendência geral

## 🌟 Diferenciais do Projeto

- ✅ **Modelo real de Machine Learning** (não apenas estatística)
- ✅ **Predição de valores futuros** com IA
- ✅ **Parametrização flexível** - analise qualquer estado brasileiro
- ✅ **Validação automática** - verifica se o estado existe nos dados
- ✅ **Dados reais do IBGE** (fonte confiável)
- ✅ **Código educacional** com comentários explicativos
- ✅ **Pipeline completo** de ML (carregamento → treino → predição)
- ✅ **Gerenciamento de memória** profissional
- ✅ **Visualização clara** dos resultados

## 📝 Estrutura dos Dados

O arquivo `dados-historicos.json` contém séries temporais de 9 estados brasileiros:

```json
[
  {
    "estado": "SP",
    "dados": [
      { "ano": 2010, "populacao": 41262199 },
      { "ano": 2015, "populacao": 44396484 },
      { "ano": 2020, "populacao": 46289333 },
      { "ano": 2025, "populacao": 44411238 }
    ]
  },
  { "estado": "MG", "dados": [...] },
  ...
]
```

**Estados disponíveis**: SP, MG, RJ, BA, PR, RS, PE, CE, ES

## 🧠 Como Funciona

### 1. **Configuração**
   - Constante `ESTADO_ALVO` define qual estado será analisado
   - Mapeamento de siglas para nomes completos
   - Validação automática de estado válido

### 2. **Carregamento de Dados**
   - Lê dados históricos de população (2010-2025) de 9 estados brasileiros
   - Utiliza o módulo `fs` para ler arquivos JSON locais
   - Extrai apenas dados do estado configurado

### 3. **Pré-processamento**
   - Extrai apenas valores numéricos (população) dos dados
   - Separa variáveis X (anos) e Y (população)

### 4. **Normalização**
   - Transforma todos os valores para o intervalo [0, 1]
   - Fórmula: `(valor - mínimo) / (máximo - mínimo)`
   - Essencial para o aprendizado equilibrado do modelo

### 5. **Conversão para Tensores**
   - Array JavaScript → Tensor 1D/2D do TensorFlow
   - Permite processamento vetorizado ultra-rápido
   - Preparação para operações de Machine Learning

### 6. **Criação do Modelo**
   - **Arquitetura**: Rede Neural Sequencial
   - **Camadas**: 1 camada densa (fully connected)
   - **Neurônios**: 1 neurônio de saída (regressão)
   - **Ativação**: Linear (sem transformação)
   - **Parâmetros**: 2 (1 peso + 1 bias)

### 7. **Treinamento**
   - **Algoritmo**: SGD (Stochastic Gradient Descent)
   - **Loss Function**: Mean Squared Error (MSE)
   - **Learning Rate**: 0.1
   - **Épocas**: 100 iterações completas
   - O modelo ajusta seus pesos para minimizar o erro

### 8. **Predição do Futuro**
   - Normaliza o ano alvo (2030)
   - Passa pelo modelo treinado
   - Desnormaliza o resultado para obter valor real
   - Calcula variação percentual e absoluta

### 9. **Cálculos Estatísticos**
   - Usa métodos nativos dos tensores: `.mean()`, `.max()`, `.min()`
   - Muito mais rápido que loops tradicionais
   - Aproveitamento de paralelização

### 10. **Gerenciamento de Memória**
   - Todos os tensores são destruídos com `.dispose()`
   - Evita memory leaks em aplicações de longa duração
   - Boa prática essencial em Machine Learning

## 📄 Licença

ISC

## 🎓 Contexto Acadêmico

Projeto desenvolvido durante a **Pós-Graduação em Engenharia de IA Aplicada**, demonstrando a aplicação prática de conceitos de:
- Tensores e operações matriciais
- Machine Learning com TensorFlow.js
- Regressão linear e otimização
- Normalização e pré-processamento de dados
- Predição de séries temporais

---

**Desenvolvido com TensorFlow.js e Node.js** 🚀  
*Transformando dados do IBGE em insights preditivos com Inteligência Artificial*
