# 📊 Analisador Estatístico de População - IBGE

Projeto de análise estatística de dados populacionais brasileiros utilizando **TensorFlow.js** para processamento de tensores em ambiente Node.js.

## 🎯 Sobre o Projeto

Este projeto demonstra o uso de **Tensores** do TensorFlow.js para realizar cálculos estatísticos sobre dados de população dos estados brasileiros (IBGE). A aplicação lê dados locais em formato JSON e utiliza operações matemáticas nativas do TensorFlow para calcular:

- **Média** das populações estaduais
- **Valor máximo** (estado mais populoso)
- **Valor mínimo** (estado menos populoso)

## 🛠️ Tecnologias Utilizadas

- **Node.js** - Runtime JavaScript
- **TensorFlow.js** - Biblioteca de Machine Learning para processamento de tensores
- **File System (fs)** - Módulo nativo do Node.js para leitura de arquivos

## 📋 Pré-requisitos

- Node.js (versão 14 ou superior)
- npm (gerenciador de pacotes)

## 🚀 Como Rodar o Projeto

### 1. Instalar as dependências

```bash
npm install
```

Este comando irá instalar todas as dependências necessárias listadas no `package.json`, incluindo o `@tensorflow/tfjs`.

### 2. Executar o projeto

```bash
node index.js
```

## 📊 Exemplo de Resultado

Ao executar o projeto, você verá no console uma saída similar a:

```
============================
Hi, looks like you are running TensorFlow.js in Node.js. To speed things up dramatically, install our node backend, visit https://github.com/tensorflow/tfjs-node for more details. 
============================
=== Análise de População dos Estados ===

Média das populações: 15.462.339
Estado mais populoso: 44.411.240 habitantes
Estado menos populoso: 3.833.712 habitantes
```

## 📁 Estrutura do Projeto

```
Projeto-00/
├── dados.json          # Dados de população dos estados brasileiros
├── index.js            # Script principal de análise
├── package.json        # Configurações e dependências do projeto
└── README.md           # Documentação do projeto
```

## 📝 Estrutura dos Dados

O arquivo `dados.json` contém um array de objetos com informações dos estados:

```json
[
  { "estado": "SP", "populacao": 44411238 },
  { "estado": "MG", "populacao": 20538718 },
  ...
]
```

## 🧠 Como Funciona

1. **Leitura de Dados**: O módulo `fs` lê o arquivo `dados.json` de forma síncrona
2. **Extração**: Os valores numéricos de população são extraídos para um array
3. **Conversão para Tensor**: O array é convertido em um Tensor 1D do TensorFlow.js
4. **Cálculos Estatísticos**: Utiliza os métodos `.mean()`, `.max()` e `.min()` dos tensores
5. **Exibição**: Os resultados são formatados e exibidos no console
6. **Limpeza**: Os tensores são liberados da memória com `.dispose()`

## 📄 Licença

ISC

---

Desenvolvido com TensorFlow.js e Node.js 🚀
