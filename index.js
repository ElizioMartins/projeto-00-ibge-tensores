const fs = require('fs');
const tf = require('@tensorflow/tfjs');

// 1. Carregamento dos Dados
// Lê o arquivo local como texto (utf-8) e converte para objeto JavaScript.
// Na vida real, isso poderia ser uma chamada de API (axios) ou um CSV gigante do IBGE.
const dadosRaw = fs.readFileSync('./dados.json', 'utf-8');
const dados = JSON.parse(dadosRaw);

// 2. Pré-processamento
// O TensorFlow é focado em matemática pura. 
// Por isso, separamos apenas os números (população), ignorando textos como siglas de estados.
const populacoes = dados.map(item => item.populacao);

// 3. Criação do Tensor
// Transforma o array simples do JS em um Tensor 1D.
// Isso prepara os dados para processamento paralelo na GPU ou otimizado na CPU,
// sendo infinitamente mais rápido que um "for loop" comum para milhões de registros.
const tensorPopulacao = tf.tensor1d(populacoes);

// 4. Processamento Matemático
// Usa as funções nativas do TensorFlow para calcular métricas de uma só vez
const media = tensorPopulacao.mean();
const maximo = tensorPopulacao.max();
const minimo = tensorPopulacao.min();

// 5. Extração e Exibição
// .dataSync()[0] é usado para "puxar" o resultado calculado de volta do mundo dos tensores 
// para uma variável comum que o Node.js consiga imprimir na tela.
console.log('=== Análise de População dos Estados ===\n');
console.log(`Média das populações: ${media.dataSync()[0].toLocaleString('pt-BR')} habitantes`);
console.log(`Estado mais populoso: ${maximo.dataSync()[0].toLocaleString('pt-BR')} habitantes`);
console.log(`Estado menos populoso: ${minimo.dataSync()[0].toLocaleString('pt-BR')} habitantes`);

// 6. Gerenciamento de Memória
// Regra de ouro da IA: se você criou um tensor e não vai mais usar, destrua-o.
// Isso evita estouro de memória (memory leak) em aplicações rodando 24/7.
tensorPopulacao.dispose();
media.dispose();
maximo.dispose();
minimo.dispose();