const fs = require('fs');
const tf = require('@tensorflow/tfjs');

// Ler o arquivo dados.json
const dadosRaw = fs.readFileSync('./dados.json', 'utf-8');
const dados = JSON.parse(dadosRaw);

// Extrair apenas os valores de população
const populacoes = dados.map(item => item.populacao);

// Converter para Tensor 1D
const tensorPopulacao = tf.tensor1d(populacoes);

// Calcular estatísticas usando métodos do TensorFlow
const media = tensorPopulacao.mean();
const maximo = tensorPopulacao.max();
const minimo = tensorPopulacao.min();

// Exibir resultados
console.log('=== Análise de População dos Estados ===\n');
console.log(`Média das populações: ${media.dataSync()[0].toLocaleString('pt-BR')}`);
console.log(`Estado mais populoso: ${maximo.dataSync()[0].toLocaleString('pt-BR')} habitantes`);
console.log(`Estado menos populoso: ${minimo.dataSync()[0].toLocaleString('pt-BR')} habitantes`);

// Liberar memória dos tensores
tensorPopulacao.dispose();
media.dispose();
maximo.dispose();
minimo.dispose();