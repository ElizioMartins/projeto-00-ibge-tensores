const fs = require('fs');
const tf = require('@tensorflow/tfjs');

// ============================================
// PARTE 1: Análise Estática (Dados de 2025)
// ============================================

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
console.log('=== Análise de População dos Estados (2025) ===\n');
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

// ============================================
// PARTE 2: Análise Temporal (Dados Históricos)
// ============================================

console.log('\n=== Análise Temporal - Evolução de São Paulo (2010-2025) ===\n');

// 7. Carregar dados históricos
const dadosHistoricosRaw = fs.readFileSync('./dados-historicos.json', 'utf-8');
const dadosHistoricos = JSON.parse(dadosHistoricosRaw);

// 8. Extrair dados de SP para análise temporal
const dadosSP = dadosHistoricos.find(item => item.estado === 'SP');
const populacoesSP = dadosSP.dados.map(d => d.populacao);
const anos = dadosSP.dados.map(d => d.ano);

// 9. Criar tensor com série temporal
const tensorTemporalSP = tf.tensor1d(populacoesSP);

// 10. Calcular crescimento ao longo do tempo
console.log('Evolução populacional de São Paulo:');
dadosSP.dados.forEach(d => {
  console.log(`  ${d.ano}: ${d.populacao.toLocaleString('pt-BR')} habitantes`);
});

// 11. Calcular taxa de crescimento média usando tensors
const crescimentoMedio = tensorTemporalSP.mean();
console.log(`\nPopulação média no período: ${crescimentoMedio.dataSync()[0].toLocaleString('pt-BR')} habitantes`);

// 12. Limpeza de memória
tensorTemporalSP.dispose();
crescimentoMedio.dispose();