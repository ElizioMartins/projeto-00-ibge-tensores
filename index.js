const fs = require('fs');
const tf = require('@tensorflow/tfjs');

// ============================================
// FUNÇÕES AUXILIARES
// ============================================

/**
 * Normaliza um array de valores para o intervalo [0, 1]
 * Fórmula: (valor - mínimo) / (máximo - mínimo)
 * Isso é crucial para que o modelo de IA aprenda de forma equilibrada,
 * evitando que valores grandes (como anos ou população) dominem o aprendizado.
 */
function normalizar(valores) {
  const min = Math.min(...valores);
  const max = Math.max(...valores);
  return {
    normalizados: valores.map(v => (v - min) / (max - min)),
    min,
    max
  };
}

/**
 * Desnormaliza um valor que estava no intervalo [0, 1]
 * Fórmula inversa: valor_normalizado * (máximo - mínimo) + mínimo
 */
function desnormalizar(valorNormalizado, min, max) {
  return valorNormalizado * (max - min) + min;
}

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

// ============================================
// PARTE 3: Normalização de Dados (Preparação para IA)
// ============================================

console.log('\n=== Normalização de Dados ===\n');

// 13. Normalizar os anos (variável X) e populações (variável Y)
// Isso transforma valores como [2010, 2015, 2020, 2025] em [0, 0.33, 0.66, 1]
// e populações gigantes em valores entre 0 e 1.
const anosNormalizados = normalizar(anos);
const populacoesNormalizadas = normalizar(populacoesSP);

console.log('Anos originais:', anos);
console.log('Anos normalizados:', anosNormalizados.normalizados.map(v => v.toFixed(2)));

console.log('\nPopulações originais:', populacoesSP.map(p => p.toLocaleString('pt-BR')));
console.log('Populações normalizadas:', populacoesNormalizadas.normalizados.map(v => v.toFixed(4)));

// 14. Exemplo de desnormalização
const primeiroAnoNormalizado = anosNormalizados.normalizados[0];
const anoOriginal = desnormalizar(primeiroAnoNormalizado, anosNormalizados.min, anosNormalizados.max);
console.log(`\nExemplo: ${primeiroAnoNormalizado.toFixed(2)} (normalizado) = ${anoOriginal} (original)`);

// ============================================
// PARTE 4: Modelo de Regressão Linear (Machine Learning!)
// ============================================

console.log('\n=== Treinamento do Modelo de IA ===\n');

// 15. Criar tensores de treino (X = anos, Y = população)
// tf.tensor2d cria uma matriz, que é o formato esperado pelo modelo
// Cada linha é um exemplo de treino: [[ano1], [ano2], [ano3], [ano4]]
const X_train = tf.tensor2d(anosNormalizados.normalizados, [anos.length, 1]);
const Y_train = tf.tensor2d(populacoesNormalizadas.normalizados, [populacoesSP.length, 1]);

// 16. Criar o modelo sequencial
// Sequential = camadas empilhadas uma após a outra
const modelo = tf.sequential();

// 17. Adicionar camada densa (neurônios totalmente conectados)
// units: 1 = um único neurônio de saída (prever 1 valor: a população)
// inputShape: [1] = recebe 1 entrada (o ano)
// activation: 'linear' = sem função de ativação, pois é regressão linear pura
modelo.add(tf.layers.dense({
  units: 1,
  inputShape: [1],
  activation: 'linear'
}));

// 18. Compilar o modelo (definir como ele vai aprender)
// optimizer: 'sgd' (Stochastic Gradient Descent) = algoritmo que ajusta os pesos
// loss: 'meanSquaredError' = mede o erro como (predição - real)²
// learningRate: 0.1 = velocidade do aprendizado (0.1 = moderada)
modelo.compile({
  optimizer: tf.train.sgd(0.1),
  loss: 'meanSquaredError'
});

console.log('📊 Arquitetura do Modelo:');
modelo.summary();

// 19. Treinar o modelo
// epochs: 100 = número de vezes que o modelo verá todos os dados
// Aqui é onde a "mágica" acontece: o modelo ajusta seus pesos internos
// para minimizar o erro entre suas previsões e os valores reais
console.log('\n🎯 Treinando modelo...');

async function treinarModelo() {
  await modelo.fit(X_train, Y_train, {
    epochs: 100,
    verbose: 0 // verbose: 0 = não mostrar progresso de cada época
  });
  
  console.log('✅ Treinamento concluído!\n');
  
  // 20. Testar o modelo com os dados de treino
  console.log('=== Testando o Modelo com Dados Conhecidos ===\n');
  
  for (let i = 0; i < anos.length; i++) {
    const anoNorm = anosNormalizados.normalizados[i];
    const predicaoNorm = modelo.predict(tf.tensor2d([anoNorm], [1, 1]));
    const predicaoReal = desnormalizar(
      predicaoNorm.dataSync()[0],
      populacoesNormalizadas.min,
      populacoesNormalizadas.max
    );
    
    console.log(`${anos[i]}: Real = ${populacoesSP[i].toLocaleString('pt-BR')}, Predito = ${Math.round(predicaoReal).toLocaleString('pt-BR')}`);
    
    predicaoNorm.dispose();
  }
  
  // 21. Limpeza de memória dos tensores de treino
  X_train.dispose();
  Y_train.dispose();
}

// Executar o treinamento
treinarModelo();
