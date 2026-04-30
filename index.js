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
// PARTE 1: Carregamento e Análise dos Dados Históricos
// ============================================

console.log('=== 📊 Preditor de População - São Paulo (IBGE) ===\n');
console.log('🔮 Análise Temporal e Predição com Machine Learning\n');
console.log('='.repeat(60) + '\n');

// 1. Carregar dados históricos
const dadosHistoricosRaw = fs.readFileSync('./dados-historicos.json', 'utf-8');
const dadosHistoricos = JSON.parse(dadosHistoricosRaw);

// 2. Extrair dados de SP para análise temporal
const dadosSP = dadosHistoricos.find(item => item.estado === 'SP');
const populacoesSP = dadosSP.dados.map(d => d.populacao);
const anos = dadosSP.dados.map(d => d.ano);

// 3. Criar tensor com série temporal
const tensorTemporalSP = tf.tensor1d(populacoesSP);

// 4. Exibir evolução temporal
console.log('📅 Evolução populacional de São Paulo:');
dadosSP.dados.forEach(d => {
  console.log(`  ${d.ano}: ${d.populacao.toLocaleString('pt-BR')} habitantes`);
});

// 5. Calcular estatísticas usando tensors
const crescimentoMedio = tensorTemporalSP.mean();
console.log(`\n📊 População média no período: ${crescimentoMedio.dataSync()[0].toLocaleString('pt-BR')} habitantes`);

// 6. Limpeza de memória
tensorTemporalSP.dispose();
crescimentoMedio.dispose();

// ============================================
// PARTE 2: Normalização de Dados (Preparação para IA)
// ============================================

console.log('\n=== Normalização de Dados ===\n');

// 7. Normalizar os anos (variável X) e populações (variável Y)
// Isso transforma valores como [2010, 2015, 2020, 2025] em [0, 0.33, 0.66, 1]
// e populações gigantes em valores entre 0 e 1.
const anosNormalizados = normalizar(anos);
const populacoesNormalizadas = normalizar(populacoesSP);

console.log('Anos originais:', anos);
console.log('Anos normalizados:', anosNormalizados.normalizados.map(v => v.toFixed(2)));

console.log('\nPopulações originais:', populacoesSP.map(p => p.toLocaleString('pt-BR')));
console.log('Populações normalizadas:', populacoesNormalizadas.normalizados.map(v => v.toFixed(4)));

// 8. Exemplo de desnormalização
const primeiroAnoNormalizado = anosNormalizados.normalizados[0];
const anoOriginal = desnormalizar(primeiroAnoNormalizado, anosNormalizados.min, anosNormalizados.max);
console.log(`\nExemplo: ${primeiroAnoNormalizado.toFixed(2)} (normalizado) = ${anoOriginal} (original)`);

// ============================================
// PARTE 3: Modelo de Regressão Linear (Machine Learning!)
// ============================================

console.log('\n=== Treinamento do Modelo de IA ===\n');

// 9. Criar tensores de treino (X = anos, Y = população)
// tf.tensor2d cria uma matriz, que é o formato esperado pelo modelo
// Cada linha é um exemplo de treino: [[ano1], [ano2], [ano3], [ano4]]
const X_train = tf.tensor2d(anosNormalizados.normalizados, [anos.length, 1]);
const Y_train = tf.tensor2d(populacoesNormalizadas.normalizados, [populacoesSP.length, 1]);

// 10. Criar o modelo sequencial
// Sequential = camadas empilhadas uma após a outra
const modelo = tf.sequential();

// 11. Adicionar camada densa (neurônios totalmente conectados)
// units: 1 = um único neurônio de saída (prever 1 valor: a população)
// inputShape: [1] = recebe 1 entrada (o ano)
// activation: 'linear' = sem função de ativação, pois é regressão linear pura
modelo.add(tf.layers.dense({
  units: 1,
  inputShape: [1],
  activation: 'linear'
}));

// 12. Compilar o modelo (definir como ele vai aprender)
// optimizer: 'sgd' (Stochastic Gradient Descent) = algoritmo que ajusta os pesos
// loss: 'meanSquaredError' = mede o erro como (predição - real)²
// learningRate: 0.1 = velocidade do aprendizado (0.1 = moderada)
modelo.compile({
  optimizer: tf.train.sgd(0.1),
  loss: 'meanSquaredError'
});

console.log('📊 Arquitetura do Modelo:');
modelo.summary();

// 13. Treinar o modelo
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
  
  // 14. Testar o modelo com os dados de treino
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
  
  // ============================================
  // PARTE 4: Predição do Futuro! (2030)
  // ============================================
  
  console.log('\n=== 🔮 Predição para 2030 ===\n');
  
  // 15. Preparar o ano 2030 para predição
  const ano2030 = 2030;
  const ano2030Normalizado = (ano2030 - anosNormalizados.min) / (anosNormalizados.max - anosNormalizados.min);
  
  // 16. Fazer a predição
  const predicao2030Norm = modelo.predict(tf.tensor2d([ano2030Normalizado], [1, 1]));
  const predicao2030Real = desnormalizar(
    predicao2030Norm.dataSync()[0],
    populacoesNormalizadas.min,
    populacoesNormalizadas.max
  );
  
  console.log(`🎯 População prevista para São Paulo em 2030: ${Math.round(predicao2030Real).toLocaleString('pt-BR')} habitantes`);
  
  // 17. Calcular a diferença em relação a 2025
  const ultimaPopulacao = populacoesSP[populacoesSP.length - 1];
  const diferencaAbsoluta = Math.round(predicao2030Real) - ultimaPopulacao;
  const diferencaPercentual = ((predicao2030Real - ultimaPopulacao) / ultimaPopulacao * 100).toFixed(2);
  
  console.log(`\n📈 Variação esperada (2025 → 2030):`);
  console.log(`   Diferença: ${diferencaAbsoluta > 0 ? '+' : ''}${diferencaAbsoluta.toLocaleString('pt-BR')} habitantes`);
  console.log(`   Percentual: ${diferencaPercentual > 0 ? '+' : ''}${diferencaPercentual}%`);
  
  // 18. Limpeza de memória
  predicao2030Norm.dispose();
  
  // ============================================
  // RESUMO FINAL
  // ============================================
  
  console.log('\n' + '='.repeat(60));
  console.log('📊 RESUMO DA ANÁLISE COM IA');
  console.log('='.repeat(60));
  console.log('\n🔢 Dados analisados:');
  console.log(`   • Estado: São Paulo (SP)`);
  console.log(`   • Período histórico: ${anos[0]} - ${anos[anos.length - 1]}`);
  console.log(`   • Pontos de dados temporais: ${anos.length}`);
  console.log(`   • População em 2025: ${populacoesSP[populacoesSP.length - 1].toLocaleString('pt-BR')} habitantes`);
  
  console.log('\n🧠 Modelo de Machine Learning:');
  console.log(`   • Tipo: Regressão Linear`);
  console.log(`   • Framework: TensorFlow.js`);
  console.log(`   • Parâmetros treináveis: 2 (peso + bias)`);
  console.log(`   • Épocas de treinamento: 100`);
  
  console.log('\n🎯 Predição Gerada:');
  console.log(`   • Ano alvo: 2030`);
  console.log(`   • População prevista: ${Math.round(predicao2030Real).toLocaleString('pt-BR')} habitantes`);
  console.log(`   • Crescimento esperado: ${diferencaPercentual}% em 5 anos`);
  
  console.log('\n💡 Insights:');
  const tendencia = diferencaAbsoluta > 0 ? 'crescimento' : 'decrescimento';
  const impacto = Math.abs(diferencaPercentual) > 10 ? 'significativo' : 'moderado';
  console.log(`   • São Paulo apresenta ${tendencia} ${impacto}`);
  console.log(`   • O modelo sugere ${diferencaAbsoluta > 0 ? 'aumento' : 'redução'} de ${Math.abs(Math.round(diferencaAbsoluta)).toLocaleString('pt-BR')} habitantes`);
  
  console.log('\n' + '='.repeat(60));
  console.log('✨ Análise concluída com sucesso!');
  console.log('='.repeat(60) + '\n');
  
  // 19. Limpeza de memória dos tensores de treino
  X_train.dispose();
  Y_train.dispose();
}

// Executar o treinamento
treinarModelo();
