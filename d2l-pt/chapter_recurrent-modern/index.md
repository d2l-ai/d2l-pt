# Redes Neurais Recorrentes Modernas
:label:`chap_modern_rnn`

Introduzimos os conceitos básicos de RNNs, que podem lidar melhor com dados de sequência. Para demonstração, implementamos modelos de linguagem baseados em RNN em dados de texto. No entanto, tais técnicas podem
não ser suficiente para os profissionais quando eles enfrentam uma ampla gama de problemas de aprendizagem de sequência
hoje em dia.

Por exemplo, um problema notável na prática é a instabilidade numérica dos RNNs. Embora tenhamos
truques de implementação aplicados, como recorte de gradiente, esse problema pode ser aliviado ainda mais com
designs mais sofisticados de modelos de sequência. Especificamente, os RNNs controlados são muito mais comuns na prática. Começaremos apresentando duas dessas redes amplamente utilizadas, chamadas de *gated recurrent units* (GRUs) e *long short-term memory* (LSTM). Além disso, vamos expandir o RNN
arquitetura com uma única camada oculta indireta que foi discutida até agora. Descreveremos arquiteturas profundas com múltiplas camadas ocultas e discutiremos o projeto bidirecional com
cálculos recorrentes para frente e para trás. Essas expansões são frequentemente adotadas em
redes recorrentes modernas. Ao explicar essas variantes RNN, continuamos a considerar o
mesmo problema de modelagem de linguagem apresentado no :numref:`chap_rnn`.

Na verdade, a modelagem de linguagem revela apenas uma pequena fração do que o aprendizado de sequência é capaz.
Em uma variedade de problemas de aprendizagem de sequência, como reconhecimento automático de fala, conversão de texto em fala,
e tradução automática, tanto as entradas quanto as saídas são sequências de comprimento arbitrário. Explicar
como ajustar este tipo de dados, tomaremos a tradução automática como exemplo e apresentaremos o
arquitetura codificador-decodificador baseada em RNNs e busca de feixe para geração de sequência.

```toc
:maxdepth: 2

gru
lstm
deep-rnn
bi-rnn
machine-translation-and-dataset
encoder-decoder
seq2seq
beam-search
```

<!--stackedit_data:
eyJoaXN0b3J5IjpbMzAwNTc5NDg1LDE1MDA1NDM1MDZdfQ==
-->