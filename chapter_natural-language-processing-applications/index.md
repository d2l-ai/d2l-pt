# Processamento de Linguagem Natural: Aplicações
:label:`chap_nlp_app`


Vimos como representar tokens de texto e treinar suas representações em :numref:`chap_nlp_pretrain`.
Essas representações de texto pré-treinadas podem ser fornecidas a vários modelos para diferentes tarefas de processamento de linguagem natural *downstream*.

Este livro não pretende cobrir as aplicações de processamento de linguagem natural de uma maneira abrangente.
Nosso foco é *como aplicar a aprendizagem de representação (profunda) de idiomas para resolver problemas de processamento de linguagem natural*.
No entanto, já discutimos várias aplicações de processamento de linguagem natural sem pré-treinamento nos capítulos anteriores,
apenas para explicar arquiteturas de aprendizado profundo.
Por exemplo, em :numref:`chap_rnn`,
contamos com RNNs para projetar modelos de linguagem para gerar textos semelhantes a novelas.
Em :numref:`chap_modern_rnn` e :numref:`chap_attention`,
também projetamos modelos baseados em RNNs e mecanismos de atenção
para tradução automática.
Dadas as representações de texto pré-treinadas,
neste capítulo, consideraremos mais duas tarefas de processamento de linguagem natural *downstream*:
análise de sentimento e inferência de linguagem natural.
Estes são aplicativos de processamento de linguagem natural populares e representativos:
o primeiro analisa um único texto e o último analisa as relações de pares de texto.

![As representações de texto pré-treinadas podem ser alimentadas para várias arquiteturas de *deep learning*  para diferentes aplicações de processamento de linguagem natural *downstream*. Este capítulo enfoca como projetar modelos para diferentes aplicações de processamento de linguagem natural *downstream*.](../img/nlp-map-app.svg)
:label:`fig_nlp-map-app`


Conforme descrito em :numref:`fig_nlp-map-app`,
este capítulo se concentra na descrição das ideias básicas de projeto de modelos de processamento de linguagem natural usando diferentes tipos de arquiteturas de aprendizado profundo, como MLPs, CNNs, RNNs e atenção.
Embora seja possível combinar qualquer representação de texto pré-treinada com qualquer arquitetura para qualquer tarefa de processamento de linguagem natural *downstream* em :numref:`fig_nlp-map-app`,
selecionamos algumas combinações representativas.
Especificamente, exploraremos arquiteturas populares baseadas em RNNs e CNNs para análise de sentimento.
Para inferência de linguagem natural, escolhemos atenção e MLPs para demonstrar como analisar pares de texto.
No final, apresentamos como ajustar um modelo BERT pré-treinado
para uma ampla gama de aplicações de processamento de linguagem natural,
como em um nível de sequência (classificação de texto único e classificação de par de texto)
e um nível de *token* (marcação de texto e resposta a perguntas).
Como um caso empírico concreto,
faremos o ajuste fino do BERT para processamento de linguagem natural.

Como apresentamos em :numref:`sec_bert`,
BERT requer mudanças mínimas de arquitetura
para uma ampla gama de aplicativos de processamento de linguagem natural.
No entanto, esse benefício vem com o custo de um ajuste fino
um grande número de parâmetros BERT para as aplicações *downstream*.
Quando o espaço ou o tempo são limitados,
aqueles modelos elaborados com base em MLPs, CNNs, RNNs e atenção
são mais viáveis.
A seguir, começamos pelo aplicativo de análise de sentimento
e ilustrar o design do modelo baseado em RNNs e CNNs, respectivamente.

```toc
:maxdepth: 2

sentiment-analysis-and-dataset
sentiment-analysis-rnn
sentiment-analysis-cnn
natural-language-inference-and-dataset
natural-language-inference-attention
finetuning-bert
natural-language-inference-bert
```

<!--stackedit_data:
eyJoaXN0b3J5IjpbLTE0MjEwMjQ2NzYsNDc5OTgzNzkzLDY5MD
Y3MjA3MF19
-->