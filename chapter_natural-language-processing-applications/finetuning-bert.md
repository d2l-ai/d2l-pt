# Ajuste Fino de BERT para Aplicações de Nível de Sequência e de Token
:label:`sec_finetuning-bert`

Nas seções anteriores deste capítulo, projetamos diferentes modelos para aplicações de processamento de linguagem natural, como os baseados em RNNs, CNNs, atenção e MLPs.
Esses modelos são úteis quando há restrição de espaço ou tempo,
no entanto, elaborar um modelo específico para cada tarefa de processamento de linguagem natural é praticamente inviável.
Em :numref:`sec_bert`, introduzimos um modelo de pré-treinamento, BERT, que requer mudanças mínimas de arquitetura para uma ampla gama de tarefas de processamento de linguagem natural.
Por um lado, na altura da sua proposta, o BERT melhorou o estado da arte em várias tarefas de processamento de linguagem natural.
Por outro lado, conforme observado em :numref:`sec_bert-pretraining`, as duas versões do modelo BERT original vêm com 110 milhões e 340 milhões de parâmetros.
Assim, quando há recursos computacionais suficientes, podemos considerar o ajuste fino do BERT para aplicativos de processamento de linguagem natural *downstream*.

A seguir, generalizamos um subconjunto de aplicações de processamento de linguagem natural como nível de sequência e nível de *token*.
No nível da sequência, apresentamos como transformar a representação BERT da entrada de texto no rótulo de saída em classificação de texto único e classificação ou regressão de par de texto.
No nível do *token*, apresentaremos brevemente novos aplicativos, como marcação de texto e resposta a perguntas, e esclareceremos como o BERT pode representar suas entradas e ser transformado em rótulos de saída.
Durante o ajuste fino, as "mudanças mínimas de arquitetura" exigidas pelo BERT em diferentes aplicativos são as camadas extras totalmente conectadas.
Durante o aprendizado supervisionado de uma aplicação *downstream*, os parâmetros das camadas extras são aprendidos do zero, enquanto todos os parâmetros no modelo BERT pré-treinado são ajustados.


## Classificação de Texto Único

* Classificação de texto único * pega uma única sequência de texto como entrada e produz seu resultado de classificação.
Além da análise de sentimento que estudamos neste capítulo,
o Corpus de Aceitabilidade Linguística (CoLA)
também é um conjunto de dados para classificação de texto único,
julgando se uma determinada frase é gramaticalmente aceitável ou não :cite:`Warstadt.Singh.Bowman.2019`.
Por exemplo, "Eu deveria estudar." é aceitável, mas "Eu deveria estudando." não é.

![Ajuste fino do BERT para aplicações de classificação de texto único, como análise de sentimento e teste de aceitabilidade linguística. Suponha que o texto único de entrada tenha seis *tokens*.](../img/bert-one-seq.svg)
:label:`fig_bert-one-seq`

:numref:`sec_bert` descreve a representação de entrada de BERT.
A sequência de entrada de BERT representa inequivocamente texto único e pares de texto, onde o *token* de classificação especial “&lt;cls&gt;” é usado para classificação de sequência e o *token* de classificação especial “&lt;sep&gt;” marca o fim de um único texto ou separa um par de texto.
Conforme mostrado em :numref:`fig_bert-one-seq`, em aplicativos de classificação de texto único, a representação BERT do *token* de classificação especial “&lt;cls&gt;” codifica as informações de toda a sequência de texto de entrada.
Como a representação do texto único de entrada, ele será alimentado em um pequeno MLP que consiste em camadas totalmente conectadas (densas) para gerar a distribuição de todos os valores de rótulo discretos.


## Classificação ou Regressão de Pares de Texto


Também examinamos a inferência da linguagem natural neste capítulo.
Pertence à *classificação de pares de texto*, um tipo de aplicativo que classifica um par de texto.

Tomando um par de texto como entrada, mas gerando um valor contínuo, *similaridade textual semântica* é uma tarefa popular de *regressão de par de texto*.
Esta tarefa mede a similaridade semântica das sentenças.
Por exemplo, no conjunto de dados Semantic Textual Similarity Benchmark, a pontuação de similaridade de um par de sentenças é uma escala ordinal que varia de 0 (sem sobreposição de significado) a 5 (significando equivalência) :cite:`Cer.Diab.Agirre.ea.2017 `.
O objetivo é prever essas pontuações.
Exemplos do conjunto de dados de referência de similaridade textual semântica incluem (sentença 1, sentença 2, pontuação de similaridade):

* "Um avião está decolando.", "Um avião está decolando.", 5.000;
* "Mulher está comendo alguma coisa.", "Mulher está comendo carne.", 3.000;
* "Uma mulher está dançando.", "Um homem está falando.", 0,000.

![Ajuste fino do BERT para aplicações de classificação ou regressão de pares de texto, como inferência de linguagem natural e similaridade textual semântica. Suponha que o par de texto de entrada tenha dois e três *tokens*.](../img/bert-two-seqs.svg)
:label:`fig_bert-two-seqs`

Comparando com a classificação de texto único em :numref:`fig_bert-one-seq`,
ajuste fino de BERT para classificação de par de texto em :numref:`fig_bert-two-seqs`
é diferente na representação de entrada.
Para tarefas de regressão de pares de texto, como semelhança textual semântica,
mudanças triviais podem ser aplicadas, como a saída de um valor de rótulo contínuo
e usando a perda quadrática média: eles são comuns para regressão.


## Marcação de Texto

Agora, vamos considerar as tarefas de nível de *token*, como *marcação de texto*,
onde cada *token* é atribuído a um rótulo.
Entre as tarefas de marcação de texto, *marcação de classe gramatical* atribui a cada palavra uma marcação de classe gramatical (por exemplo, adjetivo e determinante) de acordo com a função da palavra na frase.
Por exemplo, de acordo com o conjunto de *tags* Penn Treebank II, a frase "*John Smith 's car is new*" ("O carro de John Smith é novo") deve ser marcada como
"NNP (substantivo, singular próprio) NNP POS (desinência possessiva) NN (substantivo, singular ou massa) VB (verbo, forma básica) JJ (adjetivo)".

![Ajuste fino do BERT para aplicativos de marcação de texto, como marcação de classes gramaticais. Suponha que o texto único de entrada tenha seis *tokens*.](../img/bert-tagging.svg)
:label:`fig_bert-tagging`

O ajuste fino do BERT para aplicações de marcação de texto é ilustrado em :numref:`fig_bert-tagging`.
Comparando com :numref:`fig_bert-one-seq`, a única distinção reside na marcação de texto, a representação BERT de *cada token* do texto de entrada é alimentado nas mesmas camadas extras totalmente conectadas para dar saída ao rótulo de o *token*, como uma *tag* de classe gramatical.



## Resposta a Perguntas

Como outro aplicativo de nível de *token*, *responder a perguntas* reflete as capacidades de compreensão de leitura.
Por exemplo, o conjunto de dados de resposta a perguntas de Stanford (SQuAD v1.1) consiste na leitura de passagens e perguntas, em que a resposta a cada pergunta é apenas um segmento de texto (extensão de texto) da passagem sobre a qual a pergunta se refere cite:`Rajpurkar.Zhang.Lopyrev.ea.2016`.
Para explicar, considere uma passagem "Alguns especialistas relatam que a eficácia de uma máscara é inconclusiva. No entanto, os fabricantes de máscaras insistem que seus produtos, como as máscaras respiratórias N95, podem proteger contra o vírus." e uma pergunta "Quem disse que as máscaras respiratórias N95 podem proteger contra o vírus?". A resposta deve ser o intervalo de texto "fabricantes de máscara" na passagem. Assim, o objetivo no SQuAD v1.1 é prever o início e o fim da extensão do texto na passagem dada um par de pergunta e passagem.

![Ajuste fino do BERT para resposta a perguntas. Suponha que o par de texto de entrada tenha dois e três *tokens*.](../img/bert-qa.svg)
:label:`fig_bert-qa`

Para ajustar o BERT para responder às perguntas, a pergunta e a passagem são compactadas como a primeira e a segunda sequência de texto, respectivamente, na entrada do BERT.
Para prever a posição do início do intervalo de texto, a mesma camada adicional totalmente conectada transformará a representação BERT de qualquer *token* da passagem da posição $i$ em uma pontuação escalar $s_i$.
Essas pontuações de todas as fichas de passagem são posteriormente transformadas pela operação *softmax* em uma distribuição de probabilidade, de modo que cada posição de ficha $i$ na passagem recebe uma probabilidade $p_i$ de ser o início do período de texto.
A previsão do final da extensão do texto é igual à anterior, exceto que os parâmetros em sua camada adicional totalmente conectada são independentes daqueles para a previsão do início.
Ao prever o final, qualquer *token* de passagem da posição $i$ é transformado pela mesma camada totalmente conectada em uma pontuação escalar $e_i$.
:numref:`fig_bert-qa` descreve o ajuste fino do BERT para responder a perguntas.

Para responder a perguntas,
o objetivo de treinamento da aprendizagem supervisionada é tão simples quanto
maximizar as probabilidades-log das posições inicial e final de verdade.
Ao prever a amplitude,
podemos calcular a pontuação $s_i + e_j$ para um intervalo válido
da posição $i$ para a posição $j$ ($i \leq j$),
e produzir o intervalo com a pontuação mais alta.


## Resumo

* BERT requer mudanças mínimas de arquitetura (camadas extras totalmente conectadas) para aplicações de processamento de linguagem natural em nível de sequência e *token*, como classificação de texto único (por exemplo, análise de sentimento e teste de aceitabilidade linguística), classificação de par de texto ou regressão (por exemplo, inferência de linguagem natural e semelhança textual semântica), marcação de texto (por exemplo, marcação de classe gramatical) e resposta a perguntas.
* Durante o aprendizado supervisionado de uma aplicação *downstream*, os parâmetros das camadas extras são aprendidos do zero, enquanto todos os parâmetros no modelo BERT pré-treinado são ajustados.



## Exercícios

1. Vamos projetar um algoritmo de mecanismo de pesquisa para artigos de notícias. Quando o sistema recebe uma consulta (por exemplo, "indústria de petróleo durante o surto de coronavírus"), ele deve retornar uma lista classificada de artigos de notícias que são mais relevantes para a consulta. Suponha que temos um grande conjunto de artigos de notícias e um grande número de consultas. Para simplificar o problema, suponha que o artigo mais relevante tenha sido rotulado para cada consulta. Como podemos aplicar a amostragem negativa (ver :numref:`subsec_negative-sampling`) e BERT no projeto do algoritmo?
1. Como podemos alavancar o BERT nos modelos de treinamento de idiomas?
1. Podemos alavancar o BERT na tradução automática?

[Discussões](https://discuss.d2l.ai/t/396)
<!--stackedit_data:
eyJoaXN0b3J5IjpbMTk2NzYxNjUwOCw0Mzk0ODgyOTUsMzE0NT
cxNjYzLC0xODAxMTk4NjIsLTM3NTkxMzg2MSwtNjM4MjE5NzMx
XX0=
-->