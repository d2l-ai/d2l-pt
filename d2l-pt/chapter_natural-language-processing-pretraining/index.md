# Processamento de linguagem natural: Pré-treinamento
:label:`chap_nlp_pretrain`

Os humanos precisam se comunicar.
A partir dessa necessidade básica da condição humana, uma vasta quantidade de texto escrito tem sido gerada diariamente.
Dado o texto rico em mídia social, aplicativos de chat, e-mails, análises de produtos, artigos de notícias, artigos de pesquisa e livros, torna-se vital permitir que os computadores os entendam para oferecer assistência ou tomar decisões com base em linguagens humanas.

O processamento de linguagem natural estuda as interações entre computadores e humanos usando linguagens naturais.
Na prática, é muito comum usar técnicas de processamento de linguagem natural para processar e analisar dados de texto (linguagem natural humana), como modelos de linguagem em :numref:`sec_language_model` e modelos de tradução automática em :numref:`sec_machine_translation`.

Para entender o texto, podemos começar com sua representação,
como tratar cada palavra ou subpalavra como um token de texto individual.
Como veremos neste capítulo,
a representação de cada token pode ser pré-treinada em um grande corpus,
usando word2vec, GloVe ou modelos de incorporação de subpalavra.
Após o pré-treinamento, a representação de cada token pode ser um vetor,
no entanto, permanece o mesmo, independentemente do contexto.
Por exemplo, a representação vetorial de "banco" é a mesma
em ambos
"vá ao banco para depositar algum dinheiro"
e
"vá ao banco para se sentar".
Assim, muitos modelos de pré-treinamento mais recentes adaptam a representação do mesmo token
para contextos diferentes.
Entre eles está o BERT, um modelo muito mais profundo baseado no codificador do transformador.
Neste capítulo, vamos nos concentrar em como pré-treinar tais representações para texto,
como destacado em :numref:`fig_nlp-map-pretrain`.

![As representações de texto pré-treinadas podem ser alimentadas para várias arquiteturas de aprendizado profundo para diferentes aplicativos de processamento de linguagem natural downstream. Este capítulo enfoca o pré-treinamento de representação de texto upstream.](../img/nlp-map-pretrain.svg)
:label:`fig_nlp-map-pretrain`

Conforme mostrado em :numref:`fig_nlp-map-pretrain`,
as representações de texto pré-treinadas podem ser alimentadas para
uma variedade de arquiteturas de aprendizado profundo para diferentes aplicativos de processamento de linguagem natural downstream.
Iremos cobri-los em :numref:`chap_nlp_app`.

```toc
:maxdepth: 2

word2vec
approx-training
word-embedding-dataset
word2vec-pretraining
glove
subword-embedding
similarity-analogy
bert
bert-dataset
bert-pretraining

```

<!--stackedit_data:
eyJoaXN0b3J5IjpbMTc2MzE4NTcwNSwxMTAzODIwNzIxXX0=
-->