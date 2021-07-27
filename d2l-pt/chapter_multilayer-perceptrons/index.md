# *Perceptrons* Multicamada
:label:`chap_perceptrons`

Neste capítulo, apresentaremos sua primeira rede verdadeiramente *profunda*.
As redes profundas mais simples são chamadas *perceptrons* multicamada,
e eles consistem em várias camadas de neurônios
cada um totalmente conectado àqueles na camada abaixo
(do qual eles recebem contribuições)
e aqueles acima (que eles, por sua vez, influenciam).
Quando treinamos modelos de alta capacidade, corremos o risco de fazer *overfitting*.
Portanto, precisaremos fornecer sua primeira introdução rigorosa
às noções de *overfitting*, *underfitting* e seleção de modelo.
Para ajudá-lo a combater esses problemas,
apresentaremos técnicas de regularização, como redução do peso e abandono escolar.
Também discutiremos questões relacionadas à estabilidade numérica e inicialização de parâmetros
que são essenciais para o treinamento bem-sucedido de redes profundas.
Durante todo o tempo, nosso objetivo é dar a você uma compreensão firme não apenas dos conceitos
mas também da prática de usar redes profundas.
No final deste capítulo,
aplicamos o que apresentamos até agora a um caso real: a previsão do preço da casa.
Nós examinamos questões relacionadas ao desempenho computacional,
escalabilidade e eficiência de nossos modelos para os capítulos subsequentes.

```toc
:maxdepth: 2

mlp
mlp-scratch
mlp-concise
underfit-overfit
weight-decay
dropout
backprop
numerical-stability-and-init
environment
kaggle-house-price
```

<!--stackedit_data:
eyJoaXN0b3J5IjpbNjAzMDE2NzhdfQ==
-->