# Redes Neurais Recorrentes (RNNs)
:label:`sec_rnn`


Em :numref:`sec_language_model` introduzimos modelos de $n$-gramas, onde a probabilidade condicional da palavra $x_t$ no passo de tempo $t$ depende apenas das $n-1$ palavras anteriores.
Se quisermos incorporar o possível efeito de palavras anteriores ao passo de tempo $t-(n-1)$ em $x_t$,
precisamos aumentar $n$.
No entanto, o número de parâmetros do modelo também aumentaria exponencialmente com ele, pois precisamos armazenar $|\mathcal{V}|^n$  números para um conjunto de vocabulário $\mathcal{V}$.
Portanto, em vez de modelar $P(x_t \mid x_{t-1}, \ldots, x_{t-n+1})$, é preferível usar um modelo de variável latente:

$$P(x_t \mid x_{t-1}, \ldots, x_1) \approx P(x_t \mid h_{t-1}),$$

onde $h_{t-1}$ é um *estado oculto* (também conhecido como uma variável oculta) que armazena as informações da sequência até o passo de tempo $t-1$.
Em geral,
o estado oculto em qualquer etapa $t$ pode ser calculado com base na entrada atual $x_ {t}$ e no estado oculto anterior $h_ {t-1}$:

$$h_t = f(x_{t}, h_{t-1}).$$
:eqlabel:`eq_ht_xt`


Para uma função suficientemente poderosa $f$ em :eqref:`eq_ht_xt`, o modelo de variável latente não é uma aproximação. Afinal, $h_t$ pode simplesmente armazenar todos os dados que observou até agora.
No entanto, isso pode tornar a computação e o armazenamento caros.

Lembre-se de que discutimos camadas ocultas com unidades ocultas em :numref:`chap_perceptrons`.
É digno de nota que
camadas ocultas e estados ocultos referem-se a dois conceitos muito diferentes.
Camadas ocultas são, conforme explicado, camadas que ficam ocultas da visualização no caminho da entrada à saída.
Estados ocultos são tecnicamente falando *entradas* para tudo o que fazemos em uma determinada etapa,
e elas só podem ser calculadas observando os dados em etapas de tempo anteriores.

*Redes neurais recorrentes* (RNNs) são redes neurais com estados ocultos. Antes de introduzir o modelo RNN, primeiro revisitamos o modelo MLP introduzido em :numref:`sec_mlp`.

## Redes Neurais sem Estados Ocultos

Vamos dar uma olhada em um MLP com uma única camada oculta.
Deixe a função de ativação da camada oculta ser $\phi$.
Dado um minibatch de exemplos $\mathbf{X} \in \mathbb{R}^{n \times d}$ com tamanho de lote $n$ e $d$ entradas, a saída da camada oculta $\mathbf{H} \in \mathbb{R}^{n \times h}$ é calculada como

$$\mathbf{H} = \phi(\mathbf{X} \mathbf{W}_{xh} + \mathbf{b}_h).$$
:eqlabel:`rnn_h_without_state`

Em :eqref:`rnn_h_without_state`, temos o parâmetro de peso $\mathbf{W}_{xh} \in \mathbb{R}^{d \times h}$, o parâmetro de polarização $\mathbf{b}_h \in \mathbb{R}^{1 \times h}$, e o número de unidades ocultas $h$, para a camada oculta.
Assim, a transmissão (ver :numref:`subsec_broadcasting`) é aplicada durante a soma.
Em seguida, a variável oculta $\mathbf{H}$ é usada como entrada da camada de saída. A camada de saída é fornecida por

$$\mathbf{O} = \mathbf{H} \mathbf{W}_{hq} + \mathbf{b}_q,$$


onde $\mathbf{O} \in \mathbb{R}^{n \times q}$ é a variável de saída, $\mathbf{W}_{hq} \in \mathbb{R}^{h \times q}$ é o parâmetro de peso, e $\mathbf{b}_q \in \mathbb{R}^{1 \times q}$ é o parâmetro de polarização da camada de saída. Se for um problema de classificação, podemos usar $\text{softmax}(\mathbf{O})$ para calcular a distribuição de probabilidade das categorias de saída.

Isso é inteiramente análogo ao problema de regressão que resolvemos anteriormente em :numref:`sec_sequence`, portanto omitimos detalhes.
Basta dizer que podemos escolher pares de rótulo de recurso aleatoriamente e aprender os parâmetros de nossa rede por meio de diferenciação automática e gradiente descendente estocástico.

## Redes Neurais Recorrentes com Estados Ocultos
:label:`subsec_rnn_w_hidden_states`


As coisas são totalmente diferentes quando temos estados ocultos. Vejamos a estrutura com mais detalhes.

Suponha que temos
um minibatch de entradas
$\mathbf{X}_t \in \mathbb{R}^{n \times d}$
no passo de tempo $t$.
Em outras palavras,
para um minibatch de exemplos de sequência $n$,
cada linha de $\mathbf{X}_t$ corresponde a um exemplo no passo de tempo $t$ da sequência.
Em seguida,
denote por $\mathbf{H}_t  \in \mathbb{R}^{n \times h}$ a variável oculta do passo de tempo $t$.
Ao contrário do MLP, aqui salvamos a variável oculta $\mathbf{H}_{t-1}$ da etapa de tempo anterior e introduzimos um novo parâmetro de peso $\mathbf{W}_{hh} \in \mathbb{R}^{h \times h}$ para descrever como usar a variável oculta da etapa de tempo anterior na etapa de tempo atual. Especificamente, o cálculo da variável oculta da etapa de tempo atual é determinado pela entrada da etapa de tempo atual junto com a variável oculta da etapa de tempo anterior:

$$\mathbf{H}_t = \phi(\mathbf{X}_t \mathbf{W}_{xh} + \mathbf{H}_{t-1} \mathbf{W}_{hh}  + \mathbf{b}_h).$$
:eqlabel:`rnn_h_with_state`


Comparado com :eqref:`rnn_h_without_state`,  :eqref:`rnn_h_with_state` adiciona mais um termo $\mathbf{H}_{t-1} \mathbf{W}_{hh}$ e assim
instancia :eqref:`eq_ht_xt`.
A partir da relação entre as variáveis ​​ocultas $\mathbf{H}_t$  e $\mathbf{H}_{t-1}$ de etapas de tempo adjacentes,
sabemos que essas variáveis ​​capturaram e retiveram as informações históricas da sequência até sua etapa de tempo atual, assim como o estado ou a memória da etapa de tempo atual da rede neural. Portanto, essa variável oculta é chamada de *estado oculto*.
Visto que o estado oculto usa a mesma definição da etapa de tempo anterior na etapa de tempo atual, o cálculo de :eqref:`rnn_h_with_state` é *recorrente*. Consequentemente, redes neurais com estados ocultos
com base em cálculos recorrentes são nomeados
*redes neurais recorrentes*.
Camadas que fazem
o cálculo de :eqref:`rnn_h_with_state`
em RNNs
são chamadas de *camadas recorrentes*.


Existem muitas maneiras diferentes de construir RNNs.
RNNs com um estado oculto definido por :eqref:`rnn_h_with_state` são muito comuns.
Para a etapa de tempo $t$,
a saída da camada de saída é semelhante à computação no MLP:

$$\mathbf{O}_t = \mathbf{H}_t \mathbf{W}_{hq} + \mathbf{b}_q.$$

Parâmetros do RNN
incluem os pesos $\mathbf{W}_{xh} \in \mathbb{R}^{d \times h}, \mathbf{W}_{hh} \in \mathbb{R}^{h \times h}$,
e o *bias* $\mathbf{b}_h \in \mathbb{R}^{1 \times h}$
da camada oculta,
junto com os pesos $\mathbf{W}_{hq} \in \mathbb{R}^{h \times q}$
e o *bias*  $\mathbf{b}_q \in \mathbb{R}^{1 \times q}$
da camada de saída.
Vale a pena mencionar que
mesmo em diferentes etapas de tempo,
Os RNNs sempre usam esses parâmetros do modelo.
Portanto, o custo de parametrização de um RNN
não cresce à medida que o número de etapas de tempo aumenta.

:numref:`fig_rnn` ilustra a lógica computacional de uma RNN em três etapas de tempo adjacentes.
A qualquer momento, passo $t$,
o cálculo do estado oculto pode ser tratado como:
i) concatenar a entrada $\mathbf{X}_t$ na etapa de tempo atual $t$ e o estado oculto $\mathbf{H}_{t-1}$ na etapa de tempo anterior $t-1$;
ii) alimentar o resultado da concatenação em uma camada totalmente conectada com a função de ativação $\phi$.
A saída dessa camada totalmente conectada é o estado oculto $\mathbf{H}_t$ do intervalo de tempo atual $t$.
Nesse caso,
os parâmetros do modelo são a concatenação de $\mathbf{W}_{xh}$ e $\mathbf{W}_{hh}$, e um *bias* de $\mathbf{b}_h$, tudo de :eqref:`rnn_h_with_state`.
O estado oculto do passo de tempo atual $t$, $\mathbf{H}_t$, participará do cálculo do estado oculto $\mathbf{H}_{t+1}$ do próximo passo de tempo $t+1$.
Além disso, $\mathbf{H}_t$ também será
alimentado na camada de saída totalmente conectada
para calcular a saída
$\mathbf{O}_t$ do passo de tempo atual $t$.

![Uma RNN com um estado oculto.](../img/rnn.svg)
:label:`fig_rnn`

Acabamos de mencionar que o cálculo de $\mathbf{X}_t \mathbf{W}_{xh} + \mathbf{H}_{t-1} \mathbf{W}_{hh}$ para o estado oculto é equivalente a
multiplicação de matriz de
concatenação de $\mathbf{X}_t$ and $\mathbf{H}_{t-1}$
e
concatenação de $\mathbf{W}_{xh}$ and $\mathbf{W}_{hh}$.
Embora isso possa ser comprovado pela matemática,
a seguir, apenas usamos um trecho de código simples para mostrar isso.
Começando por,
definir as matrizes `X`,` W_xh`, `H` e` W_hh`, cujas formas são (3, 1), (1, 4), (3, 4) e (4, 4), respectivamente.
Multiplicando `X` por` W_xh`, e `H` por` W_hh`, respectivamente e, em seguida, adicionando essas duas multiplicações,
obtemos uma matriz de forma (3, 4).

```{.python .input}
from d2l import mxnet as d2l
from mxnet import np, npx
npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
```

```{.python .input}
#@tab mxnet, pytorch
X, W_xh = d2l.normal(0, 1, (3, 1)), d2l.normal(0, 1, (1, 4))
H, W_hh = d2l.normal(0, 1, (3, 4)), d2l.normal(0, 1, (4, 4))
d2l.matmul(X, W_xh) + d2l.matmul(H, W_hh)
```

```{.python .input}
#@tab tensorflow
X, W_xh = d2l.normal((3, 1), 0, 1), d2l.normal((1, 4), 0, 1)
H, W_hh = d2l.normal((3, 4), 0, 1), d2l.normal((4, 4), 0, 1)
d2l.matmul(X, W_xh) + d2l.matmul(H, W_hh)
```

Agora vamos concatenar as matrizes `X` e` H`
ao longo das colunas (eixo 1),
e as matrizes
`W_xh` e` W_hh` ao longo das linhas (eixo 0).
Essas duas concatenações
resulta em
matrizes de forma (3, 5)
e da forma (5, 4), respectivamente.
Multiplicando essas duas matrizes concatenadas,
obtemos a mesma matriz de saída de forma (3, 4)
como acima.

```{.python .input}
#@tab all
d2l.matmul(d2l.concat((X, H), 1), d2l.concat((W_xh, W_hh), 0))
```

## Modelos de Linguagem em Nível de Caracteres Baseados em RNN

Lembre-se que para a modelagem de linguagem em :numref:`sec_language_model`,
pretendemos prever o próximo token com base em
os tokens atuais e passados,
assim, mudamos a sequência original em um token
como os rótulos.
Bengio et al. propuseram primeiro
usar uma rede neural para modelagem de linguagem :cite:`Bengio.Ducharme.Vincent.ea.2003`.
A seguir, ilustramos como os RNNs podem ser usadas para construir um modelo de linguagem.
Deixe o tamanho do minibatch ser um e a sequência do texto ser "máquina".
Para simplificar o treinamento nas seções subsequentes,
nós tokenizamos o texto em caracteres em vez de palavras
e considere um *modelo de linguagem em nível de caractere*.
 :numref:`fig_rnn_train` demonstra como prever o próximo caractere com base nos caracteres atuais e anteriores através de um RNN para modelagem de linguagem em nível de caractere.

![Um modelo de linguagem de nível de caractere baseado no RNN. As sequências de entrada e rótulo são "machin" e "achine", respectivamente.](../img/rnn-train.svg)
:label:`fig_rnn_train`


Durante o processo de treinamento,
executamos uma operação softmax na saída da camada de saída para cada etapa de tempo e, em seguida, usamos a perda de entropia cruzada para calcular o erro entre a saída do modelo e o rótulo.
Devido ao cálculo recorrente do estado oculto na camada oculta, a saída da etapa de tempo 3 em :numref:`fig_rnn_train`,
$\mathbf{O}_3$, é determinada pela sequência de texto "m", "a" e "c". Como o próximo caractere da sequência nos dados de treinamento é "h", a perda de tempo da etapa 3 dependerá da distribuição de probabilidade do próximo caractere gerado com base na sequência de características "m", "a", "c" e o rótulo "h" desta etapa de tempo.

Na prática, cada token é representado por um vetor $d$-dimensional e usamos um tamanho de batch $n>1$. Portanto, a entrada $ \ mathbf X_t $ no passo de tempo $ t $ será uma matriz $\mathbf X_t$, que é idêntica ao que discutimos em :numref:`subsec_rnn_w_hidden_states`.


## Perplexidade
:label:`subsec_perplexity`


Por último, vamos discutir sobre como medir a qualidade do modelo de linguagem, que será usado para avaliar nossos modelos baseados em RNN nas seções subsequentes.
Uma maneira é verificar o quão surpreendente é o texto.
Um bom modelo de linguagem é capaz de prever com
tokens de alta precisão que veremos a seguir.
Considere as seguintes continuações da frase "Está chovendo", conforme proposto por diferentes modelos de linguagem:

1. "Está chovendo lá fora"
1. "Está chovendo bananeira"
1. "Está chovendo piouw; kcj pwepoiut"

Em termos de qualidade, o exemplo 1 é claramente o melhor. As palavras são sensatas e logicamente coerentes.
Embora possa não refletir com muita precisão qual palavra segue semanticamente ("em São Francisco" e "no inverno" seriam extensões perfeitamente razoáveis), o modelo é capaz de capturar qual tipo de palavra se segue.
O exemplo 2 é consideravelmente pior ao produzir uma extensão sem sentido. No entanto, pelo menos o modelo aprendeu como soletrar palavras e algum grau de correlação entre as palavras. Por último, o exemplo 3 indica um modelo mal treinado que não ajusta os dados adequadamente.


Podemos medir a qualidade do modelo calculando a probabilidade da sequência.
Infelizmente, esse é um número difícil de entender e difícil de comparar.
Afinal, as sequências mais curtas têm muito mais probabilidade de ocorrer do que as mais longas,
portanto, avaliando o modelo na magnum opus de Tolstoy
*Guerra e paz* produzirá inevitavelmente uma probabilidade muito menor do que, digamos, na novela de Saint-Exupéry *O Pequeno Príncipe*. O que falta equivale a uma média.

A teoria da informação é útil aqui.
Definimos entropia, surpresa e entropia cruzada
quando introduzimos a regressão softmax
(:numref:`subsec_info_theory_basics`)
e mais sobre a teoria da informação é discutido no [apêndice online sobre teoria da informação](https://d2l.ai/chapter_apencha-mathematics-for-deep-learning/information-theory.html).
Se quisermos compactar o texto, podemos perguntar sobre
prever o próximo token dado o conjunto atual de tokens.
Um modelo de linguagem melhor deve nos permitir prever o próximo token com mais precisão.
Assim, deve permitir-nos gastar menos bits na compressão da sequência.
Então, podemos medi-lo pela perda de entropia cruzada média
sobre todos os $n$ tokens de uma sequência:

$$\frac{1}{n} \sum_{t=1}^n -\log P(x_t \mid x_{t-1}, \ldots, x_1),$$
:eqlabel:`eq_avg_ce_for_lm`

onde $P$ é dado por um modelo de linguagem e $x_t$ é o token real observado no passo de tempo $t$ da sequência.
Isso torna o desempenho em documentos de comprimentos diferentes comparáveis. Por razões históricas, os cientistas do processamento de linguagem natural preferem usar uma quantidade chamada *perplexidade*. Em poucas palavras, é a exponencial de :eqref:`eq_avg_ce_for_lm`:

$$\exp\left(-\frac{1}{n} \sum_{t=1}^n \log P(x_t \mid x_{t-1}, \ldots, x_1)\right).$$


A perplexidade pode ser melhor entendida como a média harmônica do número de escolhas reais que temos ao decidir qual ficha escolher a seguir. Vejamos alguns casos:

* No melhor cenário, o modelo sempre estima perfeitamente a probabilidade do token de rótulo como 1. Nesse caso, a perplexidade do modelo é 1.
* No pior cenário, o modelo sempre prevê a probabilidade do token de rótulo como 0. Nessa situação, a perplexidade é infinita positiva.
* Na linha de base, o modelo prevê uma distribuição uniforme de todos os tokens disponíveis do vocabulário. Nesse caso, a perplexidade é igual ao número de tokens exclusivos do vocabulário. Na verdade, se armazenássemos a sequência sem nenhuma compressão, seria o melhor que poderíamos fazer para codificá-la. Conseqüentemente, isso fornece um limite superior não trivial que qualquer modelo útil deve superar.

Nas seções a seguir, implementaremos RNNs
para modelos de linguagem em nível de personagem e usaremos perplexidade
para avaliar tais modelos.


## Resumo

* Uma rede neural que usa computação recorrente para estados ocultos é chamada de rede neural recorrente (RNN).
* O estado oculto de uma RNN pode capturar informações históricas da sequência até o intervalo de tempo atual.
* O número de parâmetros do modelo RNN não aumenta com o aumento do número de etapas de tempo.
* Podemos criar modelos de linguagem em nível de caractere usando um RNN.
* Podemos usar a perplexidade para avaliar a qualidade dos modelos de linguagem.

## Exercícios

1. Se usarmos uma RNN para prever o próximo caractere em uma sequência de texto, qual é a dimensão necessária para qualquer saída?
1. Por que as RNNs podem expressar a probabilidade condicional de um token em algum intervalo de tempo com base em todos os tokens anteriores na sequência de texto?
1. O que acontece com o gradiente se você retropropaga através de uma longa sequência?
1. Quais são alguns dos problemas associados ao modelo de linguagem descrito nesta seção?


:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/337)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1050)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1051)
:end_tab:
<!--stackedit_data:
eyJoaXN0b3J5IjpbNTM1ODI1MzU3LDE3ODA4MTczMjUsMjAwMz
Y0MTY1Niw2ODIxMDI5MDAsLTgxMDY0OTA2MV19
-->