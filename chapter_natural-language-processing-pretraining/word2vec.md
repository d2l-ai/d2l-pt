# Incorporação de Palavras (word2vec)
:label:`sec_word2vec`

Uma linguagem natural é um sistema complexo que usamos para expressar significados. Nesse sistema, as palavras são a unidade básica do significado linguístico. Como o próprio nome indica, um vetor de palavras é um vetor usado para representar uma palavra. Também pode ser considerado o vetor de características de uma palavra. A técnica de mapear palavras em vetores de números reais também é conhecida como incorporação de palavras. Nos últimos anos, a incorporação de palavras tornou-se gradualmente um conhecimento básico no processamento de linguagem natural.

## Por que não usar vetores one-hot?

Usamos vetores one-hot para representar palavras (caracteres são palavras) em
:numref:`sec_rnn_scratch`.
Lembre-se de que quando assumimos o número de palavras diferentes em um
dicionário (o tamanho do dicionário) é $N$, cada palavra pode corresponder uma a uma
com inteiros consecutivos de 0 a $N-1$. Esses inteiros que correspondem a
as palavras são chamadas de índices das palavras. Assumimos que o índice de uma palavra
é $i$. A fim de obter a representação vetorial one-hot da palavra, criamos
um vetor de 0s com comprimento de $N$ e defina o elemento $i$ como 1. Desta forma,
cada palavra é representada como um vetor de comprimento $N$ que pode ser usado diretamente por
a rede neural.

Embora os vetores de uma palavra quente sejam fáceis de construir, eles geralmente não são uma boa escolha. Uma das principais razões é que os vetores de uma palavra quente não podem expressar com precisão a semelhança entre palavras diferentes, como a semelhança de cosseno que usamos comumente. Para os vetores $\mathbf{x}, \mathbf{y} \in \mathbb{R}^d$, suas semelhanças de cosseno são os cossenos dos ângulos entre eles:

$$\frac{\mathbf{x}^\top \mathbf{y}}{\|\mathbf{x}\| \|\mathbf{y}\|} \in [-1, 1].$$

Uma vez que a similaridade de cosseno entre os vetores one-hot de quaisquer duas palavras diferentes é 0, é difícil usar o vetor one-hot para representar com precisão a similaridade entre várias palavras diferentes.

[Word2vec](https://code.google.com/archive/p/word2vec/) é uma ferramenta que viemos
para resolver o problema acima. Ele representa cada palavra com um
vetor de comprimento fixo e usa esses vetores para melhor indicar a similaridade e
relações de analogia entre palavras diferentes. A ferramenta Word2vec contém dois
modelos: skip-gram :cite:`Mikolov.Sutskever.Chen.ea.2013` e bolsa contínua de
words (CBOW) :cite:`Mikolov.Chen.Corrado.ea.2013`. Em seguida, vamos dar um
observe os dois modelos e seus métodos de treinamento.

## O Modelo Skip-Gram

O modelo skip-gram assume que uma palavra pode ser usada para gerar as palavras que a cercam em uma sequência de texto. Por exemplo, assumimos que a sequência de texto é "o", "homem", "ama", "seu" e "filho". Usamos "amores" como palavra-alvo central e definimos o tamanho da janela de contexto para 2. Conforme mostrado em :numref:`fig_skip_gram`, dada a palavra-alvo central "amores", o modelo de grama de salto está preocupado com a probabilidade condicional para gerando as palavras de contexto, "o", "homem", "seu" e "filho", que estão a uma distância de no máximo 2 palavras, que é

$$P(\textrm{"the"},\textrm{"man"},\textrm{"his"},\textrm{"son"}\mid\textrm{"loves"}).$$

Assumimos que, dada a palavra-alvo central, as palavras de contexto são geradas independentemente umas das outras. Neste caso, a fórmula acima pode ser reescrita como

$$P(\textrm{"the"}\mid\textrm{"loves"})\cdot P(\textrm{"man"}\mid\textrm{"loves"})\cdot P(\textrm{"his"}\mid\textrm{"loves"})\cdot P(\textrm{"son"}\mid\textrm{"loves"}).$$

![O modelo skip-gram se preocupa com a probabilidade condicional de gerar palavras de contexto para uma determinada palavra-alvo central.](../img/skip-gram.svg)
:label:`fig_skip_gram`

No modelo skip-gram, cada palavra é representada como dois vetores de dimensão $d$, que são usados para calcular a probabilidade condicional. Assumimos que a palavra está indexada como $i$ no dicionário, seu vetor é representado como $\mathbf{v}_i\in\mathbb{R}^d$ quando é a palavra alvo central, e $\mathbf{u}_i\in\mathbb{R}^d$ quando é uma palavra de contexto. Deixe a palavra alvo central $w_c$ e a palavra de contexto $w_o$ serem indexadas como $c$ e $o$ respectivamente no dicionário. A probabilidade condicional de gerar a palavra de contexto para a palavra alvo central fornecida pode ser obtida executando uma operação softmax no produto interno do vetor:

$$P(w_o \mid w_c) = \frac{\text{exp}(\mathbf{u}_o^\top \mathbf{v}_c)}{ \sum_{i \in \mathcal{V}} \text{exp}(\mathbf{u}_i^\top \mathbf{v}_c)},$$

onde o índice de vocabulário definido $\mathcal{V} = \{0, 1, \ldots, |\mathcal{V}|-1\}$. Suponha que uma sequência de texto de comprimento $T$ seja fornecida, onde a palavra no passo de tempo $t$ é denotada como $w^{(t)}$. Suponha que as palavras de contexto sejam geradas independentemente, dadas as palavras centrais. Quando o tamanho da janela de contexto é $m$, a função de verossimilhança do modelo skip-gram é a probabilidade conjunta de gerar todas as palavras de contexto dadas qualquer palavra central

$$ \prod_{t=1}^{T} \prod_{-m \leq j \leq m,\ j \neq 0} P(w^{(t+j)} \mid w^{(t)}),$$

Aqui, qualquer intervalo de tempo menor que 1 ou maior que $T$ pode ser ignorado.

### Treinamento do modelo Skip-Gram

Os parâmetros do modelo skip-gram são o vetor da palavra alvo central e o vetor da palavra de contexto para cada palavra individual. No processo de treinamento, aprenderemos os parâmetros do modelo maximizando a função de verossimilhança, também conhecida como estimativa de máxima verossimilhança. Isso é equivalente a minimizar a seguinte função de perda:

$$ - \sum_{t=1}^{T} \sum_{-m \leq j \leq m,\ j \neq 0} \text{log}\, P(w^{(t+j)} \mid w^{(t)}).$$


Se usarmos o SGD, em cada iteração vamos escolher uma subsequência mais curta por meio de amostragem aleatória para calcular a perda para essa subsequência e, em seguida, calcular o gradiente para atualizar os parâmetros do modelo. A chave do cálculo de gradiente é calcular o gradiente da probabilidade condicional logarítmica para o vetor de palavras central e o vetor de palavras de contexto. Por definição, primeiro temos


$$\log P(w_o \mid w_c) =
\mathbf{u}_o^\top \mathbf{v}_c - \log\left(\sum_{i \in \mathcal{V}} \text{exp}(\mathbf{u}_i^\top \mathbf{v}_c)\right).$$

Por meio da diferenciação, podemos obter o gradiente $\mathbf{v}_c$ da fórmula acima.

$$
\begin{aligned}
\frac{\partial \text{log}\, P(w_o \mid w_c)}{\partial \mathbf{v}_c}
&= \mathbf{u}_o - \frac{\sum_{j \in \mathcal{V}} \exp(\mathbf{u}_j^\top \mathbf{v}_c)\mathbf{u}_j}{\sum_{i \in \mathcal{V}} \exp(\mathbf{u}_i^\top \mathbf{v}_c)}\\
&= \mathbf{u}_o - \sum_{j \in \mathcal{V}} \left(\frac{\text{exp}(\mathbf{u}_j^\top \mathbf{v}_c)}{ \sum_{i \in \mathcal{V}} \text{exp}(\mathbf{u}_i^\top \mathbf{v}_c)}\right) \mathbf{u}_j\\
&= \mathbf{u}_o - \sum_{j \in \mathcal{V}} P(w_j \mid w_c) \mathbf{u}_j.
\end{aligned}
$$

Seu cálculo obtém a probabilidade condicional para todas as palavras no dicionário dada a palavra alvo central $w_c$. Em seguida, usamos o mesmo método para obter os gradientes para outros vetores de palavras.

Após o treinamento, para qualquer palavra do dicionário com índice $i$, vamos obter seus conjuntos de vetores de duas palavras $\mathbf{v}_i$ e $\mathbf{u}_i$. Em aplicações de processamento de linguagem natural, o vetor de palavra-alvo central no modelo skip-gram é geralmente usado como o vetor de representação de uma palavra.


## O modelo do conjunto contínuo de palavras (CBOW)

O modelo de conjunto contínuo de palavras (CBOW) é semelhante ao modelo skip-gram. A maior diferença é que o modelo CBOW assume que a palavra-alvo central é gerada com base nas palavras do contexto antes e depois dela na sequência de texto. Com a mesma sequência de texto "o", "homem", "ama", "seu" e "filho", em que "ama" é a palavra alvo central, dado um tamanho de janela de contexto de 2, o modelo CBOW se preocupa com a probabilidade condicional de gerar a palavra de destino "ama" com base nas palavras de contexto "o", "homem", "seu" e "filho" (conforme mostrado em :numref:`fig_cbow`), como

$$P(\textrm{"loves"}\mid\textrm{"the"},\textrm{"man"},\textrm{"his"},\textrm{"son"}).$$

![O modelo CBOW se preocupa com a probabilidade condicional de gerar a palavra-alvo central a partir de determinadas palavras de contexto.  ](../img/cbow.svg)
:label:`fig_cbow`

Como há várias palavras de contexto no modelo CBOW, calcularemos a média de seus vetores de palavras e usaremos o mesmo método do modelo skip-gram para calcular a probabilidade condicional. Assumimos que $\mathbf{v_i}\in\mathbb{R}^d$ e $\mathbf{u_i}\in\mathbb{R}^d$ são o vetor de palavra de contexto e vetor de palavra-alvo central da palavra com index $i$ no dicionário (observe que os símbolos são opostos aos do modelo skip-gram). Deixe a palavra alvo central $w_c$ ser indexada como $c$, e as palavras de contexto $w_{o_1}, \ldots, w_{o_{2m}}$ sejam indexadas como $o_1, \ldots, o_{2m}$ no dicionário. Assim, a probabilidade condicional de gerar uma palavra-alvo central a partir da palavra de contexto fornecida é

$$P(w_c \mid w_{o_1}, \ldots, w_{o_{2m}}) = \frac{\text{exp}\left(\frac{1}{2m}\mathbf{u}_c^\top (\mathbf{v}_{o_1} + \ldots, + \mathbf{v}_{o_{2m}}) \right)}{ \sum_{i \in \mathcal{V}} \text{exp}\left(\frac{1}{2m}\mathbf{u}_i^\top (\mathbf{v}_{o_1} + \ldots, + \mathbf{v}_{o_{2m}}) \right)}.$$

Para resumir, denote  $\mathcal{W}_o= \{w_{o_1}, \ldots, w_{o_{2m}}\}$, e $\bar{\mathbf{v}}_o = \left(\mathbf{v}_{o_1} + \ldots, + \mathbf{v}_{o_{2m}} \right)/(2m)$. A equação acima pode ser simplificada como

$$P(w_c \mid \mathcal{W}_o) = \frac{\exp\left(\mathbf{u}_c^\top \bar{\mathbf{v}}_o\right)}{\sum_{i \in \mathcal{V}} \exp\left(\mathbf{u}_i^\top \bar{\mathbf{v}}_o\right)}.$$

Dada uma sequência de texto de comprimento $T$, assumimos que a palavra no passo de tempo $t$ é $w^{(t)}$, e o tamanho da janela de contexto é $m$. A função de verossimilhança do modelo CBOW é a probabilidade de gerar qualquer palavra-alvo central a partir das palavras de contexto.

$$ \prod_{t=1}^{T}  P(w^{(t)} \mid  w^{(t-m)}, \ldots, w^{(t-1)}, w^{(t+1)}, \ldots, w^{(t+m)}).$$

### Treinamento de modelo CBOW 

O treinamento do modelo CBOW é bastante semelhante ao treinamento do modelo skip-gram. A estimativa de máxima verossimilhança do modelo CBOW é equivalente a minimizar a função de perda.

$$  -\sum_{t=1}^T  \text{log}\, P(w^{(t)} \mid  w^{(t-m)}, \ldots, w^{(t-1)}, w^{(t+1)}, \ldots, w^{(t+m)}).$$

Note que

$$\log\,P(w_c \mid \mathcal{W}_o) = \mathbf{u}_c^\top \bar{\mathbf{v}}_o - \log\,\left(\sum_{i \in \mathcal{V}} \exp\left(\mathbf{u}_i^\top \bar{\mathbf{v}}_o\right)\right).$$

Por meio da diferenciação, podemos calcular o logaritmo da probabilidade condicional do gradiente de qualquer vetor de palavra de contexto $\mathbf{v}_{o_i}$($i = 1, \ldots, 2m$) na fórmula acima.

$$\frac{\partial \log\, P(w_c \mid \mathcal{W}_o)}{\partial \mathbf{v}_{o_i}} = \frac{1}{2m} \left(\mathbf{u}_c - \sum_{j \in \mathcal{V}} \frac{\exp(\mathbf{u}_j^\top \bar{\mathbf{v}}_o)\mathbf{u}_j}{ \sum_{i \in \mathcal{V}} \text{exp}(\mathbf{u}_i^\top \bar{\mathbf{v}}_o)} \right) = \frac{1}{2m}\left(\mathbf{u}_c - \sum_{j \in \mathcal{V}} P(w_j \mid \mathcal{W}_o) \mathbf{u}_j \right).$$

Em seguida, usamos o mesmo método para obter os gradientes para outros vetores de palavras. Ao contrário do modelo skip-gram, geralmente usamos o vetor de palavras de contexto como o vetor de representação de uma palavra no modelo CBOW.

## Sumário

* Um vetor de palavras é um vetor usado para representar uma palavra. A técnica de mapear palavras em vetores de números reais também é conhecida como incorporação de palavras.
* Word2vec inclui o saco contínuo de palavras (CBOW) e modelos de grama de salto. O modelo skip-gram assume que as palavras de contexto são geradas com base na palavra-alvo central. O modelo CBOW assume que a palavra-alvo central é gerada com base nas palavras do contexto.


## Exercícios

1. Qual é a complexidade computacional de cada gradiente? Se o dicionário contiver um grande volume de palavras, que problemas isso causará?
1. Existem algumas frases fixas no idioma inglês que consistem em várias palavras, como "nova york". Como você pode treinar seus vetores de palavras? Dica: Veja a seção 4 do artigo Word2vec :cite:`Mikolov.Sutskever.Chen.ea.2013`.
1. Use o modelo skip-gram como um exemplo para pensar sobre o design de um modelo word2vec. Qual é a relação entre o produto interno de dois vetores de palavras e a semelhança de cosseno no modelo de grama de salto? Para um par de palavras com significado semântico próximo, por que é provável que sua similaridade de cosseno de vetor de palavras seja alta?



[Discussão](https://discuss.d2l.ai/t/381)
<!--stackedit_data:
eyJoaXN0b3J5IjpbMTgxOTM1NDEzN119
-->