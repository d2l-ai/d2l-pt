# Teoria da Informação
:label:`sec_information_theory`


O universo está transbordando de informações. A informação fornece uma linguagem comum entre fendas disciplinares: do Soneto de Shakespeare ao artigo de pesquisadores sobre Cornell ArXiv, da impressão de Noite Estrelada de Van Gogh à Sinfonia nº 5 da música de Beethoven, da primeira linguagem de programação Plankalkül à máquina de última geração algoritmos de aprendizagem. Tudo deve seguir as regras da teoria da informação, não importa o formato. Com a teoria da informação, podemos medir e comparar quanta informação está presente em diferentes sinais. Nesta seção, investigaremos os conceitos fundamentais da teoria da informação e as aplicações da teoria da informação no *machine learning*.

Antes de começar, vamos descrever a relação entre o *machine learning* e a teoria da informação. O *machine learning* tem como objetivo extrair sinais interessantes de dados e fazer previsões críticas. Por outro lado, a teoria da informação estuda a codificação, decodificação, transmissão e manipulação de informações. Como resultado, a teoria da informação fornece uma linguagem fundamental para discutir o processamento da informação em sistemas aprendidos por máquina. Por exemplo, muitos aplicativos de *machine learning* usam a perda de entropia cruzada conforme descrito em :numref:`sec_softmax`. Essa perda pode ser derivada diretamente de considerações teóricas da informação.


## Informação


Comecemos com a "alma" da teoria da informação: informação. *Informações* podem ser codificadas em qualquer coisa com uma sequência particular de um ou mais formatos de codificação. Suponha que nos encarreguemos de tentar definir uma noção de informação. Qual poderia ser nosso ponto de partida?

Considere o seguinte experimento mental. Temos um amigo com um baralho de cartas. Eles vão embaralhar o baralho, virar algumas cartas e nos contar declarações sobre as cartas. Tentaremos avaliar o conteúdo informativo de cada declaração.

Primeiro, eles viram um cartão e nos dizem: "Vejo um cartão". Isso não nos fornece nenhuma informação. Já tínhamos a certeza de que era esse o caso, por isso esperamos que a informação seja zero.

Em seguida, eles viram um cartão e dizem: "Vejo um coração". Isso nos fornece algumas informações, mas na realidade existem apenas $4$ diferentes processos possíveis, cada um igualmente provável, portanto, não estamos surpresos com esse resultado. Esperamos que seja qual for a medida de informação, este evento tenha baixo conteúdo informativo.

Em seguida, eles viram uma carta e dizem: "Este é o $3$ de espadas." Esta é mais informações. Na verdade, havia $52$ resultados igualmente prováveis, e nosso amigo nos disse qual era. Esta deve ser uma quantidade média de informações.

Vamos levar isso ao extremo lógico. Suponha que finalmente eles virem todas as cartas do baralho e leiam toda a sequência do baralho embaralhado. Existem $52!$ Pedidos diferentes no baralho, novamente todos igualmente prováveis, portanto, precisamos de muitas informações para saber qual é.

Qualquer noção de informação que desenvolvemos deve estar de acordo com esta intuição. De fato, nas próximas seções aprenderemos como calcular que esses eventos têm 0\text{ bits}$, $2\text{ bits}$, $~5.7\text{ bits}$, e $~225.6\text{ bits}$ de informações respectivamente.


Se lermos esses experimentos mentais, veremos uma ideia natural. Como ponto de partida, ao invés de nos preocuparmos com o conhecimento, podemos partir da ideia de que a informação representa o grau de surpresa ou a possibilidade abstrata do evento. Por exemplo, se quisermos descrever um evento incomum, precisamos de muitas informações. Para um evento comum, podemos não precisar de muitas informações.

Em 1948, Claude E. Shannon publicou *A Mathematical Theory of Communication* :cite:`Shannon.1948` que estabelece a teoria da informação. Em seu artigo, Shannon introduziu o conceito de entropia de informação pela primeira vez. Começaremos nossa jornada aqui.


### Autoinformação


Visto que a informação incorpora a possibilidade abstrata de um evento, como mapeamos a possibilidade para o número de bits? Shannon introduziu a terminologia *bit* como a unidade de informação, que foi originalmente criada por John Tukey. Então, o que é um "bit" e por que o usamos para medir as informações? Historicamente, um transmissor antigo só pode enviar ou receber dois tipos de código: $0$ e $1$. Na verdade, a codificação binária ainda é de uso comum em todos os computadores digitais modernos. Desta forma, qualquer informação é codificada por uma série de $0$ e $1$. E, portanto, uma série de dígitos binários de comprimento $n$ contém $n$ bits de informação.

Agora, suponha que para qualquer série de códigos, cada $0$ ou $1$ ocorra com uma probabilidade de $\frac{1}{2}$. Portanto, um evento $X$ com uma série de códigos de comprimento $n$, ocorre com uma probabilidade de $\frac{1}{2^n}$. Ao mesmo tempo, como mencionamos antes, essa série contém $n$ bits de informação. Então, podemos generalizar para uma função matemática que pode transferir a probabilidade $p$ para o número de bits? Shannon deu a resposta definindo *autoinformação*

$$I(X) = - \log_2 (p),$$

como os *bits* de informação que recebemos para este evento $X$. Observe que sempre usaremos logaritmos de base 2 nesta seção. Para simplificar, o restante desta seção omitirá o subscrito 2 na notação logarítmica, ou seja, $\log(.)$ Sempre se refere a $\log_2(.)$. Por exemplo, o código "0010" tem uma autoinformação

$$I(\text{"0010"}) = - \log (p(\text{"0010"})) = - \log \left( \frac{1}{2^4} \right) = 4 \text{ bits}.$$

Podemos calcular a autoinformação conforme mostrado abaixo. Antes disso, vamos primeiro importar todos os pacotes necessários nesta seção.

```{.python .input}
from mxnet import np
from mxnet.metric import NegativeLogLikelihood
from mxnet.ndarray import nansum
import random

def self_information(p):
    return -np.log2(p)

self_information(1 / 64)
```

```{.python .input}
#@tab pytorch
import torch
from torch.nn import NLLLoss

def nansum(x):
    # Define nansum, as pytorch doesn't offer it inbuilt.
    return x[~torch.isnan(x)].sum()

def self_information(p):
    return -torch.log2(torch.tensor(p)).item()

self_information(1 / 64)
```

```{.python .input}
#@tab tensorflow
import tensorflow as tf

def log2(x):
    return tf.math.log(x) / tf.math.log(2.)

def nansum(x):
    return tf.reduce_sum(tf.where(tf.math.is_nan(
        x), tf.zeros_like(x), x), axis=-1)

def self_information(p):
    return -log2(tf.constant(p)).numpy()

self_information(1 / 64)
```

## Entropia

Como a autoinformação mede apenas a informação de um único evento discreto, precisamos de uma medida mais generalizada para qualquer variável aleatória de distribuição discreta ou contínua.


### Motivação para Entropia


Vamos tentar ser específicos sobre o que queremos. Esta será uma declaração informal do que é conhecido como *axiomas da entropia de Shannon*. Acontece que a seguinte coleção de afirmações de senso comum nos força a uma definição única de informação. Uma versão formal desses axiomas, junto com vários outros, pode ser encontrada em :cite:`Csiszar.2008`.

1. A informação que ganhamos ao observar uma variável aleatória não depende do que chamamos de elementos, ou da presença de elementos adicionais que têm probabilidade zero.
2. A informação que ganhamos ao observar duas variáveis ​​aleatórias não é mais do que a soma das informações que ganhamos ao observá-las separadamente. Se eles forem independentes, então é exatamente a soma.
3. A informação obtida ao observar (quase) certos eventos é (quase) zero.

Embora provar esse fato esteja além do escopo de nosso texto, é importante saber que isso determina de maneira única a forma que a entropia deve assumir. A única ambiguidade que eles permitem está na escolha das unidades fundamentais, que na maioria das vezes é normalizada fazendo a escolha que vimos antes de que a informação fornecida por um único cara ou coroa é um bit.

### Definição

Para qualquer variável aleatória $X$ que segue uma distribuição de probabilidade $P$ com uma função de densidade de probabilidade (pdf) ou uma função de massa de probabilidade (pmf) $p(x)$, medimos a quantidade esperada de informação por meio de *entropia* ( ou *entropia de Shannon*)

$$H(X) = - E_{x \sim P} [\log p(x)].$$
:eqlabel:`eq_ent_def`


Para ser específico, se $X$ é discreto, $H(X) = - \sum_i p_i \log p_i \text{, where } p_i = P(X_i).$$

Caso contrário, se $X$ for contínuo, também nos referimos à entropia como *entropia diferencial*

$$H(X) = - \int_x p(x) \log p(x) \; dx.$$

Podemos definir entropia como a seguir.

```{.python .input}
def entropy(p):
    entropy = - p * np.log2(p)
    # Operator nansum will sum up the non-nan number
    out = nansum(entropy.as_nd_ndarray())
    return out

entropy(np.array([0.1, 0.5, 0.1, 0.3]))
```

```{.python .input}
#@tab pytorch
def entropy(p):
    entropy = - p * torch.log2(p)
    # Operator nansum will sum up the non-nan number
    out = nansum(entropy)
    return out

entropy(torch.tensor([0.1, 0.5, 0.1, 0.3]))
```

```{.python .input}
#@tab tensorflow
def entropy(p):
    return nansum(- p * log2(p))

entropy(tf.constant([0.1, 0.5, 0.1, 0.3]))
```

### Interpretações


Você pode estar curioso: na definição de entropia :eqref:`eq_ent_def`, por que usamos uma expectativa de um logaritmo negativo? Aqui estão algumas intuições.

Primeiro, por que usamos uma função *logaritmo* $\log$? Suponha que $p(x) = f_1(x) f_2(x) \ldots, f_n(x)$, onde cada função componente $f_i(x)$ é independente uma da outra. Isso significa que cada $f_i(x)$ contribui de forma independente para a informação total obtida de $p(x)$. Conforme discutido acima, queremos que a fórmula da entropia seja aditiva sobre as variáveis ​​aleatórias independentes. Felizmente, $\log$ pode naturalmente transformar um produto de distribuições de probabilidade em uma soma dos termos individuais.

Em seguida, por que usamos um $\log$ *negativo*? Intuitivamente, eventos mais frequentes devem conter menos informações do que eventos menos comuns, uma vez que geralmente obtemos mais informações de um caso incomum do que de um caso comum. No entanto, $\log$ está aumentando monotonicamente com as probabilidades e, de fato, negativo para todos os valores em $[0, 1]$. Precisamos construir uma relação monotonicamente decrescente entre a probabilidade dos eventos e sua entropia, que idealmente será sempre positiva (pois nada do que observarmos deve nos forçar a esquecer o que conhecemos). Portanto, adicionamos um sinal negativo na frente da função $\log$.

Por último, de onde vem a função *expectation*? Considere uma variável aleatória $X$. Podemos interpretar a autoinformação ($-\log(p)$) como a quantidade de *surpresa* que temos ao ver um determinado resultado. Na verdade, à medida que a probabilidade se aproxima de zero, a surpresa torna-se infinita. Da mesma forma, podemos interpretar a entropia como a quantidade média de surpresa ao observar $X$. Por exemplo, imagine que um sistema de caça-níqueis emita símbolos estatísticos independentemente ${s_1, \ldots, s_k}$ com probabilidades ${p_1, \ldots, p_k}$ respectivamente. Então, a entropia deste sistema é igual à auto-informação média da observação de cada saída, ou seja,

$$H(S) = \sum_i {p_i \cdot I(s_i)} = - \sum_i {p_i \cdot \log p_i}.$$



### Propriedades da Entropia


Pelos exemplos e interpretações acima, podemos derivar as seguintes propriedades de entropia :eqref:`eq_ent_def`. Aqui, nos referimos a $X$ como um evento e $P$ como a distribuição de probabilidade de $X$.

* A entropia não é negativa, ou seja, $H(X) \geq 0, \forall X$.

* Se $X \sim P$ com uma f.d.p. ou um f.m.p. $p(x)$, e tentamos estimar $P$ por uma nova distribuição de probabilidade $Q$ com uma f.d.p. ou um f.m.p. $q(x)$, então $$H(X) = - E_{x \sim P} [\log p(x)] \leq  - E_{x \sim P} [\log q(x)], \text{ com igualdade se e somente se } P = Q.$$ Alternativamente, $H(X)$ fornece um limite inferior do número médio de bits necessários para codificar símbolos extraídos de $P$.

* Se $X \sim P$, então $x$ transporta a quantidade máxima de informações se espalhar uniformemente entre todos os resultados possíveis. Especificamente, se a distribuição de probabilidade $P$ é discreta com $k$-classe $\{p_1, \ldots, p_k \}$, então $H(X) \leq \log(k), \text{ com igualdade se e somente se  } p_i = \frac{1}{k}, \forall i.$$ Se $P$ é uma variável aleatória contínua, então a história se torna muito mais complicada. No entanto, se impormos adicionalmente que $P$ é suportado em um intervalo finito (com todos os valores entre $0$ e $1$), então $ P $ tem a entropia mais alta se for a distribuição uniforme nesse intervalo.


## Informcação Mútua


Anteriormente, definimos entropia de uma única variável aleatória $X$, e a entropia de um par de variáveis aleatórias $(X, Y)$? Podemos pensar nessas técnicas como uma tentativa de responder ao seguinte tipo de pergunta: "Quais informações estão contidas em $X$ e $Y$ juntos, comparados a cada um separadamente? Existem informações redundantes ou todas são exclusivas?"

Para a discussão a seguir, sempre usamos $(X, Y)$ como um par de variáveis aleatórias que segue uma distribuição de probabilidade conjunta $P$ com uma f.d.p. ou uma f.m.p. $p_{X, Y}(x, y)$, enquanto $X$ e $Y$ seguem a distribuição de probabilidade  $p_X(x)$ e $p_Y(y)$, respectivamente.


### Entropia Conjunta

Semelhante à entropia de uma única variável aleatória :eqref:`eq_ent_def`, definimos a *entropia conjunta* $H(X, Y)$ de um par de variáveis aleatórias $(X, Y)$ como

$$H(X, Y) = −E_{(x, y) \sim P} [\log p_{X, Y}(x, y)]. $$
:eqlabel:`eq_joint_ent_def`

Precisamente, por um lado, se $(X, Y)$ é um par de variáveis aleatórias discretas, então

$$H(X, Y) = - \sum_{x} \sum_{y} p_{X, Y}(x, y) \log p_{X, Y}(x, y).$$

Por outro lado, se $(X, Y)$ é um par de variáveis aleatórias contínuas, então definimos a *entropia conjunta diferencial* como

$$H(X, Y) = - \int_{x, y} p_{X, Y}(x, y) \ \log p_{X, Y}(x, y) \;dx \;dy.$$

Podemos pensar em :eqref:`eq_joint_ent_def` como nos dizendo a aleatoriedade total no par de variáveis aleatórias. Como um par de extremos, se $X = Y$ são duas variáveis aleatórias idênticas, então as informações do par são exatamente as informações de uma e temos $H(X, Y) = H(X) = H(Y)$. No outro extremo, se $X$ e $Y$ são independentes, então $H(X, Y) = H(X) + H(Y)$. Na verdade, sempre teremos que a informação contida em um par de variáveis aleatórias não é menor que a entropia de qualquer uma das variáveis aleatórias e não mais que a soma de ambas.

$$
H(X), H(Y) \le H(X, Y) \le H(X) + H(Y).
$$

Vamos implementar a entropia conjunta do zero.

```{.python .input}
def joint_entropy(p_xy):
    joint_ent = -p_xy * np.log2(p_xy)
    # Operator nansum will sum up the non-nan number
    out = nansum(joint_ent.as_nd_ndarray())
    return out

joint_entropy(np.array([[0.1, 0.5], [0.1, 0.3]]))
```

```{.python .input}
#@tab pytorch
def joint_entropy(p_xy):
    joint_ent = -p_xy * torch.log2(p_xy)
    # nansum will sum up the non-nan number
    out = nansum(joint_ent)
    return out

joint_entropy(torch.tensor([[0.1, 0.5], [0.1, 0.3]]))
```

```{.python .input}
#@tab tensorflow
def joint_entropy(p_xy):
    joint_ent = -p_xy * log2(p_xy)
    # nansum will sum up the non-nan number
    out = nansum(joint_ent)
    return out

joint_entropy(tf.constant([[0.1, 0.5], [0.1, 0.3]]))
```

Observe que este é o mesmo *código* de antes, mas agora o interpretamos de maneira diferente, como trabalhando na distribuição conjunta das duas variáveis aleatórias.


### Entropia Condicional


A entropia conjunta definida acima da quantidade de informações contidas em um par de variáveis ​​aleatórias. Isso é útil, mas muitas vezes não é o que nos preocupa. Considere a configuração de aprendizado de máquina. Tomemos $X$ como a variável aleatória (ou vetor de variáveis ​​aleatórias) que descreve os valores de pixel de uma imagem e $Y$ como a variável aleatória que é o rótulo da classe. $X$ deve conter informações substanciais - uma imagem natural é algo complexo. No entanto, as informações contidas em $Y$ uma vez que a imagem foi exibida devem ser baixas. Na verdade, a imagem de um dígito já deve conter as informações sobre qual dígito ele é, a menos que seja ilegível. Assim, para continuar a estender nosso vocabulário da teoria da informação, precisamos ser capazes de raciocinar sobre o conteúdo da informação em uma variável aleatória condicional a outra.

Na teoria da probabilidade, vimos a definição da *probabilidade condicional* para medir a relação entre as variáveis. Agora queremos definir analogamente a *entropia condicional* $H(Y \mid X)$.  Podemos escrever isso como

$$ H(Y \mid X) = - E_{(x, y) \sim P} [\log p(y \mid x)],$$
:eqlabel:`eq_cond_ent_def`

onde $p(y \mid x) = \frac{p_{X, Y}(x, y)}{p_X(x)}$ é a probabilidade condicional. Especificamente, se $(X, Y)$ é um par de variáveis aleatórias discretas, então

$$H(Y \mid X) = - \sum_{x} \sum_{y} p(x, y) \log p(y \mid x).$$

Se $ (X, Y) $ é um par de variáveis aleatórias contínuas, então a *entropia condicional diferencial* é similarmente definida como

$$H(Y \mid X) = - \int_x \int_y p(x, y) \ \log p(y \mid x) \;dx \;dy.$$


Agora é natural perguntar, como a *entropia condicional* $H(Y \mid X)$ se relaciona com a entropia $H(X)$ e a entropia conjunta $H(X, Y)$? Usando as definições acima, podemos expressar isso de forma clara:

$$H(Y \mid X) = H(X, Y) - H(X).$$


Isso tem uma interpretação intuitiva: a informação em $Y$ dada $X$ ($H(Y \mid X)$) é a mesma que a informação em $X$ e $Y$ juntos ($H(X, Y)$) menos as informações já contidas em $X$. Isso nos dá as informações em $Y$, que também não são representadas em $X$.

Agora, vamos implementar a entropia condicional :eqref:`eq_cond_ent_def` do zero.

```{.python .input}
def conditional_entropy(p_xy, p_x):
    p_y_given_x = p_xy/p_x
    cond_ent = -p_xy * np.log2(p_y_given_x)
    # Operator nansum will sum up the non-nan number
    out = nansum(cond_ent.as_nd_ndarray())
    return out

conditional_entropy(np.array([[0.1, 0.5], [0.2, 0.3]]), np.array([0.2, 0.8]))
```

```{.python .input}
#@tab pytorch
def conditional_entropy(p_xy, p_x):
    p_y_given_x = p_xy/p_x
    cond_ent = -p_xy * torch.log2(p_y_given_x)
    # nansum will sum up the non-nan number
    out = nansum(cond_ent)
    return out

conditional_entropy(torch.tensor([[0.1, 0.5], [0.2, 0.3]]),
                    torch.tensor([0.2, 0.8]))
```

```{.python .input}
#@tab tensorflow
def conditional_entropy(p_xy, p_x):
    p_y_given_x = p_xy/p_x
    cond_ent = -p_xy * log2(p_y_given_x)
    # nansum will sum up the non-nan number
    out = nansum(cond_ent)
    return out

conditional_entropy(tf.constant([[0.1, 0.5], [0.2, 0.3]]),
                    tf.constant([0.2, 0.8]))
```

### Informação mútua


Dada a configuração anterior de variáveis ​​aleatórias $(X, Y)$, você pode se perguntar: "Agora que sabemos quanta informação está contida em $Y$, mas não em $X$, podemos igualmente perguntar quanta informação é compartilhada entre $X$ e $Y$? " A resposta será a * informação mútua * de $(X, Y)$, que escreveremos como $I(X, Y)$.

Em vez de mergulhar direto na definição formal, vamos praticar nossa intuição tentando primeiro derivar uma expressão para a informação mútua inteiramente baseada em termos que construímos antes. Queremos encontrar a informação compartilhada entre duas variáveis ​​aleatórias. Uma maneira de tentar fazer isso é começar com todas as informações contidas em $X$ e $Y$ juntas e, em seguida, retirar as partes que não são compartilhadas. As informações contidas em $X$ e $Y$ juntas são escritas como $H(X, Y)$. Queremos subtrair disso as informações contidas em $X$, mas não em $Y$, e as informações contidas em $Y$, mas não em $X$. Como vimos na seção anterior, isso é dado por $H(X \mid Y)$ e $H(Y \mid X)$ respectivamente. Assim, temos que a informação mútua deve ser

$$
I(X, Y) = H(X, Y) - H(Y \mid X) − H(X \mid Y).
$$

Na verdade, esta é uma definição válida para a informação mútua. Se expandirmos as definições desses termos e combiná-los, um pouco de álgebra mostra que isso é o mesmo que

$$I(X, Y) = E_{x} E_{y} \left\{ p_{X, Y}(x, y) \log\frac{p_{X, Y}(x, y)}{p_X(x) p_Y(y)} \right\}. $$
:eqlabel:`eq_mut_ent_def`


Podemos resumir todas essas relações na imagem :numref:`fig_mutual_information`. É um excelente teste de intuição ver por que as seguintes afirmações também são equivalentes a $I(X, Y)$.

* $H(X) − H(X \mid Y)$
* $H(Y) − H(Y \mid X)$
* $H(X) + H(Y) − H(X, Y)$

![Relação da informação mútua com entropia conjunta e entropia condicional.](../img/mutual-information.svg)
:label:`fig_mutual_information`



De muitas maneiras, podemos pensar na informação mútua :eqref:`eq_mut_ent_def` como uma extensão de princípio do coeficiente de correlação que vimos em :numref:`sec_random_variables`. Isso nos permite pedir não apenas relações lineares entre variáveis, mas também o máximo de informações compartilhadas entre as duas variáveis aleatórias de qualquer tipo.

Agora, vamos implementar informações mútuas do zero.

```{.python .input}
def mutual_information(p_xy, p_x, p_y):
    p = p_xy / (p_x * p_y)
    mutual = p_xy * np.log2(p)
    # Operator nansum will sum up the non-nan number
    out = nansum(mutual.as_nd_ndarray())
    return out

mutual_information(np.array([[0.1, 0.5], [0.1, 0.3]]),
                   np.array([0.2, 0.8]), np.array([[0.75, 0.25]]))
```

```{.python .input}
#@tab pytorch
def mutual_information(p_xy, p_x, p_y):
    p = p_xy / (p_x * p_y)
    mutual = p_xy * torch.log2(p)
    # Operator nansum will sum up the non-nan number
    out = nansum(mutual)
    return out

mutual_information(torch.tensor([[0.1, 0.5], [0.1, 0.3]]),
                   torch.tensor([0.2, 0.8]), torch.tensor([[0.75, 0.25]]))
```

```{.python .input}
#@tab tensorflow
def mutual_information(p_xy, p_x, p_y):
    p = p_xy / (p_x * p_y)
    mutual = p_xy * log2(p)
    # Operator nansum will sum up the non-nan number
    out = nansum(mutual)
    return out

mutual_information(tf.constant([[0.1, 0.5], [0.1, 0.3]]),
                   tf.constant([0.2, 0.8]), tf.constant([[0.75, 0.25]]))
```

### Propriedades da Informação Mútua


Em vez de memorizar a definição de informação mútua :eqref:`eq_mut_ent_def`, você só precisa ter em mente suas propriedades notáveis:

* A informação mútua é simétrica, ou seja, $I(X, Y) = I(Y, X)$.
* As informações mútuas não são negativas, ou seja, $I(X, Y) \geq 0$.
* $I(X, Y) = 0$ se e somente se $X$ e $Y$ são independentes. Por exemplo, se $X$ e $Y$ são independentes, saber $Y$ não fornece nenhuma informação sobre $X$ e vice-versa, portanto, suas informações mútuas são zero.
* Alternativamente, se $X$ é uma função invertível de $Y$, então $Y$ e $X$ compartilham todas as informações e $$I(X, Y) = H(Y) = H(X).$$

### Informações Mútuas Pontuais

Quando trabalhamos com entropia no início deste capítulo, fomos capazes de fornecer uma interpretação de $-\log(p_X(x))$ como *surpresos* com o resultado particular. Podemos dar uma interpretação semelhante ao termo logarítmico nas informações mútuas, que muitas vezes é referido como as *informações mútuas pontuais*:

$$\mathrm{pmi}(x, y) = \log\frac{p_{X, Y}(x, y)}{p_X(x) p_Y(y)}.$$
:eqlabel:`eq_pmi_def`


Podemos pensar em :eqref:`eq_pmi_def` medindo o quanto mais ou menos provável a combinação específica de resultados $x$ e $y$ são comparados com o que esperaríamos para resultados aleatórios independentes. Se for grande e positivo, esses dois resultados específicos ocorrem com muito mais frequência do que em comparação com o acaso (*nota*: o denominador é $p_X(x) p_Y(y)$ que é a probabilidade de os dois resultados serem independente), ao passo que, se for grande e negativo, representa os dois resultados que acontecem muito menos do que esperaríamos ao acaso.

Isso nos permite interpretar as informações mútuas :eqref:`eq_mut_ent_def` como a quantidade média que ficamos surpresos ao ver dois resultados ocorrendo juntos em comparação com o que esperaríamos se fossem independentes.

### Aplicações de Informação Mútua


As informações mútuas podem ser um pouco abstratas em sua definição pura. Como isso se relaciona ao *machine learning*? No processamento de linguagem natural, um dos problemas mais difíceis é a *resolução da ambigüidade*, ou a questão do significado de uma palavra não ser claro no contexto. Por exemplo, recentemente uma manchete no noticiário relatou que "Amazon está pegando fogo". Você pode se perguntar se a empresa Amazon tem um prédio em chamas ou se a floresta amazônica está em chamas.

Nesse caso, informações mútuas podem nos ajudar a resolver essa ambiguidade. Primeiro encontramos o grupo de palavras em que cada uma tem uma informação mútua relativamente grande com a empresa Amazon, como e-commerce, tecnologia e online. Em segundo lugar, encontramos outro grupo de palavras em que cada uma tem uma informação mútua relativamente grande com a floresta amazônica, como chuva, floresta e tropical. Quando precisamos desambiguar "Amazon", podemos comparar qual grupo tem mais ocorrência no contexto da palavra Amazon. Nesse caso, o artigo descreveria a floresta e deixaria o contexto claro.


## Divergência de Kullback–Leibler


Conforme discutimos em :numref:`sec_linear-algebra`, podemos usar normas para medir a distância entre dois pontos no espaço de qualquer dimensionalidade. Gostaríamos de poder fazer uma tarefa semelhante com distribuições de probabilidade. Há muitas maneiras de fazer isso, mas a teoria da informação oferece uma das melhores. Agora exploramos a *divergência de Kullback-Leibler (KL)*, que fornece uma maneira de medir se duas distribuições estão próximas ou não.


### Definição

Dada uma variável aleatória $X$ que segue a distribuição de probabilidade $P$ com uma f.d.p. ou um f.m.p. $p(x)$, e estimamos $P$ por outra distribuição de probabilidade $Q$ com uma f.d.p. ou uma f.m.o. $q(x)$. Então a divergência *Kullback-Leibler(KL)* (ou *entropia relativa*) entre $P$ e $Q$ é

$$D_{\mathrm{KL}}(P\|Q) = E_{x \sim P} \left[ \log \frac{p(x)}{q(x)} \right].$$
:eqlabel:`eq_kl_def`


Tal como acontece com a informação mútua pontua l:eqref:`eq_pmi_def`, podemos fornecer novamente uma interpretação do termo logarítmico: $-\log \frac{q(x)}{p(x)} = -\log(q(x)) - (-\log(p(x)))$ será grande e positivo se virmos $x$ com muito mais frequência abaixo de $P$ do que esperaríamos para $Q$, e grande e negativo se virmos o resultado muito menor do que o esperado. Dessa forma, podemos interpretá-lo como nossa surpresa *relativa* ao observar o resultado, em comparação com o quão surpresos ficaríamos observando-o a partir de nossa distribuição de referência.

Vamos implementar a divergência KL do zero.

```{.python .input}
def kl_divergence(p, q):
    kl = p * np.log2(p / q)
    out = nansum(kl.as_nd_ndarray())
    return out.abs().asscalar()
```

```{.python .input}
#@tab pytorch
def kl_divergence(p, q):
    kl = p * torch.log2(p / q)
    out = nansum(kl)
    return out.abs().item()
```

```{.python .input}
#@tab tensorflow
def kl_divergence(p, q):
    kl = p * log2(p / q)
    out = nansum(kl)
    return tf.abs(out).numpy()
```

### Propriedades da Divergência de KL


Vamos dar uma olhada em algumas propriedades da divergência KL :eqref:`eq_kl_def`.

* A divergência KL é não simétrica, ou seja, existem $P, Q$ tais que $$D_{\mathrm{KL}}(P\|Q) \neq D_{\mathrm{KL}}(Q\|P).$$
* A divergência KL não é negativa, ou seja,$$D_{\mathrm{KL}}(P\|Q) \geq 0.$$ Observe que a igualdade é válida apenas quando $P = Q$.
* Se existe um $x$ tal que $p(x) > 0$ e $q(x) = 0$, então $D_{\mathrm{KL}}(P\|Q) = \infty$.
* Existe uma relação estreita entre divergência KL e informação mútua. Além da relação mostrada em :numref:`fig_mutual_information`, $I(X, Y)$ também é numericamente equivalente com os seguintes termos:
    1. $D_{\mathrm{KL}}(P(X, Y)  \ \| \ P(X)P(Y))$;
    1. $E_Y \{ D_{\mathrm{KL}}(P(X \mid Y) \ \| \ P(X)) \}$;
    1. $E_X \{ D_{\mathrm{KL}}(P(Y \mid X) \ \| \ P(Y)) \}$.

Para o primeiro termo, interpretamos a informação mútua como a divergência KL entre $P(X, Y)$ e o produto de $P(X)$ e $P(Y)$ e, portanto, é uma medida de quão diferente é a junta distribuição é da distribuição se eles fossem independentes. Para o segundo termo, a informação mútua nos diz a redução média na incerteza sobre $Y$ que resulta do aprendizado do valor da distribuição de $X$. Semelhante ao terceiro mandato.


### Exemplo


Vejamos um exemplo de brinquedo para ver a não simetria explicitamente.

Primeiro, vamos gerar e classificar três tensores de comprimento $10.000$: um tensor objetivo $p$ que segue uma distribuição normal $N(0, 1)$,  e dois tensores candidatos $q_1$ e $q_2$ que seguem distribuições normais $N(-1, 1)$ e $N(1, 1)$ respectivamente.

```{.python .input}
random.seed(1)

nd_len = 10000
p = np.random.normal(loc=0, scale=1, size=(nd_len, ))
q1 = np.random.normal(loc=-1, scale=1, size=(nd_len, ))
q2 = np.random.normal(loc=1, scale=1, size=(nd_len, ))

p = np.array(sorted(p.asnumpy()))
q1 = np.array(sorted(q1.asnumpy()))
q2 = np.array(sorted(q2.asnumpy()))
```

```{.python .input}
#@tab pytorch
torch.manual_seed(1)

tensor_len = 10000
p = torch.normal(0, 1, (tensor_len, ))
q1 = torch.normal(-1, 1, (tensor_len, ))
q2 = torch.normal(1, 1, (tensor_len, ))

p = torch.sort(p)[0]
q1 = torch.sort(q1)[0]
q2 = torch.sort(q2)[0]
```

```{.python .input}
#@tab tensorflow
tensor_len = 10000
p = tf.random.normal((tensor_len, ), 0, 1)
q1 = tf.random.normal((tensor_len, ), -1, 1)
q2 = tf.random.normal((tensor_len, ), 1, 1)

p = tf.sort(p)
q1 = tf.sort(q1)
q2 = tf.sort(q2)
```

Como $q_1$ e $q_2$ são simétricos em relação ao eixo y (ou seja, $x=0$), esperamos um valor semelhante de divergência KL entre $D_{\mathrm{KL}}(p\|q_1)$ e $D_{\mathrm{KL}}(p\|q_2)$. Como você pode ver abaixo, há apenas menos de 3% de desconto entre $D_{\mathrm{KL}}(p\|q_1)$ e $D_{\mathrm{KL}}(p\|q_2)$.

```{.python .input}
#@tab all
kl_pq1 = kl_divergence(p, q1)
kl_pq2 = kl_divergence(p, q2)
similar_percentage = abs(kl_pq1 - kl_pq2) / ((kl_pq1 + kl_pq2) / 2) * 100

kl_pq1, kl_pq2, similar_percentage
```

Em contraste, você pode descobrir que $D_{\mathrm{KL}}(q_2 \|p)$ e $D_{\mathrm{KL}}(p \| q_2)$ estão muito desviados, com cerca de 40% de desconto como mostrado abaixo.

```{.python .input}
#@tab all
kl_q2p = kl_divergence(q2, p)
differ_percentage = abs(kl_q2p - kl_pq2) / ((kl_q2p + kl_pq2) / 2) * 100

kl_q2p, differ_percentage
```

## Entropia Cruzada


Se você está curioso sobre as aplicações da teoria da informação no aprendizado profundo, aqui está um exemplo rápido. Definimos a distribuição verdadeira $P$ com distribuição de probabilidade $p(x)$, e a distribuição estimada $Q$ com distribuição de probabilidade $q(x)$, e as usaremos no restante desta seção.

Digamos que precisamos resolver um problema de classificação binária com base em $n$ exemplos de dados dados {$x_1, \ldots, x_n$}. Suponha que codifiquemos $1$ e $0$ como o rótulo de classe positivo e negativo $y_i$ respectivamente, e nossa rede neural seja parametrizada por $\theta$. Se quisermos encontrar o melhor $\theta$ de forma que $\hat{y}_i= p_{\theta}(y_i \mid x_i)$, é natural aplicar a abordagem de log-*likelihood* máxima como foi visto em :numref:`sec_maximum_likelihood`. Para ser específico, para rótulos verdadeiros $y_i$ e previsões $\hat{y}_i= p_{\theta}(y_i \mid x_i)$, a probabilidade de ser classificado como positivo é $\pi_i= p_{\theta}(y_i = 1 \mid x_i)$. Portanto, a função de log-*likelihood* seria

$$
\begin{aligned}
l(\theta) &= \log L(\theta) \\
  &= \log \prod_{i=1}^n \pi_i^{y_i} (1 - \pi_i)^{1 - y_i} \\
  &= \sum_{i=1}^n y_i \log(\pi_i) + (1 - y_i) \log (1 - \pi_i). \\
\end{aligned}
$$


Maximizar a função log-*likelihood* $l(\theta)$é idêntico a minimizar $- l(\theta)$ e, portanto, podemos encontrar o melhor $\theta$  aqui. Para generalizar a perda acima para quaisquer distribuições, também chamamos $- l(\theta)$ a *perda de entropia cruzada*$\mathrm{CE}(y, \hat{y})$,  onde $y$ segue a verdadeira distribuição $P$ e $\hat{y}$  segue a distribuição estimada $Q$.

Tudo isso foi derivado trabalhando do ponto de vista de máxima verossimilhança. No entanto, se olharmos com atenção, podemos ver que termos como $\log(\pi_i)$ entraram em nosso cálculo, o que é uma indicação sólida de que podemos entender a expressão de um ponto de vista teórico da informação.


### Definição formal

Como a divergência KL, para uma variável aleatória $X$, também podemos medir a divergência entre a distribuição de estimativa $Q$ e a distribuição verdadeira $P$ via *entropia cruzada*,

$$\mathrm{CE}(P, Q) = - E_{x \sim P} [\log(q(x))].$$
:eqlabel:`eq_ce_def`

Ao usar as propriedades da entropia discutidas acima, também podemos interpretá-la como a soma da entropia $H(P)$ e a divergência KL entre $P$ e $Q$, ou seja,

$$\mathrm{CE} (P, Q) = H(P) + D_{\mathrm{KL}}(P\|Q).$$


Podemos implementar a perda de entropia cruzada conforme abaixo.

```{.python .input}
def cross_entropy(y_hat, y):
    ce = -np.log(y_hat[range(len(y_hat)), y])
    return ce.mean()
```

```{.python .input}
#@tab pytorch
def cross_entropy(y_hat, y):
    ce = -torch.log(y_hat[range(len(y_hat)), y])
    return ce.mean()
```

```{.python .input}
#@tab tensorflow
def cross_entropy(y_hat, y):
    ce = -tf.math.log(y_hat[:, :len(y)])
    return tf.reduce_mean(ce)
```

Agora defina dois tensores para os rótulos e previsões e calcule a perda de entropia cruzada deles.

```{.python .input}
labels = np.array([0, 2])
preds = np.array([[0.3, 0.6, 0.1], [0.2, 0.3, 0.5]])

cross_entropy(preds, labels)
```

```{.python .input}
#@tab pytorch
labels = torch.tensor([0, 2])
preds = torch.tensor([[0.3, 0.6, 0.1], [0.2, 0.3, 0.5]])

cross_entropy(preds, labels)
```

```{.python .input}
#@tab tensorflow
labels = tf.constant([0, 2])
preds = tf.constant([[0.3, 0.6, 0.1], [0.2, 0.3, 0.5]])

cross_entropy(preds, labels)
```

### Propriedades


Como mencionado no início desta seção, entropia cruzada :eqref:`eq_ce_def` pode ser usada para definir uma função de perda no problema de otimização. Acontece que os seguintes são equivalentes:

1. Maximizar a probabilidade preditiva de $Q$ para a distribuição $P$, (ou seja, $E_{x
\sim P} [\log (q(x))]$);
1. Minimizar a entropia cruzada $\mathrm{CE} (P, Q)$;
1. Minimizar a divergência KL $D_{\mathrm{KL}}(P\|Q)$.

A definição de entropia cruzada prova indiretamente a relação equivalente entre o objetivo 2 e o objetivo 3, desde que a entropia dos dados verdadeiros $H(P)$ seja constante.


### Entropia Cruzada como Função Objetiva da Classificação Multiclasse


Se mergulharmos profundamente na função objetivo de classificação com perda de entropia cruzada $\mathrm{CE}$, descobriremos que minimizar $\mathrm{CE}$ é equivalente a maximizar a função log-*likelihood* $L$.

Para começar, suponha que recebamos um conjunto de dados com $n$ exemplos e ele possa ser classificado em $k$-classes. Para cada exemplo de dados $i$, representamos qualquer rótulo $k$-class $\mathbf{y}_i = (y_{i1}, \ldots, y_{ik})$ por *codificação one-hot*. Para ser mais específico, se o exemplo $i$ pertence à classe $j$, definimos a $j$-ésima entrada como $1$ e todos os outros componentes como $0$, ou seja,

$$ y_{ij} = \begin{cases}1 & j \in J; \\ 0 &\text{otherwise.}\end{cases}$$

Por exemplo, se um problema de classificação multiclasse contém três classes $A$, $B$ e $C$, os rótulos $\mathbf{y}_i$ podem ser codificados em {$A: (1, 0, 0); B: (0, 1, 0); C: (0, 0, 1)$}.



Suponha que nossa rede neural seja parametrizada por $\theta$. Para vetores de rótulos verdadeiros $\mathbf{y}_i$ e previsões $$\hat{\mathbf{y}}_i= p_{\theta}(\mathbf{y}_i \mid \mathbf{x}_i) = \sum_{j=1}^k y_{ij} p_{\theta} (y_{ij}  \mid  \mathbf{x}_i).$$ 

Portanto, a *perda de entropia cruzada* seria

$$
\mathrm{CE}(\mathbf{y}, \hat{\mathbf{y}}) = - \sum_{i=1}^n \mathbf{y}_i \log \hat{\mathbf{y}}_i
 = - \sum_{i=1}^n \sum_{j=1}^k y_{ij} \log{p_{\theta} (y_{ij}  \mid  \mathbf{x}_i)}.\\
$$

Por outro lado, também podemos abordar o problema por meio da estimativa de máxima verossimilhança. Para começar, vamos apresentar rapidamente uma distribuição multinoulli de classe $k$. É uma extensão da distribuição Bernoulli de classe binária para multiclasse. Se uma variável aleatória $\mathbf{z} = (z_{1}, \ldots, z_{k})$ segue uma distribuição $k$-class *multinoulli* com probabilidades $\mathbf{p} =$ ($p_{1}, \ldots, p_{k}$), ou seja, $$p(\mathbf{z}) = p(z_1, \ldots, z_k) = \mathrm{Multi} (p_1, \ldots, p_k), \text{ onde } \sum_{i=1}^k p_i = 1,$$ então a função de massa de probabilidade conjunta (f.m.p.) de $\mathbf{z}$  é
$$\mathbf{p}^\mathbf{z} = \prod_{j=1}^k p_{j}^{z_{j}}.$$


Pode-se ver que o rótulo de cada exemplo de dados, $\mathbf{y}_i$, segue uma distribuição multinoulli de $k$-classes com probabilidades $\boldsymbol{\pi} =$ ($\pi_{1}, \ldots, \pi_{k}$). Portanto, a junta f.m.p. de cada exemplo de dados $\mathbf{y}_i$ é $\mathbf{\pi}^{\mathbf{y}_i} = \prod_{j=1}^k \pi_{j}^{y_{ij}}.$
Portanto, a função de log-*likelihood* seria

$$
\begin{aligned}
l(\theta)
 = \log L(\theta)
 = \log \prod_{i=1}^n \boldsymbol{\pi}^{\mathbf{y}_i}
 = \log \prod_{i=1}^n \prod_{j=1}^k \pi_{j}^{y_{ij}}
 = \sum_{i=1}^n \sum_{j=1}^k y_{ij} \log{\pi_{j}}.\\
\end{aligned}
$$


Já que na estimativa de máxima verossimilhança, maximizamos a função objetivo $l(\theta)$ tendo $\pi_{j} = p_{\theta} (y_{ij}  \mid  \mathbf{x}_i)$. Portanto, para qualquer classificação multiclasse, maximizar a função log-verossimilhança $l(\theta)$ acima é equivalente a minimizar a perda de CE $\mathrm{CE}(y, \hat{y})$.


Para testar a prova acima, vamos aplicar a medida integrada `NegativeLogLikelihood`. Usando os mesmos `rótulos` e` preds` do exemplo anterior, obteremos a mesma perda numérica do exemplo anterior até a casa decimal 5.

```{.python .input}
nll_loss = NegativeLogLikelihood()
nll_loss.update(labels.as_nd_ndarray(), preds.as_nd_ndarray())
nll_loss.get()
```

```{.python .input}
#@tab pytorch
# Implementation of CrossEntropy loss in pytorch combines nn.LogSoftmax() and
# nn.NLLLoss()
nll_loss = NLLLoss()
loss = nll_loss(torch.log(preds), labels)
loss
```

```{.python .input}
#@tab tensorflow
def nll_loss(y_hat, y):
    # Convert labels to binary class matrix.
    y = tf.keras.utils.to_categorical(y, num_classes=3)
    # Since tf.keras.losses.binary_crossentropy returns the mean
    # over the last axis, we calculate the sum here.
    return tf.reduce_sum(
        tf.keras.losses.binary_crossentropy(y, y_hat, from_logits=True))

loss = nll_loss(tf.math.log(preds), labels)
loss
```

## Resumo

* A teoria da informação é um campo de estudo sobre codificação, decodificação, transmissão e manipulação de informações.
* Entropia é a unidade para medir quanta informação é apresentada em diferentes sinais.
* A divergência KL também pode medir a divergência entre duas distribuições.
* A entropia cruzada pode ser vista como uma função objetivo da classificação multiclasse. Minimizar a perda de entropia cruzada é equivalente a maximizar a função de log-*likelihood*.


## Exercícios

1. Verifique se os exemplos de cartas da primeira seção realmente possuem a entropia reivindicada.
1. Mostre que a divergência KL $D(p\|q)$ é não negativa para todas as distribuições $p$ e $q$. Dica: use a desigualdade de Jensen, ou seja, use o fato de que $-\log x$ é uma função convexa.
1. Vamos calcular a entropia de algumas fontes de dados:
     * Suponha que você esteja observando a saída gerada por um macaco em uma máquina de escrever. O macaco pressiona qualquer uma das $44$ teclas da máquina de escrever aleatoriamente (você pode assumir que ele ainda não descobriu nenhuma tecla especial ou a tecla shift). Quantos bits de aleatoriedade por personagem você observa?
     * Por estar infeliz com o macaco, você o substituiu por um compositor bêbado. É capaz de gerar palavras, embora não de forma coerente. Em vez disso, ele escolhe uma palavra aleatória de um vocabulário de $2.000$ palavras. Vamos supor que o comprimento médio de uma palavra seja de $4,5$ letras em inglês. Quantos bits de aleatoriedade por personagem você observa agora?
   * Ainda insatisfeito com o resultado, você substitui o compositor por um modelo de linguagem de alta qualidade. O modelo de linguagem pode atualmente obter uma perplexidade tão baixa quanto $15$ pontos por palavra. O caractere *perplexidade* de um modelo de linguagem é definido como o inverso da média geométrica de um conjunto de probabilidades, cada probabilidade corresponde a um caractere da palavra. Para ser específico, se o comprimento de uma determinada palavra é $l$, então $\mathrm{PPL}(\text{word}) = \left[\prod_i p(\text{character}_i)\right]^{ -\frac{1}{l}} = \exp \left[ - \frac{1}{l} \sum_i{\log p(\text{character}_i)} \right].$ Suponha que a palavra de teste tem 4,5 letras, quantos bits de aleatoriedade por caractere você observa agora?
1. Explique intuitivamente porque $I(X, Y) = H(X) - H(X|Y)$. Então, mostre que isso é verdade expressando ambos os lados como uma expectativa com relação à distribuição conjunta.
1. Qual é a Divergência KL entre as duas distribuições Gaussianas $\mathcal{N}(\mu_1, \sigma_1^2)$ e $\mathcal{N}(\mu_2, \sigma_2^2)$?


:begin_tab:`mxnet`
[Discussões](https://discuss.d2l.ai/t/420)
:end_tab:

:begin_tab:`pytorch`
[Discussões](https://discuss.d2l.ai/t/1104)
:end_tab:

:begin_tab:`tensorflow`
[Discussões](https://discuss.d2l.ai/t/1105)
:end_tab:
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTczMDIyMTcxMCwtMjA3NTc2MzYzNCwzOD
Q4NTQxNzksMTA2ODQ2NDQ3OCw3ODczNDE3NzgsLTEwOTM2NDc0
NzIsLTQ0MzQ3Nzk0MCwxNzgxMTI2Mjc5LDQwOTUxMDI4MSwyMT
Q1OTgwOTMyLDI3NTIzMDA2LC0xMjA1NTU1MzQyLDE1ODgzODU4
NThdfQ==
-->