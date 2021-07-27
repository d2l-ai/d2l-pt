# Canais de Múltiplas Entradas e Saídas
:label:`sec_channels`


Embora tenhamos descrito os vários canais
que compõem cada imagem (por exemplo, imagens coloridas têm os canais RGB padrão
para indicar a quantidade de vermelho, verde e azul) e camadas convolucionais para vários canais em :numref:`subsec_why-conv-channels`,
até agora, simplificamos todos os nossos exemplos numéricos
trabalhando com apenas uma única entrada e um único canal de saída.
Isso nos permitiu pensar em nossas entradas, *kernels* de convolução,
e saídas, cada um como tensores bidimensionais.

Quando adicionamos canais a isto,
nossas entradas e representações ocultas
ambas se tornam tensores tridimensionais.
Por exemplo, cada imagem de entrada RGB tem a forma $3\times h\times w$.
Referimo-nos a este eixo, com um tamanho de 3, como a dimensão do *canal*.
Nesta seção, daremos uma olhada mais detalhada
em núcleos de convolução com múltiplos canais de entrada e saída.

## Canais de Entrada Múltiplos


Quando os dados de entrada contêm vários canais,
precisamos construir um *kernel* de convolução
com o mesmo número de canais de entrada que os dados de entrada,
para que possa realizar correlação cruzada com os dados de entrada.
Supondo que o número de canais para os dados de entrada seja $c_i$,
o número de canais de entrada do *kernel* de convolução também precisa ser $c_i$. Se a forma da janela do nosso kernel de convolução é $k_h\times k_w$,
então quando $c_i=1$, podemos pensar em nosso kernel de convolução
apenas como um tensor bidimensional de forma $k_h\times k_w$.

No entanto, quando $c_i>1$, precisamos de um kernel
que contém um tensor de forma $k_h\times k_w$ para *cada* canal de entrada. Concatenando estes $c_i$ tensores juntos
produz um kernel de convolução de forma $c_i\times k_h\times k_w$.
Uma vez que o *kernel* de entrada e convolução tem cada um $c_i$ canais,
podemos realizar uma operação de correlação cruzada
no tensor bidimensional da entrada
e o tensor bidimensional do núcleo de convolução
para cada canal, adicionando os resultados $c_i$ juntos
(somando os canais)
para produzir um tensor bidimensional.
Este é o resultado de uma correlação cruzada bidimensional
entre uma entrada multicanal e
um *kernel* de convolução com vários canais de entrada.

Em :numref:`fig_conv_multi_in`, demonstramos um exemplo
de uma correlação cruzada bidimensional com dois canais de entrada.
As partes sombreadas são o primeiro elemento de saída
bem como os elementos tensores de entrada e kernel usados ​​para o cálculo de saída:
$(1\times1+2\times2+4\times3+5\times4)+(0\times0+1\times1+3\times2+4\times3)=56$.

![Cálculo de correlação cruzada com 2 canais de entrada.](../img/conv-multi-in.svg)
:label:`fig_conv_multi_in`


Para ter certeza de que realmente entendemos o que está acontecendo aqui,
podemos implementar operações de correlação cruzada com vários canais de entrada.
Observe que tudo o que estamos fazendo é realizar uma operação de correlação cruzada
por canal e depois somando os resultados.

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
#@tab mxnet, pytorch
def corr2d_multi_in(X, K):
    # First, iterate through the 0th dimension (channel dimension) of `X` and
    # `K`. Then, add them together
    return sum(d2l.corr2d(x, k) for x, k in zip(X, K))
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf

def corr2d_multi_in(X, K):
    # First, iterate through the 0th dimension (channel dimension) of `X` and
    # `K`. Then, add them together
    return tf.reduce_sum([d2l.corr2d(x, k) for x, k in zip(X, K)], axis=0)
```

Podemos construir o tensor de entrada `X` e o tensor do kernel` K`
correspondendo aos valores em :numref:`fig_conv_multi_in`
para validar a saída da operação de correlação cruzada.

```{.python .input}
#@tab all
X = d2l.tensor([[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
               [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])
K = d2l.tensor([[[0.0, 1.0], [2.0, 3.0]], [[1.0, 2.0], [3.0, 4.0]]])

corr2d_multi_in(X, K)
```

## Canais de Saída Múltiplos


Independentemente do número de canais de entrada,
até agora acabamos sempre com um canal de saída.
No entanto, como discutimos em :numref:`subsec_why-conv-channels`,
é essencial ter vários canais em cada camada.
Nas arquiteturas de rede neural mais populares,
na verdade, aumentamos a dimensão do canal
à medida que subimos na rede neural,
normalmente reduzindo as amostras para compensar a resolução espacial
para maior *profundidade do canal*.
Intuitivamente, você pode pensar em cada canal
como respondendo a algum conjunto diferente de *features*.
A realidade é um pouco mais complicada do que as interpretações mais ingênuas dessa intuição, uma vez que as representações não são aprendidas de forma independente, mas sim otimizadas para serem úteis em conjunto.
Portanto, pode não ser que um único canal aprenda um detector de bordas, mas sim que alguma direção no espaço do canal corresponde à detecção de bordas.


Denote por $c_i$ e $c_o$ o número
dos canais de entrada e saída, respectivamente,
e sejam $k_h$ e $k_w$ a altura e a largura do *kernel*.
Para obter uma saída com vários canais,
podemos criar um tensor de kernel
da forma $c_i\times k_h\times k_w$
para *cada* canal de saída.
Nós os concatenamos na dimensão do canal de saída,
de modo que a forma do núcleo de convolução
é $c_o\times c_i\times k_h\times k_w$.
Em operações de correlação cruzada,
o resultado em cada canal de saída é calculado
do *kernel* de convolução correspondente a esse canal de saída
e recebe a entrada de todos os canais no tensor de entrada.

Implementamos uma função de correlação cruzada
para calcular a saída de vários canais, conforme mostrado abaixo.

```{.python .input}
#@tab all
def corr2d_multi_in_out(X, K):
    # Iterate through the 0th dimension of `K`, and each time, perform
    # cross-correlation operations with input `X`. All of the results are
    # stacked together
    return d2l.stack([corr2d_multi_in(X, k) for k in K], 0)
```

Construímos um *kernel* de convolução com 3 canais de saída
concatenando o tensor do kernel `K` com` K + 1`
(mais um para cada elemento em `K`) e` K + 2`.

```{.python .input}
#@tab all
K = d2l.stack((K, K + 1, K + 2), 0)
K.shape
```

Abaixo, realizamos operações de correlação cruzada
no tensor de entrada `X` com o tensor do kernel` K`.
Agora a saída contém 3 canais.
O resultado do primeiro canal é consistente
com o resultado do tensor de entrada anterior `X`
e o canal de múltiplas entradas,
*kernel* do canal de saída única.

```{.python .input}
#@tab all
corr2d_multi_in_out(X, K)
```

## Camada Convolucional $1\times 1$


No início, uma convolução $1 \times 1$, ou seja, $k_h = k_w = 1$,
não parece fazer muito sentido.
Afinal, uma convolução correlaciona pixels adjacentes.
Uma convolução $1 \times 1$ obviamente não faz isso.
No entanto, são operações populares que às vezes são incluídas
nos projetos de redes profundas complexas.
Vejamos com alguns detalhes o que ele realmente faz.

Como a janela mínima é usada,
a convolução $1\times 1$ perde a capacidade
de camadas convolucionais maiores
para reconhecer padrões que consistem em interações
entre os elementos adjacentes nas dimensões de altura e largura.
O único cálculo da convolução $1\times 1$ ocorre
na dimensão do canal.

:numref:`fig_conv_1x1` mostra o cálculo de correlação cruzada
usando o kernel de convolução $1\times 1$
com 3 canais de entrada e 2 canais de saída.
Observe que as entradas e saídas têm a mesma altura e largura.
Cada elemento na saída é derivado
de uma combinação linear de elementos *na mesma posição*
na imagem de entrada.
Você poderia pensar na camada convolucional $1\times 1$
como constituindo uma camada totalmente conectada aplicada em cada localização de pixel
para transformar os valores de entrada correspondentes $c_i$ em valores de saída $c_o$.
Porque esta ainda é uma camada convolucional,
os pesos são vinculados à localização do pixel.
Assim, a camada convolucional $1\times 1$ requer pesos $c_o\times c_i$ 
(mais o *bias*).



![O cálculo de correlação cruzada usa o *kernel* de convolução $1\times 1$  com 3 canais de entrada e 2 canais de saída. A entrada e a saída têm a mesma altura e largura.](../img/conv-1x1.svg)
:label:`fig_conv_1x1`

Vamos verificar se isso funciona na prática:
implementamos uma convolução $1 \times 1$
usando uma camada totalmente conectada.
A única coisa é que precisamos fazer alguns ajustes
para a forma de dados antes e depois da multiplicação da matriz.

```{.python .input}
#@tab all
def corr2d_multi_in_out_1x1(X, K):
    c_i, h, w = X.shape
    c_o = K.shape[0]
    X = d2l.reshape(X, (c_i, h * w))
    K = d2l.reshape(K, (c_o, c_i))
    Y = d2l.matmul(K, X)  # Matrix multiplication in the fully-connected layer
    return d2l.reshape(Y, (c_o, h, w))
```

Ao realizar convolução $1\times 1$ ,
a função acima é equivalente à função de correlação cruzada implementada anteriormente `corr2d_multi_in_out`.
Vamos verificar isso com alguns dados de amostra.

```{.python .input}
#@tab mxnet, pytorch
X = d2l.normal(0, 1, (3, 3, 3))
K = d2l.normal(0, 1, (2, 3, 1, 1))
```

```{.python .input}
#@tab tensorflow
X = d2l.normal((3, 3, 3), 0, 1)
K = d2l.normal((2, 3, 1, 1), 0, 1)
```

```{.python .input}
#@tab all
Y1 = corr2d_multi_in_out_1x1(X, K)
Y2 = corr2d_multi_in_out(X, K)
assert float(d2l.reduce_sum(d2l.abs(Y1 - Y2))) < 1e-6
```

## Resumo

* Vários canais podem ser usados para estender os parâmetros do modelo da camada convolucional.
* A camada convolucional $1\times 1$ é equivalente à camada totalmente conectada, quando aplicada por pixel.
* A camada convolucional $1\times 1$ é normalmente usada para ajustar o número de canais entre as camadas de rede e para controlar a complexidade do modelo.


## Exercícios

1. Suponha que temos dois *kernels* de convolução de tamanho $k_1$ e $k_2$, respectivamente (sem não linearidade entre eles).
    1. Prove que o resultado da operação pode ser expresso por uma única convolução.
    1. Qual é a dimensionalidade da convolução única equivalente?
    1. O inverso é verdadeiro?
1. Assuma uma entrada de forma $c_i\times h\times w$ e um *kernel* de convolução de forma $c_o\times c_i\times k_h\times k_w$, preenchimento de $(p_h, p_w)$, e passo de $(s_h, s_w)$.
    1. Qual é o custo computacional (multiplicações e adições) para a propagação direta?
    1. Qual é a pegada de memória?
    1. Qual é a pegada de memória para a computação reversa?
    1. Qual é o custo computacional para a retropropagação?
1. Por que fator o número de cálculos aumenta se dobrarmos o número de canais de entrada $c_i$ e o número de canais de saída $c_o$? O que acontece se dobrarmos o preenchimento?
1. Se a altura e largura de um *kernel* de convolução é $k_h=k_w=1$,, qual é a complexidade computacional da propagação direta?
1. As variáveis ​​`Y1` e` Y2` no último exemplo desta seção são exatamente as mesmas? Porque?
1. Como você implementaria convoluções usando a multiplicação de matrizes quando a janela de convolução não é $1\times 1$?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/69)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/70)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/273)
:end_tab:
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTE2NjY1MDUzNDUsOTcxMDkyNjUxLDE4Nz
c1NTE2ODQsLTQwNTQwMzkwNCwtMTEzODU1Njc0LDIyMzY2OTYz
MywxMDg0Nzk1MTk3LDEwOTYzOTg3NjVdfQ==
-->