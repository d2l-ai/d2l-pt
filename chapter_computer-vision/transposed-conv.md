# Convolução Transposta
:label:`sec_transposed_conv`

As camadas que apresentamos até agora para redes neurais convolucionais, incluindo camadas convolucionais (:numref:`sec_conv_layer`) e camadas de pooling (:numref:`sec_pooling`), geralmente reduzem a largura e altura de entrada ou as mantêm inalteradas. Aplicativos como segmentação semântica (:numref:`sec_semantic_segmentation`) e redes adversárias geradoras (:numref:`sec_dcgan`), no entanto, exigem prever valores para cada pixel e, portanto, precisam aumentar a largura e altura de entrada. A convolução transposta, também chamada de convolução fracionada :cite:`Dumoulin.Visin.2016` ou deconvolução :cite:`Long.Shelhamer.Darrell.2015`, serve a este propósito.

```{.python .input}
from mxnet import np, npx, init
from mxnet.gluon import nn
from d2l import mxnet as d2l

npx.set_np()
```

```{.python .input}
#@tab pytorch
import torch
from torch import nn
from d2l import torch as d2l
```

## Convolução Transposta 2D Básica

Vamos considerar um caso básico em que os canais de entrada e saída são 1, com 0 preenchimento e 1 passo. :numref:`fig_trans_conv` ilustra como a convolução transposta com um *kernel* $2\times 2$ é calculada na matriz de entrada $2\times 2$.

![Camada de convolução transposta com um *kernel* $2\times 2$.](../img/trans-conv.svg)
:label:`fig_trans_conv`

Podemos implementar essa operação fornecendo o *kernel* da matriz $K$ e a entrada da matriz $X$.

```{.python .input}
#@tab all
def trans_conv(X, K):
    h, w = K.shape
    Y = d2l.zeros((X.shape[0] + h - 1, X.shape[1] + w - 1))
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Y[i: i + h, j: j + w] += X[i, j] * K
    return Y
```


Lembre-se de que a convolução calcula os resultados por `Y[i, j] = (X[i: i + h, j: j + w] * K).sum()` (consulte `corr2d` em :numref:`sec_conv_layer`), que resume os valores de entrada por meio do *kernel*. Enquanto a convolução transposta transmite valores de entrada por meio do *kernel*, o que resulta em uma forma de saída maior.

Verifique os resultados em :numref:`fig_trans_conv`.

```{.python .input}
#@tab all
X = d2l.tensor([[0., 1], [2, 3]])
K = d2l.tensor([[0., 1], [2, 3]])
trans_conv(X, K)
```

:begin_tab:`mxnet`
Ou podemos usar `nn.Conv2DTranspose` para obter os mesmos resultados. Como `nn.Conv2D`, tanto a entrada quanto o *kernel* devem ser tensores 4-D.
:end_tab:

:begin_tab:`pytorch`
Ou podemos usar `nn.ConvTranspose2d` para obter os mesmos resultados. Como `nn.Conv2d`, tanto a entrada quanto o *kernel* devem ser tensores 4-D.
:end_tab:

```{.python .input}
X, K = X.reshape(1, 1, 2, 2), K.reshape(1, 1, 2, 2)
tconv = nn.Conv2DTranspose(1, kernel_size=2)
tconv.initialize(init.Constant(K))
tconv(X)
```

```{.python .input}
#@tab pytorch
X, K = X.reshape(1, 1, 2, 2), K.reshape(1, 1, 2, 2)
tconv = nn.ConvTranspose2d(1, 1, kernel_size=2, bias=False)
tconv.weight.data = K
tconv(X)
```

## Preenchimento, Passos e Canais

Aplicamos elementos de preenchimento à entrada em convolução, enquanto eles são aplicados à saída em convolução transposta. Um preenchimento $1\times 1$ significa que primeiro calculamos a saída como normal e, em seguida, removemos as primeiras/últimas linhas e colunas.

```{.python .input}
tconv = nn.Conv2DTranspose(1, kernel_size=2, padding=1)
tconv.initialize(init.Constant(K))
tconv(X)
```

```{.python .input}
#@tab pytorch
tconv = nn.ConvTranspose2d(1, 1, kernel_size=2, padding=1, bias=False)
tconv.weight.data = K
tconv(X)
```

Da mesma forma, os avanços também são aplicados às saídas.

```{.python .input}
tconv = nn.Conv2DTranspose(1, kernel_size=2, strides=2)
tconv.initialize(init.Constant(K))
tconv(X)
```

```{.python .input}
#@tab pytorch
tconv = nn.ConvTranspose2d(1, 1, kernel_size=2, stride=2, bias=False)
tconv.weight.data = K
tconv(X)
```


A extensão multicanal da convolução transposta é igual à convolução. Quando a entrada tem vários canais, denotados por $c_i$, a convolução transposta atribui uma matriz de *kernel* $k_h\times k_w$ a cada canal de entrada. Se a saída tem um tamanho de canal $c_o$, então temos um *kernel* $c_i\times k_h\times k_w$ para cada canal de saída.


Como resultado, se alimentarmos $X$ em uma camada convolucional $f$ para calcular $Y=f(X)$ e criarmos uma camada de convolução transposta $g$ com os mesmos hiperparâmetros de $f$, exceto para o conjunto de canais de saída para ter o tamanho do canal de $X$, então $g(Y)$ deve ter o mesmo formato que $X$. Deixe-nos verificar esta afirmação.

```{.python .input}
X = np.random.uniform(size=(1, 10, 16, 16))
conv = nn.Conv2D(20, kernel_size=5, padding=2, strides=3)
tconv = nn.Conv2DTranspose(10, kernel_size=5, padding=2, strides=3)
conv.initialize()
tconv.initialize()
tconv(conv(X)).shape == X.shape
```

```{.python .input}
#@tab pytorch
X = torch.rand(size=(1, 10, 16, 16))
conv = nn.Conv2d(10, 20, kernel_size=5, padding=2, stride=3)
tconv = nn.ConvTranspose2d(20, 10, kernel_size=5, padding=2, stride=3)
tconv(conv(X)).shape == X.shape
```

## Analogia à Transposição de Matriz

A convolução transposta leva o nome da transposição da matriz. Na verdade, as operações de convolução também podem ser realizadas por multiplicação de matrizes. No exemplo abaixo, definimos uma entrada $X$ $3\times 3$ com *kernel* $K$ $2\times 2$, e então usamos `corr2d` para calcular a saída da convolução.

```{.python .input}
#@tab all
X = d2l.arange(9.0).reshape(3, 3)
K = d2l.tensor([[0, 1], [2, 3]])
Y = d2l.corr2d(X, K)
Y
```

A seguir, reescrevemos o *kernel* de convolução $K$ como uma matriz $W$. Sua forma será $(4, 9)$, onde a linha $i^\mathrm{th}$ presente aplicando o *kernel* à entrada para gerar o $i^\mathrm{th}$ elemento de saída.

```{.python .input}
#@tab all
def kernel2matrix(K):
    k, W = d2l.zeros(5), d2l.zeros((4, 9))
    k[:2], k[3:5] = K[0, :], K[1, :]
    W[0, :5], W[1, 1:6], W[2, 3:8], W[3, 4:] = k, k, k, k
    return W

W = kernel2matrix(K)
W
```

Então, o operador de convolução pode ser implementado por multiplicação de matriz com remodelagem adequada.

```{.python .input}
Y == np.dot(W, X.reshape(-1)).reshape(2, 2)
```

```{.python .input}
#@tab pytorch
Y == torch.mv(W, X.reshape(-1)).reshape(2, 2)
```

Podemos implementar a convolução transposta como uma multiplicação de matriz reutilizando `kernel2matrix`. Para reutilizar o $W$ gerado, construímos uma entrada $2\times 2$, de modo que a matriz de peso correspondente terá uma forma $(9, 4)$, que é $W^\top$. Deixe-nos verificar os resultados.

```{.python .input}
X = np.array([[0, 1], [2, 3]])
Y = trans_conv(X, K)
Y == np.dot(W.T, X.reshape(-1)).reshape(3, 3)
```

```{.python .input}
#@tab pytorch
X = torch.tensor([[0.0, 1], [2, 3]])
Y = trans_conv(X, K)
Y == torch.mv(W.T, X.reshape(-1)).reshape(3, 3)
```

## Resumo

* Em comparação com as convoluções que reduzem as entradas por meio de *kernels*, as convoluções transpostas transmitem as entradas.
* Se uma camada de convolução reduz a largura e altura de entrada em $n_w$ e $h_h$ tempo, respectivamente. Então, uma camada de convolução transposta com os mesmos tamanhos de *kernel*, preenchimento e passos aumentará a largura e altura de entrada em $n_w$ e $h_h$, respectivamente.
* Podemos implementar operações de convolução pela multiplicação da matriz, as convoluções transpostas correspondentes podem ser feitas pela multiplicação da matriz transposta.

## Exercícios

1. É eficiente usar a multiplicação de matrizes para implementar operações de convolução? Por quê?

:begin_tab:`mxnet`
[Discussões](https://discuss.d2l.ai/t/376)
:end_tab:

:begin_tab:`pytorch`
[Discussões](https://discuss.d2l.ai/t/1450)
:end_tab:
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTExNTE1Njg5MDEsLTE0MzgzMjU5MzUsMT
U0MjgyNDE5Ml19
-->