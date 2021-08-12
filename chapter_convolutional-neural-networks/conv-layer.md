# Convolução para Imagens
:label:`sec_conv_layer`

Agora que entendemos como as camadas convolucionais funcionam na teoria,
estamos prontos para ver como eles funcionam na prática.
Com base na nossa motivação de redes neurais convolucionais
como arquiteturas eficientes para explorar a estrutura em dados de imagem,
usamos imagens como nosso exemplo de execução.


## A Operação de Correlação Cruzada


Lembre-se de que, estritamente falando, as camadas convolucionais
são um nome impróprio, uma vez que as operações que elas expressam
são descritos com mais precisão como correlações cruzadas.
Com base em nossas descrições de camadas convolucionais em :numref:`sec_why-conv`,
em tal camada, um tensor de entrada
e um tensor de *kernel* são combinados
para produzir um tensor de saída por meio de uma operação de correlação cruzada.

Vamos ignorar os canais por enquanto e ver como isso funciona
com dados bidimensionais e representações ocultas.
Em :numref:`fig_correlation`,
a entrada é um tensor bidimensional
com altura de 3 e largura de 3.
Marcamos a forma do tensor como $3 \times 3$ or ($3$, $3$).
A altura e a largura do *kernel* são 2.
A forma da *janela do kernel* (ou *janela de convolução*)
é dada pela altura e largura do *kernel*
(aqui é $2 \times 2$).

![Operação de correlação cruzada bidimensional. As partes sombreadas são o primeiro elemento de saída, bem como os elementos tensores de entrada e *kernel* usados para o cálculo de saída: $0\times0+1\times1+3\times2+4\times3=19$.](../img/correlation.svg)
:label:`fig_correlation`

Na operação de correlação cruzada bidimensional,
começamos com a janela de convolução posicionada
no canto superior esquerdo do tensor de entrada
e o deslizamos pelo tensor de entrada,
ambos da esquerda para a direita e de cima para baixo.
Quando a janela de convolução desliza para uma determinada posição,
o subtensor de entrada contido nessa janela
e o tensor do *kernel* são multiplicados elemento a elemento
e o tensor resultante é resumido
produzindo um único valor escalar.
Este resultado fornece o valor do tensor de saída
no local correspondente.
Aqui, o tensor de saída tem uma altura de 2 e largura de 2
e os quatro elementos são derivados de
a operação de correlação cruzada bidimensional:

$$
0\times0+1\times1+3\times2+4\times3=19,\\
1\times0+2\times1+4\times2+5\times3=25,\\
3\times0+4\times1+6\times2+7\times3=37,\\
4\times0+5\times1+7\times2+8\times3=43.
$$

Observe que ao longo de cada eixo, o tamanho da saída
é ligeiramente menor que o tamanho de entrada.
Como o *kernel* tem largura e altura maiores que um,
só podemos calcular corretamente a correlação cruzada
para locais onde o *kernel* se encaixa totalmente na imagem,
o tamanho da saída é dado pelo tamanho da entrada $n_h \times n_w$
menos o tamanho do *kernel* de convolução $k_h \times k_w$
através da

$$(n_h-k_h+1) \times (n_w-k_w+1).$$

Este é o caso, pois precisamos de espaço suficiente
para "deslocar" o *kernel* de convolução na imagem.
Mais tarde, veremos como manter o tamanho inalterado
preenchendo a imagem com zeros em torno de seu limite
para que haja espaço suficiente para mudar o *kernel*.
Em seguida, implementamos este processo na função `corr2d`,
que aceita um tensor de entrada `X` e um tensor de *kernel* `K`
e retorna um tensor de saída `Y`.

```{.python .input}
from d2l import mxnet as d2l
from mxnet import autograd, np, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
```

```{.python .input}
#@tab mxnet, pytorch
def corr2d(X, K):  #@save
    """Compute 2D cross-correlation."""
    h, w = K.shape
    Y = d2l.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = d2l.reduce_sum((X[i: i + h, j: j + w] * K))
    return Y
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf

def corr2d(X, K):  #@save
    """Compute 2D cross-correlation."""
    h, w = K.shape
    Y = tf.Variable(tf.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1)))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j].assign(tf.reduce_sum(
                X[i: i + h, j: j + w] * K))
    return Y
```

Podemos construir o tensor de entrada `X` e o tensor do kernel` K`
from :numref:`fig_correlation`
para validar o resultado da implementação acima
da operação de correlação cruzada bidimensional.

```{.python .input}
#@tab all
X = d2l.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
K = d2l.tensor([[0.0, 1.0], [2.0, 3.0]])
corr2d(X, K)
```

## Camadas Convolucionais


Uma camada convolucional correlaciona a entrada e o *kernel*
e adiciona um *bias* escalar para produzir uma saída.
Os dois parâmetros de uma camada convolucional
são o *kernel* e o *bias* escalar.
Ao treinar modelos com base em camadas convolucionais,
normalmente inicializamos os *kernels* aleatoriamente,
assim como faríamos com uma camada totalmente conectada.

Agora estamos prontos para implementar uma camada convolucional bidimensional
com base na função `corr2d` definida acima.
Na função construtora `__init__`,
declaramos `weight` e` bias` como os dois parâmetros do modelo.
A função de propagação direta
chama a função `corr2d` e adiciona o viés.

```{.python .input}
class Conv2D(nn.Block):
    def __init__(self, kernel_size, **kwargs):
        super().__init__(**kwargs)
        self.weight = self.params.get('weight', shape=kernel_size)
        self.bias = self.params.get('bias', shape=(1,))

    def forward(self, x):
        return corr2d(x, self.weight.data()) + self.bias.data()
```

```{.python .input}
#@tab pytorch
class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return corr2d(x, self.weight) + self.bias
```

```{.python .input}
#@tab tensorflow
class Conv2D(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def build(self, kernel_size):
        initializer = tf.random_normal_initializer()
        self.weight = self.add_weight(name='w', shape=kernel_size,
                                      initializer=initializer)
        self.bias = self.add_weight(name='b', shape=(1, ),
                                    initializer=initializer)

    def call(self, inputs):
        return corr2d(inputs, self.weight) + self.bias
```

Na convolução
$h \times w$ 
ou um *kernel* de convolução $h \times w$
a altura e a largura do *kernel* de convolução são $h$ e $w$, respectivamente.
Também nos referimos a
uma camada convolucional com um kernel de convolução $h \times w$
simplesmente como uma camada convolucional $h \times w$


## Detecção de Borda de Objeto em Imagens

Vamos analisar uma aplicação simples de uma camada convolucional:
detectar a borda de um objeto em uma imagem
encontrando a localização da mudança de pixel.
Primeiro, construímos uma "imagem" de $6\times 8$ pixels.
As quatro colunas do meio são pretas (0) e as demais são brancas (1).

```{.python .input}
#@tab mxnet, pytorch
X = d2l.ones((6, 8))
X[:, 2:6] = 0
X
```

```{.python .input}
#@tab tensorflow
X = tf.Variable(tf.ones((6, 8)))
X[:, 2:6].assign(tf.zeros(X[:, 2:6].shape))
X
```

Em seguida, construímos um kernel `K` com uma altura de 1 e uma largura de 2.
Quando realizamos a operação de correlação cruzada com a entrada,
se os elementos horizontalmente adjacentes forem iguais,
a saída é 0. Caso contrário, a saída é diferente de zero.

```{.python .input}
#@tab all
K = d2l.tensor([[1.0, -1.0]])
```

Estamos prontos para realizar a operação de correlação cruzada
com os argumentos `X` (nossa entrada) e` K` (nosso kernel).
Como você pode ver, detectamos 1 para a borda do branco ao preto
e -1 para a borda do preto ao branco.
Todas as outras saídas assumem o valor 0.

```{.python .input}
#@tab all
Y = corr2d(X, K)
Y
```

Agora podemos aplicar o kernel à imagem transposta.
Como esperado, ele desaparece. O kernel `K` detecta apenas bordas verticais.

```{.python .input}
#@tab all
corr2d(d2l.transpose(X), K)
```

## Aprendendo um Kernel


Projetar um detector de borda por diferenças finitas `[1, -1]` é legal
se sabemos que é exatamente isso que estamos procurando.
No entanto, quando olhamos para *kernels* maiores,
e considere camadas sucessivas de convoluções,
pode ser impossível especificar
exatamente o que cada filtro deve fazer manualmente.

Agora vamos ver se podemos aprender o *kernel* que gerou `Y` de` X`
olhando apenas para os pares de entrada--saída.
Primeiro construímos uma camada convolucional
e inicializamos seu *kernel* como um tensor aleatório.
A seguir, em cada iteração, usaremos o erro quadrático
para comparar `Y` com a saída da camada convolucional.
Podemos então calcular o gradiente para atualizar o *kernel*.
Por uma questão de simplicidade,
na sequência
nós usamos a classe embutida
para camadas convolucionais bidimensionais
e ignorar o *bias*.

```{.python .input}
# Construct a two-dimensional convolutional layer with 1 output channel and a
# kernel of shape (1, 2). For the sake of simplicity, we ignore the bias here
conv2d = nn.Conv2D(1, kernel_size=(1, 2), use_bias=False)
conv2d.initialize()

# The two-dimensional convolutional layer uses four-dimensional input and
# output in the format of (example, channel, height, width), where the batch
# size (number of examples in the batch) and the number of channels are both 1
X = X.reshape(1, 1, 6, 8)
Y = Y.reshape(1, 1, 6, 7)

for i in range(10):
    with autograd.record():
        Y_hat = conv2d(X)
        l = (Y_hat - Y) ** 2
    l.backward()
    # Update the kernel
    conv2d.weight.data()[:] -= 3e-2 * conv2d.weight.grad()
    if (i + 1) % 2 == 0:
        print(f'batch {i + 1}, loss {float(l.sum()):.3f}')
```

```{.python .input}
#@tab pytorch
# Construct a two-dimensional convolutional layer with 1 output channel and a
# kernel of shape (1, 2). For the sake of simplicity, we ignore the bias here
conv2d = nn.Conv2d(1,1, kernel_size=(1, 2), bias=False)

# The two-dimensional convolutional layer uses four-dimensional input and
# output in the format of (example channel, height, width), where the batch
# size (number of examples in the batch) and the number of channels are both 1
X = X.reshape((1, 1, 6, 8))
Y = Y.reshape((1, 1, 6, 7))

for i in range(10):
    Y_hat = conv2d(X)
    l = (Y_hat - Y) ** 2
    conv2d.zero_grad()
    l.sum().backward()
    # Update the kernel
    conv2d.weight.data[:] -= 3e-2 * conv2d.weight.grad
    if (i + 1) % 2 == 0:
        print(f'batch {i + 1}, loss {l.sum():.3f}')
```

```{.python .input}
#@tab tensorflow
# Construct a two-dimensional convolutional layer with 1 output channel and a
# kernel of shape (1, 2). For the sake of simplicity, we ignore the bias here
conv2d = tf.keras.layers.Conv2D(1, (1, 2), use_bias=False)

# The two-dimensional convolutional layer uses four-dimensional input and
# output in the format of (example channel, height, width), where the batch
# size (number of examples in the batch) and the number of channels are both 1
X = tf.reshape(X, (1, 6, 8, 1))
Y = tf.reshape(Y, (1, 6, 7, 1))

Y_hat = conv2d(X)
for i in range(10):
    with tf.GradientTape(watch_accessed_variables=False) as g:
        g.watch(conv2d.weights[0])
        Y_hat = conv2d(X)
        l = (abs(Y_hat - Y)) ** 2
        # Update the kernel
        update = tf.multiply(3e-2, g.gradient(l, conv2d.weights[0]))
        weights = conv2d.get_weights()
        weights[0] = conv2d.weights[0] - update
        conv2d.set_weights(weights)
        if (i + 1) % 2 == 0:
            print(f'batch {i + 1}, loss {tf.reduce_sum(l):.3f}')
```

Observe que o erro caiu para um valor pequeno após 10 iterações. Agora daremos uma olhada no tensor do *kernel* que aprendemos.

```{.python .input}
d2l.reshape(conv2d.weight.data(), (1, 2))
```

```{.python .input}
#@tab pytorch
d2l.reshape(conv2d.weight.data, (1, 2))
```

```{.python .input}
#@tab tensorflow
d2l.reshape(conv2d.get_weights()[0], (1, 2))
```

Indeed, the learned kernel tensor is remarkably close
to the kernel tensor `K` we defined earlier.

## Correlação Cruzada e Convolução


Lembre-se de nossa observação de :numref:`sec_why-conv` da correspondência
entre as operações de correlação cruzada e convolução.
Aqui, vamos continuar a considerar as camadas convolucionais bidimensionais.
E se essas camadas
realizar operações de convolução estritas
conforme definido em :eqref:`eq_2d-conv-discrete`
em vez de correlações cruzadas?
Para obter a saída da operação de *convolução* estrita, precisamos apenas inverter o tensor do *kerne*l bidimensional tanto horizontal quanto verticalmente e, em seguida, executar a operação de *correlação cruzada* com o tensor de entrada.

É digno de nota que, uma vez que os *kernels* são aprendidos a partir de dados no aprendizado profundo,
as saídas das camadas convolucionais permanecem inalteradas
não importa se tais camadas
executam
as operações de convolução estrita
ou as operações de correlação cruzada.


Para ilustrar isso, suponha que uma camada convolucional execute *correlação cruzada* e aprenda o *kernel* em :numref:`fig_correlation`, que é denotado como a matriz $\mathbf{K}$ aqui.
Supondo que outras condições permaneçam inalteradas,
quando esta camada executa *convolução* estrita em vez disso,
o *kernel* aprendido $\mathbf{K}'$ será o mesmo que $\mathbf{K}$
depois que $\mathbf{K}'$ is  é
invertido horizontalmente e verticalmente.
Quer dizer,
quando a camada convolucional
executa *convolução* estrita
para a entrada em :numref:`fig_correlation`
e $\mathbf{K}'$,
a mesma saída em :numref:`fig_correlation`
(correlação cruzada da entrada e $\mathbf{K}$)
será obtida.

De acordo com a terminologia padrão da literatura de *deep learning*,
continuaremos nos referindo à operação de correlação cruzada
como uma convolução, embora, estritamente falando, seja ligeiramente diferente.
Além do mais,
usamos o termo *elemento* para nos referirmos a
uma entrada (ou componente) de qualquer tensor que representa uma representação de camada ou um *kernel* de convolução.


## Mapa de Características e Campo Receptivo


Conforme descrito em :numref:`subsec_why-conv-channels`,
a saída da camada convolucional em
:numref:`fig_correlation`
às vezes é chamada de *mapa de características*,
pois pode ser considerado como
as representações aprendidas (características)
nas dimensões espaciais (por exemplo, largura e altura)
para a camada subsequente.
Nas CNNs,
para qualquer elemento $x$ de alguma camada,
seu *campo receptivo* refere-se a
todos os elementos (de todas as camadas anteriores)
que pode afetar o cálculo de $x$
durante a propagação direta.
Observe que o campo receptivo
pode ser maior do que o tamanho real da entrada.

Vamos continuar a usar :numref:`fig_correlation` para explicar o campo receptivo.
Dado o *kernel* de convolução $2 \times 2$
o campo receptivo do elemento de saída sombreado (de valor $19$)
são
os quatro elementos na parte sombreada da entrada.
Agora, vamos denotar a saída $2 \times 2$
como $\mathbf{Y}$
e considere uma CNN mais profunda
com uma camada convolucional adicional $2 \times 2$ que leva $\mathbf{Y}$
como sua entrada, produzindo
um único elemento $z$.
Nesse caso,
o campo receptivo de $z$
em $\mathbf{Y}$ inclui todos os quatro elementos de $\mathbf{Y}$,
enquanto
o campo receptivo
na entrada inclui todos os nove elementos de entrada.
Por isso,
quando qualquer elemento em um mapa de recursos
precisa de um campo receptivo maior
para detectar recursos de entrada em uma área mais ampla,
podemos construir uma rede mais profunda.



## Resumo

* O cálculo central de uma camada convolucional bidimensional é uma operação de correlação cruzada bidimensional. Em sua forma mais simples, isso executa uma operação de correlação cruzada nos dados de entrada bidimensionais e no *kernel* e, em seguida, adiciona um *bias*.
* Podemos projetar um *kernel* para detectar bordas em imagens.
* Podemos aprender os parâmetros do *kernel* a partir de dados.
* Com os *kernels* aprendidos a partir dos dados, as saídas das camadas convolucionais permanecem inalteradas, independentemente das operações realizadas por essas camadas (convolução estrita ou correlação cruzada).
* Quando qualquer elemento em um mapa de características precisa de um campo receptivo maior para detectar características mais amplas na entrada, uma rede mais profunda pode ser considerada.


## Exercícios

1. Construa uma imagem `X` com bordas diagonais.
     1. O que acontece se você aplicar o *kernel* `K` nesta seção a ele?
     1. O que acontece se você transpõe `X`?
     1. O que acontece se você transpõe `K`?
1. Quando você tenta encontrar automaticamente o gradiente para a classe `Conv2D` que criamos, que tipo de mensagem de erro você vê?
1. Como você representa uma operação de correlação cruzada como uma multiplicação de matriz, alterando os tensores de entrada e *kernel*?
1. Projete alguns *kernels* manualmente.
     1. Qual é a forma de um *kernel* para a segunda derivada?
     1. Qual é o *kernel* de uma integral?
     1. Qual é o tamanho mínimo de um *kernel* para obter uma derivada de grau $d$?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/65)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/66)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/271)
:end_tab:
<!--stackedit_data:
eyJoaXN0b3J5IjpbMTczNzg0NzMzNywxMjA2NTU0NTU3LC0xNj
c3ODMxMzQ3LC0xOTg4NjQzNDc1LC0xODE4NDc0NzM1LC0xNDk0
NjAxMTEyLC0yNDIxOTgyNzcsLTY3NDg1MTc2OSw1NTkxMzU0NT
AsMTk4NDk3OTc5N119
-->