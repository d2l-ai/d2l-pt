# Redes Densamente Conectadas (DenseNet)

A ResNet mudou significativamente a visão de como parametrizar as funções em redes profundas. *DenseNet* (rede convolucional densa) é, até certo ponto, a extensão lógica disso :cite:`Huang.Liu.Van-Der-Maaten.ea.2017`.
Para entender como chegar a isso, façamos um pequeno desvio para a matemática.

## De ResNet para DenseNet

Lembre-se da expansão de Taylor para funções. Para o ponto $x = 0$, pode ser escrito como

$$f(x) = f(0) + f'(0) x + \frac{f''(0)}{2!}  x^2 + \frac{f'''(0)}{3!}  x^3 + \ldots.$$

O ponto principal é que ele decompõe uma função em termos de ordem cada vez mais elevados. De maneira semelhante, o ResNet decompõe funções em

$$f(\mathbf{x}) = \mathbf{x} + g(\mathbf{x}).$$

Ou seja, o ResNet decompõe $f$ em um termo linear simples e um termo mais complexo
não linear.
E se quisermos capturar (não necessariamente adicionar) informações além de dois termos?
Uma solução foi
DenseNet :cite:`Huang.Liu.Van-Der-Maaten.ea.2017`.

![A principal diferença entre ResNet (esquerda) e DenseNet (direita) em conexões de camada cruzada: uso de adição e uso de concatenação. ](../img/densenet-block.svg)
:label:`fig_densenet_block`

Conforme mostrado em :numref:`fig_densenet_block`, a principal diferença entre ResNet e DenseNet é que, no último caso, as saídas são *concatenadas* (denotadas por $[,]$) em vez de adicionadas.
Como resultado, realizamos um mapeamento de $\mathbf {x}$ para seus valores após aplicar uma sequência cada vez mais complexa de funções:

$$\mathbf{x} \to \left[
\mathbf{x},
f_1(\mathbf{x}),
f_2([\mathbf{x}, f_1(\mathbf{x})]), f_3([\mathbf{x}, f_1(\mathbf{x}), f_2([\mathbf{x}, f_1(\mathbf{x})])]), \ldots\right].$$

No final, todas essas funções são combinadas no MLP para reduzir o número de recursos novamente. Em termos de implementação, isso é bastante simples:
em vez de adicionar termos, nós os concatenamos. O nome DenseNet surge do fato de o gráfico de dependência entre as variáveis se tornar bastante denso. A última camada de tal cadeia está densamente conectada a todas as camadas anteriores. As conexões densas são mostradas em :numref:`fig_densenet`.

![Conexões densas na DenseNet.](../img/densenet.svg)
:label:`fig_densenet`


Os principais componentes que compõem uma DenseNet são *blocos densos* e *camadas de transição*. O primeiro define como as entradas e saídas são concatenadas, enquanto o último controla o número de canais para que não seja muito grande.

## Blocos Densos

A DenseNet usa a "normalização, ativação e convolução em lote" modificada
estrutura do ResNet (veja o exercício em :numref:`sec_resnet`).
Primeiro, implementamos essa estrutura de bloco de convolução.

```{.python .input}
from d2l import mxnet as d2l
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()

def conv_block(num_channels):
    blk = nn.Sequential()
    blk.add(nn.BatchNorm(),
            nn.Activation('relu'),
            nn.Conv2D(num_channels, kernel_size=3, padding=1))
    return blk
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn

def conv_block(input_channels, num_channels):
    return nn.Sequential(
        nn.BatchNorm2d(input_channels), nn.ReLU(),
        nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1))
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf

class ConvBlock(tf.keras.layers.Layer):
    def __init__(self, num_channels):
        super(ConvBlock, self).__init__()
        self.bn = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()
        self.conv = tf.keras.layers.Conv2D(
            filters=num_channels, kernel_size=(3, 3), padding='same')

        self.listLayers = [self.bn, self.relu, self.conv]

    def call(self, x):
        y = x
        for layer in self.listLayers.layers:
            y = layer(y)
        y = tf.keras.layers.concatenate([x,y], axis=-1)
        return y
```

Um *bloco denso* consiste em vários blocos de convolução, cada um usando o mesmo número de canais de saída. Na propagação direta, entretanto, concatenamos a entrada e a saída de cada bloco de convolução na dimensão do canal.

```{.python .input}
class DenseBlock(nn.Block):
    def __init__(self, num_convs, num_channels, **kwargs):
        super().__init__(**kwargs)
        self.net = nn.Sequential()
        for _ in range(num_convs):
            self.net.add(conv_block(num_channels))

    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            # Concatenate the input and output of each block on the channel
            # dimension
            X = np.concatenate((X, Y), axis=1)
        return X
```

```{.python .input}
#@tab pytorch
class DenseBlock(nn.Module):
    def __init__(self, num_convs, input_channels, num_channels):
        super(DenseBlock, self).__init__()
        layer = []
        for i in range(num_convs):
            layer.append(conv_block(
                num_channels * i + input_channels, num_channels))
        self.net = nn.Sequential(*layer)

    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            # Concatenate the input and output of each block on the channel
            # dimension
            X = torch.cat((X, Y), dim=1)
        return X
```

```{.python .input}
#@tab tensorflow
class DenseBlock(tf.keras.layers.Layer):
    def __init__(self, num_convs, num_channels):
        super(DenseBlock, self).__init__()
        self.listLayers = []
        for _ in range(num_convs):
            self.listLayers.append(ConvBlock(num_channels))

    def call(self, x):
        for layer in self.listLayers.layers:
            x = layer(x)
        return x
```

No exemplo a seguir,
definimos uma instância `DenseBlock` com 2 blocos de convolução de 10 canais de saída.
Ao usar uma entrada com 3 canais, obteremos uma saída com $3+2\times 10=23$ canais. O número de canais de bloco de convolução controla o crescimento do número de canais de saída em relação ao número de canais de entrada. Isso também é conhecido como *taxa de crescimento*.

```{.python .input}
blk = DenseBlock(2, 10)
blk.initialize()
X = np.random.uniform(size=(4, 3, 8, 8))
Y = blk(X)
Y.shape
```

```{.python .input}
#@tab pytorch
blk = DenseBlock(2, 3, 10)
X = torch.randn(4, 3, 8, 8)
Y = blk(X)
Y.shape
```

```{.python .input}
#@tab tensorflow
blk = DenseBlock(2, 10)
X = tf.random.uniform((4, 8, 8, 3))
Y = blk(X)
Y.shape
```

## Camadas de Transição

Uma vez que cada bloco denso aumentará o número de canais, adicionar muitos deles levará a um modelo excessivamente complexo. Uma *camada de transição* é usada para controlar a complexidade do modelo. Ele reduz o número de canais usando a camada convolucional $1\times 1$ e divide pela metade a altura e a largura da camada de pooling média com uma distância de 2, reduzindo ainda mais a complexidade do modelo.

```{.python .input}
def transition_block(num_channels):
    blk = nn.Sequential()
    blk.add(nn.BatchNorm(), nn.Activation('relu'),
            nn.Conv2D(num_channels, kernel_size=1),
            nn.AvgPool2D(pool_size=2, strides=2))
    return blk
```

```{.python .input}
#@tab pytorch
def transition_block(input_channels, num_channels):
    return nn.Sequential(
        nn.BatchNorm2d(input_channels), nn.ReLU(),
        nn.Conv2d(input_channels, num_channels, kernel_size=1),
        nn.AvgPool2d(kernel_size=2, stride=2))
```

```{.python .input}
#@tab tensorflow
class TransitionBlock(tf.keras.layers.Layer):
    def __init__(self, num_channels, **kwargs):
        super(TransitionBlock, self).__init__(**kwargs)
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()
        self.conv = tf.keras.layers.Conv2D(num_channels, kernel_size=1)
        self.avg_pool = tf.keras.layers.AvgPool2D(pool_size=2, strides=2)

    def call(self, x):
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.conv(x)
        return self.avg_pool(x)
```

Aplique uma camada de transição com 10 canais à saída do bloco denso no exemplo anterior. Isso reduz o número de canais de saída para 10 e divide a altura e a largura pela metade.

```{.python .input}
blk = transition_block(10)
blk.initialize()
blk(Y).shape
```

```{.python .input}
#@tab pytorch
blk = transition_block(23, 10)
blk(Y).shape
```

```{.python .input}
#@tab tensorflow
blk = TransitionBlock(10)
blk(Y).shape
```

## Modelo DenseNet 

A seguir, construiremos um modelo DenseNet. A DenseNet usa primeiro a mesma camada convolucional única e camada máxima de pooling que no ResNet.

```{.python .input}
net = nn.Sequential()
net.add(nn.Conv2D(64, kernel_size=7, strides=2, padding=3),
        nn.BatchNorm(), nn.Activation('relu'),
        nn.MaxPool2D(pool_size=3, strides=2, padding=1))
```

```{.python .input}
#@tab pytorch
b1 = nn.Sequential(
    nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
    nn.BatchNorm2d(64), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
```

```{.python .input}
#@tab tensorflow
def block_1():
    return tf.keras.Sequential([
       tf.keras.layers.Conv2D(64, kernel_size=7, strides=2, padding='same'),
       tf.keras.layers.BatchNormalization(),
       tf.keras.layers.ReLU(),
       tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')])
```

Então, semelhante aos quatro módulos compostos de blocos residuais que o ResNet usa,
A DenseNet usa quatro blocos densos.
Semelhante ao ResNet, podemos definir o número de camadas convolucionais usadas em cada bloco denso. Aqui, nós o definimos como 4, consistente com o modelo ResNet-18 em :numref:`sec_resnet`. Além disso, definimos o número de canais (ou seja, taxa de crescimento) para as camadas convolucionais no bloco denso para 32, de modo que 128 canais serão adicionados a cada bloco denso.

No ResNet, a altura e a largura são reduzidas entre cada módulo por um bloco residual com uma distância de 2. Aqui, usamos a camada de transição para reduzir pela metade a altura e a largura e pela metade o número de canais.

```{.python .input}
# `num_channels`: the current number of channels
num_channels, growth_rate = 64, 32
num_convs_in_dense_blocks = [4, 4, 4, 4]

for i, num_convs in enumerate(num_convs_in_dense_blocks):
    net.add(DenseBlock(num_convs, growth_rate))
    # This is the number of output channels in the previous dense block
    num_channels += num_convs * growth_rate
    # A transition layer that halves the number of channels is added between
    # the dense blocks
    if i != len(num_convs_in_dense_blocks) - 1:
        num_channels //= 2
        net.add(transition_block(num_channels))
```

```{.python .input}
#@tab pytorch
# `num_channels`: the current number of channels
num_channels, growth_rate = 64, 32
num_convs_in_dense_blocks = [4, 4, 4, 4]
blks = []
for i, num_convs in enumerate(num_convs_in_dense_blocks):
    blks.append(DenseBlock(num_convs, num_channels, growth_rate))
    # This is the number of output channels in the previous dense block
    num_channels += num_convs * growth_rate
    # A transition layer that halves the number of channels is added between
    # the dense blocks
    if i != len(num_convs_in_dense_blocks) - 1:
        blks.append(transition_block(num_channels, num_channels // 2))
        num_channels = num_channels // 2
```

```{.python .input}
#@tab tensorflow
def block_2():
    net = block_1()
    # `num_channels`: the current number of channels
    num_channels, growth_rate = 64, 32
    num_convs_in_dense_blocks = [4, 4, 4, 4]

    for i, num_convs in enumerate(num_convs_in_dense_blocks):
        net.add(DenseBlock(num_convs, growth_rate))
        # This is the number of output channels in the previous dense block
        num_channels += num_convs * growth_rate
        # A transition layer that halves the number of channels is added
        # between the dense blocks
        if i != len(num_convs_in_dense_blocks) - 1:
            num_channels //= 2
            net.add(TransitionBlock(num_channels))
    return net
```

Semelhante ao ResNet, uma camada de pooling global e uma camada totalmente conectada são conectadas na extremidade para produzir a saída.

```{.python .input}
net.add(nn.BatchNorm(),
        nn.Activation('relu'),
        nn.GlobalAvgPool2D(),
        nn.Dense(10))
```

```{.python .input}
#@tab pytorch
net = nn.Sequential(
    b1, *blks,
    nn.BatchNorm2d(num_channels), nn.ReLU(),
    nn.AdaptiveMaxPool2d((1, 1)),
    nn.Flatten(),
    nn.Linear(num_channels, 10))
```

```{.python .input}
#@tab tensorflow
def net():
    net = block_2()
    net.add(tf.keras.layers.BatchNormalization())
    net.add(tf.keras.layers.ReLU())
    net.add(tf.keras.layers.GlobalAvgPool2D())
    net.add(tf.keras.layers.Flatten())
    net.add(tf.keras.layers.Dense(10))
    return net
```

## Treinamento

Como estamos usando uma rede mais profunda aqui, nesta seção, reduziremos a altura e largura de entrada de 224 para 96 para simplificar o cálculo.

```{.python .input}
#@tab all
lr, num_epochs, batch_size = 0.1, 10, 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
```

## Sumário

* Em termos de conexões entre camadas, ao contrário do ResNet, onde entradas e saídas são adicionadas, o DenseNet concatena entradas e saídas na dimensão do canal.
* Os principais componentes que compõem o DenseNet são blocos densos e camadas de transição.
* Precisamos manter a dimensionalidade sob controle ao compor a rede, adicionando camadas de transição que reduzem o número de canais novamente.

## Exercícios

1. Por que usamos pooling médio em vez de pooling máximo na camada de transição?
1. Uma das vantagens mencionadas no artigo da DenseNet é que seus parâmetros de modelo são menores que os do ResNet. Por que isso acontece?
1. Um problema pelo qual a DenseNet foi criticada é o alto consumo de memória.
     1. Este é realmente o caso? Tente alterar a forma de entrada para $224\times 224$ para ver o consumo real de memória da GPU.
     1. Você consegue pensar em um meio alternativo de reduzir o consumo de memória? Como você precisa mudar a estrutura?
1. Implemente as várias versões da DenseNet apresentadas na Tabela 1 do artigo da DenseNet :cite:`Huang.Liu.Van-Der-Maaten.ea.2017`.
1. Projete um modelo baseado em MLP aplicando a ideia DenseNet. Aplique-o à tarefa de previsão do preço da habitação em :numref:`sec_kaggle_house`.
:begin_tab:`mxnet`

[Discussão](https://discuss.d2l.ai/t/87)
:end_tab:

:begin_tab:`pytorch`
[Discussão](https://discuss.d2l.ai/t/88)
:end_tab:

:begin_tab:`tensorflow`
[Discussão](https://discuss.d2l.ai/t/331)
:end_tab:
<!--stackedit_data:
eyJoaXN0b3J5IjpbMTU4NDQ5NzczMywxODU5NDAxOTcxXX0=
-->