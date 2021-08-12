# Redes Usando Blocos (VGG)
:label:`sec_vgg`

Enquanto AlexNet ofereceu evidências empíricas de que CNNs profundas
pode alcançar bons resultados, não forneceu um modelo geral
para orientar os pesquisadores subsequentes na concepção de novas redes.
Nas seções a seguir, apresentaremos vários conceitos heurísticos
comumente usado para projetar redes profundas.

O progresso neste campo reflete aquele no design de chips
onde os engenheiros deixaram de colocar transistores
para elementos lógicos para blocos lógicos.
Da mesma forma, o projeto de arquiteturas de rede neural
tornou-se progressivamente mais abstrato,
com pesquisadores deixando de pensar em termos de
neurônios individuais para camadas inteiras,
e agora para blocos, repetindo padrões de camadas.

A ideia de usar blocos surgiu pela primeira vez a partir do
[Grupo de Geometria Visual](http://www.robots.ox.ac.uk/~vgg/) (VGG)
na Universidade de Oxford,
em sua rede de mesmo nome *VGG*.
É fácil implementar essas estruturas repetidas no código
com qualquer estrutura moderna de aprendizado profundo usando loops e sub-rotinas.

## VGG Blocks

O bloco de construção básico das CNNs clássicas
é uma sequência do seguinte:
(i) uma camada convolucional
com preenchimento para manter a resolução,
(ii) uma não linearidade, como um ReLU,
(iii) uma camada de pooling tal
como uma camada de pooling máxima.
Um bloco VGG consiste em uma sequência de camadas convolucionais,
seguido por uma camada de *pooling* máxima para *downsampling* espacial.
No artigo VGG original :cite:`Simonyan.Zisserman.2014`,
Os autores
convoluções empregadas com $3 \times 3$ kernels com preenchimento de 1 (mantendo a altura e largura)
e $2 \times 2$ *pool* máximo com passo de 2
(reduzindo pela metade a resolução após cada bloco).
No código abaixo, definimos uma função chamada `vgg_block`
para implementar um bloco VGG.

: begin_tab: `mxnet, tensorflow`
A função leva dois argumentos
correspondendo ao número de camadas convolucionais `num_convs`
e o número de canais de saída `num_channels`.
: end_tab:

: begin_tab: `pytorch`
A função leva três argumentos correspondentes ao número
de camadas convolucionais `num_convs`, o número de canais de entrada `in_channels`
e o número de canais de saída `out_channels`.
: end_tab:

```{.python .input}
from d2l import mxnet as d2l
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()

def vgg_block(num_convs, num_channels):
    blk = nn.Sequential()
    for _ in range(num_convs):
        blk.add(nn.Conv2D(num_channels, kernel_size=3,
                          padding=1, activation='relu'))
    blk.add(nn.MaxPool2D(pool_size=2, strides=2))
    return blk
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn

def vgg_block(num_convs, in_channels, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels,
                                kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
    return nn.Sequential(*layers)
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf

def vgg_block(num_convs, num_channels):
    blk = tf.keras.models.Sequential()
    for _ in range(num_convs):
        blk.add(tf.keras.layers.Conv2D(num_channels,kernel_size=3,
                                    padding='same',activation='relu'))
    blk.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
    return blk
```

## Camadas VGG 

Como AlexNet e LeNet,
a rede VGG pode ser dividida em duas partes:
o primeiro consistindo principalmente de camadas convolucionais e de *pooling*
e a segunda consistindo em camadas totalmente conectadas.
Isso é descrito em :numref:`fig_vgg`.

![De AlexNet a VGG que é projetado a partir de blocos de construção.](../img/vgg.svg)
:width:`400px`
:label:`fig_vgg`

A parte convolucional da rede conecta vários blocos VGG de :numref:`fig_vgg` (também definido na função` vgg_block`)
em sucessão.
A seguinte variável `conv_arch` consiste em uma lista de tuplas (uma por bloco),
onde cada um contém dois valores: o número de camadas convolucionais
e o número de canais de saída,
quais são precisamente os argumentos necessários para chamar
a função `vgg_block`.
A parte totalmente conectada da rede VGG é idêntica à coberta no AlexNet.

A rede VGG original tinha 5 blocos convolucionais,
entre os quais os dois primeiros têm uma camada convolucional cada
e os três últimos contêm duas camadas convolucionais cada.
O primeiro bloco tem 64 canais de saída
e cada bloco subsequente dobra o número de canais de saída,
até que esse número chegue a 512.
Uma vez que esta rede usa 8 camadas convolucionais
e 3 camadas totalmente conectadas, geralmente chamado de VGG-11.

```{.python .input}
#@tab all
conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
```

O código a seguir implementa VGG-11. Esta é uma simples questão de executar um loop for sobre `conv_arch`.

```{.python .input}
def vgg(conv_arch):
    net = nn.Sequential()
    # A parte convolucional
    for (num_convs, num_channels) in conv_arch:
        net.add(vgg_block(num_convs, num_channels))
    # A parte totalmente conectada
    net.add(nn.Dense(4096, activation='relu'), nn.Dropout(0.5),
            nn.Dense(4096, activation='relu'), nn.Dropout(0.5),
            nn.Dense(10))
    return net

net = vgg(conv_arch)
```

```{.python .input}
#@tab pytorch
def vgg(conv_arch):
    conv_blks = []
    in_channels = 1
    # A parte convolucional
    for (num_convs, out_channels) in conv_arch:
        conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels

    return nn.Sequential(
        *conv_blks, nn.Flatten(),
        # A parte totalmente conectada
        nn.Linear(out_channels * 7 * 7, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 10))

net = vgg(conv_arch)
```

```{.python .input}
#@tab tensorflow
def vgg(conv_arch):
    net = tf.keras.models.Sequential()
    # A parte convolucional
    for (num_convs, num_channels) in conv_arch:
        net.add(vgg_block(num_convs, num_channels))
    # A parte totalmente conectada
    net.add(tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(4096, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(4096, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10)]))
    return net

net = vgg(conv_arch)
```

A seguir, construiremos um exemplo de dados de canal único
com altura e largura de 224 para observar a forma de saída de cada camada.

```{.python .input}
net.initialize()
X = np.random.uniform(size=(1, 1, 224, 224))
for blk in net:
    X = blk(X)
    print(blk.name, 'output shape:\t', X.shape)
```

```{.python .input}
#@tab pytorch
X = torch.randn(size=(1, 1, 224, 224))
for blk in net:
    X = blk(X)
    print(blk.__class__.__name__,'output shape:\t',X.shape)
```

```{.python .input}
#@tab tensorflow
X = tf.random.uniform((1, 224, 224, 1))
for blk in net.layers:
    X = blk(X)
    print(blk.__class__.__name__,'output shape:\t', X.shape)
```

Como você pode ver, dividimos a altura e a largura em cada bloco,
finalmente alcançando uma altura e largura de 7
antes de achatar as representações
para processamento pela parte totalmente conectada da rede.

## Treinamento

Como o VGG-11 é mais pesado em termos computacionais do que o AlexNet
construímos uma rede com um número menor de canais.
Isso é mais do que suficiente para o treinamento em Fashion-MNIST.

```{.python .input}
#@tab mxnet, pytorch
ratio = 4
small_conv_arch = [(pair[0], pair[1] // ratio) for pair in conv_arch]
net = vgg(small_conv_arch)
```

```{.python .input}
#@tab tensorflow
ratio = 4
small_conv_arch = [(pair[0], pair[1] // ratio) for pair in conv_arch]
# Lembre-se de que esta deve ser uma função que será passada para `d2l.train_ch6()`
# para que a construção/compilação do modelo precise estar dentro de `strategy.scope()` 
# a fim de utilizar os dispositivos CPU/GPU que temos
net = lambda: vgg(small_conv_arch)
```

Além de usar uma taxa de aprendizado um pouco maior,
o processo de treinamento do modelo é semelhante ao do AlexNet em :numref:`sec_alexnet`.

```{.python .input}
#@tab all
lr, num_epochs, batch_size = 0.05, 10, 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
```

## Sumário

* VGG-11 constrói uma rede usando blocos convolucionais reutilizáveis. Diferentes modelos de VGG podem ser definidos pelas diferenças no número de camadas convolucionais e canais de saída em cada bloco.
* O uso de blocos leva a representações muito compactas da definição da rede. Ele permite um projeto eficiente de redes complexas.
* Em seu artigo VGG, Simonyan e Ziserman experimentaram várias arquiteturas. Em particular, eles descobriram que várias camadas de convoluções profundas e estreitas (ou seja, $3 \times 3$) eram mais eficazes do que menos camadas de convoluções mais largas.

## Exercícios

1. Ao imprimir as dimensões das camadas, vimos apenas 8 resultados, em vez de 11. Para onde foram as informações das 3 camadas restantes?
2. Comparado com o AlexNet, o VGG é muito mais lento em termos de computação e também precisa de mais memória GPU. Analise as razões disso.
3. Tente alterar a altura e a largura das imagens no Fashion-MNIST de 224 para 96. Que influência isso tem nos experimentos?
4. Consulte a Tabela 1 no artigo VGG :cite:`Simonyan.Zisserman.2014` para construir outros modelos comuns, como VGG-16 ou VGG-19.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/77)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/78)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/277)
:end_tab:
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTIwMTk1NzExNzksMTgzMjAxMzg4OF19
-->