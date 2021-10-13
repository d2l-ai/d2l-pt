# O *Dataset* de Classificação de Imagens
:label:`sec_fashion_mnist`


(~~ O dataset MNIST, embora seja muito simples, é um dataset de referência e amplamente usado para classificação de imagens. Para os exemplos na sequencia usaremos o dataset Fashion-MNIST que é semelhante, mas um pouco mais complexo ~~)

Um dos *datasets* amplamente usados para classificação de imagens é o conjunto de dados MNIST :cite:`LeCun.Bottou.Bengio.ea.1998`.
Embora tenha tido uma boa execução como um conjunto de dados de referência,
mesmo os modelos simples pelos padrões atuais alcançam uma precisão de classificação acima de 95%,
tornando-o inadequado para distinguir entre modelos mais fortes e mais fracos.
Hoje, o MNIST serve mais como verificação de sanidade do que como referência.
Para aumentar um pouco a aposta, concentraremos nossa discussão nas próximas seções
no *dataset* Fashion-MNIST, qualitativamente semelhante, mas comparativamente complexo :cite:`Xiao.Rasul.Vollgraf.2017`, que foi lançado em 2017.

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import gluon
import sys

d2l.use_svg_display()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
import torchvision
from torchvision import transforms
from torch.utils import data

d2l.use_svg_display()
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf

d2l.use_svg_display()
```

## Lendo o Dataset

Nós podemos [**baixar e ler o *dataset* Fashion-MNIST na memória por meio das funções integradas na estrutura.**]

```{.python .input}
mnist_train = gluon.data.vision.FashionMNIST(train=True)
mnist_test = gluon.data.vision.FashionMNIST(train=False)
```

```{.python .input}
#@tab pytorch
# `ToTensor` converts the image data from PIL type to 32-bit floating point
# tensors. It divides all numbers by 255 so that all pixel values are between
# 0 and 1
trans = transforms.ToTensor()
mnist_train = torchvision.datasets.FashionMNIST(
    root="../data", train=True, transform=trans, download=True)
mnist_test = torchvision.datasets.FashionMNIST(
    root="../data", train=False, transform=trans, download=True)
```

```{.python .input}
#@tab tensorflow
mnist_train, mnist_test = tf.keras.datasets.fashion_mnist.load_data()
```

O Fashion-MNIST consiste em imagens de 10 categorias, cada uma representada
por 6.000 imagens no conjunto de dados de treinamento e por 1.000 no conjunto de dados de teste.
Um *dataset de teste* (ou *conjunto de teste*) é usado para avaliar o desempenho do modelo e não para treinamento.
Consequentemente, o conjunto de treinamento e o conjunto de teste
contém 60.000 e 10.000 imagens, respectivamente.

```{.python .input}
#@tab mxnet, pytorch
len(mnist_train), len(mnist_test)
```

```{.python .input}
#@tab tensorflow
len(mnist_train[0]), len(mnist_test[0])
```

A altura e a largura de cada imagem de entrada são 28 pixels.
Observe que o *dataset* consiste em imagens em tons de cinza, cujo número de canais é 1.
Para resumir, ao longo deste livro
armazenamos a forma de qualquer imagem com altura $h$ largura $w$ pixels como $h \times w$ or ($h$, $w$).

```{.python .input}
#@tab all
mnist_train[0][0].shape
```


[~~Duas funções utilitárias para visualizar o conjunto de dados~~]

As imagens no Fashion-MNIST estão associadas às seguintes categorias:
t-shirt, calças, pulôver, vestido, casaco, sandália, camisa, tênis, bolsa e bota.
A função a seguir converte entre índices de rótulos numéricos e seus nomes em texto.

```{.python .input}
#@tab all
def get_fashion_mnist_labels(labels):  #@save
    """Return text labels for the Fashion-MNIST dataset."""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]
```

Agora podemos criar uma função para visualizar esses exemplos.

```{.python .input}
#@tab mxnet, tensorflow
def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):  #@save
    """Plot a list of images."""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        ax.imshow(d2l.numpy(img))
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes
```

```{.python .input}
#@tab pytorch
def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):  #@save
    """Plot a list of images."""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            # Tensor Image
            ax.imshow(img.numpy())
        else:
            # PIL Image
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes
```

Aqui estão [**as imagens e seus *labels correspondentes**] (no texto)
para os primeiros exemplos no *dataset* de treinamento.

```{.python .input}
X, y = mnist_train[:18]

print(X.shape)
show_images(X.squeeze(axis=-1), 2, 9, titles=get_fashion_mnist_labels(y));
```

```{.python .input}
#@tab pytorch
X, y = next(iter(data.DataLoader(mnist_train, batch_size=18)))
show_images(X.reshape(18, 28, 28), 2, 9, titles=get_fashion_mnist_labels(y));
```

```{.python .input}
#@tab tensorflow
X = tf.constant(mnist_train[0][:18])
y = tf.constant(mnist_train[1][:18])
show_images(X, 2, 9, titles=get_fashion_mnist_labels(y));
```

## Lendo um *Minibatch*

Para tornar nossa vida mais fácil ao ler os conjuntos de treinamento e teste,
usamos o iterador de dados integrado em vez de criar um do zero.
Lembre-se de que a cada iteração, um carregador de dados
[**lê um minibatch de dados com tamanho `batch_size` cada vez.**]
Também misturamos aleatoriamente os exemplos para o iterador de dados de treinamento.

```{.python .input}
batch_size = 256

def get_dataloader_workers():  #@save
    """Use 4 processes to read the data except for Windows."""
    return 0 if sys.platform.startswith('win') else 4

# `ToTensor` converts the image data from uint8 to 32-bit floating point. It
# divides all numbers by 255 so that all pixel values are between 0 and 1
transformer = gluon.data.vision.transforms.ToTensor()
train_iter = gluon.data.DataLoader(mnist_train.transform_first(transformer),
                                   batch_size, shuffle=True,
                                   num_workers=get_dataloader_workers())
```

```{.python .input}
#@tab pytorch
batch_size = 256

def get_dataloader_workers():  #@save
    """Use 4 processes to read the data."""
    return 4

train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True,
                             num_workers=get_dataloader_workers())
```

```{.python .input}
#@tab tensorflow
batch_size = 256
train_iter = tf.data.Dataset.from_tensor_slices(
    mnist_train).batch(batch_size).shuffle(len(mnist_train[0]))
```

Vejamos o tempo que leva para ler os dados de treinamento.

```{.python .input}
#@tab all
timer = d2l.Timer()
for X, y in train_iter:
    continue
f'{timer.stop():.2f} sec'
```

## Juntando Tudo

Agora definimos [**a função `load_data_fashion_mnist`
que obtém e lê o *dataset* Fashion-MNIST.**]
Ele retorna os iteradores de dados para o conjunto de treinamento e o conjunto de validação.
Além disso, ele aceita um argumento opcional para redimensionar imagens para outra forma.

```{.python .input}
def load_data_fashion_mnist(batch_size, resize=None):  #@save
    """Download the Fashion-MNIST dataset and then load it into memory."""
    dataset = gluon.data.vision
    trans = [dataset.transforms.ToTensor()]
    if resize:
        trans.insert(0, dataset.transforms.Resize(resize))
    trans = dataset.transforms.Compose(trans)
    mnist_train = dataset.FashionMNIST(train=True).transform_first(trans)
    mnist_test = dataset.FashionMNIST(train=False).transform_first(trans)
    return (gluon.data.DataLoader(mnist_train, batch_size, shuffle=True,
                                  num_workers=get_dataloader_workers()),
            gluon.data.DataLoader(mnist_test, batch_size, shuffle=False,
                                  num_workers=get_dataloader_workers()))
```

```{.python .input}
#@tab pytorch
def load_data_fashion_mnist(batch_size, resize=None):  #@save
    """Download the Fashion-MNIST dataset and then load it into memory."""
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
        root="../data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(
        root="../data", train=False, transform=trans, download=True)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True,
                            num_workers=get_dataloader_workers()),
            data.DataLoader(mnist_test, batch_size, shuffle=False,
                            num_workers=get_dataloader_workers()))
```

```{.python .input}
#@tab tensorflow
def load_data_fashion_mnist(batch_size, resize=None):   #@save
    """Download the Fashion-MNIST dataset and then load it into memory."""
    mnist_train, mnist_test = tf.keras.datasets.fashion_mnist.load_data()
    # Divide all numbers by 255 so that all pixel values are between
    # 0 and 1, add a batch dimension at the last. And cast label to int32
    process = lambda X, y: (tf.expand_dims(X, axis=3) / 255,
                            tf.cast(y, dtype='int32'))
    resize_fn = lambda X, y: (
        tf.image.resize_with_pad(X, resize, resize) if resize else X, y)
    return (
        tf.data.Dataset.from_tensor_slices(process(*mnist_train)).batch(
            batch_size).shuffle(len(mnist_train[0])).map(resize_fn),
        tf.data.Dataset.from_tensor_slices(process(*mnist_test)).batch(
            batch_size).map(resize_fn))
```

Abaixo testamos o recurso de redimensionamento de imagem da função `load_data_fashion_mnist`
especificando o argumento `resize`.

```{.python .input}
#@tab all
train_iter, test_iter = load_data_fashion_mnist(32, resize=64)
for X, y in train_iter:
    print(X.shape, X.dtype, y.shape, y.dtype)
    break
```

Agora estamos prontos para trabalhar com o *dataset* Fashion-MNIST nas seções a seguir.

## Resumo

* Fashion-MNIST é um *dataset* de classificação de vestuário que consiste em imagens que representam 10 categorias. Usaremos esse conjunto de dados nas seções e capítulos subsequentes para avaliar vários algoritmos de classificação.
* Armazenamos a forma de qualquer imagem com altura $h$ largura $w$ pixels como $h \times w$ or ($h$, $w$).
* Os iteradores de dados são um componente chave para um desempenho eficiente. Conte com iteradores de dados bem implementados que exploram a computação de alto desempenho para evitar desacelerar o ciclo de treinamento.


## Exercícios

1. A redução de `batch_size` (por exemplo, para 1) afeta o desempenho de leitura?
1. O desempenho do iterador de dados é importante. Você acha que a implementação atual é rápida o suficiente? Explore várias opções para melhorá-lo.
1. Verifique a documentação online da API do *framework*. Quais outros conjuntos de dados estão disponíveis?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/48)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/49)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/224)
:end_tab:
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTE1NDYwOTU2MDUsMjE0ODM2OTIxXX0=
-->
