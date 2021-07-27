# Implementação de Perceptrons Multicamadas do Zero
:label:`sec_mlp_scratch`

Agora que caracterizamos
perceptrons multicamadas (MLPs) matematicamente,
vamos tentar implementar um nós mesmos. Para comparar com nossos resultados anteriores
alcançado com regressão *softmax*
(:numref:`sec_softmax_scratch`),
vamos continuar a trabalhar com
o conjunto de dados de classificação de imagens Fashion-MNIST
(:numref:`sec_fashion_mnist`).

```{.python .input}
from d2l import mxnet as d2l
from mxnet import gluon, np, npx
npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
```

```{.python .input}
#@tab all
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
```

## Inicializando os Parâmetros do Modelo


Lembre-se de que o Fashion-MNIST contém 10 classes,
e que cada imagem consiste em uma graded $28 \times 28 = 784$ de valores de pixel em tons de cinza.
Novamente, vamos desconsiderar a estrutura espacial
entre os pixels por enquanto,
então podemos pensar nisso simplesmente como um conjunto de dados de classificação
com 784 características de entrada e 10 classes.
Para começar, iremos [**implementar um MLP
com uma camada oculta e 256 unidades ocultas.**]
Observe que podemos considerar essas duas quantidades
como hiperparâmetros.
Normalmente, escolhemos larguras de camada em potências de 2,
que tendem a ser computacionalmente eficientes porque
de como a memória é alocada e endereçada no hardware.

Novamente, iremos representar nossos parâmetros com vários tensores.
Observe que *para cada camada*, devemos acompanhar
uma matriz de ponderação e um vetor de polarização.
Como sempre, alocamos memória
para os gradientes da perda com relação a esses parâmetros.

```{.python .input}
num_inputs, num_outputs, num_hiddens = 784, 10, 256

W1 = np.random.normal(scale=0.01, size=(num_inputs, num_hiddens))
b1 = np.zeros(num_hiddens)
W2 = np.random.normal(scale=0.01, size=(num_hiddens, num_outputs))
b2 = np.zeros(num_outputs)
params = [W1, b1, W2, b2]

for param in params:
    param.attach_grad()
```

```{.python .input}
#@tab pytorch
num_inputs, num_outputs, num_hiddens = 784, 10, 256

W1 = nn.Parameter(torch.randn(
    num_inputs, num_hiddens, requires_grad=True) * 0.01)
b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))
W2 = nn.Parameter(torch.randn(
    num_hiddens, num_outputs, requires_grad=True) * 0.01)
b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))

params = [W1, b1, W2, b2]
```

```{.python .input}
#@tab tensorflow
num_inputs, num_outputs, num_hiddens = 784, 10, 256

W1 = tf.Variable(tf.random.normal(
    shape=(num_inputs, num_hiddens), mean=0, stddev=0.01))
b1 = tf.Variable(tf.zeros(num_hiddens))
W2 = tf.Variable(tf.random.normal(
    shape=(num_hiddens, num_outputs), mean=0, stddev=0.01))
b2 = tf.Variable(tf.random.normal([num_outputs], stddev=.01))

params = [W1, b1, W2, b2]
```

## Função de Ativação

Para ter certeza de que sabemos como tudo funciona,
iremos [**implementar a ativação ReLU**] nós mesmos
usar a função máxima em vez de
invocar a função embutida `relu` diretamente.

```{.python .input}
def relu(X):
    return np.maximum(X, 0)
```

```{.python .input}
#@tab pytorch
def relu(X):
    a = torch.zeros_like(X)
    return torch.max(X, a)
```

```{.python .input}
#@tab tensorflow
def relu(X):
    return tf.math.maximum(X, 0)
```

## Modelo

Porque estamos desconsiderando a estrutura espacial,
nós `remodelamos` cada imagem bidimensional em
um vetor plano de comprimento `num_inputs`.
Finalmente, nós (**implementamos nosso modelo**)
com apenas algumas linhas de código.

```{.python .input}
def net(X):
    X = d2l.reshape(X, (-1, num_inputs))
    H = relu(np.dot(X, W1) + b1)
    return np.dot(H, W2) + b2
```

```{.python .input}
#@tab pytorch
def net(X):
    X = d2l.reshape(X, (-1, num_inputs))
    H = relu(X@W1 + b1)  # Here '@' stands for matrix multiplication
    return (H@W2 + b2)
```

```{.python .input}
#@tab tensorflow
def net(X):
    X = d2l.reshape(X, (-1, num_inputs))
    H = relu(tf.matmul(X, W1) + b1)
    return tf.matmul(H, W2) + b2
```

## Função de Perda

Para garantir estabilidade numérica,
e porque já implementamos
a função* softmax* do zero
(:numref:`sec_softmax_scratch`),
alavancamos a função integrada de APIs de alto nível
para calcular o *softmax* e a perda de entropia cruzada.
Lembre-se de nossa discussão anterior sobre essas complexidades
em :numref:`subsec_softmax-implementation-revisited`.
Nós encorajamos o leitor interessado
a examinar o código-fonte para a função de perda
para aprofundar seu conhecimento dos detalhes de implementação.

```{.python .input}
loss = gluon.loss.SoftmaxCrossEntropyLoss()
```

```{.python .input}
#@tab pytorch
loss = nn.CrossEntropyLoss()
```

```{.python .input}
#@tab tensorflow
def loss(y_hat, y):
    return tf.losses.sparse_categorical_crossentropy(
        y, y_hat, from_logits=True)
```

## Trainamento

Felizmente, [**o loop de treinamento para MLPs
é exatamente igual à regressão *softmax*.**]
Aproveitando o pacote `d2l` novamente,
chamamos a função `train_ch3`
(ver :numref:`sec_softmax_scratch`),
definindo o número de épocas para 10
e a taxa de aprendizagem para 0,1.

```{.python .input}
num_epochs, lr = 10, 0.1
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs,
              lambda batch_size: d2l.sgd(params, lr, batch_size))
```

```{.python .input}
#@tab pytorch
num_epochs, lr = 10, 0.1
updater = torch.optim.SGD(params, lr=lr)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)
```

```{.python .input}
#@tab tensorflow
num_epochs, lr = 10, 0.1
updater = d2l.Updater([W1, W2, b1, b2], lr)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)
```

Para avaliar o modelo aprendido,
nós [**aplicamos em alguns dados de teste**].

```{.python .input}
#@tab all
d2l.predict_ch3(net, test_iter)
```

## Resumo

* Vimos que implementar um MLP simples é fácil, mesmo quando feito manualmente.
* No entanto, com um grande número de camadas, implementar MLPs do zero ainda pode ser complicado (por exemplo, nomear e controlar os parâmetros do nosso modelo).


## Exercícios

1. Altere o valor do hiperparâmetro `num_hiddens` e veja como esse hiperparâmetro influencia seus resultados. Determine o melhor valor deste hiperparâmetro, mantendo todos os outros constantes.
1. Experimente adicionar uma camada oculta adicional para ver como isso afeta os resultados.
1. Como mudar a taxa de aprendizado altera seus resultados? Corrigindo a arquitetura do modelo e outros hiperparâmetros (incluindo o número de épocas), qual taxa de aprendizado oferece os melhores resultados?
1. Qual é o melhor resultado que você pode obter otimizando todos os hiperparâmetros (taxa de aprendizagem, número de épocas, número de camadas ocultas, número de unidades ocultas por camada) em conjunto?
1. Descreva por que é muito mais difícil lidar com vários hiperparâmetros.
1. Qual é a estratégia mais inteligente que você pode imaginar para estruturar uma pesquisa em vários hiperparâmetros?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/92)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/93)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/227)
:end_tab:
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTEyMjA0MjIzOTIsLTE2OTg0NTgxNDIsLT
EzOTM0NjY2NzIsMTUyOTMxNzE3NF19
-->