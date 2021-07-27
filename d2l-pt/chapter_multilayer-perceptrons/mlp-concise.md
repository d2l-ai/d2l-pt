# Implementação Concisa de *Perceptrons* Multicamadas
:label:`sec_mlp_concise`

As you might expect, by (**relying on the high-level APIs,
we can implement MLPs even more concisely.**)

```{.python .input}
from d2l import mxnet as d2l
from mxnet import gluon, init, npx
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
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
```

## Modelo

Em comparação com nossa implementação concisa
de implementação de regressão *softmax*
(:numref:`sec_softmax_concise`),
a única diferença é que adicionamos
*duas* camadas totalmente conectadas
(anteriormente, adicionamos *uma*).
A primeira é [**nossa camada oculta**],
que (**contém 256 unidades ocultas
e aplica a função de ativação ReLU**).
A segunda é nossa camada de saída.

```{.python .input}
net = nn.Sequential()
net.add(nn.Dense(256, activation='relu'),
        nn.Dense(10))
net.initialize(init.Normal(sigma=0.01))
```

```{.python .input}
#@tab pytorch
net = nn.Sequential(nn.Flatten(),
                    nn.Linear(784, 256),
                    nn.ReLU(),
                    nn.Linear(256, 10))

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights);
```

```{.python .input}
#@tab tensorflow
net = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(10)])
```

[**O loop de treinamento**] é exatamente o mesmo
como quando implementamos a regressão *softmax*.
Essa modularidade nos permite separar
questões relativas à arquitetura do modelo
a partir de considerações ortogonais.

```{.python .input}
batch_size, lr, num_epochs = 256, 0.1, 10
loss = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
```

```{.python .input}
#@tab pytorch
batch_size, lr, num_epochs = 256, 0.1, 10
loss = nn.CrossEntropyLoss()
trainer = torch.optim.SGD(net.parameters(), lr=lr)
```

```{.python .input}
#@tab tensorflow
batch_size, lr, num_epochs = 256, 0.1, 10
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
trainer = tf.keras.optimizers.SGD(learning_rate=lr)
```

```{.python .input}
#@tab all
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
```

## Resumo

* Usando APIs de alto nível, podemos implementar MLPs de forma muito mais concisa.
* Para o mesmo problema de classificação, a implementação de um MLP é a mesma da regressão *softmax*, exceto para camadas ocultas adicionais com funções de ativação.

## Exercícios

1. Tente adicionar diferentes números de camadas ocultas (você também pode modificar a taxa de aprendizagem). Qual configuração funciona melhor?
1. Experimente diferentes funções de ativação. Qual funciona melhor?
1. Experimente diferentes esquemas para inicializar os pesos. Qual método funciona melhor?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/94)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/95)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/262)
:end_tab:
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTU5MDk2OTMwN119
-->