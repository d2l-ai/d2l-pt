# Implementação Concisa de Regressão Linear
:label:`sec_linear_concise`


Amplo e intenso interesse em *deep learning* nos últimos anos
inspiraram empresas, acadêmicos e amadores
para desenvolver uma variedade de estruturas de código aberto maduras
para automatizar o trabalho repetitivo de implementação
algoritmos de aprendizagem baseados em gradiente.
Em :numref:`sec_linear_scratch`, contamos apenas com
(i) tensores para armazenamento de dados e álgebra linear;
e (ii) auto diferenciação para cálculo de gradientes.
Na prática, porque iteradores de dados, funções de perda, otimizadores,
e camadas de rede neural
são tão comuns que as bibliotecas modernas também implementam esses componentes para nós.

Nesta seção, (**mostraremos como implementar
o modelo de regressão linear**) de:numref:`sec_linear_scratch`
(**de forma concisa, usando APIs de alto nível**) de estruturas de *deep learning*.


## Gerando the Dataset

Para começar, vamos gerar o mesmo conjunto de dados como em
:numref:`sec_linear_scratch`.

```{.python .input}
from d2l import mxnet as d2l
from mxnet import autograd, gluon, np, npx
npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import numpy as np
import torch
from torch.utils import data
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import numpy as np
import tensorflow as tf
```

```{.python .input}
#@tab all
true_w = d2l.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)
```

## Lendo o Dataset

Em vez de usar nosso próprio iterador,
podemos [**chamar a API existente em uma estrutura para ler os dados.**]
Passamos *`features`* e *`labels`* como argumentos e especificamos *`batch_size`*
ao instanciar um objeto iterador de dados.
Além disso, o valor booleano `is_train`
indica se ou não
queremos que o objeto iterador de dados embaralhe os dados
em cada época (passe pelo conjunto de dados).

```{.python .input}
def load_array(data_arrays, batch_size, is_train=True):  #@save
    """Construct a Gluon data iterator."""
    dataset = gluon.data.ArrayDataset(*data_arrays)
    return gluon.data.DataLoader(dataset, batch_size, shuffle=is_train)
```

```{.python .input}
#@tab pytorch
def load_array(data_arrays, batch_size, is_train=True):  #@save
    """Construct a PyTorch data iterator."""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)
```

```{.python .input}
#@tab tensorflow
def load_array(data_arrays, batch_size, is_train=True):  #@save
    """Construct a TensorFlow data iterator."""
    dataset = tf.data.Dataset.from_tensor_slices(data_arrays)
    if is_train:
        dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(batch_size)
    return dataset
```

```{.python .input}
#@tab all
batch_size = 10
data_iter = load_array((features, labels), batch_size)
```

Now we can use `data_iter` in much the same way as we called
the `data_iter` function in :numref:`sec_linear_scratch`.
To verify that it is working, we can read and print
the first minibatch of examples.
Comparing with :numref:`sec_linear_scratch`,
here we use `iter` to construct a Python iterator and use `next` to obtain the first item from the iterator.

```{.python .input}
#@tab all
next(iter(data_iter))
```

## Definindo o Modelo


Quando implementamos a regressão linear do zero
em :numref:`sec_linear_scratch`,
definimos nossos parâmetros de modelo explicitamente
e codificamos os cálculos para produzir saída
usando operações básicas de álgebra linear.
Você *deveria* saber como fazer isso.
Mas quando seus modelos ficam mais complexos,
e uma vez que você tem que fazer isso quase todos os dias,
você ficará feliz com a ajuda.
A situação é semelhante a codificar seu próprio blog do zero.
Fazer uma ou duas vezes é gratificante e instrutivo,
mas você seria um péssimo desenvolvedor da web
se toda vez que você precisava de um blog você passava um mês
reinventando tudo.

Para operações padrão, podemos [**usar as camadas predefinidas de uma estrutura,**]
o que nos permite focar especialmente
nas camadas usadas para construir o modelo
em vez de ter que se concentrar na implementação.
Vamos primeiro definir uma variável de modelo `net`,
que se refere a uma instância da classe `Sequential`.
A classe `Sequential` define um contêiner
para várias camadas que serão encadeadas.
Dados dados de entrada, uma instância `Sequential` passa por
a primeira camada, por sua vez passando a saída
como entrada da segunda camada e assim por diante.
No exemplo a seguir, nosso modelo consiste em apenas uma camada,
portanto, não precisamos realmente de `Sequencial`.
Mas como quase todos os nossos modelos futuros
envolverão várias camadas,
vamos usá-lo de qualquer maneira apenas para familiarizá-lo
com o fluxo de trabalho mais padrão.

Lembre-se da arquitetura de uma rede de camada única, conforme mostrado em :numref:`fig_single_neuron`.
Diz-se que a camada está *totalmente conectada*
porque cada uma de suas entradas está conectada a cada uma de suas saídas
por meio de uma multiplicação de matriz-vetor.

:begin_tab:`mxnet`
No Gluon, a camada totalmente conectada é definida na classe `Densa`.
Uma vez que queremos apenas gerar uma única saída escalar,
nós definimos esse número para 1.

É importante notar que, por conveniência,
Gluon não exige que especifiquemos
a forma de entrada para cada camada.
Então, aqui, não precisamos dizer ao Gluon
quantas entradas vão para esta camada linear.
Quando tentamos primeiro passar dados por meio de nosso modelo,
por exemplo, quando executamos `net (X)` mais tarde,
o Gluon irá inferir automaticamente o número de entradas para cada camada.
Descreveremos como isso funciona com mais detalhes posteriormente.
:end_tab:

: begin_tab: `pytorch`
No PyTorch, a camada totalmente conectada é definida na classe `Linear`. Observe que passamos dois argumentos para `nn.Linear`. O primeiro especifica a dimensão do recurso de entrada, que é 2, e o segundo é a dimensão do recurso de saída, que é um escalar único e, portanto, 1.
:end_tab:

:begin_tab:`tensorflow`
No Keras, a camada totalmente conectada é definida na classe `Dense`. Como queremos gerar apenas uma única saída escalar, definimos esse número como 1.

É importante notar que, por conveniência,
Keras não exige que especifiquemos
a forma de entrada para cada camada.
Então, aqui, não precisamos dizer a Keras
quantas entradas vão para esta camada linear.
Quando tentamos primeiro passar dados por meio de nosso modelo,
por exemplo, quando executamos `net (X)` mais tarde,
Keras inferirá automaticamente o número de entradas para cada camada.
Descreveremos como isso funciona com mais detalhes posteriormente.
:end_tab:

```{.python .input}
# `nn` is an abbreviation for neural networks
from mxnet.gluon import nn
net = nn.Sequential()
net.add(nn.Dense(1))
```

```{.python .input}
#@tab pytorch
# `nn` is an abbreviation for neural networks
from torch import nn
net = nn.Sequential(nn.Linear(2, 1))
```

```{.python .input}
#@tab tensorflow
# `keras` is the high-level API for TensorFlow
net = tf.keras.Sequential()
net.add(tf.keras.layers.Dense(1))
```

## Inicializando os Parâmetros do Modelo


Antes de usar `net`, precisamos (**inicializar os parâmetros do modelo,**)
como os pesos e *bias* no modelo de regressão linear.
As estruturas de *deep learning* geralmente têm uma maneira predefinida de inicializar os parâmetros.
Aqui especificamos que cada parâmetro de peso
deve ser amostrado aleatoriamente a partir de uma distribuição normal
com média 0 e desvio padrão 0,01.
O parâmetro bias será inicializado em zero.

:begin_tab:`mxnet`
Vamos importar o módulo *`initializer`* do MXNet.
Este módulo fornece vários métodos para inicialização de parâmetros do modelo.
Gluon disponibiliza `init` como um atalho (abreviatura)
para acessar o pacote `initializer`.
Nós apenas especificamos como inicializar o peso chamando `init.Normal (sigma = 0,01)`.
Os parâmetros de polarização são inicializados em zero por padrão.
:end_tab:

:begin_tab:`pytorch`
As we have specified the input and output dimensions when constructing `nn.Linear`. Now we access the parameters directly to specify their initial values. We first locate the layer by `net[0]`, which is the first layer in the network, and then use the `weight.data` and `bias.data` methods to access the parameters. Next we use the replace methods `normal_` and `fill_` to overwrite parameter values.
:end_tab:

:begin_tab:`tensorflow`
O módulo *`initializers`* no TensorFlow fornece vários métodos para a inicialização dos parâmetros do modelo. A maneira mais fácil de especificar o método de inicialização no Keras é ao criar a camada especificando *`kernel_initializer`*. Aqui, recriamos o `net` novamente.
:end_tab:

```{.python .input}
from mxnet import init
net.initialize(init.Normal(sigma=0.01))
```

```{.python .input}
#@tab pytorch
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)
```

```{.python .input}
#@tab tensorflow
initializer = tf.initializers.RandomNormal(stddev=0.01)
net = tf.keras.Sequential()
net.add(tf.keras.layers.Dense(1, kernel_initializer=initializer))
```

:begin_tab:`mxnet`
O código acima pode parecer simples, mas você deve observar
que algo estranho está acontecendo aqui.
Estamos inicializando parâmetros para uma rede
mesmo que Gluon ainda não saiba
quantas dimensões a entrada terá!
Pode ser 2 como em nosso exemplo ou pode ser 2.000.
Gluon nos permite fugir com isso porque, nos bastidores,
a inicialização é, na verdade, *adiada*.
A inicialização real ocorrerá apenas
quando tentamos, pela primeira vez, passar dados pela rede.
Apenas tome cuidado para lembrar que, uma vez que os parâmetros
ainda não foram inicializados,
não podemos acessá-los ou manipulá-los.
:end_tab:

:begin_tab:`pytorch`

:end_tab:

:begin_tab:`tensorflow`
O código acima pode parecer simples, mas você deve observar
que algo estranho está acontecendo aqui.
Estamos inicializando parâmetros para uma rede
mesmo que o Keras ainda não saiba
quantas dimensões a entrada terá!
Pode ser 2 como em nosso exemplo ou pode ser 2.000.
O Keras nos permite fugir do problema com isso porque, nos bastidores,
a inicialização é, na verdade, *adiada*.
A inicialização real ocorrerá apenas
quando tentamos, pela primeira vez, passar dados pela rede.
Apenas tome cuidado para lembrar que, uma vez que os parâmetros
ainda não foram inicializados,
não podemos acessá-los ou manipulá-los.
:end_tab:

## Definindo a Função de Perda

:begin_tab:`mxnet`
No Gluon, o módulo `loss` define várias funções de perda.
Neste exemplo, usaremos a implementação de perda quadrática do  Gluon (`L2Loss`).
:end_tab:

:begin_tab:`pytorch`
[**A classe `MSELoss` calcula o erro quadrático médio, também conhecido como norma $ L_2 $ quadrada.**]
Por padrão, ela retorna a perda média sobre os exemplos.
:end_tab:

:begin_tab:`tensorflow`
A classe `MeanSquaredError` calcula o erro quadrático médio, também conhecido como norma $L_2$ quadrada.
Por padrão, ela retorna a perda média sobre os exemplos.
:end_tab:

```{.python .input}
loss = gluon.loss.L2Loss()
```

```{.python .input}
#@tab pytorch
loss = nn.MSELoss()
```

```{.python .input}
#@tab tensorflow
loss = tf.keras.losses.MeanSquaredError()
```

## Definindo o Algoritmo de Otimização

:begin_tab:`mxnet`
O gradiente descendente estocástico de *minibatch* é uma ferramenta padrão
para otimizar redes neurais
e assim o Gluon o apoia ao lado de uma série de
variações desse algoritmo por meio de sua classe *`Trainer`*.
Quando instanciamos o *`Trainer`*,
iremos especificar os parâmetros para otimizar
(que pode ser obtido em nosso modelo `net` via` net.collect_params ()`),
o algoritmo de otimização que desejamos usar (`sgd`),
e um dicionário de hiperparâmetros
exigido por nosso algoritmo de otimização.
O gradiente descendente estocástico de *minibatch* requer apenas que definamos o valor *`learning_rate`*, que é definido como 0,03 aqui.
:end_tab:

:begin_tab:`pytorch`
O gradiente descendente estocástico de *minibatch* é uma ferramenta padrão
para otimizar redes neurais
e, portanto, PyTorch o suporta ao lado de uma série de
variações deste algoritmo no módulo `optim`.
Quando nós (**instanciamos uma instância `SGD`,**)
iremos especificar os parâmetros para otimizar
(podem ser obtidos de nossa rede via `net.parameters ()`), com um dicionário de hiperparâmetros
exigido por nosso algoritmo de otimização.
O gradiente descendente estocástico de *minibatch* requer apenas que definamos o valor `lr`, que é definido como 0,03 aqui.
:end_tab:

:begin_tab:`tensorflow`
O gradiente descendente estocástico de *minibatch* é uma ferramenta padrão
para otimizar redes neurais
e, portanto, Keras oferece suporte ao lado de uma série de
variações deste algoritmo no módulo `otimizadores`.
O gradiente descendente estocástico de *minibatch* requer apenas que
definamos o valor `learning_rate`, que é definido como 0,03 aqui.
:end_tab:

```{.python .input}
from mxnet import gluon
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.03})
```

```{.python .input}
#@tab pytorch
trainer = torch.optim.SGD(net.parameters(), lr=0.03)
```

```{.python .input}
#@tab tensorflow
trainer = tf.keras.optimizers.SGD(learning_rate=0.03)
```

## Treinamento


Você deve ter notado que expressar nosso modelo por meio
APIs de alto nível de uma estrutura de *deep learning*
requer comparativamente poucas linhas de código.
Não tivemos que alocar parâmetros individualmente,
definir nossa função de perda ou implementar o gradiente descendente estocástico de *minibatch*.
Assim que começarmos a trabalhar com modelos muito mais complexos,
as vantagens das APIs de alto nível aumentarão consideravelmente.
No entanto, uma vez que temos todas as peças básicas no lugar,
[**o loop de treinamento em si é surpreendentemente semelhante
ao que fizemos ao implementar tudo do zero.**]

Para refrescar sua memória: para anguns números de épocas,
faremos uma passagem completa sobre o conjunto de dados (*`train_data`*),
pegando iterativamente um *minibatch* de entradas
e os *labels* de verdade fundamental correspondentes.
Para cada *minibatch*, passamos pelo seguinte ritual:

* Gerar previsões chamando `net (X)` e calcular a perda `l` (a propagação direta).
* Calcular gradientes executando a retropropagação.
* Atualizar os parâmetros do modelo invocando nosso otimizador.

Para uma boa medida, calculamos a perda após cada época e a imprimimos para monitorar o progresso.

```{.python .input}
num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        with autograd.record():
            l = loss(net(X), y)
        l.backward()
        trainer.step(batch_size)
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l.mean().asnumpy():f}')
```

```{.python .input}
#@tab pytorch
num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X) ,y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')
```

```{.python .input}
#@tab tensorflow
num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        with tf.GradientTape() as tape:
            l = loss(net(X, training=True), y)
        grads = tape.gradient(l, net.trainable_variables)
        trainer.apply_gradients(zip(grads, net.trainable_variables))
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')
```

Abaixo, nós [**comparamos os parâmetros do modelo aprendidos pelo treinamento em dados finitos
e os parâmetros reais**] que geraram nosso *dataset*.
Para acessar os parâmetros,
primeiro acessamos a camada que precisamos de `net`
e, em seguida, acessamos os pesos e a polarização dessa camada.
Como em nossa implementação do zero,
observe que nossos parâmetros estimados são
perto de suas contrapartes verdadeiras.

```{.python .input}
w = net[0].weight.data()
print(f'error in estimating w: {true_w - d2l.reshape(w, true_w.shape)}')
b = net[0].bias.data()
print(f'error in estimating b: {true_b - b}')
```

```{.python .input}
#@tab pytorch
w = net[0].weight.data
print('error in estimating w:', true_w - d2l.reshape(w, true_w.shape))
b = net[0].bias.data
print('error in estimating b:', true_b - b)
```

```{.python .input}
#@tab tensorflow
w = net.get_weights()[0]
print('error in estimating w', true_w - d2l.reshape(w, true_w.shape))
b = net.get_weights()[1]
print('error in estimating b', true_b - b)
```

## Resumo

:begin_tab:`mxnet`
* Usando o Gluon, podemos implementar modelos de forma muito mais concisa.
* No Gluon, o módulo `data` fornece ferramentas para processamento de dados, o módulo` nn` define um grande número de camadas de rede neural e o módulo `loss` define muitas funções de perda comuns.
* O módulo `inicializador` do MXNet fornece vários métodos para inicialização dos parâmetros do modelo.
* A dimensionalidade e o armazenamento são inferidos automaticamente, mas tome cuidado para não tentar acessar os parâmetros antes de eles serem inicializados.
:end_tab:

:begin_tab:`pytorch`
* Usando as APIs de alto nível do PyTorch, podemos implementar modelos de forma muito mais concisa.
* No PyTorch, o módulo `data` fornece ferramentas para processamento de dados, o módulo` nn` define um grande número de camadas de rede neural e funções de perda comuns.
* Podemos inicializar os parâmetros substituindo seus valores por métodos que terminam com `_`.
:end_tab:

:begin_tab:`tensorflow`
* Usando as APIs de alto nível do TensorFlow, podemos implementar modelos de maneira muito mais concisa.
* No TensorFlow, o módulo `data` fornece ferramentas para processamento de dados, o módulo` keras` define um grande número de camadas de rede neural e funções de perda comuns.
* O módulo *`initializers`* do TensorFlow fornece vários métodos para a inicialização dos parâmetros do modelo.
* A dimensionalidade e o armazenamento são inferidos automaticamente (mas tome cuidado para não tentar acessar os parâmetros antes de serem inicializados).
:end_tab:

## Exercícios

:begin_tab:`mxnet`
1. Se substituirmos `l = loss (output, y)` por `l = loss (output, y).mean()`, precisamos alterar `trainer.step(batch_size)` para `trainer.step(1)`para que o código se comporte de forma idêntica. Por quê?
1. Revise a documentação do MXNet para ver quais funções de perda e métodos de inicialização são fornecidos nos módulos `gluon.loss` e` init`. Substitua a perda pela perda de Huber.
1. Como você acessa o gradiente de `dense.weight`?

[Discussions](https://discuss.d2l.ai/t/44)
:end_tab:

:begin_tab:`pytorch`
1. Se substituirmos `nn.MSELoss (*reduction* = 'sum')` por `nn.MSELoss ()`, como podemos alterar a taxa de aprendizagem para que o código se comporte de forma idêntica. Por quê?
1. Revise a documentação do PyTorch para ver quais funções de perda e métodos de inicialização são fornecidos. Substitua a perda pela perda de Huber.
1. Como você acessa o gradiente de `net[0].weight`?

[Discussions](https://discuss.d2l.ai/t/45)
:end_tab:

:begin_tab:`tensorflow`
1. Revise a documentação do TensorFlow para ver quais funções de perda e métodos de inicialização são fornecidos. Substitua a perda pela perda de Huber.

[Discussions](https://discuss.d2l.ai/t/204)
:end_tab:
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTUxMjc1MTc4MiwxMzgxNzE4MjMxLDE1MT
Y2Nzk5ODgsLTMwMDU2OTMzNSwtNTUxNzY3NTUsMzYzNjY2MzMs
LTY1Mjk5MTk1OCwtMjE0NTk5NTMwN119
-->