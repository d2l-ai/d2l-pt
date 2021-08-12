# Linear Regression Implementation from Scratch
:label:`sec_linear_scratch`

Agora que você entende as principais ideias por trás da regressão linear,
podemos começar a trabalhar por meio de uma implementação prática no código.
Nesta seção, (**vamos implementar todo o método do zero,
incluindo o pipeline de dados, o modelo,
a função de perda e o otimizador de descida gradiente estocástico do minibatch.**)
Embora as estruturas modernas de *deep learning* possam automatizar quase todo esse trabalho,
implementar coisas do zero é a única maneira
para ter certeza de que você realmente sabe o que está fazendo.
Além disso, quando chega a hora de personalizar modelos,
definindo nossas próprias camadas ou funções de perda,
entender como as coisas funcionam nos bastidores será útil.
Nesta seção, contaremos apenas com tensores e diferenciação automática.
Posteriormente, apresentaremos uma implementação mais concisa,
aproveitando sinos e assobios de *frameworks* de *deep learning*.

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, np, npx
import random
npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
import random
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
import random
```

## Gerando o Dataset

Para manter as coisas simples, iremos [**construir um conjunto de dados artificial
de acordo com um modelo linear com ruído aditivo.**]
Nossa tarefa será recuperar os parâmetros deste modelo
usando o conjunto finito de exemplos contidos em nosso conjunto de dados.
Manteremos os dados em baixa dimensão para que possamos visualizá-los facilmente.
No seguinte *snippet* de código, geramos um conjunto de dados
contendo 1000 exemplos, cada um consistindo em 2 *features*
amostrado a partir de uma distribuição normal padrão.
Assim, nosso conjunto de dados sintético será uma matriz
$\mathbf{X}\in\mathbb{R}^{1000\times 2}$.

(**Os verdadeiros parâmetros que geram nosso conjunto de dados serão
$\mathbf{w} = [2, -3,4]^\top$ e $b = 4,2$,
e**) nossos rótulos sintéticos serão atribuídos de acordo
ao seguinte modelo linear com o termo de ruído $\epsilon$:

(**$$\mathbf{y}= \mathbf{X} \mathbf{w} + b + \mathbf\epsilon.$$**)

Você pode pensar em $\epsilon$ como um potencial de captura
erros de medição nos recursos e rótulos.
Vamos assumir que as premissas padrão são válidas e, portanto,
que $\epsilon$ obedece a uma distribuição normal com média 0.
Para tornar nosso problema mais fácil, definiremos seu desvio padrão em 0,01.
O código a seguir gera nosso conjunto de dados sintético.

```{.python .input}
#@tab mxnet, pytorch
def synthetic_data(w, b, num_examples):  #@save
    """Generate y = Xw + b + noise."""
    X = d2l.normal(0, 1, (num_examples, len(w)))
    y = d2l.matmul(X, w) + b
    y += d2l.normal(0, 0.01, y.shape)
    return X, d2l.reshape(y, (-1, 1))
```

```{.python .input}
#@tab tensorflow
def synthetic_data(w, b, num_examples):  #@save
    """Generate y = Xw + b + noise."""
    X = d2l.zeros((num_examples, w.shape[0]))
    X += tf.random.normal(shape=X.shape)
    y = d2l.matmul(X, tf.reshape(w, (-1, 1))) + b
    y += tf.random.normal(shape=y.shape, stddev=0.01)
    y = d2l.reshape(y, (-1, 1))
    return X, y
```

```{.python .input}
#@tab all
true_w = d2l.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)
```

Observe que [**cada linha em `features` consiste em um exemplo de dados bidimensionais
e que cada linha em `labels` consiste em um valor de rótulo unidimensional (um escalar).**]

```{.python .input}
#@tab all
print('features:', features[0],'\nlabel:', labels[0])
```

Ao gerar um gráfico de dispersão usando o segundo recurso `features [:, 1]` e `labels`,
podemos observar claramente a correlação linear entre os dois.

```{.python .input}
#@tab all
d2l.set_figsize()
# The semicolon is for displaying the plot only
d2l.plt.scatter(d2l.numpy(features[:, 1]), d2l.numpy(labels), 1);
```

## Lendo o *Dataset*


Lembre-se de que os modelos de treinamento consistem em
fazer várias passagens sobre o *dataset*,
pegando um *minibatch* de exemplos por vez,
e usando-los para atualizar nosso modelo.
Uma vez que este processo é tão fundamental
para treinar algoritmos de a*machine learning*,
vale a pena definir uma função de utilidade
para embaralhar o conjunto de dados e acessá-lo em *minibatches*.

No código a seguir, nós [**definimos a função `data_iter`**] (~~que~~)
para demonstrar uma possível implementação dessa funcionalidade.
A função (**leva um tamanho de amostra, uma matriz de *features*,
e um vetor de *labels*, produzindo *minibatches* do tamanho `batch_size`. **)
Cada *minibatch* consiste em uma tupla de *features* e *labels*.

```{.python .input}
#@tab mxnet, pytorch
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    # The examples are read at random, in no particular order
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = d2l.tensor(
            indices[i: min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]
```

```{.python .input}
#@tab tensorflow
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    # The examples are read at random, in no particular order
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        j = tf.constant(indices[i: min(i + batch_size, num_examples)])
        yield tf.gather(features, j), tf.gather(labels, j)
```


Em geral, queremos usar *minibatches* de tamanhos razoáveis
para aproveitar as vantagens do hardware da GPU,
que se destaca em operações de paralelização.
Porque cada exemplo pode ser alimentado por meio de nossos modelos em paralelo
e o gradiente da função de perda para cada exemplo também pode ser tomado em paralelo,
GPUs nos permitem processar centenas de exemplos em pouco mais tempo
do que pode demorar para processar apenas um único exemplo.

Para construir alguma intuição, vamos ler e imprimir
o primeiro pequeno lote de exemplos de dados.
A forma dos recursos em cada *minibatch* nos diz
o tamanho do *minibatch* e o número de recursos de entrada.
Da mesma forma, nosso *minibatch* de rótulos terá uma forma dada por `batch_size`.

```{.python .input}
#@tab all
batch_size = 10

for X, y in data_iter(batch_size, features, labels):
    print(X, '\n', y)
    break
```

Conforme executamos a iteração, obtemos *minibatches* distintos
sucessivamente até que todo o conjunto de dados se esgote (tente isto).
Embora a iteração implementada acima seja boa para fins didáticos,
é ineficiente de maneiras que podem nos colocar em apuros em problemas reais.
Por exemplo, requer que carreguemos todos os dados na memória
e que realizamos muitos acessos aleatórios à memória.
Os iteradores integrados implementados em uma estrutura de *deep learning*
são consideravelmente mais eficientes e podem lidar
com dados armazenados em arquivos e dados alimentados por meio de fluxos de dados.


## Initializing Model Parameters

[**Antes de começarmos a otimizar os parâmetros do nosso modelo**] por gradiente descendente estocástico de *minibatch*,
(**precisamos ter alguns parâmetros em primeiro lugar.**)
No código a seguir, inicializamos os pesos por amostragem
números aleatórios de uma distribuição normal com média 0
e um desvio padrão de 0,01, e definindo a tendência para 0.

```{.python .input}
w = np.random.normal(0, 0.01, (2, 1))
b = np.zeros(1)
w.attach_grad()
b.attach_grad()
```

```{.python .input}
#@tab pytorch
w = torch.normal(0, 0.01, size=(2,1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
```

```{.python .input}
#@tab tensorflow
w = tf.Variable(tf.random.normal(shape=(2, 1), mean=0, stddev=0.01),
                trainable=True)
b = tf.Variable(tf.zeros(1), trainable=True)
```


Depois de inicializar nossos parâmetros,
nossa próxima tarefa é atualizá-los até
eles se ajustam aos nossos dados suficientemente bem.
Cada atualização requer a obtenção do gradiente
da nossa função de perda no que diz respeito aos parâmetros.
Dado este gradiente, podemos atualizar cada parâmetro
na direção que pode reduzir a perda.

Uma vez que ninguém quer calcular gradientes explicitamente
(isso é entediante e sujeito a erros),
usamos diferenciação automática,
conforme apresentado em :numref:`sec_autograd`, para calcular o gradiente.


## Definindo o Modelo

Em seguida, devemos [**definir nosso modelo,
relacionando suas entradas e parâmetros com suas saídas.**]
Lembre-se de que, para calcular a saída do modelo linear,
simplesmente pegamos o produto escalar vetor-matriz
dos recursos de entrada $\mathbf{X}$ e os pesos do modelo $\mathbf{w}$,
e adicione o *offset* $b$ a cada exemplo.
Observe que abaixo de $\mathbf{Xw}$ está um vetor e $b$ é um escalar.
Lembre-se do mecanismo de transmissão conforme descrito em :numref:`subsec_broadcasting`.
Quando adicionamos um vetor e um escalar,
o escalar é adicionado a cada componente do vetor.

```{.python .input}
#@tab all
def linreg(X, w, b):  #@save
    """The linear regression model."""
    return d2l.matmul(X, w) + b
```

## Definindo a Função de Perda

Uma vez que [**atualizar nosso modelo requer tomar
o gradiente de nossa função de perda,**]
devemos (**definir a função de perda primeiro.**)
Aqui vamos usar a função de perda quadrada
conforme descrito em :numref:`sec_linear_regression`.
Na implementação, precisamos transformar o valor verdadeiro `y`
na forma do valor previsto `y_hat`.
O resultado retornado pela seguinte função
também terá a mesma forma de `y_hat`.

```{.python .input}
#@tab all
def squared_loss(y_hat, y):  #@save
    """Squared loss."""
    return (y_hat - d2l.reshape(y, y_hat.shape)) ** 2 / 2
```

## Definindo o Algoritmo de Otimização


Como discutimos em :numref:`sec_linear_regression`,
a regressão linear tem uma solução de forma fechada.
No entanto, este não é um livro sobre regressão linear:
é um livro sobre *deep learning*.
Uma vez que nenhum dos outros modelos que este livro apresenta
pode ser resolvido analiticamente, aproveitaremos esta oportunidade para apresentar seu primeiro exemplo de trabalho de gradiente descendente estocástico de *minibatch*.
[~~Apesar da regressão linear ter uma solução de forma fechada, outros modelos neste livro não têm. Aqui, introduzimos o gradiente descendente estocástico de *minibatch*~~]

Em cada etapa, usando um *minibatch* retirado aleatoriamente de nosso conjunto de dados,
vamos estimar o gradiente da perda em relação aos nossos parâmetros.
A seguir, vamos atualizar nossos parâmetros
na direção que pode reduzir a perda.
O código a seguir aplica a atualização da descida gradiente estocástica do *minibatch*,
dado um conjunto de parâmetros, uma taxa de aprendizagem e um tamanho de *batch*.
O tamanho da etapa de atualização é determinado pela taxa de aprendizagem `lr`.
Como nossa perda é calculada como a soma do *minibatch* de exemplos,
normalizamos o tamanho do nosso passo pelo tamanho do *batch* (`batch_size`),
de modo que a magnitude de um tamanho de passo típico
não depende muito de nossa escolha do tamanho do lote.

```{.python .input}
def sgd(params, lr, batch_size):  #@save
    """Minibatch stochastic gradient descent."""
    for param in params:
        param[:] = param - lr * param.grad / batch_size
```

```{.python .input}
#@tab pytorch
def sgd(params, lr, batch_size):  #@save
    """Minibatch stochastic gradient descent."""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()
```

```{.python .input}
#@tab tensorflow
def sgd(params, grads, lr, batch_size):  #@save
    """Minibatch stochastic gradient descent."""
    for param, grad in zip(params, grads):
        param.assign_sub(lr*grad/batch_size)
```

## Treinamento


Agora que temos todas as peças no lugar,
estamos prontos para [**implementar o *loop* de treinamento principal.**]
É crucial que você entenda este código
porque você verá loops de treinamento quase idênticos
repetidamente ao longo de sua carreira de *deep learning*.

Em cada iteração, pegaremos um *minibatch* de exemplos de treinamento,
e os passamos por nosso modelo para obter um conjunto de previsões.
Depois de calcular a perda, iniciamos a passagem para trás pela rede,
armazenando os gradientes em relação a cada parâmetro.
Finalmente, chamaremos o algoritmo de otimização de `sgd`
para atualizar os parâmetros do modelo.

Em resumo, vamos executar o seguinte loop:

* Inicializar parâmetros $(\mathbf{w}, b)$
* Repetir até terminar
    * Computar gradiente $\mathbf{g} \leftarrow \partial_{(\mathbf{w},b)} \frac{1}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} l(\mathbf{x}^{(i)}, y^{(i)}, \mathbf{w}, b)$
    * Atualizar parâmetros $(\mathbf{w}, b) \leftarrow (\mathbf{w}, b) - \eta \mathbf{g}$

Em cada *época*,
iremos iterar por todo o conjunto de dados
(usando a função `data_iter`) uma vez
passando por todos os exemplos no conjunto de dados de treinamento
(assumindo que o número de exemplos seja divisível pelo tamanho do lote).
O número de épocas `num_epochs` e a taxa de aprendizagem` lr` são hiperparâmetros,
que definimos aqui como 3 e 0,03, respectivamente.
Infelizmente, definir hiperparâmetros é complicado
e requer alguns ajustes por tentativa e erro.
Excluímos esses detalhes por enquanto, mas os revisamos
mais tarde em
:numref:`chap_optimization`.

```{.python .input}
#@tab all
lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss
```

```{.python .input}
for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        with autograd.record():
            l = loss(net(X, w, b), y)  # Minibatch loss in `X` and `y`
        # Because `l` has a shape (`batch_size`, 1) and is not a scalar
        # variable, the elements in `l` are added together to obtain a new
        # variable, on which gradients with respect to [`w`, `b`] are computed
        l.backward()
        sgd([w, b], lr, batch_size)  # Update parameters using their gradient
    train_l = loss(net(features, w, b), labels)
    print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')
```

```{.python .input}
#@tab pytorch
for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y)  # Minibatch loss in `X` and `y`
        # Compute gradient on `l` with respect to [`w`, `b`]
        l.sum().backward()
        sgd([w, b], lr, batch_size)  # Update parameters using their gradient
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')
```

```{.python .input}
#@tab tensorflow
for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        with tf.GradientTape() as g:
            l = loss(net(X, w, b), y)  # Minibatch loss in `X` and `y`
        # Compute gradient on l with respect to [`w`, `b`]
        dw, db = g.gradient(l, [w, b])
        # Update parameters using their gradient
        sgd([w, b], [dw, db], lr, batch_size)
    train_l = loss(net(features, w, b), labels)
    print(f'epoch {epoch + 1}, loss {float(tf.reduce_mean(train_l)):f}')
```

Neste caso, porque nós mesmos sintetizamos o conjunto de dados,
sabemos exatamente quais são os verdadeiros parâmetros.
Assim, podemos [**avaliar nosso sucesso no treinamento
comparando os parâmetros verdadeiros
com aqueles que aprendemos**] através de nosso ciclo de treinamento.
Na verdade, eles acabam sendo muito próximos um do outro.

```{.python .input}
#@tab all
print(f'error in estimating w: {true_w - d2l.reshape(w, true_w.shape)}')
print(f'error in estimating b: {true_b - b}')
```

Observe que não devemos tomar isso como garantido
que somos capazes de recuperar os parâmetros perfeitamente.
No entanto, no *machine learning*, normalmente estamos menos preocupados
com a recuperação de verdadeiros parâmetros subjacentes,
e mais preocupados com parâmetros que levam a previsões altamente precisas.
Felizmente, mesmo em problemas de otimização difíceis,
o gradiente descendente estocástico pode muitas vezes encontrar soluções notavelmente boas,
devido em parte ao fato de que, para redes profundas,
existem muitas configurações dos parâmetros
que levam a uma previsão altamente precisa.


## Resumo

* Vimos como uma rede profunda pode ser implementada e otimizada do zero, usando apenas tensores e diferenciação automática, sem a necessidade de definir camadas ou otimizadores sofisticados.
* Esta seção apenas arranha a superfície do que é possível. Nas seções a seguir, descreveremos modelos adicionais com base nos conceitos que acabamos de apresentar e aprenderemos como implementá-los de forma mais concisa.


## Exercícios

1. O que aconteceria se inicializássemos os pesos para zero. O algoritmo ainda funcionaria?
1. Suponha que você seja
    [Georg Simon Ohm](https://en.wikipedia.org/wiki/Georg_Ohm) tentando inventar um modelo entre tensão e corrente. Você poderia usar a diferenciação automática para aprender os parâmetros do seu modelo?
1. Você pode usar a [Lei de Planck](https://en.wikipedia.org/wiki/Planck%27s_law) para determinar a temperatura de um objeto usando densidade de energia espectral?
1. Quais são os problemas que você pode encontrar se quiser calcular as derivadas secundárias? Como você os consertaria?
1. Por que a função `reshape` é necessária na função` squared_loss`?
1. Experimente usar diferentes taxas de aprendizagem para descobrir a rapidez com que o valor da função de perda diminui.
1. Se o número de exemplos não pode ser dividido pelo tamanho do lote, o que acontece com o comportamento da função `data_iter`?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/42)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/43)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/201)
:end_tab:
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTMyODI5MzE5NSwtMTYzNTkwMDA0Myw4MT
EzMTUxOTQsMTYwOTI1NjAxLC0xMzQ1MjY5NzA5LDE0OTY1OTE3
MCwtMTk5MzU2NjkzLC05ODg5NjkwMTYsLTEyMjk2NzU0OTddfQ
==
-->