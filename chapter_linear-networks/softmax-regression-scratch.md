# Implementação da Regressão *Softmax*  do Zero
:label:`sec_softmax_scratch`

(**Assim como implementamos a regressão linear do zero, acreditamos que**)
regressão *softmax*
é igualmente fundamental e
(**você deve saber os detalhes sangrentos de**)
(~~*regressão softmax*~~)
como implementá-lo sozinho.
Vamos trabalhar com o *dataset* Fashion-MNIST, recém-introduzido em :numref:`sec_fashion_mnist`,
configurando um iterador de dados com *batch size* 256.

```{.python .input}
from d2l import mxnet as d2l
from mxnet import autograd, np, npx, gluon
from IPython import display
npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from IPython import display
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
from IPython import display
```

```{.python .input}
#@tab all
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
```

## Inicializando os Parâmetros do Modelo


Como em nosso exemplo de regressão linear,
cada exemplo aqui será representado por um vetor de comprimento fixo.
Cada exemplo no conjunto de dados bruto é uma imagem $28 \times 28$.
Nesta seção, [**vamos nivelar cada imagem,
tratando-os como vetores de comprimento 784.**]
No futuro, falaremos sobre estratégias mais sofisticadas
para explorar a estrutura espacial em imagens,
mas, por enquanto, tratamos cada localização de pixel como apenas outro recurso.

Lembre-se de que na regressão *softmax*,
temos tantas saídas quanto classes.
(**Como nosso conjunto de dados tem 10 classes,
nossa rede terá uma dimensão de saída de 10.**)
Consequentemente, nossos pesos constituirão uma matriz $784 \times 10$
e os *bias* constituirão um vetor-linha $1 \times 10$.
Tal como acontece com a regressão linear, vamos inicializar nossos pesos `W`
com ruído Gaussiano e nossos *bias* com o valor inicial 0.
```{.python .input}
num_inputs = 784
num_outputs = 10

W = np.random.normal(0, 0.01, (num_inputs, num_outputs))
b = np.zeros(num_outputs)
W.attach_grad()
b.attach_grad()
```

```{.python .input}
#@tab pytorch
num_inputs = 784
num_outputs = 10

W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)
```

```{.python .input}
#@tab tensorflow
num_inputs = 784
num_outputs = 10

W = tf.Variable(tf.random.normal(shape=(num_inputs, num_outputs),
                                 mean=0, stddev=0.01))
b = tf.Variable(tf.zeros(num_outputs))
```

## Definindo a Operação do *Softmax*

Antes de implementar o modelo de regressão do *softmax*,
vamos revisar brevemente como o operador de soma funciona
ao longo de dimensões específicas em um tensor,
conforme discutido em: numref :numref:`subseq_lin-alg-reduction` e :numref:`subseq_lin-alg-non-reduction`.
[**Dada uma matriz `X`, podemos somar todos os elementos (por padrão) ou apenas
sobre elementos no mesmo eixo,**]
ou seja, a mesma coluna (eixo 0) ou a mesma linha (eixo 1).
Observe que se `X` é um tensor com forma (2, 3)
e somamos as colunas,
o resultado será um vetor com forma (3,).
Ao invocar o operador de soma,
podemos especificar para manter o número de eixos no tensor original,
em vez de reduzir a dimensão que resumimos.
Isso resultará em um tensor bidimensional com forma (1, 3).

```{.python .input}
#@tab pytorch
X = d2l.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
d2l.reduce_sum(X, 0, keepdim=True), d2l.reduce_sum(X, 1, keepdim=True)
```

```{.python .input}
#@tab mxnet, tensorflow
X = d2l.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
d2l.reduce_sum(X, 0, keepdims=True), d2l.reduce_sum(X, 1, keepdims=True)
```

Agora estamos prontos para (**implementar a operação do *softmax* **).
Lembre-se de que o *softmax* consiste em três etapas:
i) exponenciamos cada termo (usando `exp`);
ii) somamos cada linha (temos uma linha por exemplo no lote)
para obter a constante de normalização para cada exemplo;
iii) dividimos cada linha por sua constante de normalização,
garantindo que o resultado seja 1.
Antes de olhar para o código, vamos lembrar
como isso parece, expresso como uma equação:

(**
$$\mathrm{softmax}(\mathbf{X})_{ij} = \frac{\exp(\mathbf{X}_{ij})}{\sum_k \exp(\mathbf{X}_{ik})}.$$
**)

O denominador, ou constante de normalização,
às vezes também é chamada de *função de partição*
(e seu logaritmo é chamado de função de partição de log).
As origens desse nome estão em [física estatística](https://en.wikipedia.org/wiki/Partition_function_ (estatística_mecânica))
onde uma equação relacionada modela a distribuição
sobre um conjunto de partículas.

```{.python .input}
#@tab mxnet, tensorflow
def softmax(X):
    X_exp = d2l.exp(X)
    partition = d2l.reduce_sum(X_exp, 1, keepdims=True)
    return X_exp / partition  # The broadcasting mechanism is applied here
```

```{.python .input}
#@tab pytorch
def softmax(X):
    X_exp = d2l.exp(X)
    partition = d2l.reduce_sum(X_exp, 1, keepdim=True)
    return X_exp / partition  # The broadcasting mechanism is applied here
```

Como você pode ver, para qualquer entrada aleatória,
[**transformamos cada elemento em um número não negativo.
Além disso, cada linha soma 1,**]
como é necessário para uma probabilidade.

```{.python .input}
#@tab mxnet, pytorch
X = d2l.normal(0, 1, (2, 5))
X_prob = softmax(X)
X_prob, d2l.reduce_sum(X_prob, 1)
```

```{.python .input}
#@tab tensorflow
X = tf.random.normal((2, 5), 0, 1)
X_prob = softmax(X)
X_prob, tf.reduce_sum(X_prob, 1)
```

Observe que embora pareça correto matematicamente,
fomos um pouco desleixados em nossa implementação
porque falhamos em tomar precauções contra estouro numérico ou estouro negativo
devido a elementos grandes ou muito pequenos da matriz.

## Definindo o Modelo

Agora que definimos a operação do *softmax*,
podemos [**implementar o modelo de regressão softmax.**]
O código a seguir define como a entrada é mapeada para a saída por meio da rede.
Observe que achatamos cada imagem original no lote
em um vetor usando a função `reshape`
antes de passar os dados pelo nosso modelo.

```{.python .input}
#@tab all
def net(X):
    return softmax(d2l.matmul(d2l.reshape(X, (-1, W.shape[0])), W) + b)
```

## Definindo a Função de Perda


Em seguida, precisamos implementar a função de perda de entropia cruzada,
conforme apresentado em :numref:`sec_softmax`.
Esta pode ser a função de perda mais comum
em todo o *deep learning* porque, no momento,
os problemas de classificação superam em muito os problemas de regressão.

Lembre-se de que a entropia cruzada leva a *log-likelihood* negativa
da probabilidade prevista atribuída ao rótulo verdadeiro.
Em vez de iterar as previsões com um *loop for* Python
(que tende a ser ineficiente),
podemos escolher todos os elementos por um único operador.
Abaixo, nós [**criamos dados de amostra `y_hat`
com 2 exemplos de probabilidades previstas em 3 classes e seus rótulos correspondentes `y`.**]
Com `y` sabemos que no primeiro exemplo a primeira classe é a previsão correta e
no segundo exemplo, a terceira classe é a verdade fundamental.
[**Usando `y` como os índices das probabilidades em` y_hat`,**]
escolhemos a probabilidade da primeira classe no primeiro exemplo
e a probabilidade da terceira classe no segundo exemplo.

```{.python .input}
#@tab mxnet, pytorch
y = d2l.tensor([0, 2])
y_hat = d2l.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
y_hat[[0, 1], y]
```

```{.python .input}
#@tab tensorflow
y_hat = tf.constant([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
y = tf.constant([0, 2])
tf.boolean_mask(y_hat, tf.one_hot(y, depth=y_hat.shape[-1]))
```

Agora podemos (**implementar a função de perda de entropia cruzada**) de forma eficiente com apenas uma linha de código.

```{.python .input}
#@tab mxnet, pytorch
def cross_entropy(y_hat, y):
    return - d2l.log(y_hat[range(len(y_hat)), y])

cross_entropy(y_hat, y)
```

```{.python .input}
#@tab tensorflow
def cross_entropy(y_hat, y):
    return -tf.math.log(tf.boolean_mask(
        y_hat, tf.one_hot(y, depth=y_hat.shape[-1])))

cross_entropy(y_hat, y)
```

## Exatidão da Classificação


Dada a distribuição de probabilidade prevista `y_hat`,
normalmente escolhemos a classe com a maior probabilidade prevista
sempre que a previsão que devemos produzir é difícil.
Na verdade, muitos aplicativos exigem que façamos uma escolha.
O Gmail deve categorizar um e-mail em "Principal", "Social", "Atualizações" ou "Fóruns".
Pode estimar probabilidades internamente,
mas no final do dia ele tem que escolher uma das classes.

Quando as previsões são consistentes com a classe de *label* `y`, elas estão corretas.
A precisão da classificação é a fração de todas as previsões corretas.
Embora possa ser difícil otimizar a precisão diretamente (não é diferenciável),
muitas vezes é a medida de desempenho que mais nos preocupa,
e quase sempre o relataremos ao treinar classificadores.

Para calcular a precisão, fazemos o seguinte.
Primeiro, se `y_hat` é uma matriz,
presumimos que a segunda dimensão armazena pontuações de predição para cada classe.
Usamos `argmax` para obter a classe prevista pelo índice para a maior entrada em cada linha.
Em seguida, [**comparamos a classe prevista com a verdade fundamental `y` elemento a elemento.**]
Uma vez que o operador de igualdade `==` é sensível aos tipos de dados,
convertemos o tipo de dados de `y_hat` para corresponder ao de` y`.
O resultado é um tensor contendo entradas de 0 (falso) e 1 (verdadeiro).
Tirar a soma resulta no número de previsões corretas.

```{.python .input}
#@tab all
def accuracy(y_hat, y):  #@save
    """Compute the number of correct predictions."""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = d2l.argmax(y_hat, axis=1)
    cmp = d2l.astype(y_hat, y.dtype) == y
    return float(d2l.reduce_sum(d2l.astype(cmp, y.dtype)))
```

Continuaremos a usar as variáveis `y_hat` e` y`
definidas antes
como as distribuições de probabilidade e *labels* previstos, respectivamente.
Podemos ver que a classe prevista no primeiro exemplo é 2
(o maior elemento da linha é 0,6 com o índice 2),
que é inconsistente com o rótulo real, 0.
A classe prevista do segundo exemplo é 2
(o maior elemento da linha é 0,5 com o índice de 2),
que é consistente com o rótulo real, 2.
Portanto, a taxa de precisão da classificação para esses dois exemplos é 0,5.

```{.python .input}
#@tab all
accuracy(y_hat, y) / len(y)
```

[**Da mesma forma, podemos avaliar a precisão da `rede` de qualquer modelo em um conjunto de dados**]
que é acessado por meio do iterador de dados `data_iter`.

```{.python .input}
#@tab mxnet, tensorflow
def evaluate_accuracy(net, data_iter):  #@save
    """Compute the accuracy for a model on a dataset."""
    metric = Accumulator(2)  # No. of correct predictions, no. of predictions
    for X, y in data_iter:
        metric.add(accuracy(net(X), y), d2l.size(y))
    return metric[0] / metric[1]
```

```{.python .input}
#@tab pytorch
def evaluate_accuracy(net, data_iter):  #@save
    """Compute the accuracy for a model on a dataset."""
    if isinstance(net, torch.nn.Module):
        net.eval()  # Set the model to evaluation mode
    metric = Accumulator(2)  # No. of correct predictions, no. of predictions
    for X, y in data_iter:
        metric.add(accuracy(net(X), y), d2l.size(y))
    return metric[0] / metric[1]
```

Aqui, `Accumulator` é uma classe utilitária para acumular somas sobre múltiplas variáveis.
Na função `evaluate_accuracy` acima,
criamos 2 variáveis na instância `Accumulator` para armazenar ambos
o número de previsões corretas e o número de previsões, respectivamente.
Ambos serão acumulados ao longo do tempo à medida que iteramos no conjunto de dados.

```{.python .input}
#@tab all
class Accumulator:  #@save
    """For accumulating sums over `n` variables."""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
```

[**PComo inicializamos o modelo `net` com pesos aleatórios,
a precisão deste modelo deve ser próxima à aleatoriedade,**]
ou seja, 0,1 para 10 classes.

```{.python .input}
#@tab all
evaluate_accuracy(net, test_iter)
```

## Treinamento

[**O *loop* de treinamento**]
para regressão *softmax* deve ser extremamente familiar
se você ler nossa implementação
de regressão linear em :numref:`sec_linear_scratch`.
Aqui, nós refatoramos a implementação para torná-la reutilizável.
Primeiro, definimos uma função para treinar por uma época.
Observe que `updater` é uma função geral para atualizar os parâmetros do modelo,
que aceita o tamanho do lote como argumento.
Pode ser um *wrapper* da função `d2l.sgd`
ou a função de otimização integrada de uma estrutura.

```{.python .input}
def train_epoch_ch3(net, train_iter, loss, updater):  #@save
    """Train a model within one epoch (defined in Chapter 3)."""
    # Sum of training loss, sum of training accuracy, no. of examples
    metric = Accumulator(3)
    if isinstance(updater, gluon.Trainer):
        updater = updater.step
    for X, y in train_iter:
        # Compute gradients and update parameters
        with autograd.record():
            y_hat = net(X)
            l = loss(y_hat, y)
        l.backward()
        updater(X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), y.size)
    # Return training loss and training accuracy
    return metric[0] / metric[2], metric[1] / metric[2]
```

```{.python .input}
#@tab pytorch
def train_epoch_ch3(net, train_iter, loss, updater):  #@save
    """The training loop defined in Chapter 3."""
    # Set the model to training mode
    if isinstance(net, torch.nn.Module):
        net.train()
    # Sum of training loss, sum of training accuracy, no. of examples
    metric = Accumulator(3)
    for X, y in train_iter:
        # Compute gradients and update parameters
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            # Using PyTorch in-built optimizer & loss criterion
            updater.zero_grad()
            l.backward()
            updater.step()
            metric.add(float(l) * len(y), accuracy(y_hat, y),
                       y.size().numel())
        else:
            # Using custom built optimizer & loss criterion
            l.sum().backward()
            updater(X.shape[0])
            metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    # Return training loss and training accuracy
    return metric[0] / metric[2], metric[1] / metric[2]
```

```{.python .input}
#@tab tensorflow
def train_epoch_ch3(net, train_iter, loss, updater):  #@save
    """The training loop defined in Chapter 3."""
    # Sum of training loss, sum of training accuracy, no. of examples
    metric = Accumulator(3)
    for X, y in train_iter:
        # Compute gradients and update parameters
        with tf.GradientTape() as tape:
            y_hat = net(X)
            # Keras implementations for loss takes (labels, predictions)
            # instead of (predictions, labels) that users might implement
            # in this book, e.g. `cross_entropy` that we implemented above
            if isinstance(loss, tf.keras.losses.Loss):
                l = loss(y, y_hat)
            else:
                l = loss(y_hat, y)
        if isinstance(updater, tf.keras.optimizers.Optimizer):
            params = net.trainable_variables
            grads = tape.gradient(l, params)
            updater.apply_gradients(zip(grads, params))
        else:
            updater(X.shape[0], tape.gradient(l, updater.params))
        # Keras loss by default returns the average loss in a batch
        l_sum = l * float(tf.size(y)) if isinstance(
            loss, tf.keras.losses.Loss) else tf.reduce_sum(l)
        metric.add(l_sum, accuracy(y_hat, y), tf.size(y))
    # Return training loss and training accuracy
    return metric[0] / metric[2], metric[1] / metric[2]
```

Antes de mostrar a implementação da função de treinamento,
definimos [**uma classe de utilitário que plota dados em animação.**]
Novamente, o objetivo é simplificar o código no restante do livro.

```{.python .input}
#@tab all
class Animator:  #@save
    """For plotting data in animation."""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        # Incrementally plot multiple lines
        if legend is None:
            legend = []
        d2l.use_svg_display()
        self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # Use a lambda function to capture arguments
        self.config_axes = lambda: d2l.set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # Add multiple data points into the figure
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)
```

[~~A função de treinamento~~]
A seguinte função de treinamento então
treina um modelo `net` em um conjunto de dados de treinamento acessado via` train_iter`
para várias épocas, que é especificado por `num_epochs`.
No final de cada época,
o modelo é avaliado em um conjunto de dados de teste acessado via `test_iter`.
Vamos aproveitar a classe `Animator` para visualizar
o progresso do treinamento.

```{.python .input}
#@tab all
def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):  #@save
    """Train a model (defined in Chapter 3)."""
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc,))
    train_loss, train_acc = train_metrics
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc
```

Como uma implementação do zero,
nós [**usamos a descida gradiente estocástica do *minibatch* **] definido em :numref:`sec_linear_scratch`
para otimizar a função de perda do modelo com uma taxa de aprendizado de 0,1.

```{.python .input}
#@tab mxnet, pytorch
lr = 0.1

def updater(batch_size):
    return d2l.sgd([W, b], lr, batch_size)
```

```{.python .input}
#@tab tensorflow
class Updater():  #@save
    """For updating parameters using minibatch stochastic gradient descent."""
    def __init__(self, params, lr):
        self.params = params
        self.lr = lr

    def __call__(self, batch_size, grads):
        d2l.sgd(self.params, grads, self.lr, batch_size)

updater = Updater([W, b], lr=0.1)
```

Agora [**treinamos o modelo com 10 épocas.**]
Observe que tanto o número de épocas (`num_epochs`),
e a taxa de aprendizagem (`lr`) são hiperparâmetros ajustáveis.
Mudando seus valores, podemos ser capazes
de aumentar a precisão da classificação do modelo.

```{.python .input}
#@tab all
num_epochs = 10
train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)
```

## Predição

Agora que o treinamento está completo,
nosso modelo está pronto para [**classificar algumas imagens.**]
Dada uma série de imagens,
vamos comparar seus *labels* reais
(primeira linha de saída de texto)
e as previsões do modelo
(segunda linha de saída de texto).

```{.python .input}
#@tab all
def predict_ch3(net, test_iter, n=6):  #@save
    """Predict labels (defined in Chapter 3)."""
    for X, y in test_iter:
        break
    trues = d2l.get_fashion_mnist_labels(y)
    preds = d2l.get_fashion_mnist_labels(d2l.argmax(net(X), axis=1))
    titles = [true +'\n' + pred for true, pred in zip(trues, preds)]
    d2l.show_images(
        d2l.reshape(X[0:n], (n, 28, 28)), 1, n, titles=titles[0:n])

predict_ch3(net, test_iter)
```

## Resumo

* Com a regressão *softmax*, podemos treinar modelos para classificação multiclasse.
* O loop de treinamento da regressão *softmax* é muito semelhante ao da regressão linear: recuperar e ler dados, definir modelos e funções de perda e treinar modelos usando algoritmos de otimização. Como você descobrirá em breve, os modelos de *deep learning* mais comuns têm procedimentos de treinamento semelhantes.

## Exercícios

1. Nesta seção, implementamos diretamente a função *softmax* com base na definição matemática da operação do *softmax*. Que problemas isso pode causar? Dica: tente calcular o tamanho de $\exp(50)$.
1. A função `cross_entropy` nesta seção foi implementada de acordo com a definição da função de perda de entropia cruzada. Qual poderia ser o problema com esta implementação? Dica: considere o domínio do logaritmo.
1. Que soluções você pode pensar para resolver os dois problemas acima?
1. É sempre uma boa ideia retornar o *label* mais provável? Por exemplo, você faria isso para diagnóstico médico?
1. Suponha que quiséssemos usar a regressão *softmax* para prever a próxima palavra com base em alguns recursos. Quais são alguns problemas que podem surgir de um vocabulário extenso?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/50)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/51)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/225)
:end_tab:
<!--stackedit_data:
eyJoaXN0b3J5IjpbNzUxOTk3Mzc0LDY4NDQzMjI2MSwtNDEwNz
A5NTcyLDExMDgzMDY5MCwtMTk1MzMzMjAxOSwxMTUzMDgxMzUs
MTYwMDk4NDQ1Niw3NDI1MTc0ODcsLTIxMzE1MDU0OTMsLTk5OT
A3ODc2NywtMTk4NDE4OTcyNV19
-->