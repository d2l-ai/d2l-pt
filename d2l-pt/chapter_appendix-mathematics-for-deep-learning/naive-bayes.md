# Naive Bayes
:label:`sec_naive_bayes`


Ao longo das seções anteriores, aprendemos sobre a teoria da probabilidade e variáveis aleatórias. Para colocar essa teoria em prática, vamos apresentar o classificador *Naive Bayes*. Isso usa apenas fundamentos probabilísticos para nos permitir realizar a classificação dos dígitos.

Aprender é fazer suposições. Se quisermos classificar um novo exemplo de dados que nunca vimos antes, temos que fazer algumas suposições sobre quais exemplos de dados são semelhantes entre si. O classificador *Naive Bayes*, um algoritmo popular e notavelmente claro, assume que todos os recursos são independentes uns dos outros para simplificar o cálculo. Nesta seção, vamos aplicar este modelo para reconhecer personagens em imagens.

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
import math
from mxnet import gluon, np, npx
npx.set_np()
d2l.use_svg_display()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import math
import torch
import torchvision
d2l.use_svg_display()
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import math
import tensorflow as tf
d2l.use_svg_display()
```

## Reconhecimento Ótico de Caracteres

MNIST :cite:`LeCun.Bottou.Bengio.ea.1998` é um dos conjuntos de dados amplamente usados. Ele contém 60.000 imagens para treinamento e 10.000 imagens para validação. Cada imagem contém um dígito escrito à mão de 0 a 9. A tarefa é classificar cada imagem no dígito correspondente.

Gluon fornece uma classe `MNIST` no módulo `data.vision` para
recupera automaticamente o conjunto de dados da Internet.
Posteriormente, o Gluon usará a cópia local já baixada.
Especificamos se estamos solicitando o conjunto de treinamento ou o conjunto de teste
definindo o valor do parâmetro `train` para `True` ou `False`, respectivamente.
Cada imagem é uma imagem em tons de cinza com largura e altura de $28$ com forma ($28$,$28$,$1$). Usamos uma transformação personalizada para remover a última dimensão do canal. Além disso, o conjunto de dados representa cada pixel por um inteiro não assinado de $8$ bits. Nós os quantificamos em recursos binários para simplificar o problema.

```{.python .input}
def transform(data, label):
    return np.floor(data.astype('float32') / 128).squeeze(axis=-1), label

mnist_train = gluon.data.vision.MNIST(train=True, transform=transform)
mnist_test = gluon.data.vision.MNIST(train=False, transform=transform)
```

```{.python .input}
#@tab pytorch
data_transform = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor()])

mnist_train = torchvision.datasets.MNIST(
    root='./temp', train=True, transform=data_transform, download=True)
mnist_test = torchvision.datasets.MNIST(
    root='./temp', train=False, transform=data_transform, download=True)
```

```{.python .input}
#@tab tensorflow
((train_images, train_labels), (
    test_images, test_labels)) = tf.keras.datasets.mnist.load_data()
```

Podemos acessar um exemplo particular, que contém a imagem e o rótulo correspondente.

```{.python .input}
image, label = mnist_train[2]
image.shape, label
```

```{.python .input}
#@tab pytorch
image, label = mnist_train[2]
image.shape, label
```

```{.python .input}
#@tab tensorflow
image, label = train_images[2], train_labels[2]
image.shape, label
```

Nosso exemplo, armazenado aqui na variável `imagem`, corresponde a uma imagem com altura e largura de $28$ pixels.

```{.python .input}
#@tab all
image.shape, image.dtype
```

Nosso código armazena o rótulo de cada imagem como um escalar. Seu tipo é um número inteiro de $32$ bits.

```{.python .input}
label, type(label), label.dtype
```

```{.python .input}
#@tab pytorch
label, type(label)
```

```{.python .input}
#@tab tensorflow
label, type(label)
```

Também podemos acessar vários exemplos ao mesmo tempo.

```{.python .input}
images, labels = mnist_train[10:38]
images.shape, labels.shape
```

```{.python .input}
#@tab pytorch
images = torch.stack([mnist_train[i][0] for i in range(10,38)], 
                     dim=1).squeeze(0)
labels = torch.tensor([mnist_train[i][1] for i in range(10,38)])
images.shape, labels.shape
```

```{.python .input}
#@tab tensorflow
images = tf.stack([train_images[i] for i in range(10, 38)], axis=0)
labels = tf.constant([train_labels[i] for i in range(10, 38)])
images.shape, labels.shape
```

Vamos visualizar esses exemplos.

```{.python .input}
#@tab all
d2l.show_images(images, 2, 9);
```

## O Modelo Probabilístico para Classificação

Em uma tarefa de classificação, mapeamos um exemplo em uma categoria. Aqui, um exemplo é uma imagem em tons de cinza $28\times 28$, e uma categoria é um dígito. (Consulte :numref:`sec_softmax` para uma explicação mais detalhada.)
Uma maneira natural de expressar a tarefa de classificação é por meio da questão probabilística: qual é o rótulo mais provável dado os recursos (ou seja, pixels de imagem)? Denote por $\mathbf x\in\mathbb R^d$ as características do exemplo e $y\in\mathbb R$ o rótulo. Aqui, os recursos são pixels de imagem, onde podemos remodelar uma imagem $2$-dimensional para um vetor de modo que $d=28^2=784$, e os rótulos são dígitos.
A probabilidade do rótulo dado as características é $p(y  \mid  \mathbf{x})$. Se pudermos calcular essas probabilidades, que são $p(y  \mid  \mathbf{x})$ para $y=0, \ldots,9$ em nosso exemplo, então o classificador produzirá a previsão $\hat{y}$ dado pela expressão:

$$\hat{y} = \mathrm{argmax} \> p(y  \mid  \mathbf{x}).$$


Infelizmente, isso requer que estimemos $p(y  \mid  \mathbf{x})$ para cada valor de $\mathbf{x} = x_1, ..., x_d$. Imagine que cada recurso pudesse assumir um dos $2$ valores. Por exemplo, o recurso $x_1 = 1$ pode significar que a palavra maçã aparece em um determinado documento e $x_1 = 0$ pode significar que não. Se tivéssemos $30$ de tais características binárias, isso significaria que precisamos estar preparados para classificar qualquer um dos $2^{30}$ (mais de 1 bilhão!) de valores possíveis do vetor de entrada $\mathbf{x}$.

Além disso, onde está o aprendizado? Se precisarmos ver todos os exemplos possíveis para prever o rótulo correspondente, não estaremos realmente aprendendo um padrão, mas apenas memorizando o conjunto de dados.

## O Classificador Naive Bayes

Felizmente, ao fazer algumas suposições sobre a independência condicional, podemos introduzir algum viés indutivo e construir um modelo capaz de generalizar a partir de uma seleção comparativamente modesta de exemplos de treinamento. Para começar, vamos usar o teorema de Bayes, para expressar o classificador como

$$\hat{y} = \mathrm{argmax}_y \> p(y  \mid  \mathbf{x}) = \mathrm{argmax}_y \> \frac{p( \mathbf{x}  \mid  y) p(y)}{p(\mathbf{x})}.$$

Observe que o denominador é o termo de normalização $p(\mathbf{x})$ que não depende do valor do rótulo $y$. Como resultado, só precisamos nos preocupar em comparar o numerador em diferentes valores de $y$. Mesmo que o cálculo do denominador fosse intratável, poderíamos escapar ignorando-o, desde que pudéssemos avaliar o numerador. Felizmente, mesmo se quiséssemos recuperar a constante de normalização, poderíamos. Sempre podemos recuperar o termo de normalização, pois $\sum_y p(y  \mid  \mathbf{x}) = 1$.

Agora, vamos nos concentrar em $p( \mathbf{x}  \mid  y)$. Usando a regra da cadeia de probabilidade, podemos expressar o termo $p( \mathbf{x}  \mid  y)$ como

$$p(x_1  \mid y) \cdot p(x_2  \mid  x_1, y) \cdot ... \cdot p( x_d  \mid  x_1, ..., x_{d-1}, y).$$

Por si só, essa expressão não nos leva mais longe. Ainda devemos estimar cerca de $2^d$ parâmetros. No entanto, se assumirmos que *as características são condicionalmente independentes umas das outras, dado o rótulo*, então de repente estamos em uma forma muito melhor, pois este termo simplifica para $\prod_i p(x_i  \mid  y)$, dando-nos o preditor

$$\hat{y} = \mathrm{argmax}_y \> \prod_{i=1}^d p(x_i  \mid  y) p(y).$$

Se pudermos estimar $p(x_i=1  \mid  y)$ para cada $i$ e $y$, e salvar seu valor em $P_{xy}[i, y]$, aqui $P_{xy}$ é uma matriz $d\times n$ com $n$ sendo o número de classes e $y\in\{1, \ldots, n\}$, então também podemos usar isso para estimar $p(x_i = 0 \mid y)$, ou seja,

$$ 
p(x_i = t_i \mid y) = 
\begin{cases}
    P_{xy}[i, y] & \text{for } t_i=1 ;\\
    1 - P_{xy}[i, y] & \text{for } t_i = 0 .
\end{cases}
$$

Além disso, estimamos $p(y)$ para cada $y$ e o salvamos em $P_y[y]$, com $P_y$ um vetor de comprimento $n$. Então, para qualquer novo exemplo $\mathbf t = (t_1, t_2, \ldots, t_d)$, poderíamos calcular

$$\begin{aligned}\hat{y} &= \mathrm{argmax}_ y \ p(y)\prod_{i=1}^d   p(x_t = t_i \mid y) \\ &= \mathrm{argmax}_y \ P_y[y]\prod_{i=1}^d \ P_{xy}[i, y]^{t_i}\, \left(1 - P_{xy}[i, y]\right)^{1-t_i}\end{aligned}$$
:eqlabel:`eq_naive_bayes_estimation`

para qualquer $y$. Portanto, nossa suposição de independência condicional levou a complexidade do nosso modelo de uma dependência exponencial no número de características $\mathcal{O}(2^dn)$ para uma dependência linear, que é $\mathcal{O}(dn)$.


## Trainamento

O problema agora é que não conhecemos $P_{xy}$ e $P_y$. Portanto, precisamos primeiro estimar seus valores dados alguns dados de treinamento. Isso é *treinar* o modelo. Estimar $P_y$ não é muito difícil. Como estamos lidando apenas com classes de $10$, podemos contar o número de ocorrências $n_y$ para cada um dos dígitos e dividi-lo pela quantidade total de dados $n$. Por exemplo, se o dígito 8 ocorre $n_8 = 5,800$ vezes e temos um total de $n = 60,000$ imagens, a estimativa de probabilidade é $p(y=8) = 0.0967$.

```{.python .input}
X, Y = mnist_train[:]  # All training examples

n_y = np.zeros((10))
for y in range(10):
    n_y[y] = (Y == y).sum()
P_y = n_y / n_y.sum()
P_y
```

```{.python .input}
#@tab pytorch
X = torch.stack([mnist_train[i][0] for i in range(len(mnist_train))], 
                dim=1).squeeze(0)
Y = torch.tensor([mnist_train[i][1] for i in range(len(mnist_train))])

n_y = torch.zeros(10)
for y in range(10):
    n_y[y] = (Y == y).sum()
P_y = n_y / n_y.sum()
P_y
```

```{.python .input}
#@tab tensorflow
X = tf.stack([train_images[i] for i in range(len(train_images))], axis=0)
Y = tf.constant([train_labels[i] for i in range(len(train_labels))])

n_y = tf.Variable(tf.zeros(10))
for y in range(10):
    n_y[y].assign(tf.reduce_sum(tf.cast(Y == y, tf.float32)))
P_y = n_y / tf.reduce_sum(n_y)
P_y
```

Agora vamos para coisas um pouco mais difíceis $P_{xy}$. Como escolhemos imagens em preto e branco, $p(x_i  \mid  y)$ denota a probabilidade de que o pixel $i$ seja ativado para a classe $y$. Assim como antes, podemos ir e contar o número de vezes $n_{iy}$ para que um evento ocorra e dividi-lo pelo número total de ocorrências de $y$, ou seja, $n_y$. Mas há algo um pouco preocupante: certos pixels podem nunca ser pretos (por exemplo, para imagens bem cortadas, os pixels dos cantos podem sempre ser brancos). Uma maneira conveniente para os estatísticos lidarem com esse problema é adicionar pseudo contagens a todas as ocorrências. Portanto, em vez de $n_{iy}$, usamos $n_{y} + 1$ e em vez de $n_y$ usamos $n_{iy}+1$. Isso também é chamado de *Suavização de Laplace*. Pode parecer ad-hoc, mas pode ser bem motivado do ponto de vista bayesiano.

```{.python .input}
n_x = np.zeros((10, 28, 28))
for y in range(10):
    n_x[y] = np.array(X.asnumpy()[Y.asnumpy() == y].sum(axis=0))
P_xy = (n_x + 1) / (n_y + 1).reshape(10, 1, 1)

d2l.show_images(P_xy, 2, 5);
```

```{.python .input}
#@tab pytorch
n_x = torch.zeros((10, 28, 28))
for y in range(10):
    n_x[y] = torch.tensor(X.numpy()[Y.numpy() == y].sum(axis=0))
P_xy = (n_x + 1) / (n_y + 1).reshape(10, 1, 1)

d2l.show_images(P_xy, 2, 5);
```

```{.python .input}
#@tab tensorflow
n_x = tf.Variable(tf.zeros((10, 28, 28)))
for y in range(10):
    n_x[y].assign(tf.cast(tf.reduce_sum(
        X.numpy()[Y.numpy() == y], axis=0), tf.float32))
P_xy = (n_x + 1) / tf.reshape((n_y + 1), (10, 1, 1))

d2l.show_images(P_xy, 2, 5);
```


Visualizando essas probabilidades de $10\times 28\times 28$ (para cada pixel de cada classe), poderíamos obter alguns dígitos de aparência média.

Agora podemos usar :eqref:`eq_naive_bayes_estimation` para prever uma nova imagem. Dado $\mathbf x$, as seguintes funções calculam $p(\mathbf x \mid y)p(y)$ para cada $y$.

```{.python .input}
def bayes_pred(x):
    x = np.expand_dims(x, axis=0)  # (28, 28) -> (1, 28, 28)
    p_xy = P_xy * x + (1 - P_xy)*(1 - x)
    p_xy = p_xy.reshape(10, -1).prod(axis=1)  # p(x|y)
    return np.array(p_xy) * P_y

image, label = mnist_test[0]
bayes_pred(image)
```

```{.python .input}
#@tab pytorch
def bayes_pred(x):
    x = x.unsqueeze(0)  # (28, 28) -> (1, 28, 28)
    p_xy = P_xy * x + (1 - P_xy)*(1 - x)
    p_xy = p_xy.reshape(10, -1).prod(dim=1)  # p(x|y)
    return p_xy * P_y

image, label = mnist_test[0]
bayes_pred(image)
```

```{.python .input}
#@tab tensorflow
def bayes_pred(x):
    x = tf.expand_dims(x, axis=0)  # (28, 28) -> (1, 28, 28)
    p_xy = P_xy * x + (1 - P_xy)*(1 - x)
    p_xy = tf.math.reduce_prod(tf.reshape(p_xy, (10, -1)), axis=1)  # p(x|y)
    return p_xy * P_y

image, label = tf.cast(train_images[0], tf.float32), train_labels[0]
bayes_pred(image)
```


Isso deu terrivelmente errado! Para descobrir o porquê, vejamos as probabilidades por pixel. Normalmente são números entre $0,001$ e $1$. Estamos multiplicando $784$ deles. Neste ponto, vale a pena mencionar que estamos calculando esses números em um computador, portanto, com um intervalo fixo para o expoente. O que acontece é que experimentamos *underflow numérico*, ou seja, a multiplicação de todos os números pequenos leva a algo ainda menor até que seja arredondado para zero. Discutimos isso como uma questão teórica em :numref:`sec_maximum_likelihood`, mas vemos o fenômeno claramente aqui na prática.

Conforme discutido nessa seção, corrigimos isso usando o fato de que $\log a b = \log a + \log b$, ou seja, mudamos para logaritmos de soma.
Mesmo se $a$ e $b$ forem números pequenos, os valores de logaritmo devem estar em uma faixa adequada.

```{.python .input}
a = 0.1
print('underflow:', a**784)
print('logarithm is normal:', 784*math.log(a))
```

```{.python .input}
#@tab pytorch
a = 0.1
print('underflow:', a**784)
print('logarithm is normal:', 784*math.log(a))
```

```{.python .input}
#@tab tensorflow
a = 0.1
print('underflow:', a**784)
print('logarithm is normal:', 784*tf.math.log(a).numpy())
```

Como o logaritmo é uma função crescente, podemos reescrever :eqref:`eq_naive_bayes_estimation` como

$$ \hat{y} = \mathrm{argmax}_y \ \log P_y[y] + \sum_{i=1}^d \Big[t_i\log P_{xy}[x_i, y] + (1-t_i) \log (1 - P_{xy}[x_i, y]) \Big].$$

Podemos implementar a seguinte versão estável:

```{.python .input}
log_P_xy = np.log(P_xy)
log_P_xy_neg = np.log(1 - P_xy)
log_P_y = np.log(P_y)

def bayes_pred_stable(x):
    x = np.expand_dims(x, axis=0)  # (28, 28) -> (1, 28, 28)
    p_xy = log_P_xy * x + log_P_xy_neg * (1 - x)
    p_xy = p_xy.reshape(10, -1).sum(axis=1)  # p(x|y)
    return p_xy + log_P_y

py = bayes_pred_stable(image)
py
```

```{.python .input}
#@tab pytorch
log_P_xy = torch.log(P_xy)
log_P_xy_neg = torch.log(1 - P_xy)
log_P_y = torch.log(P_y)

def bayes_pred_stable(x):
    x = x.unsqueeze(0)  # (28, 28) -> (1, 28, 28)
    p_xy = log_P_xy * x + log_P_xy_neg * (1 - x)
    p_xy = p_xy.reshape(10, -1).sum(axis=1)  # p(x|y)
    return p_xy + log_P_y

py = bayes_pred_stable(image)
py
```

```{.python .input}
#@tab tensorflow
log_P_xy = tf.math.log(P_xy)
# TODO: Look into why this returns infs
log_P_xy_neg = tf.math.log(1 - P_xy)
log_P_y = tf.math.log(P_y)

def bayes_pred_stable(x):
    x = tf.expand_dims(x, axis=0)  # (28, 28) -> (1, 28, 28)
    p_xy = log_P_xy * x + log_P_xy_neg * (1 - x)
    p_xy = tf.math.reduce_sum(tf.reshape(p_xy, (10, -1)), axis=1)  # p(x|y)
    return p_xy + log_P_y

py = bayes_pred_stable(image)
py
```

Podemos agora verificar se a previsão está correta.

```{.python .input}
# Convert label which is a scalar tensor of int32 dtype
# to a Python scalar integer for comparison
py.argmax(axis=0) == int(label)
```

```{.python .input}
#@tab pytorch
py.argmax(dim=0) == label
```

```{.python .input}
#@tab tensorflow
tf.argmax(py, axis=0) == label
```

Se agora prevermos alguns exemplos de validação, podemos ver o classificador Bayes funciona muito bem.

```{.python .input}
def predict(X):
    return [bayes_pred_stable(x).argmax(axis=0).astype(np.int32) for x in X]

X, y = mnist_test[:18]
preds = predict(X)
d2l.show_images(X, 2, 9, titles=[str(d) for d in preds]);
```

```{.python .input}
#@tab pytorch
def predict(X):
    return [bayes_pred_stable(x).argmax(dim=0).type(torch.int32).item() 
            for x in X]

X = torch.stack([mnist_train[i][0] for i in range(10,38)], dim=1).squeeze(0)
y = torch.tensor([mnist_train[i][1] for i in range(10,38)])
preds = predict(X)
d2l.show_images(X, 2, 9, titles=[str(d) for d in preds]);
```

```{.python .input}
#@tab tensorflow
def predict(X):
    return [tf.cast(tf.argmax(bayes_pred_stable(x), axis=0), tf.int32).numpy()
            for x in X]

X = tf.stack(
    [tf.cast(train_images[i], tf.float32) for i in range(10, 38)], axis=0)
y = tf.constant([train_labels[i] for i in range(10, 38)])
preds = predict(X)
# TODO: The preds are not correct due to issues with bayes_pred_stable()
d2l.show_images(X, 2, 9, titles=[str(d) for d in preds]);
```

Finalmente, vamos calcular a precisão geral do classificador.

```{.python .input}
X, y = mnist_test[:]
preds = np.array(predict(X), dtype=np.int32)
float((preds == y).sum()) / len(y)  # Validation accuracy
```

```{.python .input}
#@tab pytorch
X = torch.stack([mnist_train[i][0] for i in range(len(mnist_test))], 
                dim=1).squeeze(0)
y = torch.tensor([mnist_train[i][1] for i in range(len(mnist_test))])
preds = torch.tensor(predict(X), dtype=torch.int32)
float((preds == y).sum()) / len(y)  # Validation accuracy
```

```{.python .input}
#@tab tensorflow
X = tf.stack([tf.cast(train_images[i], tf.float32) for i in range(
    len(test_images))], axis=0)
y = tf.constant([train_labels[i] for i in range(len(test_images))])
preds = tf.constant(predict(X), dtype=tf.int32)
# TODO: The accuracy is not correct due to issues with bayes_pred_stable()
tf.reduce_sum(tf.cast(preds == y, tf.float32)) / len(y)  # Validation accuracy
```

Redes profundas modernas alcançam taxas de erro de menos de $0,01$. O desempenho relativamente baixo é devido às suposições estatísticas incorretas que fizemos em nosso modelo: presumimos que cada pixel é gerado *independentemente*, dependendo apenas do rótulo. Claramente, não é assim que os humanos escrevem dígitos, e essa suposição errada levou à queda de nosso classificador excessivamente ingênuo (Bayes).

## Resumo
* Usando a regra de Bayes, um classificador pode ser feito assumindo que todas as características observadas são independentes.
* Este classificador pode ser treinado em um conjunto de dados contando o número de ocorrências de combinações de rótulos e valores de pixel.
* Esse classificador foi o padrão ouro por décadas para tarefas como detecção de spam.

## Exercícios
1. Considere o conjunto de dados $[[0,0], [0,1], [1,0], [1,1]]$ com rótulos dados pelo XOR dos dois elementos $[0,1,1,0]$. Quais são as probabilidades de um classificador Naive Bayes construído neste conjunto de dados. Classifica com sucesso nossos pontos? Se não, quais premissas são violadas?
1. Suponha que não usamos a suavização de Laplace ao estimar as probabilidades e um exemplo de dados chegou no momento do teste que continha um valor nunca observado no treinamento. Qual seria a saída do modelo?
1. O classificador Naive Bayes é um exemplo específico de uma rede Bayesiana, onde a dependência de variáveis aleatórias é codificada com uma estrutura de grafo. Embora a teoria completa esteja além do escopo desta seção (consulte :cite:`Koller.Friedman.2009` para detalhes completos), explique por que permitir a dependência explícita entre as duas variáveis de entrada no modelo XOR permite a criação de um classificador de sucesso .


:begin_tab:`mxnet`
[Discussões](https://discuss.d2l.ai/t/418)
:end_tab:

:begin_tab:`pytorch`
[Discussões](https://discuss.d2l.ai/t/1100)
:end_tab:

:begin_tab:`tensorflow`
[Discussões](https://discuss.d2l.ai/t/1101)
:end_tab:
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTQxMjQ5NTM2NCwtOTA2ODIwNjc5LC0zMz
gzMzU5NjksLTIxMDU1MTI3MzMsMTMyMzkzOTkxMCw3MDU3NzM2
MDAsMTEzNDQwMTkyOF19
-->