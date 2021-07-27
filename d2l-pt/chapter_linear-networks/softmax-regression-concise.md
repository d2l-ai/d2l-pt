# Implementação Concisa da Regressão *Softmax*
:label:`sec_softmax_concise`



(**APIs de alto nível tal como**)
os *frameworks* de *deep learning*
(**tornaram muito mais fácil de implementar a regressão linear**)
em :numref:`sec_linear_concise`,
(**encontraremos de forma semelhante**) (~~aqui~~) (ou possivelmente mais)
conveniente, implementar modelos de classificação. Vamos ficar com o conjunto de dados *Fashion-MNIST*
e manter o tamanho do lote em 256 como em :numref:`sec_softmax_scratch`.

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

```{.python .input}
#@tab all
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
```

## Inicializando os Parâmetros do Modelo

Conforme mencionado em: numref: `sec_softmax`,
[**a camada de saída da regressão *softmax*
é uma camada totalmente conectada.**]
Portanto, para implementar nosso modelo,
só precisamos adicionar uma camada totalmente conectada
com 10 saídas para nosso `Sequential`.
Novamente, aqui, o `Sequential` não é realmente necessário,
mas podemos também criar o hábito, pois será onipresente
ao implementar modelos profundos.
Novamente, inicializamos os pesos aleatoriamente
com média zero e desvio padrão 0,01.

```{.python .input}
net = nn.Sequential()
net.add(nn.Dense(10))
net.initialize(init.Normal(sigma=0.01))
```

```{.python .input}
#@tab pytorch
# PyTorch does not implicitly reshape the inputs. Thus we define the flatten
# layer to reshape the inputs before the linear layer in our network
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights);
```

```{.python .input}
#@tab tensorflow
net = tf.keras.models.Sequential()
net.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
weight_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01)
net.add(tf.keras.layers.Dense(10, kernel_initializer=weight_initializer))
```

## Implementação do *Softmax* Revisitada
:label:`subsec_softmax-implementation-revisited`


No exemplo anterior de :numref:`sec_softmax_scratch`,
calculamos a saída do nosso modelo
e então executamos esta saída através da perda de entropia cruzada.
Matematicamente, isso é uma coisa perfeitamente razoável de se fazer.
No entanto, de uma perspectiva computacional,
a exponenciação pode ser uma fonte de problemas de estabilidade numérica.

Lembre-se de que a função *softmax* calcula
$\hat y_j = \frac{\exp(o_j)}{\sum_k \exp(o_k)}$,
onde $\hat y_j$ é o elemento $j^\mathrm{th}$ da distribuição de probabilidade prevista $\hat{\mathbf{y}}$
e $o_j$ é o elemento $j^\mathrm{th}$ dos *logits*
$\mathbf{o}$.
Se alguns dos $o_k$ forem muito grandes (ou seja, muito positivos),
então $\exp(o_k)$ pode ser maior que o maior número,
podemos ter para certos tipos de dados (ou seja, *estouro*).
Isso tornaria o denominador (e/ou numerador) `inf` (infinito)
e acabamos encontrando 0, `inf` ou` nan` (não um número) para $\hat y_j$.
Nessas situações, não obtemos uma definição bem definida
valor de retorno para entropia cruzada.


Um truque para contornar isso é primeiro subtrair $\max(o_k)$
de todos $o_k$ antes de prosseguir com o cálculo do *softmax*.
Você pode ver que este deslocamento de cada $o_k$ por um fator constante
não altera o valor de retorno de *softmax*:
$$
\begin{aligned}
\hat y_j & =  \frac{\exp(o_j - \max(o_k))\exp(\max(o_k))}{\sum_k \exp(o_k - \max(o_k))\exp(\max(o_k))} \\
& = \frac{\exp(o_j - \max(o_k))}{\sum_k \exp(o_k - \max(o_k))}.
\end{aligned}
$$



Após a etapa de subtração e normalização,
pode ser possível que alguns  $o_j - \max(o_k)$ tenham grandes valores negativos
e assim que o $\exp(o_j - \max(o_k))$ correspondente assumirá valores próximos a zero.
Eles podem ser arredondados para zero devido à precisão finita (ou seja, *underflow*),
tornando $\hat y_j$ zero e dando-nos `-inf` para $\log(\hat y_j)$.
Alguns passos abaixo na *backpropagation*,
podemos nos encontrar diante de uma tela cheia
dos temidos resultados `nan`.

Felizmente, somos salvos pelo fato de que
embora estejamos computando funções exponenciais,
em última análise, pretendemos levar seu log
(ao calcular a perda de entropia cruzada).
Combinando esses dois operadores
*softmax* e entropia cruzada juntos,
podemos escapar dos problemas de estabilidade numérica
que poderia nos atormentar durante a *backpropagation*.
Conforme mostrado na equação abaixo, evitamos calcular $\exp(o_j - \max(o_k))$
e podemos usar $o_j - \max(o_k)$ diretamente devido ao cancelamento em $\log(\exp(\cdot))$:

$$
\begin{aligned}
\log{(\hat y_j)} & = \log\left( \frac{\exp(o_j - \max(o_k))}{\sum_k \exp(o_k - \max(o_k))}\right) \\
& = \log{(\exp(o_j - \max(o_k)))}-\log{\left( \sum_k \exp(o_k - \max(o_k)) \right)} \\
& = o_j - \max(o_k) -\log{\left( \sum_k \exp(o_k - \max(o_k)) \right)}.
\end{aligned}
$$

Queremos manter a função *softmax* convencional acessível
no caso de querermos avaliar as probabilidades de saída por nosso modelo.
Mas em vez de passar probabilidades de *softmax* para nossa nova função de perda,
nós vamos apenas
[**passar os *logits* e calcular o *softmax* e seu log
tudo de uma vez dentro da função de perda de entropia cruzada,**]
que faz coisas inteligentes como o ["Truque LogSumExp"](https://en.wikipedia.org/wiki/LogSumExp).

```{.python .input}
loss = gluon.loss.SoftmaxCrossEntropyLoss()
```

```{.python .input}
#@tab pytorch
loss = nn.CrossEntropyLoss()
```

```{.python .input}
#@tab tensorflow
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
```

## Otimização do Algoritmo

Aqui, nós (**usamos gradiente descendente estocástico de *minibatch***)
com uma taxa de aprendizado de 0,1 como o algoritmo de otimização.
Observe que este é o mesmo que aplicamos no exemplo de regressão linear
e ilustra a aplicabilidade geral dos otimizadores.

```{.python .input}
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.1})
```

```{.python .input}
#@tab pytorch
trainer = torch.optim.SGD(net.parameters(), lr=0.1)
```

```{.python .input}
#@tab tensorflow
trainer = tf.keras.optimizers.SGD(learning_rate=.1)
```

## Trainamento

Em seguida, [**chamamos a função de treinamento definida**] (~~anteriormente~~) em :numref:`sec_softmax_scratch` para treinar o modelo.

```{.python .input}
#@tab all
num_epochs = 10
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
```

Como antes, este algoritmo converge para uma solução
que atinge uma precisão decente,
embora desta vez com menos linhas de código do que antes.


## Resumo

* Usando APIs de alto nível, podemos implementar a regressão *softmax* de forma muito mais concisa.
* De uma perspectiva computacional, a implementação da regressão *softmax* tem complexidades. Observe que, em muitos casos, uma estrutura de *deep learning* toma precauções adicionais além desses truques mais conhecidos para garantir a estabilidade numérica, salvando-nos de ainda mais armadilhas que encontraríamos se tentássemos codificar todos os nossos modelos do zero na prática.

## Exercícios

1. Tente ajustar os hiperparâmetros, como *batch size*, número de épocas e taxa de aprendizado, para ver quais são os resultados.
2. Aumente o número de épocas de treinamento. Por que a precisão do teste pode diminuir depois de um tempo? Como poderíamos consertar isso?
3. 
:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/52)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/53)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/260)
:end_tab:
<!--stackedit_data:
eyJoaXN0b3J5IjpbODcwNTQ4NzYzLC0xNzQ5MTUzMjIwLC0xNj
IyMTg2ODIyXX0=
-->