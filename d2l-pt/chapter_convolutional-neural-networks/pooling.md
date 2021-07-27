# *Pooling*
:label:`sec_pooling`



Muitas vezes, conforme processamos imagens, queremos gradualmente
reduzir a resolução espacial de nossas representações ocultas,
agregando informações para que
quanto mais alto subimos na rede,
maior o campo receptivo (na entrada)
ao qual cada nó oculto é sensível.

Muitas vezes, nossa tarefa final faz alguma pergunta global sobre a imagem,
por exemplo, *contém um gato?*
Então, normalmente, as unidades de nossa camada final devem ser sensíveis
para toda a entrada.
Ao agregar informações gradualmente, produzindo mapas cada vez mais grosseiros,
alcançamos esse objetivo de, em última análise, aprendendo uma representação global,
enquanto mantém todas as vantagens das camadas convolucionais nas camadas intermediárias de processamento.

Além disso, ao detectar recursos de nível inferior, como bordas
(conforme discutido em :numref:`sec_conv_layer`),
frequentemente queremos que nossas representações sejam um tanto invariáveis ​​à tradução.
Por exemplo, se pegarmos a imagem `X`
com uma delimitação nítida entre preto e branco
e deslocarmos a imagem inteira em um pixel para a direita,
ou seja, `Z [i, j] = X [i, j + 1]`,
então a saída para a nova imagem `Z` pode ser muito diferente.
A borda terá deslocado um pixel.
Na realidade, os objetos dificilmente ocorrem exatamente no mesmo lugar.
Na verdade, mesmo com um tripé e um objeto estacionário,
a vibração da câmera devido ao movimento do obturador
pode mudar tudo em um pixel ou mais
(câmeras de última geração são carregadas com recursos especiais para resolver esse problema).

Esta seção apresenta *camadas de pooling*,
que servem ao duplo propósito de
mitigando a sensibilidade das camadas convolucionais à localização
e de representações de *downsampling* espacialmente.

## *Pooling* Máximo e *Pooling* Médio


Como camadas convolucionais, operadores de *pooling*
consistem em uma janela de formato fixo que é deslizada
todas as regiões na entrada de acordo com seu passo,
computando uma única saída para cada local percorrido
pela janela de formato fixo (também conhecida como *janela de pooling*).
No entanto, ao contrário do cálculo de correlação cruzada
das entradas e grãos na camada convolucional,
a camada de *pooling* não contém parâmetros (não há *kernel*).
Em vez disso, os operadores de *pooling* são determinísticos,
normalmente calculando o valor máximo ou médio
dos elementos na janela de *pooling*.
Essas operações são chamadas de *pooling máximo* (*pooling máximo* para breve)
e *pooling médio*, respectivamente.


Em ambos os casos, como com o operador de correlação cruzada,
podemos pensar na janela de *pooling*
começando da parte superior esquerda do tensor de entrada
e deslizando pelo tensor de entrada da esquerda para a direita e de cima para baixo.
Em cada local que atinge a janela de *pooling*,
ele calcula o máximo ou o médio
valor do subtensor de entrada na janela,
dependendo se o *pooling* máximo ou médio é empregado.

![Pooling máximo com uma forma de janela de pool de $2\times 2$. As partes sombreadas são o primeiro elemento de saída, bem como os elementos tensores de entrada usados para o cálculo de saída: $\max(0, 1, 3, 4)=4$.](../img/pooling.svg)
:label:`fig_pooling`

O tensor de saída em :numref:`fig_pooling` tem uma altura de 2 e uma largura de 2.
Os quatro elementos são derivados do valor máximo em cada janela de *pooling*:

$$
\max(0, 1, 3, 4)=4,\\
\max(1, 2, 4, 5)=5,\\
\max(3, 4, 6, 7)=7,\\
\max(4, 5, 7, 8)=8.\\
$$


Uma camada de *pooling* com uma forma de janela de pool de $p \times q$
é chamado de camada de *pooling* $p \times q$
A operação de *pooling* é chamada $p \times q$ *pooling*.

Vamos retornar ao exemplo de detecção de borda do objeto
mencionado no início desta seção.
Agora vamos usar a saída da camada convolucional
como entrada para $2\times 2$ *pooling* máximo.
Defina a entrada da camada convolucional como `X` e a saída da camada de pooling como` Y`. Se os valores de `X [i, j]` e `X [i, j + 1]` são ou não diferentes,
ou `X [i, j + 1]` e `X [i, j + 2]` são diferentes,
a camada de pool sempre produz `Y [i, j] = 1`.
Ou seja, usando a camada de pooling máxima $2\times 2$
ainda podemos detectar se o padrão reconhecido pela camada convolucional
move no máximo um elemento em altura ou largura.

No código abaixo, implementamos a propagação direta
da camada de *pooling* na função `pool2d`.
Esta função é semelhante à função `corr2d`
in: numref: `sec_conv_layer`.
No entanto, aqui não temos *kernel*, computando a saída
como o máximo ou a média de cada região na entrada.

```{.python .input}
from d2l import mxnet as d2l
from mxnet import np, npx
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
#@tab mxnet, pytorch
def pool2d(X, pool_size, mode='max'):
    p_h, p_w = pool_size
    Y = d2l.zeros((X.shape[0] - p_h + 1, X.shape[1] - p_w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i, j] = X[i: i + p_h, j: j + p_w].max()
            elif mode == 'avg':
                Y[i, j] = X[i: i + p_h, j: j + p_w].mean()
    return Y
```

```{.python .input}
#@tab tensorflow
import tensorflow as tf

def pool2d(X, pool_size, mode='max'):
    p_h, p_w = pool_size
    Y = tf.Variable(tf.zeros((X.shape[0] - p_h + 1, X.shape[1] - p_w +1)))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i, j].assign(tf.reduce_max(X[i: i + p_h, j: j + p_w]))
            elif mode =='avg':
                Y[i, j].assign(tf.reduce_mean(X[i: i + p_h, j: j + p_w]))
    return Y
```

Podemos construir o tensor de entrada `X` em :numref:`fig_pooling`  para validar a saída da camada de *pooling* máximo bidimensional.

```{.python .input}
#@tab all
X = d2l.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
pool2d(X, (2, 2))
```

Além disso, experimentamos a camada de *pooling* média.

```{.python .input}
#@tab all
pool2d(X, (2, 2), 'avg')
```

## Preenchimento e Passos


Tal como acontece com as camadas convolucionais, camadas de *pooling*
também podem alterar a forma de saída.
E como antes, podemos alterar a operação para obter uma forma de saída desejada
preenchendo a entrada e ajustando o passo.
Podemos demonstrar o uso de preenchimento e passos
em camadas de agrupamento por meio da camada de agrupamento máximo bidimensional integrada do framework de *deep learning*.
Primeiro construímos um tensor de entrada `X` cuja forma tem quatro dimensões,
onde o número de exemplos (tamanho do lote) e o número de canais são ambos 1.

:begin_tab:`tensorflow`
É importante notar que o *tensorflow*
prefere e é otimizado para *as últimas* entradas dos canais.
:end_tab:

```{.python .input}
#@tab mxnet, pytorch
X = d2l.reshape(d2l.arange(16, dtype=d2l.float32), (1, 1, 4, 4))
X
```


```{.python .input}
#@tab tensorflow
X = d2l.reshape(d2l.arange(16, dtype=d2l.float32), (1, 4, 4, 1))
X
```

Por padrão, o passo e a janela de *pooling* na instância da classe interna do *framework*
têm a mesma forma.
Abaixo, usamos uma janela de *pooling* de forma `(3, 3)`,
portanto, obtemos uma forma de passo de `(3, 3)` por padrão.

```{.python .input}
pool2d = nn.MaxPool2D(3)
# Because there are no model parameters in the pooling layer, we do not need
# to call the parameter initialization function
pool2d(X)
```

```{.python .input}
#@tab pytorch
pool2d = nn.MaxPool2d(3)
pool2d(X)
```

```{.python .input}
#@tab tensorflow
pool2d = tf.keras.layers.MaxPool2D(pool_size=[3, 3])
pool2d(X)
```

O passo e o preenchimento podem ser especificados manualmente.

```{.python .input}
pool2d = nn.MaxPool2D(3, padding=1, strides=2)
pool2d(X)
```

```{.python .input}
#@tab pytorch
pool2d = nn.MaxPool2d(3, padding=1, stride=2)
pool2d(X)
```

```{.python .input}
#@tab tensorflow
paddings = tf.constant([[0, 0], [1,0], [1,0], [0,0]])
X_padded = tf.pad(X, paddings, "CONSTANT")
pool2d = tf.keras.layers.MaxPool2D(pool_size=[3, 3], padding='valid',
                                   strides=2)
pool2d(X_padded)
```




:begin_tab:`mxnet`
Claro, podemos especificar uma janela de *pooling* retangular arbitrária
e especificar o preenchimento e o passo para altura e largura, respectivamente.
:end_tab:

:begin_tab: `pytorch`
Claro, podemos especificar uma janela de *pooling* retangular arbitrária
e especificar o preenchimento e o passo para altura e largura, respectivamente.
Para `nn.MaxPool2D`, o preenchimento deve ser menor que a metade do kernel_size.
Se a condição não for atendida, podemos primeiro preenchemos a entrada usando
`nn.functional.pad` e, em seguida, o passamos para a camada de *pooling*.
:end_tab:

:begin_tab: `tensorflow`
Claro, podemos especificar uma janela de *pooling* retangular arbitrária
e especificar o preenchimento e o passo para altura e largura, respectivamente.
No TensorFlow, para implementar um preenchimento de 1 em todo o tensor, uma função projetada para preenchimento
deve ser invocada usando `tf.pad`. Isso implementará o preenchimento necessário e permitirá que o supracitado (3, 3) agrupamento com uma passada (2, 2) para realizar
semelhantes aos do PyTorch e MXNet. Ao preencher desta forma, a variável embutida `padding` deve ser definida como `válida`.
:end_tab:

```{.python .input}
pool2d = nn.MaxPool2D((2, 3), padding=(1, 2), strides=(2, 3))
pool2d(X)
```

```{.python .input}
#@tab pytorch
X_pad = nn.functional.pad(X, (2, 2, 1, 1))
pool2d = nn.MaxPool2d((2, 3), stride=(2, 3))
pool2d(X_pad)
```

```{.python .input}
#@tab tensorflow
paddings = tf.constant([[0, 0], [1, 1], [2, 1], [0, 0]])
X_padded = tf.pad(X, paddings, "CONSTANT")

pool2d = tf.keras.layers.MaxPool2D(pool_size=[2, 3], padding='valid',
                                   strides=(2,3))
pool2d(X_padded)
```


## Canais Múltiplos


Ao processar dados de entrada multicanal,
a camada de *pooling* agrupa cada canal de entrada separadamente,
em vez de somar as entradas nos canais
como em uma camada convolucional.
Isso significa que o número de canais de saída para a camada de *pooling*
é igual ao número de canais de entrada.
Abaixo, vamos concatenar os tensores `X` e` X + 1`
na dimensão do canal para construir uma entrada com 2 canais.

:begin_tab: `tensorflow`
Observe que isso exigirá um
concatenação ao longo da última dimensão do TensorFlow devido à sintaxe dos últimos canais.
:end_tab:

```{.python .input}
#@tab mxnet, pytorch
X = d2l.concat((X, X + 1), 1)
X
```

```{.python .input}
#@tab tensorflow
X = tf.concat([X, X + 1], 3)  # Concatenate along `dim=3` due to channels-last syntax
```

Como podemos ver, o número de canais de saída ainda é 2 após o *pooling*.

```{.python .input}
pool2d = nn.MaxPool2D(3, padding=1, strides=2)
pool2d(X)
```

```{.python .input}
#@tab pytorch
pool2d = nn.MaxPool2d(3, padding=1, stride=2)
pool2d(X)
```

```{.python .input}
#@tab tensorflow
paddings = tf.constant([[0, 0], [1,0], [1,0], [0,0]])
X_padded = tf.pad(X, paddings, "CONSTANT")
pool2d = tf.keras.layers.MaxPool2D(pool_size=[3, 3], padding='valid',
                                   strides=2)
pool2d(X_padded)

```

:begin_tab: `tensorflow`
Observe que a saída para o *pooling* de tensorflow parece à primeira vista ser diferente, no entanto
numericamente, os mesmos resultados são apresentados como MXNet e PyTorch.
A diferença está na dimensionalidade, e na leitura do
a saída verticalmente produz a mesma saída que as outras implementações.
:end_tab:

## Resumo

* Pegando os elementos de entrada na janela de agrupamento, a operação de agrupamento máxima atribui o valor máximo como a saída e a operação de agrupamento média atribui o valor médio como a saída.
* Um dos principais benefícios de uma camada de *pooling* é aliviar a sensibilidade excessiva da camada convolucional ao local.
* Podemos especificar o preenchimento e a passada para a camada de *pooling*.
* O agrupamento máximo, combinado com uma passada maior do que 1, pode ser usado para reduzir as dimensões espaciais (por exemplo, largura e altura).
* O número de canais de saída da camada de *pooling* é igual ao número de canais de entrada.


## Exercícios

1. Você pode implementar o *pooling* médio como um caso especial de uma camada de convolução? Se sim, faça.
1. Você pode implementar o *pooling* máximo como um caso especial de uma camada de convolução? Se for assim, faça.
1. Qual é o custo computacional da camada de *pooling*? Suponha que a entrada para a camada de *pooling* seja do tamanho  $c\times h\times w$, a janela de *pooling* tem um formato de $p_h\times p_w$ com um preenchimento de $(p_h, p_w)$ e um passo de $(s_h, s_w)$.
1. Por que você espera que o *pooling* máximo e o *pooling* médio funcionem de maneira diferente?
1. Precisamos de uma camada mínima de *pooling* separada? Você pode substituí-la por outra operação?
1. Existe outra operação entre o *pooling* médio e máximo que você possa considerar (dica: lembre-se do *softmax*)? Por que não é tão popular?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/71)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/72)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/274)
:end_tab:
<!--stackedit_data:
eyJoaXN0b3J5IjpbMjAzMjIwMDQ0NiwtMTIwOTA5MTA3NiwtMT
EwMDAzNzAzOCwxODk0MDMzMjM3LC0xMzA5OTExODZdfQ==
-->