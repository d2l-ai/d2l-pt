# Preenchimento e Saltos
:label:`sec_padding`



No exemplo anterior de :numref:`fig_correlation`,
nossa entrada tinha altura e largura de 3
e nosso núcleo de convolução tinha altura e largura de 2,
produzindo uma representação de saída com dimensão $2\times2$.
Como generalizamos em :numref:`sec_conv_layer`,
assumindo que
a forma de entrada é $n_h\times n_w$
e a forma do kernel de convolução é $k_h\times k_w$,
então a forma de saída será
$(n_h-k_h+1) \times (n_w-k_w+1)$.
Portanto, a forma de saída da camada convolucional
é determinada pela forma da entrada
e a forma do núcleo de convolução.

Em vários casos, incorporamos técnicas,
incluindo preenchimento e convoluções com saltos,
que afetam o tamanho da saída.
Como motivação, note que uma vez que os *kernels* geralmente
têm largura e altura maiores que $1$,
depois de aplicar muitas convoluções sucessivas,
tendemos a acabar com resultados que são
consideravelmente menor do que nossa entrada.
Se começarmos com uma imagem de $240 \times 240$ pixels,
$10$ camadas de $5 \times 5$ convoluções
reduzem a imagem para $200 \times 200$ pixels,
cortando $30 \%$ da imagem e com ela
obliterando qualquer informação interessante
nos limites da imagem original.
*Preenchimento* é a ferramenta mais popular para lidar com esse problema.

In other cases, we may want to reduce the dimensionality drastically,
e.g., if we find the original input resolution to be unwieldy.
*Strided convolutions* are a popular technique that can help in these instances.

## Preenchimento

Conforme descrito acima, um problema complicado ao aplicar camadas convolucionais
é que tendemos a perder pixels no perímetro de nossa imagem.
Uma vez que normalmente usamos pequenos *kernels*,
para qualquer convolução dada,
podemos perder apenas alguns pixels,
mas isso pode somar conforme aplicamos
muitas camadas convolucionais sucessivas.
Uma solução direta para este problema
é adicionar pixels extras de preenchimento ao redor do limite de nossa imagem de entrada,
aumentando assim o tamanho efetivo da imagem.
Normalmente, definimos os valores dos pixels extras para zero.
Em :numref:`img_conv_pad`, preenchemos uma entrada $3 \times 3$,
aumentando seu tamanho para $5 \times 5$.
A saída correspondente então aumenta para uma matriz $4 \times 4$
As partes sombreadas são o primeiro elemento de saída, bem como os elementos tensores de entrada e kernel usados para o cálculo de saída: $0\times0+0\times1+0\times2+0\times3=0$.

![Correlação cruzada bidimensional com preenchimento.](../img/conv-pad.svg)
:label:`img_conv_pad`

Em geral, se adicionarmos um total de $p_h$ linhas de preenchimento
(cerca de metade na parte superior e metade na parte inferior)
e um total de $p_w$ colunas de preenchimento
(cerca de metade à esquerda e metade à direita),
a forma de saída será

$$(n_h-k_h+p_h+1)\times(n_w-k_w+p_w+1).$$


Isso significa que a altura e largura da saída
aumentará em $p_h$ e $p_w$, respectivamente.

Em muitos casos, queremos definir $p_h=k_h-1$ e $p_w=k_w-1$
para dar à entrada e saída a mesma altura e largura.
Isso tornará mais fácil prever a forma de saída de cada camada
ao construir a rede.
Supondo que $k_h$ seja estranho aqui,
vamos preencher $p_h/2$ linhas em ambos os lados da altura.
Se $k_h$ for par, uma possibilidade é
juntar $\lceil p_h/2\rceil$ linhas no topo da entrada
e $\lfloor p_h/2\rfloor$ linhas na parte inferior.
Vamos preencher ambos os lados da largura da mesma maneira.


CNNs geralmente usam *kernels* de convolução
com valores de altura e largura ímpares, como 1, 3, 5 ou 7.
Escolher tamanhos ímpares de *kernel* tem o benefício
que podemos preservar a dimensionalidade espacial
enquanto preenche com o mesmo número de linhas na parte superior e inferior,
e o mesmo número de colunas à esquerda e à direita.

Além disso, esta prática de usar *kernels* estranhos
e preenchimento para preservar precisamente a dimensionalidade
oferece um benefício administrativo.
Para qualquer tensor bidimensional `X`,
quando o tamanho do *kernel* é estranho
e o número de linhas e colunas de preenchimento
em todos os lados são iguais,
produzindo uma saída com a mesma altura e largura da entrada,
sabemos que a saída `Y [i, j]` é calculada
por correlação cruzada do kernel de entrada e convolução
com a janela centralizada em `X [i, j]`.

No exemplo a seguir, criamos uma camada convolucional bidimensional
com altura e largura de 3
e aplique 1 pixel de preenchimento em todos os lados.
Dada uma entrada com altura e largura de 8,
descobrimos que a altura e a largura da saída também é 8.

```{.python .input}
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()

# For convenience, we define a function to calculate the convolutional layer.
# This function initializes the convolutional layer weights and performs
# corresponding dimensionality elevations and reductions on the input and
# output
def comp_conv2d(conv2d, X):
    conv2d.initialize()
    # Here (1, 1) indicates that the batch size and the number of channels
    # are both 1
    X = X.reshape((1, 1) + X.shape)
    Y = conv2d(X)
    # Exclude the first two dimensions that do not interest us: examples and
    # channels
    return Y.reshape(Y.shape[2:])

# Note that here 1 row or column is padded on either side, so a total of 2
# rows or columns are added
conv2d = nn.Conv2D(1, kernel_size=3, padding=1)
X = np.random.uniform(size=(8, 8))
comp_conv2d(conv2d, X).shape
```

```{.python .input}
#@tab pytorch
import torch
from torch import nn

# We define a convenience function to calculate the convolutional layer. This
# function initializes the convolutional layer weights and performs
# corresponding dimensionality elevations and reductions on the input and
# output
def comp_conv2d(conv2d, X):
    # Here (1, 1) indicates that the batch size and the number of channels
    # are both 1
    X = X.reshape((1, 1) + X.shape)
    Y = conv2d(X)
    # Exclude the first two dimensions that do not interest us: examples and
    # channels
    return Y.reshape(Y.shape[2:])
# Note that here 1 row or column is padded on either side, so a total of 2
# rows or columns are added
conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1)
X = torch.rand(size=(8, 8))
comp_conv2d(conv2d, X).shape
```

```{.python .input}
#@tab tensorflow
import tensorflow as tf

# We define a convenience function to calculate the convolutional layer. This
# function initializes the convolutional layer weights and performs
# corresponding dimensionality elevations and reductions on the input and
# output
def comp_conv2d(conv2d, X):
    # Here (1, 1) indicates that the batch size and the number of channels
    # are both 1
    X = tf.reshape(X, (1, ) + X.shape + (1, ))
    Y = conv2d(X)
    # Exclude the first two dimensions that do not interest us: examples and
    # channels
    return tf.reshape(Y, Y.shape[1:3])
# Note that here 1 row or column is padded on either side, so a total of 2
# rows or columns are added
conv2d = tf.keras.layers.Conv2D(1, kernel_size=3, padding='same')
X = tf.random.uniform(shape=(8, 8))
comp_conv2d(conv2d, X).shape
```

Quando a altura e largura do núcleo de convolução são diferentes,
podemos fazer com que a saída e a entrada tenham a mesma altura e largura
definindo diferentes números de preenchimento para altura e largura.

```{.python .input}
# Here, we use a convolution kernel with a height of 5 and a width of 3. The
# padding numbers on either side of the height and width are 2 and 1,
# respectively
conv2d = nn.Conv2D(1, kernel_size=(5, 3), padding=(2, 1))
comp_conv2d(conv2d, X).shape
```

```{.python .input}
#@tab pytorch
# Here, we use a convolution kernel with a height of 5 and a width of 3. The
# padding numbers on either side of the height and width are 2 and 1,
# respectively
conv2d = nn.Conv2d(1, 1, kernel_size=(5, 3), padding=(2, 1))
comp_conv2d(conv2d, X).shape
```

```{.python .input}
#@tab tensorflow
# Here, we use a convolution kernel with a height of 5 and a width of 3. The
# padding numbers on either side of the height and width are 2 and 1,
# respectively
conv2d = tf.keras.layers.Conv2D(1, kernel_size=(5, 3), padding='same')
comp_conv2d(conv2d, X).shape
```

## Saltos


Ao calcular a correlação cruzada,
começamos com a janela de convolução
no canto superior esquerdo do tensor de entrada,
e o deslizamos sobre todos os locais para baixo e para a direita.
Nos exemplos anteriores, optamos por deslizar um elemento de cada vez.
No entanto, às vezes, seja para eficiência computacional
ou porque desejamos reduzir a resolução,
movemos nossa janela mais de um elemento por vez,
pulando os locais intermediários.

Nos referimos ao número de linhas e colunas percorridas por slide como o *salto*.
Até agora, usamos saltos de 1, tanto para altura quanto para largura.
Às vezes, podemos querer dar um salto maior.
:numref:`img_conv_stride` mostra uma operação de correlação cruzada bidimensional
com um salto de 3 na vertical e 2 na horizontal.
As partes sombreadas são os elementos de saída, bem como os elementos tensores de entrada e *kernel* usados ​​para o cálculo de saída: $0\times0+0\times1+1\times2+2\times3=8$, $0\times0+6\times1+0\times2+0\times3=6$.
Podemos ver que quando o segundo elemento da primeira coluna é gerado,
a janela de convolução desliza três fileiras para baixo.
A janela de convolução desliza duas colunas para a direita
quando o segundo elemento da primeira linha é gerado.
Quando a janela de convolução continua a deslizar duas colunas para a direita na entrada,
não há saída porque o elemento de entrada não pode preencher a janela
(a menos que adicionemos outra coluna de preenchimento).

![Correlação cruzada com passos de 3 e 2 para altura e largura, respectivamente.](../img/conv-stride.svg)
:label:`img_conv_stride`

Em geral, quando o salto para a altura é $s_h$
e a distância para a largura é $s_w$, a forma de saída é

$$\lfloor(n_h-k_h+p_h+s_h)/s_h\rfloor \times \lfloor(n_w-k_w+p_w+s_w)/s_w\rfloor.$$


Se definirmos $p_h=k_h-1$ e $p_w=k_w-1$,
então a forma de saída será simplificada para
$\lfloor(n_h+s_h-1)/s_h\rfloor \times \lfloor(n_w+s_w-1)/s_w\rfloor$.
Indo um passo adiante, se a altura e largura de entrada
são divisíveis pelos saltos na altura e largura,
então a forma de saída será $(n_h/s_h) \times (n_w/s_w)$.

Abaixo, definimos os saltos de altura e largura para 2,
reduzindo assim pela metade a altura e a largura da entrada.

```{.python .input}
conv2d = nn.Conv2D(1, kernel_size=3, padding=1, strides=2)
comp_conv2d(conv2d, X).shape
```

```{.python .input}
#@tab pytorch
conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=2)
comp_conv2d(conv2d, X).shape
```

```{.python .input}
#@tab tensorflow
conv2d = tf.keras.layers.Conv2D(1, kernel_size=3, padding='same', strides=2)
comp_conv2d(conv2d, X).shape
```

A seguir, veremos um exemplo um pouco mais complicado.

```{.python .input}
conv2d = nn.Conv2D(1, kernel_size=(3, 5), padding=(0, 1), strides=(3, 4))
comp_conv2d(conv2d, X).shape
```

```{.python .input}
#@tab pytorch
conv2d = nn.Conv2d(1, 1, kernel_size=(3, 5), padding=(0, 1), stride=(3, 4))
comp_conv2d(conv2d, X).shape
```

```{.python .input}
#@tab tensorflow
conv2d = tf.keras.layers.Conv2D(1, kernel_size=(3,5), padding='valid',
                                strides=(3, 4))
comp_conv2d(conv2d, X).shape
```

Por uma questão de brevidade, quando o número de preenchimento
em ambos os lados da altura e largura de entrada são $p_h$ e$p_w$ respectivamente, chamamos o preenchimento $(p_h, p_w)$.
Especificamente, quando $p_h = p_w = p$, o preenchimento é $p$.
Quando os saltos de altura e largura são $s_h$ e $s_w$, respectivamente,
chamamos o salto de $(s_h, s_w)$.
Especificamente, quando $s_h = s_w = s$, , o salto é $s$.
Por padrão, o preenchimento é 0 e a salto é 1.
Na prática, raramente usamos saltos não homogêneos ou preenchimento,
ou seja, geralmente temos $p_h = p_w$ e $s_h = s_w$.

## Resumo

* O preenchimento pode aumentar a altura e a largura da saída. Isso geralmente é usado para dar à saída a mesma altura e largura da entrada.
* Os saltos podem reduzir a resolução da saída, por exemplo, reduzindo a altura e largura da saída para apenas $1/n$ da altura e largura da entrada ($n$ é um número inteiro maior que $1$).
* Preenchimento e saltos podem ser usados para ajustar a dimensionalidade dos dados de forma eficaz.

## Exercises

1. Para o último exemplo nesta seção, use matemática para calcular a forma de saída para ver se é consistente com o resultado experimental.
1. Experimente outras combinações de preenchimento e saltos nos experimentos desta seção.
1. Para sinais de áudio, a que corresponde um salto de 2?
1. Quais são os benefícios computacionais de uma salto maior que 1?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/67)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/68)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/272)
:end_tab:
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTE0NDU4MTAyODcsMjEzMjMzMzgwMiwtND
Y5Mjc4NjA3LC0xODk2Nzc1NDk1LC0xNDgzMTY2NjEwLDE1NTY5
NTczNDgsMTk3NDUwNDg5MiwtOTA0ODM3MzZdfQ==
-->