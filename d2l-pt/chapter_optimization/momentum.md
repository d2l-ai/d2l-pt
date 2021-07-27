# Momentum
:label:`sec_momentum`

Em :numref:`sec_sgd`, revisamos o que acontece ao realizar a descida do gradiente estocástico, ou seja, ao realizar a otimização onde apenas uma variante barulhenta do gradiente está disponível. Em particular, notamos que, para gradientes ruidosos, precisamos ser extremamente cautelosos ao escolher a taxa de aprendizado em face do ruído. Se diminuirmos muito rapidamente, a convergência para. Se formos tolerantes demais, não conseguiremos convergir para uma solução boa o suficiente, pois o ruído continua nos afastando da otimização.

## Fundamentos

Nesta seção, exploraremos algoritmos de otimização mais eficazes, especialmente para certos tipos de problemas de otimização que são comuns na prática.

### Médias com vazamento

A seção anterior nos viu discutindo o minibatch SGD como um meio de acelerar a computação. Também teve o bom efeito colateral de que a média dos gradientes reduziu a quantidade de variância. O minibatch SGD pode ser calculado por:

$$ \ mathbf {g} _ {t, t-1} = \ partial _ {\ mathbf {w}}  \frac{1}{|\mathcal{B}_t|} \sum_{i \in \mathcal{B}_t} f(\ mathbf {x}_{ i}, \ mathbf {w }_{t-1}) = \frac{1}{|\mathcal{B}_t|} \sum_{i \in \mathcal{B}_t} \mathbf{h}_{i, t-1}.
$$

Para manter a notação simples, aqui usamos $\mathbf{h}_{i, t-1} = \partial_{\mathbf{w}} f(\mathbf{x}_i, \mathbf{w}_{t-1})$ como o SGD para a amostra $i$ usando os pesos atualizados no tempo $ t-1 $.
Seria bom se pudéssemos nos beneficiar do efeito da redução da variância, mesmo além da média dos gradientes em um minibatch. Uma opção para realizar esta tarefa é substituir o cálculo do gradiente por uma "média com vazamento":

$$\mathbf{v}_t = \beta \mathbf{v}_{t-1} + \mathbf{g}_{t, t-1}$$

por algum $\beta \in (0, 1)$. Isso substitui efetivamente o gradiente instantâneo por um que foi calculado em vários gradientes *anteriores*. $\mathbf{v}$ é chamado *momentum*. Ele acumula gradientes anteriores semelhantes a como uma bola pesada rolando pela paisagem da função objetivo se integra às forças passadas. Para ver o que está acontecendo com mais detalhes, vamos expandir $\mathbf{v}_t$ recursivamente em

$$\begin{aligned}
\mathbf{v}_t = \beta^2 \mathbf{v}_{t-2} + \beta \mathbf{g}_{t-1, t-2} + \mathbf{g}_{t, t-1}
= \ldots, = \sum_{\tau = 0}^{t-1} \beta^{\tau} \mathbf{g}_{t-\tau, t-\tau-1}.
\end{aligned}$$


$\beta$ grande equivale a uma média de longo alcance, enquanto $\beta$ pequeno equivale a apenas uma ligeira correção em relação a um método de gradiente. A nova substituição de gradiente não aponta mais para a direção da descida mais íngreme em uma instância particular, mas sim na direção de uma média ponderada de gradientes anteriores. Isso nos permite obter a maioria dos benefícios da média de um lote sem o custo de realmente calcular os gradientes nele. Iremos revisitar este procedimento de média com mais detalhes posteriormente.

O raciocínio acima formou a base para o que agora é conhecido como métodos de gradiente *acelerado*, como gradientes com momentum. Eles têm o benefício adicional de serem muito mais eficazes nos casos em que o problema de otimização é mal condicionado (ou seja, onde há algumas direções onde o progresso é muito mais lento do que em outras, parecendo um desfiladeiro estreito). Além disso, eles nos permitem calcular a média dos gradientes subsequentes para obter direções de descida mais estáveis. Na verdade, o aspecto da aceleração, mesmo para problemas convexos sem ruído, é uma das principais razões pelas quais o momentum funciona e por que funciona tão bem.

Como seria de esperar, devido ao seu momentum de eficácia, é um assunto bem estudado em otimização para aprendizado profundo e além. Veja, por exemplo, o belo [artigo expositivo](https://distill.pub/2017/momentum/) por :cite:`Goh.2017` para uma análise aprofundada e animação interativa. Foi proposto por :cite:`Polyak.1964`. :cite:`Nesterov.2018` tem uma discussão teórica detalhada no contexto da otimização convexa. O momentum no aprendizado profundo é conhecido por ser benéfico há muito tempo. Veja, por exemplo, a discussão de :cite:`Sutskever.Martens.Dahl.ea.2013` para obter detalhes.

### Um problema mal condicionado

Para obter uma melhor compreensão das propriedades geométricas do método do momento, revisitamos a descida do gradiente, embora com uma função objetivo significativamente menos agradável. Lembre-se de que em :numref:`sec_gd` usamos $f(\mathbf{x}) = x_1^2 + 2 x_2^2$, ou seja, um objetivo elipsóide moderadamente distorcido. Distorcemos esta função ainda mais estendendo-a na direção $x_1$ por meio de

$$f(\mathbf{x}) = 0.1 x_1^2 + 2 x_2^2.$$

Como antes, $f$ tem seu mínimo em $(0, 0)$. Esta função é *muito* plana na direção de $x_1$. Vamos ver o que acontece quando executamos a descida gradiente como antes nesta nova função. Escolhemos uma taxa de aprendizagem de $0,4$.

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import np, npx
npx.set_np()

eta = 0.4
def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2
def gd_2d(x1, x2, s1, s2):
    return (x1 - eta * 0.2 * x1, x2 - eta * 4 * x2, 0, 0)

d2l.show_trace_2d(f_2d, d2l.train_2d(gd_2d))
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch

eta = 0.4
def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2
def gd_2d(x1, x2, s1, s2):
    return (x1 - eta * 0.2 * x1, x2 - eta * 4 * x2, 0, 0)

d2l.show_trace_2d(f_2d, d2l.train_2d(gd_2d))
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf

eta = 0.4
def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2
def gd_2d(x1, x2, s1, s2):
    return (x1 - eta * 0.2 * x1, x2 - eta * 4 * x2, 0, 0)

d2l.show_trace_2d(f_2d, d2l.train_2d(gd_2d))
```

Por construção, o gradiente na direção $x_2$ é *muito* maior e muda muito mais rapidamente do que na direção $x_1$ horizontal. Portanto, estamos presos entre duas escolhas indesejáveis: se escolhermos uma pequena taxa de aprendizado, garantimos que a solução não diverge na direção $x_2$, mas estamos sobrecarregados com uma convergência lenta na direção $x_1$. Por outro lado, com uma grande taxa de aprendizado, progredimos rapidamente na direção $x_1$, mas divergimos em $x_2$. O exemplo abaixo ilustra o que acontece mesmo após um ligeiro aumento na taxa de aprendizagem de $0,4$ para $0,6$. A convergência na direção $x_1$ melhora, mas a qualidade geral da solução é muito pior.

```{.python .input}
#@tab all
eta = 0.6
d2l.show_trace_2d(f_2d, d2l.train_2d(gd_2d))
```

###  Método Momentum

O método do momento nos permite resolver o problema de descida gradiente descrito
acima de. Olhando para o traço de otimização acima, podemos intuir que calcular a média de gradientes em relação ao passado funcionaria bem. Afinal, na direção $x_1$, isso agregará gradientes bem alinhados, aumentando assim a distância que percorremos a cada passo. Por outro lado, na direção $x_2$ onde os gradientes oscilam, um gradiente agregado reduzirá o tamanho do passo devido às oscilações que se cancelam.
Usar $\mathbf{v}_t$ em vez do gradiente $\mathbf{g}_t$ produz as seguintes equações de atualização:

$$
\begin{aligned}
\mathbf{v}_t &\leftarrow \beta \mathbf{v}_{t-1} + \mathbf{g}_{t, t-1}, \\
\mathbf{x}_t &\leftarrow \mathbf{x}_{t-1} - \eta_t \mathbf{v}_t.
\end{aligned}
$$

Observe que para $\beta = 0$ recuperamos a descida gradiente regular. Antes de nos aprofundarmos nas propriedades matemáticas, vamos dar uma olhada rápida em como o algoritmo se comporta na prática.

```{.python .input}
#@tab all
def momentum_2d(x1, x2, v1, v2):
    v1 = beta * v1 + 0.2 * x1
    v2 = beta * v2 + 4 * x2
    return x1 - eta * v1, x2 - eta * v2, v1, v2

eta, beta = 0.6, 0.5
d2l.show_trace_2d(f_2d, d2l.train_2d(momentum_2d))
```

Como podemos ver, mesmo com a mesma taxa de aprendizado que usamos antes, o momentum ainda converge bem. Vamos ver o que acontece quando diminuímos o parâmetro momentum. Reduzi-lo para $\beta = 0,25$ leva a uma trajetória que quase não converge. No entanto, é muito melhor do que sem momentum (quando a solução diverge).

```{.python .input}
#@tab all
eta, beta = 0.6, 0.25
d2l.show_trace_2d(f_2d, d2l.train_2d(momentum_2d))
```

Observe que podemos combinar momentum com SGD e, em particular, minibatch-SGD. A única mudança é que, nesse caso, substituímos os gradientes $\mathbf{g}_{t, t-1}$ por $\mathbf{g}_t$. Por último, por conveniência, inicializamos $\mathbf{v}_0 = 0$ no momento $t=0$. Vejamos o que a média de vazamento realmente faz com as atualizações.

### Peso Efetivo da Amostra

Lembre-se de que $\mathbf{v}_t = \sum_{\tau = 0}^{t-1} \beta^{\tau} \mathbf{g}_{t-\tau, t-\tau-1}$. No limite, os termos somam $\sum_{\tau=0}^\infty \beta^\tau = \frac{1}{1-\beta}$. Em outras palavras, em vez de dar um passo de tamanho $\eta$ em GD ou SGD, damos um passo de tamanho $\frac{\eta}{1-\beta}$ enquanto, ao mesmo tempo, lidamos com um potencial muito direção de descida melhor comportada. Esses são dois benefícios em um. Para ilustrar como a ponderação se comporta para diferentes escolhas de $\beta$, considere o diagrama abaixo.

```{.python .input}
#@tab all
d2l.set_figsize()
betas = [0.95, 0.9, 0.6, 0]
for beta in betas:
    x = d2l.numpy(d2l.arange(40))
    d2l.plt.plot(x, beta ** x, label=f'beta = {beta:.2f}')
d2l.plt.xlabel('time')
d2l.plt.legend();
```

## Experimentos Práticos

Vamos ver como o momentum funciona na prática, ou seja, quando usado no contexto de um otimizador adequado. Para isso, precisamos de uma implementação um pouco mais escalonável.

### Implementação do zero

Em comparação com (minibatch) SGD, o método de momentum precisa manter um conjunto de variáveis auxiliares, ou seja, a velocidade. Tem a mesma forma dos gradientes (e variáveis do problema de otimização). Na implementação abaixo, chamamos essas variáveis de `estados`.

```{.python .input}
#@tab mxnet,pytorch
def init_momentum_states(feature_dim):
    v_w = d2l.zeros((feature_dim, 1))
    v_b = d2l.zeros(1)
    return (v_w, v_b)
```

```{.python .input}
#@tab tensorflow
def init_momentum_states(features_dim):
    v_w = tf.Variable(d2l.zeros((features_dim, 1)))
    v_b = tf.Variable(d2l.zeros(1))
    return (v_w, v_b)
```

```{.python .input}
def sgd_momentum(params, states, hyperparams):
    for p, v in zip(params, states):
        v[:] = hyperparams['momentum'] * v + p.grad
        p[:] -= hyperparams['lr'] * v
```

```{.python .input}
#@tab pytorch
def sgd_momentum(params, states, hyperparams):
    for p, v in zip(params, states):
        with torch.no_grad():
            v[:] = hyperparams['momentum'] * v + p.grad
            p[:] -= hyperparams['lr'] * v
        p.grad.data.zero_()
```

```{.python .input}
#@tab tensorflow
def sgd_momentum(params, grads, states, hyperparams):
    for p, v, g in zip(params, states, grads):
            v[:].assign(hyperparams['momentum'] * v + g)
            p[:].assign(p - hyperparams['lr'] * v)
```

Vamos ver como isso funciona na prática.

```{.python .input}
#@tab all
def train_momentum(lr, momentum, num_epochs=2):
    d2l.train_ch11(sgd_momentum, init_momentum_states(feature_dim),
                   {'lr': lr, 'momentum': momentum}, data_iter,
                   feature_dim, num_epochs)

data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
train_momentum(0.02, 0.5)
```

Quando aumentamos o hiperparâmetro de momento `momentum` para 0,9, resulta em um tamanho de amostra efetivo significativamente maior de $\frac{1}{1 - 0.9} = 10$. Reduzimos ligeiramente a taxa de aprendizagem para $0,01$ para manter os assuntos sob controle.

```{.python .input}
#@tab all
train_momentum(0.01, 0.9)
```

A redução da taxa de aprendizagem resolve ainda mais qualquer questão de problemas de otimização não suave. Configurá-lo como $0,005$ produz boas propriedades de convergência.

```{.python .input}
#@tab all
train_momentum(0.005, 0.9)
```

### Implementação concisa

Há muito pouco a fazer no Gluon, uma vez que o solucionador `sgd` padrão já tem o momentum embutido. A configuração dos parâmetros correspondentes produz uma trajetória muito semelhante.

```{.python .input}
d2l.train_concise_ch11('sgd', {'learning_rate': 0.005, 'momentum': 0.9},
                       data_iter)
```

```{.python .input}
#@tab pytorch
trainer = torch.optim.SGD
d2l.train_concise_ch11(trainer, {'lr': 0.005, 'momentum': 0.9}, data_iter)
```

```{.python .input}
#@tab tensorflow
trainer = tf.keras.optimizers.SGD
d2l.train_concise_ch11(trainer, {'learning_rate': 0.005, 'momentum': 0.9},
                       data_iter)
```

## Análise teórica

Até agora, o exemplo 2D de $f(x) = 0.1 x_1^2 + 2 x_2^2$ parecia bastante artificial. Veremos agora que isso é na verdade bastante representativo dos tipos de problemas que podemos encontrar, pelo menos no caso de minimizar funções objetivas quadráticas convexas.

### Funções quadráticas convexas

Considere a função

$$h(\mathbf{x}) = \frac{1}{2} \mathbf{x}^\top \mathbf{Q} \mathbf{x} + \mathbf{x}^\top \mathbf{c} + b.$$

Esta é uma função quadrática geral. Para matrizes definidas positivas $\mathbf{Q} \succ 0$, ou seja, para matrizes com autovalores positivos, tem um minimizador em $\mathbf{x}^* = -\mathbf{Q}^{-1} \mathbf{c}$ com valor mínimo $b - \frac{1}{2} \mathbf{c}^\top \mathbf{Q}^{-1} \mathbf{c}$. Portanto, podemos reescrever $h$ como

$$h(\mathbf{x}) = \frac{1}{2} (\mathbf{x} - \mathbf{Q}^{-1} \mathbf{c})^\top \mathbf{Q} (\mathbf{x} - \mathbf{Q}^{-1} \mathbf{c}) + b - \frac{1}{2} \mathbf{c}^\top \mathbf{Q}^{-1} \mathbf{c}.$$

O gradiente é dado por $\partial_{\mathbf{x}} f(\mathbf{x}) = \mathbf{Q} (\mathbf{x} - \mathbf{Q}^{-1} \mathbf{c})$. Ou seja, é dada pela distância entre $\mathbf{x}$ e o minimizador, multiplicada por $\mathbf{Q}$. Consequentemente, também o momento é uma combinação linear de termos $\mathbf{Q} (\mathbf{x}_t - \mathbf{Q}^{-1} \mathbf{c})$.

Uma vez que $\mathbf{Q}$ é definido positivo, pode ser decomposto em seu auto-sistema via $\mathbf{Q} = \mathbf{O}^\top \boldsymbol{\Lambda} \mathbf{O}$ para um ortogonal ( rotação) matriz $\mathbf{O}$ e uma matriz diagonal $\boldsymbol{\Lambda}$ de autovalores positivos. Isso nos permite realizar uma mudança de variáveis de $\mathbf{x}$ para $\mathbf{z} := \mathbf{O} (\mathbf{x} - \mathbf{Q}^{-1} \mathbf{c})$ para obter uma expressão muito simplificada:

$$h(\mathbf{z}) = \frac{1}{2} \mathbf{z}^\top \boldsymbol{\Lambda} \mathbf{z} + b'.$$

Aqui $c' = b - \frac{1}{2} \mathbf{c}^\top \mathbf{Q}^{-1} \mathbf{c}$. Uma vez que $\mathbf{O}$ é apenas uma matriz ortogonal, isso não perturba os gradientes de uma forma significativa. Expresso em termos de $\mathbf{z}$ gradiente, a descida torna-se

$$\mathbf{z}_t = \mathbf{z}_{t-1} - \boldsymbol{\Lambda} \mathbf{z}_{t-1} = (\mathbf{I} - \boldsymbol{\Lambda}) \mathbf{z}_{t-1}.$$

O fato importante nesta expressão é que a descida gradiente *não se mistura* entre diferentes espaços auto. Ou seja, quando expresso em termos do autossistema de $\mathbf{Q}$, o problema de otimização ocorre de maneira coordenada. Isso também vale para o momento.

$$\begin{aligned}
\mathbf{v}_t & = \beta \mathbf{v}_{t-1} + \boldsymbol{\Lambda} \mathbf{z}_{t-1} \\
\mathbf{z}_t & = \mathbf{z}_{t-1} - \eta \left(\beta \mathbf{v}_{t-1} + \boldsymbol{\Lambda} \mathbf{z}_{t-1}\right) \\
    & = (\mathbf{I} - \eta \boldsymbol{\Lambda}) \mathbf{z}_{t-1} - \eta \beta \mathbf{v}_{t-1}.
\end{aligned}$$

Ao fazer isso, acabamos de provar o seguinte teorema: Gradiente descendente com e sem momento para uma função quadrática convexa se decompõe em otimização coordenada na direção dos vetores próprios da matriz quadrática.

### Funções Escalares

Dado o resultado acima, vamos ver o que acontece quando minimizamos a função $f(x) = \frac{\lambda}{2} x^2$. Para descida gradiente, temos

$$x_{t+1} = x_t - \eta \lambda x_t = (1 - \eta \lambda) x_t.$$

Sempre que $|1 - \eta \lambda| < 1$ esta otimização converge a uma taxa exponencial, pois após $t$ passos temos $x_t = (1 - \eta \lambda)^t x_0$. Isso mostra como a taxa de convergência melhora inicialmente à medida que aumentamos a taxa de aprendizado $\eta$ até $\eta \lambda = 1$. Além disso, as coisas divergem e para $\eta \lambda > 2$ o problema de otimização diverge.

```{.python .input}
#@tab all
lambdas = [0.1, 1, 10, 19]
eta = 0.1
d2l.set_figsize((6, 4))
for lam in lambdas:
    t = d2l.numpy(d2l.arange(20))
    d2l.plt.plot(t, (1 - eta * lam) ** t, label=f'lambda = {lam:.2f}')
d2l.plt.xlabel('time')
d2l.plt.legend();
```

Para analisar a convergência no caso de momentum, começamos reescrevendo as equações de atualização em termos de dois escalares: um para $x$ e outro para o momentum $v$. Isso produz:

$$
\begin{bmatrix} v_{t+1} \\ x_{t+1} \end{bmatrix} =
\begin{bmatrix} \beta & \lambda \\ -\eta \beta & (1 - \eta \lambda) \end{bmatrix}
\begin{bmatrix} v_{t} \\ x_{t} \end{bmatrix} = \mathbf{R}(\beta, \eta, \lambda) \begin{bmatrix} v_{t} \\ x_{t} \end{bmatrix}.
$$

Usamos $\mathbf{R}$ para denotar o comportamento de convergência que rege $2 \times 2$. Após $t$ passos, a escolha inicial $[v_0, x_0]$ torna-se $\mathbf{R}(\beta, \eta, \lambda)^t [v_0, x_0]$. Consequentemente, cabe aos autovalores de $\mathbf{R}$ determinar a velocidade de convergência. Veja o [Post do Destill](https://distill.pub/2017/momentum/) de :cite:`Goh.2017` para uma ótima animação e :cite:`Flammarion.Bach.2015` para uma análise detalhada. Pode-se mostrar que $0 < \eta \lambda < 2 + 2 \beta$ momentum converge. Este é um intervalo maior de parâmetros viáveis quando comparado a $0 < \eta \lambda < 2$ para descida de gradiente. Também sugere que, em geral, grandes valores de $\beta$ são desejáveis. Mais detalhes requerem uma boa quantidade de detalhes técnicos e sugerimos que o leitor interessado consulte as publicações originais.

## Sumário

* Momentum substitui gradientes por uma média com vazamento em relação aos gradientes anteriores. Isso acelera a convergência significativamente.
* É desejável tanto para descida gradiente sem ruído quanto para descida gradiente estocástica (ruidosa).
* O momentum evita a paralisação do processo de otimização, que é muito mais provável de ocorrer na descida do gradiente estocástico.
* O número efetivo de gradientes é dado por $\frac{1}{1-\beta}$ devido à redução exponenciada de dados anteriores.
* No caso de problemas quadráticos convexos, isso pode ser analisado explicitamente em detalhes.
* A implementação é bastante direta, mas exige que armazenemos um vetor de estado adicional (momentum $\mathbf{v}$).

## Exercícios

1. Use outras combinações de hiperparâmetros de momentum e taxas de aprendizagem e observe e analise os diferentes resultados experimentais.
1. Experimente GD e momentum para um problema quadrático onde você tem vários autovalores, ou seja, $f(x) = \frac{1}{2} \sum_i \lambda_i x_i^2$ ou seja $\lambda_i = 2^{-i}$. Trace como os valores de $x$ diminuem para a inicialização $x_i = 1$.
1. Derive o valor mínimo e minimizador para $h(\mathbf{x}) = \frac{1}{2} \mathbf{x}^\top \mathbf{Q} \mathbf{x} + \mathbf{x}^\top \mathbf{c} + b$.
1. O que muda quando executamos SGD com momentum? O que acontece quando usamos minibatch SGD com momentum? Experimentar com os parâmetros?

:begin_tab:`mxnet`
[Discussão](https://discuss.d2l.ai/t/354)
:end_tab:

:begin_tab:`pytorch`
[Discussão](https://discuss.d2l.ai/t/1070)
:end_tab:


:begin_tab:`tensorflow`
[Discussão](https://discuss.d2l.ai/t/1071)
:end_tab:
<!--stackedit_data:
eyJoaXN0b3J5IjpbNjAzMTUxMjM5LDIwNzIzMzY5NzgsMTIzMD
gxOTc4OSwxMjcxMzM4NzIzXX0=
-->