# Distribuições
:label:`sec_distributions`

Agora que aprendemos como trabalhar com probabilidade tanto na configuração discreta quanto na contínua, vamos conhecer algumas das distribuições comuns encontradas. Dependendo da área de *machine learning*, podemos precisar estar familiarizados com muito mais delas ou, para algumas áreas de *deep learning*, possivelmente nenhuma. Esta é, no entanto, uma boa lista básica para se familiarizar. Vamos primeiro importar algumas bibliotecas comuns.

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from IPython import display
from math import erf, factorial
import numpy as np
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
from IPython import display
from math import erf, factorial
import torch

torch.pi = torch.acos(torch.zeros(1)) * 2  # Define pi in torch
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
from IPython import display
from math import erf, factorial
import tensorflow as tf
import tensorflow_probability as tfp

tf.pi = tf.acos(tf.zeros(1)) * 2  # Define pi in TensorFlow
```

## Bernoulli

Esta é a variável aleatória mais simples normalmente encontrada. Esta variável aleatória codifica um lançamento de moeda que dá $1$ com probabilidade $p$ e $0$ com probabilidade $1-p$. Se tivermos uma variável aleatória $X$ com esta distribuição, vamos escrever

$$
X \sim \mathrm{Bernoulli}(p).
$$

A função de distribuição cumulativa é

$$F(x) = \begin{cases} 0 & x < 0, \\ 1-p & 0 \le x < 1, \\ 1 & x >= 1 . \end{cases}$$
:eqlabel:`eq_bernoulli_cdf`

A função de massa de probabilidade está representada abaixo.

```{.python .input}
#@tab all
p = 0.3

d2l.set_figsize()
d2l.plt.stem([0, 1], [1 - p, p], use_line_collection=True)
d2l.plt.xlabel('x')
d2l.plt.ylabel('p.m.f.')
d2l.plt.show()
```

Agora, vamos representar graficamente a função de distribuição cumulativa :eqref:`eq_bernoulli_cdf`.

```{.python .input}
x = np.arange(-1, 2, 0.01)

def F(x):
    return 0 if x < 0 else 1 if x > 1 else 1 - p

d2l.plot(x, np.array([F(y) for y in x]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab pytorch
x = torch.arange(-1, 2, 0.01)

def F(x):
    return 0 if x < 0 else 1 if x > 1 else 1 - p

d2l.plot(x, torch.tensor([F(y) for y in x]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab tensorflow
x = tf.range(-1, 2, 0.01)

def F(x):
    return 0 if x < 0 else 1 if x > 1 else 1 - p

d2l.plot(x, tf.constant([F(y) for y in x]), 'x', 'c.d.f.')
```

Se $X \sim \mathrm{Bernoulli}(p)$, então:

* $\mu_X = p$,
* $\sigma_X^2 = p(1-p)$.

Podemos amostrar uma matriz de forma arbitrária de uma variável aleatória de Bernoulli como segue.

```{.python .input}
1*(np.random.rand(10, 10) < p)
```

```{.python .input}
#@tab pytorch
1*(torch.rand(10, 10) < p)
```

```{.python .input}
#@tab tensorflow
tf.cast(tf.random.uniform((10, 10)) < p, dtype=tf.float32)
```

## Uniforme e Discreta

A próxima variável aleatória comumente encontrada é uma uniforme discreta. Para nossa discussão aqui, assumiremos que é suportada nos inteiros $\{1, 2, \ldots, n\}$, entretanto qualquer outro conjunto de valores pode ser escolhido livremente. O significado da palavra *uniforme* neste contexto é que todos os valores possíveis são igualmente prováveis. A probabilidade para cada valor $i \in \{1, 2, 3, \ldots, n\}$ é $p_i = \frac{1}{n}$. Vamos denotar uma variável aleatória $X$ com esta distribuição como

$$
X \sim U(n).
$$

A função de distribuição cumulativa é

$$F(x) = \begin{cases} 0 & x < 1, \\ \frac{k}{n} & k \le x < k+1 \text{ with } 1 \le k < n, \\ 1 & x >= n . \end{cases}$$
:eqlabel:`eq_discrete_uniform_cdf`

Deixe-nos primeiro representar graficamente a função de massa de probabilidade.

```{.python .input}
#@tab all
n = 5

d2l.plt.stem([i+1 for i in range(n)], n*[1 / n], use_line_collection=True)
d2l.plt.xlabel('x')
d2l.plt.ylabel('p.m.f.')
d2l.plt.show()
```

Agora, vamos representar graficamente a função de distribuição cumulativa: eqref:`eq_discrete_uniform_cdf`.

```{.python .input}
x = np.arange(-1, 6, 0.01)

def F(x):
    return 0 if x < 1 else 1 if x > n else np.floor(x) / n

d2l.plot(x, np.array([F(y) for y in x]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab pytorch
x = torch.arange(-1, 6, 0.01)

def F(x):
    return 0 if x < 1 else 1 if x > n else torch.floor(x) / n

d2l.plot(x, torch.tensor([F(y) for y in x]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab tensorflow
x = tf.range(-1, 6, 0.01)

def F(x):
    return 0 if x < 1 else 1 if x > n else tf.floor(x) / n

d2l.plot(x, [F(y) for y in x], 'x', 'c.d.f.')
```

If $X \sim U(n)$, then:

* $\mu_X = \frac{1+n}{2}$,
* $\sigma_X^2 = \frac{n^2-1}{12}$.

Podemos amostrar uma matriz de forma arbitrária a partir de uma variável aleatória uniforme discreta como segue.

```{.python .input}
np.random.randint(1, n, size=(10, 10))
```

```{.python .input}
#@tab pytorch
torch.randint(1, n, size=(10, 10))
```

```{.python .input}
#@tab tensorflow
tf.random.uniform((10, 10), 1, n, dtype=tf.int32)
```

## Uniforme e Contínua

A seguir, vamos discutir a distribuição uniforme contínua. A ideia por trás dessa variável aleatória é que, se aumentarmos $n$ na distribuição uniforme discreta e, em seguida, escaloná-la para caber no intervalo $[a, b]$, abordaremos uma variável aleatória contínua que apenas escolhe um valor arbitrário em $[a, b]$ todos com probabilidade igual. Vamos denotar esta distribuição como

$$
X \sim U(a, b).
$$

A função de densidade de probabilidade é

$$p(x) = \begin{cases} \frac{1}{b-a} & x \in [a, b], \\ 0 & x \not\in [a, b].\end{cases}$$
:eqlabel:`eq_cont_uniform_pdf`

A função de distribuição cumulativa é

$$F(x) = \begin{cases} 0 & x < a, \\ \frac{x-a}{b-a} & x \in [a, b], \\ 1 & x >= b . \end{cases}$$
:eqlabel:`eq_cont_uniform_cdf`

Vamos primeiro representar graficamente a função de densidade de probabilidade :eqref:`eq_cont_uniform_pdf`.

```{.python .input}
a, b = 1, 3

x = np.arange(0, 4, 0.01)
p = (x > a)*(x < b)/(b - a)

d2l.plot(x, p, 'x', 'p.d.f.')
```

```{.python .input}
#@tab pytorch
a, b = 1, 3

x = torch.arange(0, 4, 0.01)
p = (x > a).type(torch.float32)*(x < b).type(torch.float32)/(b-a)
d2l.plot(x, p, 'x', 'p.d.f.')
```

```{.python .input}
#@tab tensorflow
a, b = 1, 3

x = tf.range(0, 4, 0.01)
p = tf.cast(x > a, tf.float32) * tf.cast(x < b, tf.float32) / (b - a)
d2l.plot(x, p, 'x', 'p.d.f.')
```

Agora, vamos representar graficamente a função de distribuição cumulativa :eqref:`eq_cont_uniform_cdf`.

```{.python .input}
def F(x):
    return 0 if x < a else 1 if x > b else (x - a) / (b - a)

d2l.plot(x, np.array([F(y) for y in x]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab pytorch
def F(x):
    return 0 if x < a else 1 if x > b else (x - a) / (b - a)

d2l.plot(x, torch.tensor([F(y) for y in x]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab tensorflow
def F(x):
    return 0 if x < a else 1 if x > b else (x - a) / (b - a)

d2l.plot(x, [F(y) for y in x], 'x', 'c.d.f.')
```

Se $X \sim U(a, b)$, então:

* $\mu_X = \frac{a+b}{2}$,
* $\sigma_X^2 = \frac{(b-a)^2}{12}$.

Podemos amostrar uma matriz de forma arbitrária a partir de uma variável aleatória uniforme da seguinte maneira. Observe que, por padrão, é uma amostra de $U(0,1)$, portanto, se quisermos um intervalo diferente, precisamos escaloná-lo.

```{.python .input}
(b - a) * np.random.rand(10, 10) + a
```

```{.python .input}
#@tab pytorch
(b - a) * torch.rand(10, 10) + a
```

```{.python .input}
#@tab tensorflow
(b - a) * tf.random.uniform((10, 10)) + a
```

## Binomial


Deixe-nos tornar as coisas um pouco mais complexas e examinar a variável aleatória *binomial*. Essa variável aleatória se origina da execução de uma sequência de $n$ experimentos independentes, cada um dos quais tem probabilidade $p$ de sucesso, e perguntando quantos sucessos esperamos ver.

Vamos expressar isso matematicamente. Cada experimento é uma variável aleatória independente $X_i$, onde usaremos $1$ para codificar o sucesso e $0$ para codificar a falha. Como cada um é um lançamento de moeda independente que é bem-sucedido com a probabilidade $p$, podemos dizer que $X_i \sim \mathrm{Bernoulli}(p)$. Então, a variável aleatória binomial é

$$
X = \sum_{i=1}^n X_i.
$$

Neste caso, vamos escrever

$$
X \sim \mathrm{Binomial}(n, p).
$$

Para obter a função de distribuição cumulativa, precisamos observar que obter exatamente $k$ sucessos podem ocorrer em $\binom{n}{k} = \frac{n!}{k!(n-k)!}$ maneiras, cada uma das quais tem uma probabilidade de $p^k(1-p)^{n-k}$ de ocorrer. Assim, a função de distribuição cumulativa é

$$F(x) = \begin{cases} 0 & x < 0, \\ \sum_{m \le k} \binom{n}{m} p^m(1-p)^{n-m}  & k \le x < k+1 \text{ with } 0 \le k < n, \\ 1 & x >= n . \end{cases}$$
:eqlabel:`eq_binomial_cdf`

Deixe-nos primeiro representar graficamente a função de massa de probabilidade.

```{.python .input}
n, p = 10, 0.2

# Compute binomial coefficient
def binom(n, k):
    comb = 1
    for i in range(min(k, n - k)):
        comb = comb * (n - i) // (i + 1)
    return comb

pmf = np.array([p**i * (1-p)**(n - i) * binom(n, i) for i in range(n + 1)])

d2l.plt.stem([i for i in range(n + 1)], pmf, use_line_collection=True)
d2l.plt.xlabel('x')
d2l.plt.ylabel('p.m.f.')
d2l.plt.show()
```

```{.python .input}
#@tab pytorch
n, p = 10, 0.2

# Compute binomial coefficient
def binom(n, k):
    comb = 1
    for i in range(min(k, n - k)):
        comb = comb * (n - i) // (i + 1)
    return comb

pmf = d2l.tensor([p**i * (1-p)**(n - i) * binom(n, i) for i in range(n + 1)])

d2l.plt.stem([i for i in range(n + 1)], pmf, use_line_collection=True)
d2l.plt.xlabel('x')
d2l.plt.ylabel('p.m.f.')
d2l.plt.show()
```

```{.python .input}
#@tab tensorflow
n, p = 10, 0.2

# Compute binomial coefficient
def binom(n, k):
    comb = 1
    for i in range(min(k, n - k)):
        comb = comb * (n - i) // (i + 1)
    return comb

pmf = tf.constant([p**i * (1-p)**(n - i) * binom(n, i) for i in range(n + 1)])

d2l.plt.stem([i for i in range(n + 1)], pmf, use_line_collection=True)
d2l.plt.xlabel('x')
d2l.plt.ylabel('p.m.f.')
d2l.plt.show()
```

Now, let us plot the cumulative distribution function :eqref:`eq_binomial_cdf`.

```{.python .input}
x = np.arange(-1, 11, 0.01)
cmf = np.cumsum(pmf)

def F(x):
    return 0 if x < 0 else 1 if x > n else cmf[int(x)]

d2l.plot(x, np.array([F(y) for y in x.tolist()]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab pytorch
x = torch.arange(-1, 11, 0.01)
cmf = torch.cumsum(pmf, dim=0)

def F(x):
    return 0 if x < 0 else 1 if x > n else cmf[int(x)]

d2l.plot(x, torch.tensor([F(y) for y in x.tolist()]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab tensorflow
x = tf.range(-1, 11, 0.01)
cmf = tf.cumsum(pmf)

def F(x):
    return 0 if x < 0 else 1 if x > n else cmf[int(x)]

d2l.plot(x, [F(y) for y in x.numpy().tolist()], 'x', 'c.d.f.')
```

Se $X \sim \mathrm{Binomial}(n, p)$, então:

* $\mu_X = np$,
* $\sigma_X^2 = np(1-p)$.

Isso decorre da linearidade do valor esperado sobre a soma das $n$ variáveis aleatórias de Bernoulli e do fato de que a variância da soma das variáveis aleatórias independentes é a soma das variâncias. Isso pode ser amostrado da seguinte maneira.

```{.python .input}
np.random.binomial(n, p, size=(10, 10))
```

```{.python .input}
#@tab pytorch
m = torch.distributions.binomial.Binomial(n, p)
m.sample(sample_shape=(10, 10))
```

```{.python .input}
#@tab tensorflow
m = tfp.distributions.Binomial(n, p)
m.sample(sample_shape=(10, 10))
```

## Poisson

Vamos agora realizar um experimento mental. Estamos parados em um ponto de ônibus e queremos saber quantos ônibus chegarão no próximo minuto. Vamos começar considerando $X^{(1)} \sim \mathrm{Bernoulli}(p)$ que é simplesmente a probabilidade de que um ônibus chegue na janela de um minuto. Para paradas de ônibus longe de um centro urbano, essa pode ser uma boa aproximação. Podemos nunca ver mais de um ônibus por minuto.

Porém, se estivermos em uma área movimentada, é possível ou mesmo provável que cheguem dois ônibus. Podemos modelar isso dividindo nossa variável aleatória em duas partes nos primeiros 30 segundos ou nos segundos 30 segundos. Neste caso, podemos escrever

$$
X^{(2)} \sim X^{(2)}_1 + X^{(2)}_2,
$$

onde $X^{(2)}$ é a soma total, e $X^{(2)}_i \sim \mathrm{Bernoulli}(p/2)$. A distribuição total é então $X^{(2)} \sim \mathrm{Binomial}(2, p/2)$.

Why stop here?  Let us continue to split that minute into $n$ parts.  By the same reasoning as above, we see that

$$X^{(n)} \sim \mathrm{Binomial}(n, p/n).$$
:eqlabel:`eq_eq_poisson_approx`

Considere essas variáveis aleatórias. Pela seção anterior, sabemos que :eqref:`eq_eq_poisson_approx` tem média $\mu_{X^{(n)}} = n(p/n) = p$, e variância $\sigma_{X^{(n)}}^2 = n(p/n)(1-(p/n)) = p(1-p/n)$. Se tomarmos $n \rightarrow \infty$, podemos ver que esses números se estabilizam em $\mu_{X^{(\infty)}} = p$, e variância $\sigma_{X^{(\infty)}}^2 = p$. Isso indica que *pode haver* alguma variável aleatória que podemos definir neste limite de subdivisão infinito.


Isso não deve ser uma surpresa, já que no mundo real podemos apenas contar o número de chegadas de ônibus, no entanto, é bom ver que nosso modelo matemático está bem definido. Essa discussão pode ser formalizada como a *lei dos eventos raros*.

Seguindo esse raciocínio com cuidado, podemos chegar ao seguinte modelo. Diremos que $X \sim \mathrm{Poisson}(\lambda)$ se for uma variável aleatória que assume os valores $\{0,1,2, \ldots\}$ com probabilidade

$$p_k = \frac{\lambda^ke^{-\lambda}}{k!}.$$
:eqlabel:`eq_poisson_mass`


O valor $\lambda > 0$ é conhecido como *taxa* (ou o parâmetro *forma*) e denota o número médio de chegadas que esperamos em uma unidade de tempo.

Podemos somar essa função de massa de probabilidade para obter a função de distribuição cumulativa.

$$F(x) = \begin{cases} 0 & x < 0, \\ e^{-\lambda}\sum_{m = 0}^k \frac{\lambda^m}{m!} & k \le x < k+1 \text{ with } 0 \le k. \end{cases}$$
:eqlabel:`eq_poisson_cdf`

Vamos primeiro representar graficamente a função de massa de probabilidade :eqref:`eq_poisson_mass`.

```{.python .input}
lam = 5.0

xs = [i for i in range(20)]
pmf = np.array([np.exp(-lam) * lam**k / factorial(k) for k in xs])

d2l.plt.stem(xs, pmf, use_line_collection=True)
d2l.plt.xlabel('x')
d2l.plt.ylabel('p.m.f.')
d2l.plt.show()
```

```{.python .input}
#@tab pytorch
lam = 5.0

xs = [i for i in range(20)]
pmf = torch.tensor([torch.exp(torch.tensor(-lam)) * lam**k
                    / factorial(k) for k in xs])

d2l.plt.stem(xs, pmf, use_line_collection=True)
d2l.plt.xlabel('x')
d2l.plt.ylabel('p.m.f.')
d2l.plt.show()
```

```{.python .input}
#@tab tensorflow
lam = 5.0

xs = [i for i in range(20)]
pmf = tf.constant([tf.exp(tf.constant(-lam)).numpy() * lam**k
                    / factorial(k) for k in xs])

d2l.plt.stem(xs, pmf, use_line_collection=True)
d2l.plt.xlabel('x')
d2l.plt.ylabel('p.m.f.')
d2l.plt.show()
```

Agora, vamos representar graficamente a função de distribuição cumulativa :eqref:`eq_poisson_cdf`.

```{.python .input}
x = np.arange(-1, 21, 0.01)
cmf = np.cumsum(pmf)
def F(x):
    return 0 if x < 0 else 1 if x > n else cmf[int(x)]

d2l.plot(x, np.array([F(y) for y in x.tolist()]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab pytorch
x = torch.arange(-1, 21, 0.01)
cmf = torch.cumsum(pmf, dim=0)
def F(x):
    return 0 if x < 0 else 1 if x > n else cmf[int(x)]

d2l.plot(x, torch.tensor([F(y) for y in x.tolist()]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab tensorflow
x = tf.range(-1, 21, 0.01)
cmf = tf.cumsum(pmf)
def F(x):
    return 0 if x < 0 else 1 if x > n else cmf[int(x)]

d2l.plot(x, [F(y) for y in x.numpy().tolist()], 'x', 'c.d.f.')
```

Como vimos acima, as médias e variações são particularmente concisas. Se $X \sim \mathrm{Poisson}(\lambda)$, então:

* $\mu_X = \lambda$,
* $\sigma_X^2 = \lambda$.

Isso pode ser amostrado da seguinte maneira.

```{.python .input}
np.random.poisson(lam, size=(10, 10))
```

```{.python .input}
#@tab pytorch
m = torch.distributions.poisson.Poisson(lam)
m.sample((10, 10))
```

```{.python .input}
#@tab tensorflow
m = tfp.distributions.Poisson(lam)
m.sample((10, 10))
```

## Gaussiana


Agora, vamos tentar um experimento diferente, mas relacionado. Digamos que estamos novamente realizando $n$ medidas independentes de $\mathrm{Bernoulli}(p)$ $X_i$. A distribuição da soma delas é $X^{(n)} \sim \mathrm{Binomial}(n, p)$. Em vez de considerar um limite à medida que $n$ aumenta e $p$ diminui, vamos corrigir $p$ e enviar $n \rightarrow \infty$. Neste caso $\mu_{X^{(n)}} = np \rightarrow \infty$ e $\sigma_{X^{(n)}}^2 = np(1-p) \rightarrow \infty$, portanto, não há razão para pensar que esse limite deva ser bem definido.

No entanto, nem toda esperança está perdida! Vamos apenas fazer com que a média e a variância sejam bem comportadas, definindo

$$
Y^{(n)} = \frac{X^{(n)} - \mu_{X^{(n)}}}{\sigma_{X^{(n)}}}.
$$

Pode-se ver que isso tem média zero e variância um e, portanto, é plausível acreditar que convergirá para alguma distribuição limitante. Se traçarmos a aparência dessas distribuições, ficaremos ainda mais convencidos de que funcionará.

```{.python .input}
p = 0.2
ns = [1, 10, 100, 1000]
d2l.plt.figure(figsize=(10, 3))
for i in range(4):
    n = ns[i]
    pmf = np.array([p**i * (1-p)**(n-i) * binom(n, i) for i in range(n + 1)])
    d2l.plt.subplot(1, 4, i + 1)
    d2l.plt.stem([(i - n*p)/np.sqrt(n*p*(1 - p)) for i in range(n + 1)], pmf,
                 use_line_collection=True)
    d2l.plt.xlim([-4, 4])
    d2l.plt.xlabel('x')
    d2l.plt.ylabel('p.m.f.')
    d2l.plt.title("n = {}".format(n))
d2l.plt.show()
```

```{.python .input}
#@tab pytorch
p = 0.2
ns = [1, 10, 100, 1000]
d2l.plt.figure(figsize=(10, 3))
for i in range(4):
    n = ns[i]
    pmf = torch.tensor([p**i * (1-p)**(n-i) * binom(n, i)
                        for i in range(n + 1)])
    d2l.plt.subplot(1, 4, i + 1)
    d2l.plt.stem([(i - n*p)/torch.sqrt(torch.tensor(n*p*(1 - p)))
                  for i in range(n + 1)], pmf,
                 use_line_collection=True)
    d2l.plt.xlim([-4, 4])
    d2l.plt.xlabel('x')
    d2l.plt.ylabel('p.m.f.')
    d2l.plt.title("n = {}".format(n))
d2l.plt.show()
```

```{.python .input}
#@tab tensorflow
p = 0.2
ns = [1, 10, 100, 1000]
d2l.plt.figure(figsize=(10, 3))
for i in range(4):
    n = ns[i]
    pmf = tf.constant([p**i * (1-p)**(n-i) * binom(n, i)
                        for i in range(n + 1)])
    d2l.plt.subplot(1, 4, i + 1)
    d2l.plt.stem([(i - n*p)/tf.sqrt(tf.constant(n*p*(1 - p)))
                  for i in range(n + 1)], pmf,
                 use_line_collection=True)
    d2l.plt.xlim([-4, 4])
    d2l.plt.xlabel('x')
    d2l.plt.ylabel('p.m.f.')
    d2l.plt.title("n = {}".format(n))
d2l.plt.show()
```


Uma coisa a observar: em comparação com o caso de Poisson, agora estamos dividindo pelo desvio padrão, o que significa que estamos comprimindo os resultados possíveis em áreas cada vez menores. Isso é uma indicação de que nosso limite não será mais discreto, mas sim contínuo.

Uma derivação do que ocorre está além do escopo deste documento, mas o *teorema do limite central* afirma que, como $n \rightarrow \infty$, isso resultará na Distribuição Gaussiana (ou as vezes na distribuição normal). Mais explicitamente, para qualquer $a, b$:

$$
\lim_{n \rightarrow \infty} P(Y^{(n)} \in [a, b]) = P(\mathcal{N}(0,1) \in [a, b]),
$$

onde dizemos que uma variável aleatória é normalmente distribuída com dada média $\mu$ e variância $\sigma^2$, escrita $X \sim \mathcal{N}(\mu, \sigma^2)$ se $X$ tem densidade

$$p_X(x) = \frac{1}{\sqrt{2\pi\sigma^2}}e^{-\frac{(x-\mu)^2}{2\sigma^2}}.$$
:eqlabel:`eq_gaussian_pdf`

Vamos primeiro representar graficamente a função de densidade de probabilidade :eqref:`eq_gaussian_pdf`.

```{.python .input}
mu, sigma = 0, 1

x = np.arange(-3, 3, 0.01)
p = 1 / np.sqrt(2 * np.pi * sigma**2) * np.exp(-(x - mu)**2 / (2 * sigma**2))

d2l.plot(x, p, 'x', 'p.d.f.')
```

```{.python .input}
#@tab pytorch
mu, sigma = 0, 1

x = torch.arange(-3, 3, 0.01)
p = 1 / torch.sqrt(2 * torch.pi * sigma**2) * torch.exp(
    -(x - mu)**2 / (2 * sigma**2))

d2l.plot(x, p, 'x', 'p.d.f.')
```

```{.python .input}
#@tab tensorflow
mu, sigma = 0, 1

x = tf.range(-3, 3, 0.01)
p = 1 / tf.sqrt(2 * tf.pi * sigma**2) * tf.exp(
    -(x - mu)**2 / (2 * sigma**2))

d2l.plot(x, p, 'x', 'p.d.f.')
```

Agora, vamos representar graficamente a função de distribuição cumulativa. Está além do escopo deste apêndice, mas a f.d.c. Gaussiana não tem uma fórmula de forma fechada em termos de funções mais elementares. Usaremos `erf`, que fornece uma maneira de calcular essa integral numericamente.

```{.python .input}
def phi(x):
    return (1.0 + erf((x - mu) / (sigma * np.sqrt(2)))) / 2.0

d2l.plot(x, np.array([phi(y) for y in x.tolist()]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab pytorch
def phi(x):
    return (1.0 + erf((x - mu) / (sigma * torch.sqrt(d2l.tensor(2.))))) / 2.0

d2l.plot(x, torch.tensor([phi(y) for y in x.tolist()]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab tensorflow
def phi(x):
    return (1.0 + erf((x - mu) / (sigma * tf.sqrt(tf.constant(2.))))) / 2.0

d2l.plot(x, [phi(y) for y in x.numpy().tolist()], 'x', 'c.d.f.')
```


Os leitores mais atentos reconhecerão alguns desses termos. Na verdade, encontramos essa integral em :numref:`sec_integral_calculus`. Na verdade, precisamos exatamente desse cálculo para ver que esse $p_X (x)$ tem área total um e, portanto, é uma densidade válida.

Nossa escolha de trabalhar com cara ou coroa tornou os cálculos mais curtos, mas nada nessa escolha foi fundamental. De fato, se tomarmos qualquer coleção de variáveis aleatórias independentes distribuídas de forma idêntica $X_i$, e formar

$$
X^{(N)} = \sum_{i=1}^N X_i.
$$

Then

$$
\frac{X^{(N)} - \mu_{X^{(N)}}}{\sigma_{X^{(N)}}}
$$


será aproximadamente gaussiana. Existem requisitos adicionais necessários para fazê-la funcionar, mais comumente $E[X^4] < \infty$, mas a filosofia é clara.

O teorema do limite central é a razão pela qual o Gaussiano é fundamental para probabilidade, estatística e aprendizado de máquina. Sempre que podemos dizer que algo que medimos é a soma de muitas pequenas contribuições independentes, podemos supor que o que está sendo medido será próximo de gaussiano.


Existem muitas outras propriedades fascinantes das gaussianas, e gostaríamos de discutir mais uma aqui. A Gaussiana é conhecida como *distribuição de entropia máxima*. Entraremos em entropia mais profundamente em :numref:`sec_information_theory`, no entanto, tudo o que precisamos saber neste ponto é que é uma medida de aleatoriedade. Em um sentido matemático rigoroso, podemos pensar no gaussiano como a escolha *mais* aleatória de variável aleatória com média e variância fixas. Portanto, se sabemos que nossa variável aleatória tem alguma média e variância, a Gaussiana é, de certo modo, a escolha de distribuição mais conservadora que podemos fazer.

Para fechar a seção, vamos lembrar que se $X \sim \mathcal{N}(\mu, \sigma^2)$, então:

* $\mu_X = \mu$,
* $\sigma_X^2 = \sigma^2$.

Podemos obter uma amostra da distribuição gaussiana (ou normal padrão), conforme mostrado abaixo.

```{.python .input}
np.random.normal(mu, sigma, size=(10, 10))
```

```{.python .input}
#@tab pytorch
torch.normal(mu, sigma, size=(10, 10))
```

```{.python .input}
#@tab tensorflow
tf.random.normal((10, 10), mu, sigma)
```

## Família Exponencial
:label:`subsec_exponential_family`

Uma propriedade compartilhada para todas as distribuições listadas acima é que todas
pertencem à conhecida como *família exponencial*. A família exponencial
é um conjunto de distribuições cuja densidade pode ser expressa no seguinte
Formato:

$$p(\mathbf{x} | \boldsymbol{\eta}) = h(\mathbf{x}) \cdot \mathrm{exp} \left( \boldsymbol{\eta}^{\top} \cdot T(\mathbf{x}) - A(\boldsymbol{\eta}) \right)$$
:eqlabel:`eq_exp_pdf`


Como essa definição pode ser um pouco sutil, vamos examiná-la de perto.

Primeiro, $h(\mathbf{x})$ é conhecido como a *medida subjacente* ou a
*medida de base*. Isso pode ser visto como uma escolha original da medida que estamos
modificando com nosso peso exponencial.

Em segundo lugar, temos o vetor $\boldsymbol{\eta} = (\eta_1, \eta_2, ..., \eta_l) \in\mathbb{R}^l$ chamado de *parâmetros naturais* ou *parâmetros canônicos*. Eles definem como a medida base será modificada. Os parâmetros naturais entram na nova medida tomando o produto escalar desses parâmetros em relação a alguma função $T(\cdot)$ de \mathbf{x}= (x_1, x_2, ..., x_n) \in\mathbb{R}^n$ e exponenciado. O vetor $T(\mathbf{x})= (T_1(\mathbf{x}), T_2(\mathbf{x}), ..., T_l(\mathbf{x}))$ é chamado de *estatísticas suficientes* para $\boldsymbol{\eta}$. Este nome é usado uma vez que a informação representada por $T(\mathbf{x})$ é suficiente para calcular a
densidade de probabilidade e nenhuma outra informação da amostra $\mathbf{x}$
é requerida.

Terceiro, temos $A(\boldsymbol{\eta})$, que é referido como a *função cumulativa*, que garante que a distribuição acima :eqref:`eq_exp_pdf`
integra-se a um, ou seja,

$$A(\boldsymbol{\eta})  = \log \left[\int h(\mathbf{x}) \cdot \mathrm{exp}
\left(\boldsymbol{\eta}^{\top} \cdot T(\mathbf{x}) \right) d\mathbf{x} \right].$$

Para sermos concretos, consideremos o gaussiano. Supondo que $\mathbf{x}$ seja
uma variável univariada, vimos que ela tinha uma densidade de

$$
\begin{aligned}
p(x | \mu, \sigma) &= \frac{1}{\sqrt{2 \pi \sigma^2}} \cdot \mathrm{exp} 
\left\{ \frac{-(x-\mu)^2}{2 \sigma^2} \right\} \\
&= \frac{1}{\sqrt{2 \pi}} \cdot \mathrm{exp} \left\{ \frac{\mu}{\sigma^2}x
-\frac{1}{2 \sigma^2} x^2 - \left( \frac{1}{2 \sigma^2} \mu^2
+\log(\sigma) \right) \right\}.
\end{aligned}
$$

Isso corresponde à definição da família exponencial com:

* *medida subjacente*: $h(x) = \frac{1}{\sqrt{2 \pi}}$,
* *parâmetros naturais*: $\boldsymbol{\eta} = \begin{bmatrix} \eta_1 \\ \eta_2
\end{bmatrix} = \begin{bmatrix} \frac{\mu}{\sigma^2} \\
\frac{1}{2 \sigma^2} \end{bmatrix}$,
* *estatísticas suficientes*: $T(x) = \begin{bmatrix}x\\-x^2\end{bmatrix}$, and
* *função cumulativa*: $A({\boldsymbol\eta}) = \frac{1}{2 \sigma^2} \mu^2 + \log(\sigma)
= \frac{\eta_1^2}{4 \eta_2} - \frac{1}{2}\log(2 \eta_2)$.

É importante notar que a escolha exata de cada um dos termos acima é um pouco
arbitrária. Na verdade, a característica importante é que a distribuição pode ser
expressa nesta forma, não na forma exata em si.

Como aludimos em :numref:`subsec_softmax_and_derivatives`, uma técnica amplamente utilizada é assumir que a saída final $\mathbf{y}$ segue uma
distribuição da família exponencial. A família exponencial é uma comum e poderosa família de distribuições encontradas com frequência no *machine learning*.


## Resumo
* Variáveis aleatórias de Bernoulli podem ser usadas para modelar eventos com um resultado sim/não.
* O modelo de distribuições uniformes discretas seleciona a partir de um conjunto finito de possibilidades.
* Distribuições uniformes contínuas selecionam a partir de um intervalo.
* As distribuições binomiais modelam uma série de variáveis aleatórias de Bernoulli e contam o número de sucessos.
* Variáveis aleatórias de Poisson modelam a chegada de eventos raros.
* Variáveis aleatórias gaussianas modelam o resultado da adição de um grande número de variáveis aleatórias independentes.
* Todas as distribuições acima pertencem à família exponencial.

## Exercícios

1. Qual é o desvio padrão de uma variável aleatória que é a diferença $X-Y$ de duas variáveis aleatórias binomiais independentes $X, Y \sim \mathrm{Binomial}(16, 1/2)$.
2. Se tomarmos uma variável aleatória de Poisson$X \sim \mathrm{Poisson}(\lambda)$ e considerar $(X - \lambda)/\sqrt{\lambda}$ como $\lambda \rightarrow \infty$, podemos mostrar que isso se torna aproximadamente gaussiano. Por que isso faz sentido?
3. Qual é a função de massa de probabilidade para uma soma de duas variáveis aleatórias uniformes discretas em $n$ elementos?


:begin_tab:`mxnet`
[Discussões](https://discuss.d2l.ai/t/417)
:end_tab:

:begin_tab:`pytorch`
[Discussões](https://discuss.d2l.ai/t/1098)
:end_tab:

:begin_tab:`tensorflow`
[Discussões](https://discuss.d2l.ai/t/1099)
:end_tab:
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTMxNTMzNzkyOCwxMDYwODc4MjYsLTIwNT
gzOTg4MDcsLTIxMzIzOTg2NjAsNDQxNzc2OTU0LDE2OTkzNjk2
NzQsLTkxNjA2NDgxMCwxMjgxNDI0NTk4LC0zODc1ODU0MzBdfQ
==
-->