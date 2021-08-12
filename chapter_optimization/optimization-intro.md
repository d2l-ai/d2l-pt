# Optimização e Deep Learning

Nesta seção, discutiremos a relação entre a otimização e o aprendizado profundo, bem como os desafios de usar a otimização no aprendizado profundo.
Para um problema de aprendizado profundo, geralmente definiremos uma *função de perda* primeiro. Uma vez que temos a função de perda, podemos usar um algoritmo de otimização na tentativa de minimizar a perda.
Na otimização, uma função de perda é frequentemente referida como a *função objetivo* do problema de otimização. Por tradição e convenção, a maioria dos algoritmos de otimização se preocupa com a *minimização*. Se alguma vez precisarmos maximizar um objetivo, há uma solução simples: basta virar o sinal no objetivo.

## Objetivos da Optimização

Embora a otimização forneça uma maneira de minimizar a função de perda para profundas
aprendizagem, em essência, as metas de otimização e aprendizagem profunda são
fundamentalmente diferente.
O primeiro se preocupa principalmente em minimizar um
objetivo, enquanto o último está preocupado em encontrar um modelo adequado, dado um
quantidade finita de dados.
Em :numref: `sec_model_selection`,
discutimos a diferença entre esses dois objetivos em detalhes.
Por exemplo,
erro de treinamento e erro de generalização geralmente diferem: uma vez que o objetivo
função do algoritmo de otimização é geralmente uma função de perda com base no
conjunto de dados de treinamento, o objetivo da otimização é reduzir o erro de treinamento.
No entanto, o objetivo do aprendizado profundo (ou mais amplamente, inferência estatística) é
reduzir o erro de generalização.
Para realizar o último, precisamos pagar
atenção ao *overfitting*, além de usar o algoritmo de otimização para
reduzir o erro de treinamento.

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mpl_toolkits import mplot3d
from mxnet import np, npx
npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import numpy as np
from mpl_toolkits import mplot3d
import torch
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import numpy as np
from mpl_toolkits import mplot3d
import tensorflow as tf
```

Para ilustrar os diferentes objetivos mencionados acima,
deixe-nos considerar
o risco empírico e o risco.
Conforme descrito
in :numref: `subsec_empirical-risk-and-risk`,
o risco empírico
é uma perda média
no conjunto de dados de treinamento
enquanto o risco é a perda esperada
em toda a população de dados.
Abaixo, definimos duas funções:
a função de risco `f`
e a função de risco empírica `g`.
Suponha que tenhamos apenas uma quantidade finita de dados de treinamento.
Como resultado, aqui `g` é menos suave do que `f`.

```{.python .input}
#@tab all
def f(x):
    return x * d2l.cos(np.pi * x)

def g(x):
    return f(x) + 0.2 * d2l.cos(5 * np.pi * x)
```

O gráfico abaixo ilustra que o mínimo do risco empírico em um conjunto de dados de treinamento pode estar em um local diferente do mínimo do risco (erro de generalização).

```{.python .input}
#@tab all
def annotate(text, xy, xytext):  #@save
    d2l.plt.gca().annotate(text, xy=xy, xytext=xytext,
                           arrowprops=dict(arrowstyle='->'))

x = d2l.arange(0.5, 1.5, 0.01)
d2l.set_figsize((4.5, 2.5))
d2l.plot(x, [f(x), g(x)], 'x', 'risk')
annotate('min of\nempirical risk', (1.0, -1.2), (0.5, -1.1))
annotate('min of risk', (1.1, -1.05), (0.95, -0.5))
```

## Desafios de otimização em Deep Learning

Neste capítulo, vamos nos concentrar especificamente no desempenho dos algoritmos de otimização para minimizar a função objetivo, ao invés de um
erro de generalização do modelo.
Em :numref: `sec_linear_regression`
distinguimos entre soluções analíticas e soluções numéricas em
problemas de otimização.
No aprendizado profundo, a maioria das funções objetivas são
complicadas e não possuem soluções analíticas. Em vez disso, devemos usar
algoritmos de otimização.
Os algoritmos de otimização neste capítulo
todos caem nisso
categoria.

Existem muitos desafios na otimização do aprendizado profundo. Alguns dos mais irritantes são mínimos locais, pontos de sela e gradientes de desaparecimento.
Vamos dar uma olhada neles.

### Minimo Local


Para qualquer função objetivo $f(x)$,
se o valor de $f(x)$ em $x$ for menor do que os valores de $f(x)$ em quaisquer outros pontos nas proximidades de $x$, então $f(x)$ poderia ser um mínimo local .
Se o valor de $f(x)$ em $x$ é o mínimo da função objetivo em todo o domínio,
então $f(x)$ é o mínimo global.

Por exemplo, dada a função

$$f(x) = x \cdot \text{cos}(\pi x) \text{ for } -1.0 \leq x \leq 2.0,$$

podemos aproximar o mínimo local e o mínimo global desta função.

```{.python .input}
#@tab all
x = d2l.arange(-1.0, 2.0, 0.01)
d2l.plot(x, [f(x), ], 'x', 'f(x)')
annotate('local minimum', (-0.3, -0.25), (-0.77, -1.0))
annotate('global minimum', (1.1, -0.95), (0.6, 0.8))
```

A função objetivo dos modelos de aprendizado profundo geralmente tem muitos ótimos locais.
Quando a solução numérica de um problema de otimização está próxima do ótimo local, a solução numérica obtida pela iteração final pode apenas minimizar a função objetivo *localmente*, ao invés de *globalmente*, conforme o gradiente das soluções da função objetivo se aproxima ou torna-se zero .
Apenas algum grau de ruído pode derrubar o parâmetro do mínimo local. Na verdade, esta é uma das propriedades benéficas de
descida gradiente estocástica do minibatch onde a variação natural dos gradientes sobre os minibatches é capaz de deslocar os parâmetros dos mínimos locais.

### Pontos de Sela

Além dos mínimos locais, os pontos de sela são outra razão para o desaparecimento dos gradientes. Um * ponto de sela * é qualquer local onde todos os gradientes de uma função desaparecem, mas que não é um mínimo global nem local.
Considere a função $f(x) = x^3$. Sua primeira e segunda derivadas desaparecem para $x=0$. A otimização pode parar neste ponto, embora não seja o mínimo.

```{.python .input}
#@tab all
x = d2l.arange(-2.0, 2.0, 0.01)
d2l.plot(x, [x**3], 'x', 'f(x)')
annotate('saddle point', (0, -0.2), (-0.52, -5.0))
```

Os pontos de sela em dimensões mais altas são ainda mais insidiosos, como mostra o exemplo abaixo. Considere a função $f(x, y) = x^2 - y^2$. Ele tem seu ponto de sela em $(0, 0)$. Este é um máximo em relação a $y$ e um mínimo em relação a $x$. Além disso, *se parece* com uma sela, que é onde essa propriedade matemática recebeu seu nome.

```{.python .input}
#@tab all
x, y = d2l.meshgrid(
    d2l.linspace(-1.0, 1.0, 101), d2l.linspace(-1.0, 1.0, 101))
z = x**2 - y**2

ax = d2l.plt.figure().add_subplot(111, projection='3d')
ax.plot_wireframe(x, y, z, **{'rstride': 10, 'cstride': 10})
ax.plot([0], [0], [0], 'rx')
ticks = [-1, 0, 1]
d2l.plt.xticks(ticks)
d2l.plt.yticks(ticks)
ax.set_zticks(ticks)
d2l.plt.xlabel('x')
d2l.plt.ylabel('y');
```

Assumimos que a entrada de uma função é um vetor $k$ -dimensional e seu
a saída é um escalar, então sua matriz Hessiana terá $k$ autovalores
(consulte o [apêndice online sobre eigendecompositions](https://d2l.ai/chapter_apencha-mathematics-for-deep-learning/eigendecomposition.html)).
A solução do
função pode ser um mínimo local, um máximo local ou um ponto de sela em um
posição onde o gradiente da função é zero:

* Quando os autovalores da matriz Hessiana da função na posição do gradiente zero são todos positivos, temos um mínimo local para a função.
* Quando os valores próprios da matriz Hessiana da função na posição do gradiente zero são todos negativos, temos um máximo local para a função.
* Quando os valores próprios da matriz Hessiana da função na posição do gradiente zero são negativos e positivos, temos um ponto de sela para a função.

Para problemas de alta dimensão, a probabilidade de que pelo menos * alguns * dos autovalores sejam negativos é bastante alta. Isso torna os pontos de sela mais prováveis do que os mínimos locais. Discutiremos algumas exceções a essa situação na próxima seção, ao introduzir a convexidade. Em suma, funções convexas são aquelas em que os autovalores do Hessiano nunca são negativos. Infelizmente, porém, a maioria dos problemas de aprendizado profundo não se enquadra nessa categoria. No entanto, é uma ótima ferramenta para estudar algoritmos de otimização.

### Gradiente de Desaparecimento

Provavelmente, o problema mais insidioso a ser encontrado é o gradiente de desaparecimento.
Lembre-se de nossas funções de ativação comumente usadas e seus derivados em: numref: `subsec_activation-functions`.
Por exemplo, suponha que queremos minimizar a função $f(x) = \tanh(x)$ e começamos em $x = 4$. Como podemos ver, o gradiente de $f$ é quase nulo.
Mais especificamente, $f'(x) = 1 - \tanh^2(x)$ e portanto $f'(4) = 0.0013$.
Conseqüentemente, a otimização ficará parada por um longo tempo antes de progredirmos. Isso acabou sendo um dos motivos pelos quais treinar modelos de aprendizado profundo era bastante complicado antes da introdução da função de ativação ReLU.

```{.python .input}
#@tab all
x = d2l.arange(-2.0, 5.0, 0.01)
d2l.plot(x, [d2l.tanh(x)], 'x', 'f(x)')
annotate('vanishing gradient', (4, 1), (2, 0.0))
```

Como vimos, a otimização para aprendizado profundo está cheia de desafios. Felizmente, existe uma gama robusta de algoritmos que funcionam bem e são fáceis de usar, mesmo para iniciantes. Além disso, não é realmente necessário encontrar a melhor solução. Ótimos locais ou mesmo soluções aproximadas deles ainda são muito úteis.

## Sumário

* Minimizar o erro de treinamento *não* garante que encontraremos o melhor conjunto de parâmetros para minimizar o erro de generalização.
* Os problemas de otimização podem ter muitos mínimos locais.
* O problema pode ter ainda mais pontos de sela, pois geralmente os problemas não são convexos.
* O desaparecimento de gradientes pode causar o travamento da otimização. Frequentemente, uma reparametrização do problema ajuda. Uma boa inicialização dos parâmetros também pode ser benéfica.


## Exercícios

1. Considere um MLP simples com uma única camada oculta de, digamos, $d$ dimensões na camada oculta e uma única saída. Mostre que, para qualquer mínimo local, existem pelo menos $d!$ Soluções equivalentes que se comportam de forma idêntica.
1. Suponha que temos uma matriz aleatória simétrica $\mathbf{M}$ onde as entradas
   $M_{ij} = M_{ji}$ são, cada um, extraídos de alguma distribuição de probabilidade
   $p_{ij}$. Além disso, assuma que $p_{ij}(x) = p_{ij}(-x)$, ou seja, que o
   a distribuição é simétrica (consulte, por exemplo :cite:`Wigner.1958` para obter detalhes).
    1. Prove que a distribuição sobre os autovalores também é simétrica. Ou seja, para qualquer autovetor $\mathbf{v}$ a probabilidade de que o autovalor associado $\lambda$ satisfaça $P(\lambda > 0) = P(\lambda < 0)$.
    1. Por que o acima *não* implica $P(\lambda > 0) = 0.5$?
1. Em que outros desafios envolvidos na otimização do aprendizado profundo você consegue pensar?
1. Suponha que você deseja equilibrar uma bola (real) em uma sela (real).
    1. Por que isso é difícil?
    1. Você pode explorar esse efeito também para algoritmos de otimização?

:begin_tab:`mxnet`
[Discussão](https://discuss.d2l.ai/t/349)
:end_tab:

:begin_tab:`pytorch`
[Discussão](https://discuss.d2l.ai/t/487)
:end_tab:

:begin_tab:`tensorflow`
[Discussão](https://discuss.d2l.ai/t/489)
:end_tab:
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTEzNTg4MTg3LDY1MjU5NzE5MiwyODIxMz
Y1MzQsLTEwMzExODk0Nl19
-->