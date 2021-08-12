# Cálculo Multivariável
:label:`sec_multivariable_calculus`

Agora que temos um entendimento bastante forte das derivadas de uma função de uma única variável, vamos voltar à nossa questão original, onde estávamos considerando uma função de perda de potencialmente bilhões de pesos.

## Diferenciação de Dimensões Superiores
O que :numref:`sec_single_variable_calculus` nos diz é que se mudarmos um desses bilhões de pesos deixando todos os outros fixos, sabemos o que vai acontecer! Isso nada mais é do que uma função de uma única variável, então podemos escrever

$$L(w_1+\epsilon_1, w_2, \ldots, w_N) \approx L(w_1, w_2, \ldots, w_N) + \epsilon_1 \frac{d}{dw_1} L(w_1, w_2, \ldots, w_N).$$
:eqlabel:`eq_part_der`


Chamaremos a derivada em uma variável enquanto fixamos a outra *derivada parcial*, e usaremos a notação $\frac{\partial}{\partial w_1}$ para a derivada em :eqref:`eq_part_der`.

Agora, vamos pegar isso e mudar $ w_2 $ um pouco para $w_2 + \epsilon_2$:

$$
\begin{aligned}
L(w_1+\epsilon_1, w_2+\epsilon_2, \ldots, w_N) & \approx L(w_1, w_2+\epsilon_2, \ldots, w_N) + \epsilon_1 \frac{\partial}{\partial w_1} L(w_1, w_2+\epsilon_2, \ldots, w_N+\epsilon_N) \\
& \approx L(w_1, w_2, \ldots, w_N) \\
& \quad + \epsilon_2\frac{\partial}{\partial w_2} L(w_1, w_2, \ldots, w_N) \\
& \quad + \epsilon_1 \frac{\partial}{\partial w_1} L(w_1, w_2, \ldots, w_N) \\
& \quad + \epsilon_1\epsilon_2\frac{\partial}{\partial w_2}\frac{\partial}{\partial w_1} L(w_1, w_2, \ldots, w_N) \\
& \approx L(w_1, w_2, \ldots, w_N) \\
& \quad + \epsilon_2\frac{\partial}{\partial w_2} L(w_1, w_2, \ldots, w_N) \\
& \quad + \epsilon_1 \frac{\partial}{\partial w_1} L(w_1, w_2, \ldots, w_N).
\end{aligned}
$$

Usamos novamente a ideia de que $\epsilon_1\epsilon_2$ é um termo de ordem superior que podemos descartar da mesma forma que descartamos $\epsilon^{2}$ na seção anterior, junto com o que vimos em :eqref:`eq_part_der`. Continuando dessa maneira, podemos escrever que

$$
L(w_1+\epsilon_1, w_2+\epsilon_2, \ldots, w_N+\epsilon_N) \approx L(w_1, w_2, \ldots, w_N) + \sum_i \epsilon_i \frac{\partial}{\partial w_i} L(w_1, w_2, \ldots, w_N).
$$

Isso pode parecer uma bagunça, mas podemos tornar isso mais familiar observando que a soma à direita parece exatamente com um produto escalar, então, se deixarmos

$$
\boldsymbol{\epsilon} = [\epsilon_1, \ldots, \epsilon_N]^\top \; \text{and} \;
\nabla_{\mathbf{x}} L = \left[\frac{\partial L}{\partial x_1}, \ldots, \frac{\partial L}{\partial x_N}\right]^\top,
$$

então

$$L(\mathbf{w} + \boldsymbol{\epsilon}) \approx L(\mathbf{w}) + \boldsymbol{\epsilon}\cdot \nabla_{\mathbf{w}} L(\mathbf{w}).$$
:eqlabel:`eq_nabla_use`


Chamaremos o vetor $\nabla_{\mathbf{w}} L$ de *gradiente* de $L$.

Equação :eqref:`eq_nabla_use` vale a pena ponderar por um momento. Tem exatamente o formato que encontramos em uma dimensão, apenas convertemos tudo para vetores e produtos escalares. Isso nos permite dizer aproximadamente como a função $L$ mudará dada qualquer perturbação na entrada. Como veremos na próxima seção, isso nos fornecerá uma ferramenta importante para compreender geometricamente como podemos aprender usando as informações contidas no gradiente.

Mas, primeiro, vejamos essa aproximação em funcionamento com um exemplo. Suponha que estejamos trabalhando com a função

$$
f(x, y) = \log(e^x + e^y) \text{ with gradient } \nabla f (x, y) = \left[\frac{e^x}{e^x+e^y}, \frac{e^y}{e^x+e^y}\right].
$$

Se olharmos para um ponto como $(0, \log(2))$, vemos que

$$
f(x, y) = \log(3) \text{ with gradient } \nabla f (x, y) = \left[\frac{1}{3}, \frac{2}{3}\right].
$$

Assim, se quisermos aproximar $f$ em $(\epsilon_1, \log(2) + \epsilon_2)$, vemos que devemos ter a instância específica de :eqref:`eq_nabla_use`:

$$
f(\epsilon_1, \log(2) + \epsilon_2) \approx \log(3) + \frac{1}{3}\epsilon_1 + \frac{2}{3}\epsilon_2.
$$

Podemos testar isso no código para ver o quão boa é a aproximação.

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from IPython import display
from mpl_toolkits import mplot3d
from mxnet import autograd, np, npx
npx.set_np()

def f(x, y):
    return np.log(np.exp(x) + np.exp(y))
def grad_f(x, y):
    return np.array([np.exp(x) / (np.exp(x) + np.exp(y)),
                     np.exp(y) / (np.exp(x) + np.exp(y))])

epsilon = np.array([0.01, -0.03])
grad_approx = f(0, np.log(2)) + epsilon.dot(grad_f(0, np.log(2)))
true_value = f(0 + epsilon[0], np.log(2) + epsilon[1])
f'approximation: {grad_approx}, true Value: {true_value}'
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
from IPython import display
from mpl_toolkits import mplot3d
import torch
import numpy as np

def f(x, y):
    return torch.log(torch.exp(x) + torch.exp(y))
def grad_f(x, y):
    return torch.tensor([torch.exp(x) / (torch.exp(x) + torch.exp(y)),
                     torch.exp(y) / (torch.exp(x) + torch.exp(y))])

epsilon = torch.tensor([0.01, -0.03])
grad_approx = f(torch.tensor([0.]), torch.log(
    torch.tensor([2.]))) + epsilon.dot(
    grad_f(torch.tensor([0.]), torch.log(torch.tensor(2.))))
true_value = f(torch.tensor([0.]) + epsilon[0], torch.log(
    torch.tensor([2.])) + epsilon[1])
f'approximation: {grad_approx}, true Value: {true_value}'
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
from IPython import display
from mpl_toolkits import mplot3d
import tensorflow as tf
import numpy as np

def f(x, y):
    return tf.math.log(tf.exp(x) + tf.exp(y))
def grad_f(x, y):
    return tf.constant([(tf.exp(x) / (tf.exp(x) + tf.exp(y))).numpy(),
                        (tf.exp(y) / (tf.exp(x) + tf.exp(y))).numpy()])

epsilon = tf.constant([0.01, -0.03])
grad_approx = f(tf.constant([0.]), tf.math.log(
    tf.constant([2.]))) + tf.tensordot(
    epsilon, grad_f(tf.constant([0.]), tf.math.log(tf.constant(2.))), axes=1)
true_value = f(tf.constant([0.]) + epsilon[0], tf.math.log(
    tf.constant([2.])) + epsilon[1])
f'approximation: {grad_approx}, true Value: {true_value}'
```

## Geometria de Gradientes e Gradiente Descendente
Considere novamente :eqref:`eq_nabla_use`:

$$
L(\mathbf{w} + \boldsymbol{\epsilon}) \approx L(\mathbf{w}) + \boldsymbol{\epsilon}\cdot \nabla_{\mathbf{w}} L(\mathbf{w}).
$$

Suponhamos que eu queira usar isso para ajudar a minimizar nossa perda $L$. Vamos entender geometricamente o algoritmo de gradiente descendente descrito pela primeira vez em :numref:`sec_autograd`. O que faremos é o seguinte:

1. Comece com uma escolha aleatória para os parâmetros iniciais $\mathbf{w}$.
2. Encontre a direção $\mathbf{v}$ que faz $L$ diminuir mais rapidamente em $\mathbf{w}$.
3. Dê um pequeno passo nessa direção: $\mathbf{w} \rightarrow \mathbf{w} + \epsilon\mathbf{v}$.
4. Repita.

A única coisa que não sabemos exatamente como fazer é calcular o vetor $\mathbf{v}$ no segundo passo. Chamaremos tal direção de *direção da descida mais íngreme*. Usando o entendimento geométrico de produtos escalares de :numref:`sec_geometry-linear-algebraic-ops`, vemos que podemos reescrever :eqref:`eq_nabla_use` como

$$
L(\mathbf{w} + \mathbf{v}) \approx L(\mathbf{w}) + \mathbf{v}\cdot \nabla_{\mathbf{w}} L(\mathbf{w}) = L(\mathbf{w}) + \|\nabla_{\mathbf{w}} L(\mathbf{w})\|\cos(\theta).
$$

Observe que seguimos nossa orientação para ter comprimento um por conveniência, e usamos $\theta$ para o ângulo entre $\mathbf{v}$ e $\nabla_{\mathbf{w}} L(\mathbf{w})$ Se quisermos encontrar a direção que diminui $L$ o mais rápido possível, queremos tornar essa expressão o mais negativa possível. A única maneira pela qual a direção que escolhemos entra nesta equação é através de $\cos(\theta)$ e, portanto, desejamos tornar esse cosseno o mais negativo possível. Agora, relembrando a forma do cosseno, podemos torná-lo o mais negativo possível, tornando $\cos(\theta) = -1$ ou equivalentemente tornando o ângulo entre o gradiente e nossa direção escolhida em $\pi$ radianos, ou equivalentemente $180$ graus. A única maneira de conseguir isso é seguir na direção oposta exata: escolha $\mathbf{v}$ para apontar na direção oposta exata para $\nabla_{\mathbf{w}} L(\mathbf{w})$!

Isso nos leva a um dos conceitos matemáticos mais importantes no aprendizado de máquina: a direção dos pontos decentes mais íngremes na direção de $-\nabla_{\mathbf{w}}L(\mathbf{w})$.  Assim, nosso algoritmo informal pode ser reescrito da seguinte maneira.

1. Comece com uma escolha aleatória para os parâmetros iniciais $\mathbf{w}$.
2. Calcule $\nabla_{\mathbf{w}} L(\mathbf{w})$.
3. Dê um pequeno passo na direção oposta: $\mathbf{w} \rightarrow \mathbf{w} - \epsilon\nabla_{\mathbf{w}} L(\mathbf{w})$.
4. Repita.


Este algoritmo básico foi modificado e adaptado de várias maneiras por muitos pesquisadores, mas o conceito central permanece o mesmo em todos eles. Usar o gradiente para encontrar a direção que diminui a perda o mais rápido possível e atualizar os parâmetros para dar um passo nessa direção.

## Uma Nota Sobre Otimização Matemática

Ao longo deste livro, enfocamos diretamente as técnicas de otimização numérica pela razão prática de que todas as funções que encontramos no ambiente de aprendizado profundo são muito complexas para serem minimizadas explicitamente.

No entanto, é um exercício útil considerar o que a compreensão geométrica que obtivemos acima nos diz sobre como otimizar funções diretamente.

Suponha que desejamos encontrar o valor de $\mathbf{x}_0$ que minimiza alguma função $L(\mathbf{x})$. Suponhamos que, além disso, alguém nos dê um valor e nos diga que é o valor que minimiza $L$. Existe algo que possamos verificar para ver se a resposta deles é plausível?

Considere novamente :eqref:`eq_nabla_use`:
$$
L(\mathbf{x}_0 + \boldsymbol{\epsilon}) \approx L(\mathbf{x}_0) + \boldsymbol{\epsilon}\cdot \nabla_{\mathbf{x}} L(\mathbf{x}_0).
$$


Se o gradiente não for zero, sabemos que podemos dar um passo na direção $-\epsilon \nabla_{\mathbf{x}} L(\mathbf{x}_0)$ para encontrar um valor de $L$ que é menor. Portanto, se realmente estamos no mínimo, não pode ser esse o caso! Podemos concluir que se $\mathbf{x}_0$ é um mínimo, então $\nabla_{\mathbf{x}} L(\mathbf{x}_0) = 0$. Chamamos pontos com $\nabla_{\mathbf{x}} L(\mathbf{x}_0) = 0$ *pontos críticos*.

Isso é bom, porque em algumas configurações raras, nós *podemos* encontrar explicitamente todos os pontos onde o gradiente é zero e encontrar aquele com o menor valor.

Para um exemplo concreto, considere a função
$$
f(x) = 3x^4 - 4x^3 -12x^2.
$$

Esta função tem derivada
$$
\frac{df}{dx} = 12x^3 - 12x^2 -24x = 12x(x-2)(x+1).
$$

A única localização possível dos mínimos está em $x = -1, 0, 2$, onde a função assume os valores $-5,0, -32$ respectivamente, e assim podemos concluir que minimizamos nossa função quando $x = 2$. Um gráfico rápido confirma isso.

```{.python .input}
x = np.arange(-2, 3, 0.01)
f = (3 * x**4) - (4 * x**3) - (12 * x**2)

d2l.plot(x, f, 'x', 'f(x)')
```

```{.python .input}
#@tab pytorch
x = torch.arange(-2, 3, 0.01)
f = (3 * x**4) - (4 * x**3) - (12 * x**2)

d2l.plot(x, f, 'x', 'f(x)')
```

```{.python .input}
#@tab tensorflow
x = tf.range(-2, 3, 0.01)
f = (3 * x**4) - (4 * x**3) - (12 * x**2)

d2l.plot(x, f, 'x', 'f(x)')
```

Isso destaca um fato importante a saber ao trabalhar teoricamente ou numericamente: os únicos pontos possíveis onde podemos minimizar (ou maximizar) uma função terão gradiente igual a zero, no entanto, nem todo ponto com gradiente zero é o verdadeiro *global* mínimo (ou máximo).

## Regra da Cadeia Multivariada
Vamos supor que temos uma função de quatro variáveis ($w, x, y$, e $z$) que podemos fazer compondo muitos termos:

$$\begin{aligned}f(u, v) & = (u+v)^{2} \\u(a, b) & = (a+b)^{2}, \qquad v(a, b) = (a-b)^{2}, \\a(w, x, y, z) & = (w+x+y+z)^{2}, \qquad b(w, x, y, z) = (w+x-y-z)^2.\end{aligned}$$
:eqlabel:`eq_multi_func_def`

Essas cadeias de equações são comuns ao trabalhar com redes neurais, portanto, tentar entender como calcular gradientes de tais funções é fundamental. Podemos começar a ver dicas visuais dessa conexão em :numref:`fig_chain-1` se dermos uma olhada em quais variáveis se relacionam diretamente entre si.

![As relações de função acima, onde os nós representam valores e as arestas mostram dependência funcional.](../img/chain-net1.svg)
:label:`fig_chain-1`

Nada nos impede de apenas compor tudo de :eqref:`eq_multi_func_def` e escrever isso

$$
f(w, x, y, z) = \left(\left((w+x+y+z)^2+(w+x-y-z)^2\right)^2+\left((w+x+y+z)^2-(w+x-y-z)^2\right)^2\right)^2.
$$

Podemos então tirar a derivada usando apenas derivadas de variável única, mas se fizéssemos isso, rapidamente nos veríamos inundados com termos, muitos dos quais são repetições! Na verdade, pode-se ver que, por exemplo:

$$
\begin{aligned}
\frac{\partial f}{\partial w} & = 2 \left(2 \left(2 (w + x + y + z) - 2 (w + x - y - z)\right) \left((w + x + y + z)^{2}- (w + x - y - z)^{2}\right) + \right.\\
& \left. \quad 2 \left(2 (w + x - y - z) + 2 (w + x + y + z)\right) \left((w + x - y - z)^{2}+ (w + x + y + z)^{2}\right)\right) \times \\
& \quad \left(\left((w + x + y + z)^{2}- (w + x - y - z)^2\right)^{2}+ \left((w + x - y - z)^{2}+ (w + x + y + z)^{2}\right)^{2}\right).
\end{aligned}
$$

Se também quiséssemos calcular $\frac{\partial f}{\partial x}$, acabaríamos com uma equação semelhante novamente com muitos termos repetidos e muitos termos repetidos *compartilhados* entre as duas derivadas. Isso representa uma enorme quantidade de trabalho desperdiçado e, se precisássemos calcular as derivadas dessa forma, toda a revolução do aprendizado profundo teria estagnado antes de começar!


Vamos resolver o problema. Começaremos tentando entender como $f$ muda quando mudamos $a$, essencialmente supondo que $w, x, y$, e $z$ não existem. Vamos raciocinar como fazíamos quando trabalhamos com gradiente pela primeira vez. Vamos pegar $a$ e adicionar uma pequena quantia $\epsilon$ a ele.

$$
\begin{aligned}
& f(u(a+\epsilon, b), v(a+\epsilon, b)) \\
\approx & f\left(u(a, b) + \epsilon\frac{\partial u}{\partial a}(a, b), v(a, b) + \epsilon\frac{\partial v}{\partial a}(a, b)\right) \\
\approx & f(u(a, b), v(a, b)) + \epsilon\left[\frac{\partial f}{\partial u}(u(a, b), v(a, b))\frac{\partial u}{\partial a}(a, b) + \frac{\partial f}{\partial v}(u(a, b), v(a, b))\frac{\partial v}{\partial a}(a, b)\right].
\end{aligned}
$$

A primeira linha segue da definição de derivada parcial e a segunda segue da definição de gradiente. É notacionalmente pesado rastrear exatamente onde avaliamos cada derivada, como na expressão $\frac{\partial f}{\partial u}(u(a, b), v(a, b))$, então frequentemente abreviamos para muito mais memorável

$$
\frac{\partial f}{\partial a} = \frac{\partial f}{\partial u}\frac{\partial u}{\partial a}+\frac{\partial f}{\partial v}\frac{\partial v}{\partial a}.
$$

É útil pensar sobre o significado do processo. Estamos tentando entender como uma função da forma $f(u(a, b), v(a, b))$ muda seu valor com uma mudança em $a$. Isso pode ocorrer de duas maneiras: há o caminho onde $a \rightarrow u \rightarrow f$ e onde $a \rightarrow v \rightarrow f$. Podemos calcular ambas as contribuições por meio da regra da cadeia: $\frac{\partial w}{\partial u} \cdot \frac{\partial u}{\partial x}$ e $\frac{\partial w}{\partial v} \cdot \frac{\partial v}{\partial x}$ respectivamente, e somados.

Imagine que temos uma rede diferente de funções onde as funções à direita dependem daquelas que estão conectadas à esquerda, como mostrado em :numref:`fig_chain-2`.

![Outro exemplo mais sutil da regra da cadeia.](../img/chain-net2.svg)
:label:`fig_chain-2`

Para calcular algo como $\frac{\partial f}{\partial y}$, precisamos somar todos (neste caso $3$) caminhos de $y$ a $f$ dando

$$
\frac{\partial f}{\partial y} = \frac{\partial f}{\partial a} \frac{\partial a}{\partial u} \frac{\partial u}{\partial y} + \frac{\partial f}{\partial u} \frac{\partial u}{\partial y} + \frac{\partial f}{\partial b} \frac{\partial b}{\partial v} \frac{\partial v}{\partial y}.
$$

Entender a regra da cadeia desta forma renderá grandes dividendos ao tentar entender como os gradientes fluem através das redes, e por que várias escolhas arquitetônicas como aquelas em LSTMs (:numref:`sec_lstm`) ou camadas residuais (:numref:`sec_resnet`) podem ajudam a moldar o processo de aprendizagem, controlando o fluxo gradiente.

## O Algoritmo de Retropropagação

Vamos retornar ao exemplo de :eqref:`eq_multi_func_def` a seção anterior onde

$$
\begin{aligned}
f(u, v) & = (u+v)^{2} \\
u(a, b) & = (a+b)^{2}, \qquad v(a, b) = (a-b)^{2}, \\
a(w, x, y, z) & = (w+x+y+z)^{2}, \qquad b(w, x, y, z) = (w+x-y-z)^2.
\end{aligned}
$$

Se quisermos calcular, digamos $\frac{\partial f}{\partial w}$, podemos aplicar a regra da cadeia multivariada para ver:

$$
\begin{aligned}
\frac{\partial f}{\partial w} & = \frac{\partial f}{\partial u}\frac{\partial u}{\partial w} + \frac{\partial f}{\partial v}\frac{\partial v}{\partial w}, \\
\frac{\partial u}{\partial w} & = \frac{\partial u}{\partial a}\frac{\partial a}{\partial w}+\frac{\partial u}{\partial b}\frac{\partial b}{\partial w}, \\
\frac{\partial v}{\partial w} & = \frac{\partial v}{\partial a}\frac{\partial a}{\partial w}+\frac{\partial v}{\partial b}\frac{\partial b}{\partial w}.
\end{aligned}
$$

Vamos tentar usar esta decomposição para calcular $\frac{\partial f}{\partial w}$. Observe que tudo o que precisamos aqui são as várias parciais de etapa única:

$$
\begin{aligned}
\frac{\partial f}{\partial u} = 2(u+v), & \quad\frac{\partial f}{\partial v} = 2(u+v), \\
\frac{\partial u}{\partial a} = 2(a+b), & \quad\frac{\partial u}{\partial b} = 2(a+b), \\
\frac{\partial v}{\partial a} = 2(a-b), & \quad\frac{\partial v}{\partial b} = -2(a-b), \\
\frac{\partial a}{\partial w} = 2(w+x+y+z), & \quad\frac{\partial b}{\partial w} = 2(w+x-y-z).
\end{aligned}
$$

Se escrevermos isso no código, isso se tornará uma expressão bastante gerenciável.

```{.python .input}
#@tab all
# Compute the value of the function from inputs to outputs
w, x, y, z = -1, 0, -2, 1
a, b = (w + x + y + z)**2, (w + x - y - z)**2
u, v = (a + b)**2, (a - b)**2
f = (u + v)**2
print(f'    f at {w}, {x}, {y}, {z} is {f}')

# Compute the single step partials
df_du, df_dv = 2*(u + v), 2*(u + v)
du_da, du_db, dv_da, dv_db = 2*(a + b), 2*(a + b), 2*(a - b), -2*(a - b)
da_dw, db_dw = 2*(w + x + y + z), 2*(w + x - y - z)

# Compute the final result from inputs to outputs
du_dw, dv_dw = du_da*da_dw + du_db*db_dw, dv_da*da_dw + dv_db*db_dw
df_dw = df_du*du_dw + df_dv*dv_dw
print(f'df/dw at {w}, {x}, {y}, {z} is {df_dw}')
```


No entanto, observe que isso ainda não facilita o cálculo de algo como $\frac{\partial f}{\partial x}$. A razão para isso é a *forma* que escolhemos para aplicar a regra da cadeia. Se observarmos o que fizemos acima, sempre mantivemos $\partial w$ no denominador quando podíamos. Desta forma, optamos por aplicar a regra da cadeia vendo como $w$ mudou todas as outras variáveis. Se é isso que queríamos, seria uma boa ideia. No entanto, pense em nossa motivação com o aprendizado profundo: queremos ver como cada parâmetro altera a *perda*. Em essência, queremos aplicar a regra da cadeia mantendo $\partial f$ no numerador sempre que pudermos!

Para ser mais explícito, observe que podemos escrever

$$
\begin{aligned}
\frac{\partial f}{\partial w} & = \frac{\partial f}{\partial a}\frac{\partial a}{\partial w} + \frac{\partial f}{\partial b}\frac{\partial b}{\partial w}, \\
\frac{\partial f}{\partial a} & = \frac{\partial f}{\partial u}\frac{\partial u}{\partial a}+\frac{\partial f}{\partial v}\frac{\partial v}{\partial a}, \\
\frac{\partial f}{\partial b} & = \frac{\partial f}{\partial u}\frac{\partial u}{\partial b}+\frac{\partial f}{\partial v}\frac{\partial v}{\partial b}.
\end{aligned}
$$

Observe que esta aplicação da regra da cadeia nos faz computar explicitamente $\frac{\partial f}{\partial u}, \frac{\partial f}{\partial v}, \frac{\partial f}{\partial a}, \frac{\partial f}{\partial b}, \; \text{e} \; \frac{\partial f}{\partial w}$. Nada nos impede de incluir também as equações:

$$
\begin{aligned}
\frac{\partial f}{\partial x} & = \frac{\partial f}{\partial a}\frac{\partial a}{\partial x} + \frac{\partial f}{\partial b}\frac{\partial b}{\partial x}, \\
\frac{\partial f}{\partial y} & = \frac{\partial f}{\partial a}\frac{\partial a}{\partial y}+\frac{\partial f}{\partial b}\frac{\partial b}{\partial y}, \\
\frac{\partial f}{\partial z} & = \frac{\partial f}{\partial a}\frac{\partial a}{\partial z}+\frac{\partial f}{\partial b}\frac{\partial b}{\partial z}.
\end{aligned}
$$

e acompanhar como $f$ muda quando mudamos *qualquer* nó em toda a rede. Vamos implementar.

```{.python .input}
#@tab all
# Compute the value of the function from inputs to outputs
w, x, y, z = -1, 0, -2, 1
a, b = (w + x + y + z)**2, (w + x - y - z)**2
u, v = (a + b)**2, (a - b)**2
f = (u + v)**2
print(f'f at {w}, {x}, {y}, {z} is {f}')

# Compute the derivative using the decomposition above
# First compute the single step partials
df_du, df_dv = 2*(u + v), 2*(u + v)
du_da, du_db, dv_da, dv_db = 2*(a + b), 2*(a + b), 2*(a - b), -2*(a - b)
da_dw, db_dw = 2*(w + x + y + z), 2*(w + x - y - z)
da_dx, db_dx = 2*(w + x + y + z), 2*(w + x - y - z)
da_dy, db_dy = 2*(w + x + y + z), -2*(w + x - y - z)
da_dz, db_dz = 2*(w + x + y + z), -2*(w + x - y - z)

# Now compute how f changes when we change any value from output to input
df_da, df_db = df_du*du_da + df_dv*dv_da, df_du*du_db + df_dv*dv_db
df_dw, df_dx = df_da*da_dw + df_db*db_dw, df_da*da_dx + df_db*db_dx
df_dy, df_dz = df_da*da_dy + df_db*db_dy, df_da*da_dz + df_db*db_dz

print(f'df/dw at {w}, {x}, {y}, {z} is {df_dw}')
print(f'df/dx at {w}, {x}, {y}, {z} is {df_dx}')
print(f'df/dy at {w}, {x}, {y}, {z} is {df_dy}')
print(f'df/dz at {w}, {x}, {y}, {z} is {df_dz}')
```


O fato de calcularmos derivadas de $f$ de volta para as entradas em vez de das entradas para as saídas (como fizemos no primeiro trecho de código acima) é o que dá a esse algoritmo seu nome: *retropropagação*. Observe que existem duas etapas:
1. Calcular o valor da função e as parciais de etapa única da frente para trás. Embora não feito acima, isso pode ser combinado em um único *passe para frente*.
2. Calcule o gradiente de $f$ de trás para frente. Chamamos isso de *passe para trás*.

Isso é precisamente o que todo algoritmo de aprendizado profundo implementa para permitir o cálculo do gradiente da perda em relação a cada peso na rede em uma passagem. É um fato surpreendente que tenhamos tal decomposição.

Para ver como encapsular isso, vamos dar uma olhada rápida neste exemplo.

```{.python .input}
# Initialize as ndarrays, then attach gradients
w, x, y, z = np.array(-1), np.array(0), np.array(-2), np.array(1)

w.attach_grad()
x.attach_grad()
y.attach_grad()
z.attach_grad()

# Do the computation like usual, tracking gradients
with autograd.record():
    a, b = (w + x + y + z)**2, (w + x - y - z)**2
    u, v = (a + b)**2, (a - b)**2
    f = (u + v)**2

# Execute backward pass
f.backward()

print(f'df/dw at {w}, {x}, {y}, {z} is {w.grad}')
print(f'df/dx at {w}, {x}, {y}, {z} is {x.grad}')
print(f'df/dy at {w}, {x}, {y}, {z} is {y.grad}')
print(f'df/dz at {w}, {x}, {y}, {z} is {z.grad}')
```

```{.python .input}
#@tab pytorch
# Initialize as ndarrays, then attach gradients
w = torch.tensor([-1.], requires_grad=True)
x = torch.tensor([0.], requires_grad=True)
y = torch.tensor([-2.], requires_grad=True)
z = torch.tensor([1.], requires_grad=True)
# Do the computation like usual, tracking gradients
a, b = (w + x + y + z)**2, (w + x - y - z)**2
u, v = (a + b)**2, (a - b)**2
f = (u + v)**2

# Execute backward pass
f.backward()

print(f'df/dw at {w.data.item()}, {x.data.item()}, {y.data.item()}, '
      f'{z.data.item()} is {w.grad.data.item()}')
print(f'df/dx at {w.data.item()}, {x.data.item()}, {y.data.item()}, '
      f'{z.data.item()} is {x.grad.data.item()}')
print(f'df/dy at {w.data.item()}, {x.data.item()}, {y.data.item()}, '
      f'{z.data.item()} is {y.grad.data.item()}')
print(f'df/dz at {w.data.item()}, {x.data.item()}, {y.data.item()}, '
      f'{z.data.item()} is {z.grad.data.item()}')
```

```{.python .input}
#@tab tensorflow
# Initialize as ndarrays, then attach gradients
w = tf.Variable(tf.constant([-1.]))
x = tf.Variable(tf.constant([0.]))
y = tf.Variable(tf.constant([-2.]))
z = tf.Variable(tf.constant([1.]))
# Do the computation like usual, tracking gradients
with tf.GradientTape(persistent=True) as t:
    a, b = (w + x + y + z)**2, (w + x - y - z)**2
    u, v = (a + b)**2, (a - b)**2
    f = (u + v)**2

# Execute backward pass
w_grad = t.gradient(f, w).numpy()
x_grad = t.gradient(f, x).numpy()
y_grad = t.gradient(f, y).numpy()
z_grad = t.gradient(f, z).numpy()

print(f'df/dw at {w.numpy()}, {x.numpy()}, {y.numpy()}, '
      f'{z.numpy()} is {w_grad}')
print(f'df/dx at {w.numpy()}, {x.numpy()}, {y.numpy()}, '
      f'{z.numpy()} is {x_grad}')
print(f'df/dy at {w.numpy()}, {x.numpy()}, {y.numpy()}, '
      f'{z.numpy()} is {y_grad}')
print(f'df/dz at {w.numpy()}, {x.numpy()}, {y.numpy()}, '
      f'{z.numpy()} is {z_grad}')
```

Tudo o que fizemos acima pode ser feito automaticamente chamando `f.backwards ()`.


## Hessians

Como no cálculo de variável única, é útil considerar derivadas de ordem superior para entender como podemos obter uma melhor aproximação de uma função do que usar apenas o gradiente.

Há um problema imediato que se encontra ao trabalhar com derivadas de funções de várias variáveis de ordem superior, que é o grande número delas. Se temos uma função $f(x_1, \ldots, x_n)$ de $n$ variáveis, então podemos tomar $n^{2}$ derivadas secundas, nomeadamente para qualquer escolha de $i$ e $j$:

$$
\frac{d^2f}{dx_idx_j} = \frac{d}{dx_i}\left(\frac{d}{dx_j}f\right).
$$

Isso é tradicionalmente montado em uma matriz chamada *Hessian*:

$$\mathbf{H}_f = \begin{bmatrix} \frac{d^2f}{dx_1dx_1} & \cdots & \frac{d^2f}{dx_1dx_n} \\ \vdots & \ddots & \vdots \\ \frac{d^2f}{dx_ndx_1} & \cdots & \frac{d^2f}{dx_ndx_n} \\ \end{bmatrix}.$$
:eqlabel:`eq_hess_def`

Nem todas as entradas desta matriz são independentes. Na verdade, podemos mostrar que, enquanto ambas * parciais mistas * (derivadas parciais em relação a mais de uma variável) existem e são contínuos, podemos dizer que para qualquer $i$ e $j$,

$$
\frac{d^2f}{dx_idx_j} = \frac{d^2f}{dx_jdx_i}.
$$


Isto segue considerando primeiro perturbar uma função na direção de $x_i$, e então perturbá-la em $x_j$ e então comparar o resultado disso com o que acontece se perturbarmos primeiro $x_j$ e então $x_i$, com o conhecimento que ambos os pedidos levam à mesma mudança final na produção de $f$.

Assim como acontece com variáveis únicas, podemos usar essas derivadas para ter uma ideia muito melhor de como a função se comporta perto de um ponto. Em particular, podemos usar isso para encontrar a quadrática de melhor ajuste próximo a um ponto $\mathbf{x}_0$, como vimos em uma única variável.

Vejamos um exemplo. Suponha que $f(x_1, x_2) = a + b_1x_1 + b_2x_2 + c_{11}x_1^{2} + c_{12}x_1x_2 + c_{22}x_2^{2}$. Esta é a forma geral de uma quadrática em duas variáveis. Se olharmos para o valor da função, seu gradiente e seu Hessian :eqref:`eq_hess_def`, tudo no ponto zero:

$$
\begin{aligned}
f(0,0) & = a, \\
\nabla f (0,0) & = \begin{bmatrix}b_1 \\ b_2\end{bmatrix}, \\
\mathbf{H} f (0,0) & = \begin{bmatrix}2 c_{11} & c_{12} \\ c_{12} & 2c_{22}\end{bmatrix},
\end{aligned}
$$

podemos obter nosso polinômio original de volta, dizendo

$$
f(\mathbf{x}) = f(0) + \nabla f (0) \cdot \mathbf{x} + \frac{1}{2}\mathbf{x}^\top \mathbf{H} f (0) \mathbf{x}.
$$

In general, if we computed this expansion any point $\mathbf{x}_0$, we see that

$$
f(\mathbf{x}) = f(\mathbf{x}_0) + \nabla f (\mathbf{x}_0) \cdot (\mathbf{x}-\mathbf{x}_0) + \frac{1}{2}(\mathbf{x}-\mathbf{x}_0)^\top \mathbf{H} f (\mathbf{x}_0) (\mathbf{x}-\mathbf{x}_0).
$$

Isso funciona para entradas de qualquer dimensão e fornece a melhor aproximação quadrática para qualquer função em um ponto. Para dar um exemplo, vamos representar graficamente a função

$$
f(x, y) = xe^{-x^2-y^2}.
$$

Pode-se calcular que o gradiente e Hessian são
$$
\nabla f(x, y) = e^{-x^2-y^2}\begin{pmatrix}1-2x^2 \\ -2xy\end{pmatrix} \; \text{and} \; \mathbf{H}f(x, y) = e^{-x^2-y^2}\begin{pmatrix} 4x^3 - 6x & 4x^2y - 2y \\ 4x^2y-2y &4xy^2-2x\end{pmatrix}.
$$

E assim, com um pouco de álgebra, veja que a aproximação quadrática em $[-1,0]^\top$ é

$$
f(x, y) \approx e^{-1}\left(-1 - (x+1) +(x+1)^2+y^2\right).
$$

```{.python .input}
# Construct grid and compute function
x, y = np.meshgrid(np.linspace(-2, 2, 101),
                   np.linspace(-2, 2, 101), indexing='ij')
z = x*np.exp(- x**2 - y**2)

# Compute approximating quadratic with gradient and Hessian at (1, 0)
w = np.exp(-1)*(-1 - (x + 1) + (x + 1)**2 + y**2)

# Plot function
ax = d2l.plt.figure().add_subplot(111, projection='3d')
ax.plot_wireframe(x, y, z, **{'rstride': 10, 'cstride': 10})
ax.plot_wireframe(x, y, w, **{'rstride': 10, 'cstride': 10}, color='purple')
d2l.plt.xlabel('x')
d2l.plt.ylabel('y')
d2l.set_figsize()
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_zlim(-1, 1)
ax.dist = 12
```

```{.python .input}
#@tab pytorch
# Construct grid and compute function
x, y = torch.meshgrid(torch.linspace(-2, 2, 101),
                   torch.linspace(-2, 2, 101))

z = x*torch.exp(- x**2 - y**2)

# Compute approximating quadratic with gradient and Hessian at (1, 0)
w = torch.exp(torch.tensor([-1.]))*(-1 - (x + 1) + 2 * (x + 1)**2 + 2 * y**2)

# Plot function
ax = d2l.plt.figure().add_subplot(111, projection='3d')
ax.plot_wireframe(x.numpy(), y.numpy(), z.numpy(),
                  **{'rstride': 10, 'cstride': 10})
ax.plot_wireframe(x.numpy(), y.numpy(), w.numpy(),
                  **{'rstride': 10, 'cstride': 10}, color='purple')
d2l.plt.xlabel('x')
d2l.plt.ylabel('y')
d2l.set_figsize()
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_zlim(-1, 1)
ax.dist = 12
```

```{.python .input}
#@tab tensorflow
# Construct grid and compute function
x, y = tf.meshgrid(tf.linspace(-2., 2., 101),
                   tf.linspace(-2., 2., 101))

z = x*tf.exp(- x**2 - y**2)

# Compute approximating quadratic with gradient and Hessian at (1, 0)
w = tf.exp(tf.constant([-1.]))*(-1 - (x + 1) + 2 * (x + 1)**2 + 2 * y**2)

# Plot function
ax = d2l.plt.figure().add_subplot(111, projection='3d')
ax.plot_wireframe(x.numpy(), y.numpy(), z.numpy(),
                  **{'rstride': 10, 'cstride': 10})
ax.plot_wireframe(x.numpy(), y.numpy(), w.numpy(),
                  **{'rstride': 10, 'cstride': 10}, color='purple')
d2l.plt.xlabel('x')
d2l.plt.ylabel('y')
d2l.set_figsize()
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_zlim(-1, 1)
ax.dist = 12
```

Isso forma a base para o Algoritmo de Newton discutido em :numref:`sec_gd`, onde realizamos a otimização numérica encontrando iterativamente a quadrática de melhor ajuste e, em seguida, minimizando exatamente essa quadrática.

## Um Pouco de Cálculo Matricial
Derivadas de funções envolvendo matrizes revelaram-se particularmente interessantes. Esta seção pode se tornar notacionalmente pesada, portanto, pode ser ignorada em uma primeira leitura, mas é útil saber como as derivadas de funções que envolvem operações de matriz comum são muitas vezes muito mais limpas do que se poderia prever inicialmente, especialmente considerando como as operações de matriz centrais são para o aprendizado profundo aplicações.


Vamos começar com um exemplo. Suponha que temos algum vetor de coluna fixo $\boldsymbol{\beta}$, e queremos obter a função de produto $f(\mathbf{x}) = \boldsymbol{\beta}^\top\mathbf{x}$, e entender como o produto escalar muda quando mudamos $\mathbf{x}$.

Um pouco de notação que será útil ao trabalhar com derivadas de matriz em ML é chamado de *derivada de matriz de layout de denominador*, onde montamos nossas derivadas parciais na forma de qualquer vetor, matriz ou tensor que esteja no denominador do diferencial. Neste caso, vamos escrever

$$
\frac{df}{d\mathbf{x}} = \begin{bmatrix}
\frac{df}{dx_1} \\
\vdots \\
\frac{df}{dx_n}
\end{bmatrix},
$$


onde combinamos a forma do vetor coluna $\mathbf{x}$.

Se escrevermos nossa função em componentes, isso é

$$
f(\mathbf{x}) = \sum_{i = 1}^{n} \beta_ix_i = \beta_1x_1 + \cdots + \beta_nx_n.
$$

Se agora tomarmos a derivada parcial em relação a $\beta_1$, note que tudo é zero, exceto o primeiro termo, que é apenas $x_1$  multiplicado por $\beta_1$,, então obtemos isso

$$
\frac{df}{dx_1} = \beta_1,
$$

ou mais geralmente isso

$$
\frac{df}{dx_i} = \beta_i.
$$

Agora podemos remontar isso em uma matriz para ver

$$
\frac{df}{d\mathbf{x}} = \begin{bmatrix}
\frac{df}{dx_1} \\
\vdots \\
\frac{df}{dx_n}
\end{bmatrix} = \begin{bmatrix}
\beta_1 \\
\vdots \\
\beta_n
\end{bmatrix} = \boldsymbol{\beta}.
$$


Isso ilustra alguns fatores sobre o cálculo de matriz que muitas vezes iremos contrariar ao longo desta seção:

* Primeiro, os cálculos ficarão bastante complicados.
* Em segundo lugar, os resultados finais são muito mais limpos do que o processo intermediário e sempre serão semelhantes ao caso de uma única variável. Neste caso, observe que $\frac{d}{dx}(bx) = b$ and $\frac{d}{d\mathbf{x}} (\boldsymbol{\beta}^\top\mathbf{x}) = \boldsymbol{\beta}$ são ambos semelhantes.
* Terceiro, as transpostas muitas vezes podem aparecer aparentemente do nada. A principal razão para isso é a convenção de que combinamos a forma do denominador, portanto, quando multiplicamos as matrizes, precisaremos fazer transposições para corresponder à forma do termo original.

Para continuar construindo a intuição, vamos tentar um cálculo um pouco mais difícil. Suponha que temos um vetor coluna $\mathbf{x}$ e uma matriz quadrada $A$ e queremos calcular

$$\frac{d}{d\mathbf{x}}(\mathbf{x}^\top A \mathbf{x}).$$
:eqlabel:`eq_mat_goal_1`

Para direcionar para uma notação mais fácil de manipular, vamos considerar este problema usando a notação de Einstein. Neste caso, podemos escrever a função como

$$
\mathbf{x}^\top A \mathbf{x} = x_ia_{ij}x_j.
$$

Para calcular nossa derivada, precisamos entender para cada $k$, qual é o valor de

$$
\frac{d}{dx_k}(\mathbf{x}^\top A \mathbf{x}) = \frac{d}{dx_k}x_ia_{ij}x_j.
$$

Pela regra do produto, isso é

$$
\frac{d}{dx_k}x_ia_{ij}x_j = \frac{dx_i}{dx_k}a_{ij}x_j + x_ia_{ij}\frac{dx_j}{dx_k}.
$$

Para um termo como $\frac{dx_i}{dx_k}$, não é difícil ver que este é um quando $i=k$ e zero caso contrário. Isso significa que todos os termos em que $i$ e $k$ são diferentes desaparecem dessa soma, de modo que os únicos termos que permanecem nessa primeira soma são aqueles em que $i=k$. O mesmo raciocínio vale para o segundo termo em que precisamos de $j=k$. Isto dá

$$
\frac{d}{dx_k}x_ia_{ij}x_j = a_{kj}x_j + x_ia_{ik}.
$$

Agora, os nomes dos índices na notação de Einstein são arbitrários --- o fato de que $i$ e $j$ são diferentes é irrelevante para este cálculo neste ponto, então podemos reindexar para que ambos usem $i$ para ver isso

$$
\frac{d}{dx_k}x_ia_{ij}x_j = a_{ki}x_i + x_ia_{ik} = (a_{ki} + a_{ik})x_i.
$$

Agora, é aqui que precisamos de um pouco de prática para ir mais longe. Vamos tentar identificar esse resultado em termos de operações de matriz. $a_{ki} + a_{ik}$ é o $k, i$-ésimo componente de $\mathbf{A} + \mathbf{A}^\top$. Isto dá

$$
\frac{d}{dx_k}x_ia_{ij}x_j = [\mathbf{A} + \mathbf{A}^\top]_{ki}x_i.
$$

Da mesma forma, este termo é agora o produto da matriz $\mathbf{A} + \mathbf{A}^\top$ pelo vetor $\mathbf{x}$, então vemos que

$$
\left[\frac{d}{d\mathbf{x}}(\mathbf{x}^\top A \mathbf{x})\right]_k = \frac{d}{dx_k}x_ia_{ij}x_j = [(\mathbf{A} + \mathbf{A}^\top)\mathbf{x}]_k.
$$

Assim, vemos que a $k$ -ésima entrada da derivada desejada de :eqref:`eq_mat_goal_1` é apenas a $k$-ésima entrada do vetor à direita e, portanto, os dois são iguais. Assim

$$
\frac{d}{d\mathbf{x}}(\mathbf{x}^\top A \mathbf{x}) = (\mathbf{A} + \mathbf{A}^\top)\mathbf{x}.
$$

Isso exigiu muito mais trabalho do que o anterior, mas o resultado final é pequeno. Mais do que isso, considere o seguinte cálculo para derivadas de variável única tradicionais:

$$
\frac{d}{dx}(xax) = \frac{dx}{dx}ax + xa\frac{dx}{dx} = (a+a)x.
$$


Equivalentemente $\frac{d}{dx}(ax^2) = 2ax = (a+a)x$. Novamente, obtemos um resultado que se parece bastante com o resultado de uma única variável, mas com uma transposição inserida.

Neste ponto, o padrão deve parecer bastante suspeito, então vamos tentar descobrir o porquê. Quando tomamos derivadas de matriz como esta, vamos primeiro supor que a expressão que obtemos será outra expressão de matriz: uma expressão que podemos escrevê-la em termos de produtos e somas de matrizes e suas transposições. Se tal expressão existir, ela precisará ser verdadeira para todas as matrizes. Em particular, ele precisará ser verdadeiro para matrizes $1 \times 1$, caso em que o produto da matriz é apenas o produto dos números, a soma da matriz é apenas a soma e a transposta não faz nada! Em outras palavras, qualquer expressão que obtivermos *deve* corresponder à expressão de variável única. Isso significa que, com alguma prática, muitas vezes pode-se adivinhar as derivadas de matriz apenas por saber como a expressão da única variável associada deve se parecer!

Vamos tentar fazer isso. Suponha que $\mathbf{X}$ é uma matriz $n \times m$, $\mathbf{U}$ é uma $n \times r$ e $\mathbf{V}$ é uma $r \times m$. Vamos tentar calcular

$$\frac{d}{d\mathbf{V}} \|\mathbf{X} - \mathbf{U}\mathbf{V}\|_2^{2} = \;?$$
:eqlabel:`eq_mat_goal_2`

Este cálculo é importante em uma área chamada fatoração de matrizes. Para nós, no entanto, é apenas uma derivada para calcular. Vamos tentar imaginar o que seria para as matrizes $1\times1$. Nesse caso, obtemos a expressão

$$
\frac{d}{dv} (x-uv)^{2}= -2(x-uv)u,
$$

onde, a derivada é bastante padrão. Se tentarmos converter isso de volta em uma expressão de matriz, obteremos

$$
\frac{d}{d\mathbf{V}} \|\mathbf{X} - \mathbf{U}\mathbf{V}\|_2^{2}= -2(\mathbf{X} - \mathbf{U}\mathbf{V})\mathbf{U}.
$$


No entanto, se olharmos para isso, não funciona bem. Lembre-se de que $\mathbf{X}$ é $n \times m$, assim como $\mathbf{U}\mathbf{V}$, então a matriz $2(\mathbf{X} - \mathbf{U}\mathbf{V})$ é $n \times m$. Por outro lado $\mathbf{U}$ é $n \times r$, e não podemos multiplicar uma matriz $n \times m$ e uma $n \times r$ porque as dimensões não combinam!

Queremos obter $\frac{d}{d\mathbf{V}}$, que tem a mesma forma de $\mathbf{V}$, que é $r \times m$. Então, de alguma forma, precisamos pegar uma matriz $n \times m$ e uma matriz $n \times r$, multiplicá-las juntas (talvez com algumas transposições) para obter uma $r \times m$. Podemos fazer isso multiplicando $U^\top$ por $(\mathbf{X} - \mathbf{U}\mathbf{V})$.. Assim, podemos adivinhar a solução para :eqref:`eq_mat_goal_2` é

$$
\frac{d}{d\mathbf{V}} \|\mathbf{X} - \mathbf{U}\mathbf{V}\|_2^{2}= -2\mathbf{U}^\top(\mathbf{X} - \mathbf{U}\mathbf{V}).
$$

Para mostrar que isso funciona, seríamos negligentes em não fornecer um cálculo detalhado. Se já acreditamos que essa regra prática funciona, fique à vontade para pular esta derivação. Para calcular

$$
\frac{d}{d\mathbf{V}} \|\mathbf{X} - \mathbf{U}\mathbf{V}\|_2^2,
$$

devemos encontrar para cada $a$ e $b$

$$
\frac{d}{dv_{ab}} \|\mathbf{X} - \mathbf{U}\mathbf{V}\|_2^{2}= \frac{d}{dv_{ab}} \sum_{i, j}\left(x_{ij} - \sum_k u_{ik}v_{kj}\right)^2.
$$

Lembrando que todas as entradas de $\mathbf{X}$ e $\mathbf{U}$ são constantes no que diz respeito a $\frac{d}{dv_{ab}}$, podemos colocar a derivada dentro da soma, e aplicar a regra da cadeia ao quadrado para obter

$$
\frac{d}{dv_{ab}} \|\mathbf{X} - \mathbf{U}\mathbf{V}\|_2^{2}= \sum_{i, j}2\left(x_{ij} - \sum_k u_{ik}v_{kj}\right)\left(-\sum_k u_{ik}\frac{dv_{kj}}{dv_{ab}} \right).
$$

Como na derivação anterior, podemos notar que $\frac{dv_{kj}}{dv_{ab}}$ só é diferente de zero se $k=a$ and $j=b$. Se qualquer uma dessas condições não for válida, o termo na soma é zero e podemos descartá-lo livremente. Nós vemos que
$$
\frac{d}{dv_{ab}} \|\mathbf{X} - \mathbf{U}\mathbf{V}\|_2^{2}= -2\sum_{i}\left(x_{ib} - \sum_k u_{ik}v_{kb}\right)u_{ia}.
$$

Uma sutileza importante aqui é que o requisito de que $k=a$ não ocorre dentro da soma interna, uma vez que $k$ é uma variável que estamos somando dentro do termo interno. Para um exemplo notacionalmente mais limpo, considere por que
$$
\frac{d}{dx_1} \left(\sum_i x_i \right)^{2}= 2\left(\sum_i x_i \right).
$$

A partir deste ponto, podemos começar a identificar os componentes da soma. Primeiro,

$$
\sum_k u_{ik}v_{kb} = [\mathbf{U}\mathbf{V}]_{ib}.
$$

Portanto, toda a expressão no interior da soma é

$$
x_{ib} - \sum_k u_{ik}v_{kb} = [\mathbf{X}-\mathbf{U}\mathbf{V}]_{ib}.
$$

Isso significa que agora podemos escrever nossa derivada como

$$
\frac{d}{dv_{ab}} \|\mathbf{X} - \mathbf{U}\mathbf{V}\|_2^{2}= -2\sum_{i}[\mathbf{X}-\mathbf{U}\mathbf{V}]_{ib}u_{ia}.
$$

Queremos que se pareça com o elemento $a, b$ de uma matriz, para que possamos usar a técnica como no exemplo anterior para chegar a uma expressão de matriz, o que significa que precisamos trocar a ordem dos índices em $u_{ia}$. Se notarmos que $u_{ia} = [\mathbf{U}^\top]_{ai}$, podemos escrever

$$
\frac{d}{dv_{ab}} \|\mathbf{X} - \mathbf{U}\mathbf{V}\|_2^{2}= -2\sum_{i} [\mathbf{U}^\top]_{ai}[\mathbf{X}-\mathbf{U}\mathbf{V}]_{ib}.
$$

Este é um produto de matriz e, portanto, podemos concluir que

$$
\frac{d}{dv_{ab}} \|\mathbf{X} - \mathbf{U}\mathbf{V}\|_2^{2}= -2[\mathbf{U}^\top(\mathbf{X}-\mathbf{U}\mathbf{V})]_{ab}.
$$

e assim podemos escrever a solução para :eqref:`eq_mat_goal_2`

$$
\frac{d}{d\mathbf{V}} \|\mathbf{X} - \mathbf{U}\mathbf{V}\|_2^{2}= -2\mathbf{U}^\top(\mathbf{X} - \mathbf{U}\mathbf{V}).
$$


Isso corresponde à solução que adivinhamos acima!

É razoável perguntar neste ponto: "Por que não posso simplesmente escrever versões de matriz de todas as regras de cálculo que aprendi? Está claro que isso ainda é mecânico. Por que não simplesmente acabamos com isso!" E de fato existem tais regras e :cite:`Petersen.Pedersen.ea.2008` fornece um excelente resumo. No entanto, devido à infinidade de maneiras pelas quais as operações de matriz podem ser combinadas em comparação com valores únicos, há muito mais regras de derivadas de matriz do que regras de variável única. Geralmente, é melhor trabalhar com os índices ou deixar para a diferenciação automática, quando apropriado.

## Resumo

* Em dimensões superiores, podemos definir gradientes que têm o mesmo propósito que os derivadas em uma dimensão. Isso nos permite ver como uma função multivariável muda quando fazemos uma pequena mudança arbitrária nas entradas.
* O algoritmo de retropropagação pode ser visto como um método de organizar a regra da cadeia multivariável para permitir o cálculo eficiente de muitas derivadas parciais.
* O cálculo matricial nos permite escrever as derivadas das expressões matriciais de maneiras concisas.

## Exercícios
1. Dado um vetor de coluna $\boldsymbol{\beta}$, calcule as derivadas de $f(\mathbf{x}) = \boldsymbol{\beta}^\top\mathbf{x}$ e $g(\mathbf{x}) = \mathbf{x}^\top\boldsymbol{\beta}$. Por que você obtém a mesma resposta?
2. Seja $\mathbf{v}$ um vetor de dimensão $n$. O que é $\frac{\partial}{\partial\mathbf{v}}\|\mathbf{v}\|_2$?
3. Seja $L(x, y) = \log(e^x + e^y)$. Calcule o gradiente. Qual é a soma dos componentes do gradiente?
4. Seja $f(x, y) = x^2y + xy^2$. Mostre que o único ponto crítico é $(0,0)$. Considerando $ f(x, x)$, determine se $(0,0)$ é máximo, mínimo ou nenhum.
5. Suponha que estejamos minimizando uma função $f(\mathbf{x}) = g(\mathbf{x}) + h(\mathbf{x})$. Como podemos interpretar geometricamente a condição de $\nabla f = 0$ em termos de $g$ e $h$?


:begin_tab:`mxnet`
[Discussões](https://discuss.d2l.ai/t/413)
:end_tab:

:begin_tab:`pytorch`
[Discussões](https://discuss.d2l.ai/t/1090)
:end_tab:


:begin_tab:`tensorflow`
[Discussões](https://discuss.d2l.ai/t/1091)
:end_tab:
<!--stackedit_data:
eyJoaXN0b3J5IjpbMjA5MTAzMTkxNCwtODAxMTMyODMzLC0xNj
IwMjA1NzE4LDgwNDA1Njc4MywtMTI5NDk1OTc1NiwtMTEzMTI5
ODk2NCw5OTg0MjEyMDIsLTE1NjA3MjY1OTEsLTU5ODA5MzcxNC
wtMTc1NjU2MjYwNCwyMTA0NzIzNzA3LC0xMjA4NzMxMjg3LC03
MzkxNjAwNzQsLTE0NTEyOTgyODYsNzIzMDUzMzksLTQ3MTQ5ND
k4MCwxMzY0NzAyOTg3XX0=
-->