# Diferenciação automática
:label:`sec_autograd`

Como já explicado em :numref:`sec_calculus`, a diferenciação é uma etapa crucial em quase todos os algoritmos de otimização de *Deep Learning*. Embora os cálculos para obter esses derivados sejam diretos, exigindo apenas alguns cálculos básicos, para modelos complexos, trabalhando as atualizações manualmente pode ser uma tarefa difícil (e muitas vezes sujeita a erros).
*Frameworks* de *Deep learning* aceleram este trabalho calculando automaticamente as derivadas, ou seja, *diferenciação automática*. Na prática, com base em nosso modelo projetado o sistema constrói um *grafo computacional*, rastreando quais dados combinados por meio de quais operações produzem a saída. A diferenciação automática permite que o sistema propague gradientes posteriormente. Aqui, propagar(do Inglês *backpropagate*) significa simplesmente traçar o gráfico computacional, preencher as derivadas parciais em relação a cada parâmetro.


## Um exemplo simples

Como exemplo, digamos que estamos interessados em (**derivar a função $y = 2\mathbf{x}^{\top}\mathbf{x}$ com respeito ao vetor coluna $\mathbf{x}$.**)
Inicialmente criamos a variável `x` e atribuimos a ela um valor inicial.

```{.python .input}
from mxnet import autograd, np, npx
npx.set_np()

x = np.arange(4.0)
x
```

```{.python .input}
#@tab pytorch
import torch

x = torch.arange(4.0)
x
```

```{.python .input}
#@tab tensorflow
import tensorflow as tf

x = tf.range(4, dtype=tf.float32)
x
```

[**Antes de calcularmos o gradiente de$y$ em relação a $\mathbf{x}$,
precisamos armazena-lo.**]
É importante que não aloquemos nova memória cada vez que tomamos uma derivada em relação a um parâmetro porque costumamos atualizar os mesmos parâmetros milhares ou milhões de vezes e podemos rapidamente ficar sem memória.
Observe que um gradiente de uma função com valor escalar com respeito a um vetor $\mathbf{x}$ tem valor vetorial e tem a mesma forma de $\mathbf{x}$.


```{.python .input}
# Alocamos memória do gradiente do vetor invocando `attach_grad`
x.attach_grad()
# Após calcularmos o gradiente como respeito a `x`, será possível
# acessa-lo vai atributo `grad`, cujo valor inicializará com 0s
x.grad
```

```{.python .input}
#@tab pytorch
x.requires_grad_(True)  # Same as `x = torch.arange(4.0, requires_grad=True)`
x.grad  # O valor padrão é None
```

```{.python .input}
#@tab tensorflow
x = tf.Variable(x)
```

(**Então calcularemos $y$.**)

```{.python .input}
# Colocaremos o código dentro do escopo `autograd.record` para contruir
# o grafo computacional
with autograd.record():
    y = 2 * np.dot(x, x)
y
```

```{.python .input}
#@tab pytorch
y = 2 * torch.dot(x, x)
y
```

```{.python .input}
#@tab tensorflow
# Gravando todos os calculos em um *tape*
with tf.GradientTape() as t:
    y = 2 * tf.tensordot(x, x, axes=1)
y
```

Uma vez que `x` é um vetor de comprimento 4,
um produto interno de `x` e` x` é realizado,
produzindo a saída escalar que atribuímos a `y`.
Em seguida, [** podemos calcular automaticamente o gradiente de `y`
com relação a cada componente de `x` **]
chamando a função de retropropagação e imprimindo o gradiente.

```{.python .input}
y.backward()
x.grad
```

```{.python .input}
#@tab pytorch
y.backward()
x.grad
```

```{.python .input}
#@tab tensorflow
x_grad = t.gradient(y, x)
x_grad
```

(**O gradiente da função $y = 2\mathbf{x}^{\top}\mathbf{x}$
em relação a $\mathbf{x}$ should be $4\mathbf{x}$.**)
Vamos verificar rapidamente se nosso gradiente desejado foi calculado corretamente.

```{.python .input}
x.grad == 4 * x
```

```{.python .input}
#@tab pytorch
x.grad == 4 * x
```

```{.python .input}
#@tab tensorflow
x_grad == 4 * x
```

[**Agora calculamos outra função de `x`.**]

```{.python .input}
with autograd.record():
    y = x.sum()
y.backward()
x.grad  # Sobrescrito pelo novo gradiente calculado
```

```{.python .input}
#@tab pytorch
# O PyTorch acumula os gradientes por padrão, precisamos
# apagar os valores anteriores

x.grad.zero_()
y = x.sum()
y.backward()
x.grad
```

```{.python .input}
#@tab tensorflow
with tf.GradientTape() as t:
    y = tf.reduce_sum(x)
t.gradient(y, x)  # Overwritten by the newly calculated gradient
```

## Retroceder para variáveis não escalares

Tecnicamente, quando `y` não é um escalar,
a interpretação mais natural da diferenciação de um vetor `y`
em relação a um vetor, `x` é uma matriz.
Para `y` e` x` de ordem superior e dimensão superior,
o resultado da diferenciação pode ser um tensor de ordem alta.

No entanto, embora esses objetos mais exóticos apareçam
em aprendizado de máquina avançado (incluindo [**em *Deep Learning***]),
com mais frequência (**quando estamos retrocedendo um vetor,**)
estamos tentando calcular as derivadas das funções de perda
para cada constituinte de um *lote* de exemplos de treinamento.
Aqui, (**nossa intenção é**) não calcular a matriz de diferenciação
mas sim (**a soma das derivadas parciais
calculado individualmente para cada exemplo**) no lote.

```{.python .input}
# Quando invocamos `backward` em uma variável de vetor valorado `y` (em função de `x`),
# uma nova variável escalar é criada somando os elementos em `y`. Então o
# gradiente daquela variável escalar em respeito a `x` é computada
with autograd.record():
    y = x * x  # `y` is a vector
y.backward()
x.grad  # Igual a y = sum(x * x)
```

```{.python .input}
#@tab pytorch
# Invocar `backward` em um não escalar requer passar um argumento `gradient`
# que especifica o gradiente da função diferenciada w.r.t `self`.
# Em nosso caso, simplesmente queremos somar as derivadas parciais, assim passando
# em um gradiente de uns é apropriado
x.grad.zero_()
y = x * x
# y.backward(torch.ones(len(x))) equivalente a:
y.sum().backward()
x.grad
```

```{.python .input}
#@tab tensorflow
with tf.GradientTape() as t:
    y = x * x
t.gradient(y, x)  # Same as `y = tf.reduce_sum(x * x)`
```

## Computação *Detaching* 

Às vezes, desejamos [**mover alguns cálculos
fora do gráfico computacional registrado.**]
Por exemplo, digamos que `y` foi calculado como uma função de` x`,
e que subsequentemente `z` foi calculado como uma função de` y` e `x`.
Agora, imagine que quiséssemos calcular
o gradiente de `z` em relação a` x`,
mas queria, por algum motivo, tratar `y` como uma constante,
e apenas leve em consideração a função
que `x` jogou após` y` foi calculado.
Aqui, podemos desanexar `y` para retornar uma nova variável `u`
que tem o mesmo valor que `y`, mas descarta qualquer informação
sobre como `y` foi calculado no grafo computacional.
Em outras palavras, o gradiente não fluirá de volta de `u` para `x`.
Assim, a seguinte função de retropropagação calcula
a derivada parcial de `z = u * x` com respeito a` x` enquanto trata `u` como uma constante,
em vez da derivada parcial de `z = x * x * x` em relação a` x`.

```{.python .input}
with autograd.record():
    y = x * x
    u = y.detach()
    z = u * x
z.backward()
x.grad == u
```

```{.python .input}
#@tab pytorch
x.grad.zero_()
y = x * x
u = y.detach()
z = u * x

z.sum().backward()
x.grad == u
```

```{.python .input}
#@tab tensorflow
# Defina `persistent=True` para executar `t.gradient` mais de uma vez
with tf.GradientTape(persistent=True) as t:
    y = x * x
    u = tf.stop_gradient(y)
    z = u * x

x_grad = t.gradient(z, x)
x_grad == u
```

Uma vez que o cálculo de `y` foi registrado,
podemos subsequentemente invocar a retropropagação em `y` para obter a derivada de` y = x * x` com respeito a `x`, que é` 2 * x`.

```{.python .input}
y.backward()
x.grad == 2 * x
```

```{.python .input}
#@tab pytorch
x.grad.zero_()
y.sum().backward()
x.grad == 2 * x
```

```{.python .input}
#@tab tensorflow
t.gradient(y, x) == 2 * x
```

## Computando o Gradiente do *Python Control Flow*

Uma vantagem de usar a diferenciação automática
é que [**mesmo se**] construir o gráfo computacional de (**uma função
requer muito trabalho com o uso do  *Python Control Flow***)
(por exemplo, condicionais, loops e chamadas de função arbitrárias),
(**ainda podemos calcular o gradiente da variável resultante.**)
No trecho a seguir, observe que
o número de iterações do loop `while`
e a avaliação da instrução `if`
ambos dependem do valor da entrada `a`.

```{.python .input}
def f(a):
    b = a * 2
    while np.linalg.norm(b) < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c
```

```{.python .input}
#@tab pytorch
def f(a):
    b = a * 2
    while b.norm() < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c
```

```{.python .input}
#@tab tensorflow
def f(a):
    b = a * 2
    while tf.norm(b) < 1000:
        b = b * 2
    if tf.reduce_sum(b) > 0:
        c = b
    else:
        c = 100 * b
    return c
```

Vamos computar o gradiente:

```{.python .input}
a = np.random.normal()
a.attach_grad()
with autograd.record():
    d = f(a)
d.backward()
```

```{.python .input}
#@tab pytorch
a = torch.randn(size=(), requires_grad=True)
d = f(a)
d.backward()
```

```{.python .input}
#@tab tensorflow
a = tf.Variable(tf.random.normal(shape=()))
with tf.GradientTape() as t:
    d = f(a)
d_grad = t.gradient(d, a)
d_grad
```

Agora podemos analisar a função `f` definida acima.
Observe que é linear por partes em sua entrada `a`.
Em outras palavras, para qualquer `a` existe algum escalar constante` k`
tal que `f (a) = k * a`, onde o valor de` k` depende da entrada `a`.
Consequentemente, `d / a` nos permite verificar se o gradiente está correto.

```{.python .input}
a.grad == d / a
```

```{.python .input}
#@tab pytorch
a.grad == d / a
```

```{.python .input}
#@tab tensorflow
d_grad == d / a
```

## Sumário

* *Frameworks* de *Deep learning* podem automatizar o cálculo de derivadas. Para usá-lo, primeiro anexamos gradientes às variáveis em relação às quais desejamos as derivadas parciais. Em seguida, registramos o cálculo de nosso valor alvo, executamos sua função para retropropagação e acessamos o gradiente resultante.


## Exercícios

1. Por que a segunda derivada é muito mais computacionalmente cara de se calcular do que a primeira derivada?
2. Depois de executar a função de retropropagação, execute-a imediatamente novamente e veja o que acontece.
3. No exemplo de fluxo de controle onde calculamos a derivada de `d` com respeito a `a`, o que aconteceria se mudássemos a variável `a` para um vetor ou matriz aleatória. Neste ponto, o resultado do cálculo `f (a)` não é mais um escalar. O que acontece com o resultado? Como analisamos isso?
4. Redesenhe um exemplo para encontrar o gradiente do *Control Flow*. Execute e analise o resultado.
5. Seja $f (x) = \ sin (x)$. Plote $f (x)$ e $\ frac {df (x)} {dx}$, onde o último é calculado sem explorar que $f '(x) = \ cos (x)$.


:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/34)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/35)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/200)
:end_tab:
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTI4NzgwMTM5OSw4ODQwMzIxNTksLTg2Mj
IyMDIxOSwtMTg3MDI0NTA0NiwtMjAyMDM0OTY2NSwxMjQ4MDc0
NTIyLDE0MDY0MzkzNDFdfQ==
-->