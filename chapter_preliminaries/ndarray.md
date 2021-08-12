# Manipulação de Dados
:label:`sec_ndarray`

Para fazer qualquer coisa, precisamos de alguma forma de armazenar e manipular dados.
Geralmente, há duas coisas importantes que precisamos fazer com os dados: (i) adquirir
eles; e (ii) processá-los assim que estiverem dentro do computador. Não há
sentido em adquirir dados sem alguma forma de armazená-los, então vamos  brincar com dados sintéticos. Para começar, apresentamos o
*array* $n$-dimensional, também chamado de *tensor*.

Se você trabalhou com NumPy, o mais amplamente utilizado
pacote de computação científica em Python,
então você achará esta seção familiar.
Não importa qual estrutura você usa,
sua *classe de tensor* (`ndarray` em MXNet,
`Tensor` em PyTorch e TensorFlow) é semelhante ao` ndarray` do NumPy com
alguns recursos interessantes.
Primeiro, a GPU é bem suportada para acelerar a computação
enquanto o NumPy suporta apenas computação de CPU.
Em segundo lugar, a classe tensor
suporta diferenciação automática.
Essas propriedades tornam a classe tensor adequada para aprendizado profundo.
Ao longo do livro, quando dizemos tensores,
estamos nos referindo a instâncias da classe tensorial, a menos que seja declarado de outra forma.

## Iniciando

Nesta seção, nosso objetivo é colocá-lo em funcionamento,
equipando você com as ferramentas básicas de matemática e computação numérica
que você desenvolverá conforme progride no livro.
Não se preocupe se você lutar para grocar alguns dos
os conceitos matemáticos ou funções de biblioteca.
As seções a seguir revisitarão este material
no contexto de exemplos práticos e irá afundar.
Por outro lado, se você já tem alguma experiência
e quiser se aprofundar no conteúdo matemático, basta pular esta seção.

:begin_tab:`mxnet`
Para começar, importamos o `np` (` numpy`) e
Módulos `npx` (` numpy_extension`) da MXNet.
Aqui, o módulo `np` inclui funções suportadas por NumPy,
enquanto o módulo `npx` contém um conjunto de extensões
desenvolvido para capacitar o *Deep Learning* em um ambiente semelhante ao NumPy.
Ao usar tensores, quase sempre invocamos a função `set_np`:
isso é para compatibilidade de processamento de tensor por outros componentes do MXNet.
:end_tab:

:begin_tab:`pytorch`
(**Para começar, importamos `torch`. Note que apesar de ser chamado PyTorch, devemos importar `torch` ao invés de `pytorch`.**)
:end_tab:

:begin_tab:`tensorflow`
Importamos `tensorflow`. Como o nome é longo, importamos abreviando `tf`.
:end_tab:

```{.python .input}
from mxnet import np, npx
npx.set_np()
```

```{.python .input}
#@tab pytorch
import torch
```

```{.python .input}
#@tab tensorflow
import tensorflow as tf
```

[**Um tensor representa uma matriz (possivelmente multidimensional) de valores numéricos.**]
Com uma dimensão, um tensor corresponde (em matemática) a um *vetor*.
Com duas dimensões, um tensor corresponde a uma * matriz *.
Tensores com mais de dois eixos não possuem
nomes matemáticos.

Para começar, podemos usar `arange` para criar um vetor linha `x`
contendo os primeiros 12 inteiros começando com 0,
embora eles sejam criados como *float* por padrão.
Cada um dos valores em um tensor é chamado de *elemento* do tensor.
Por exemplo, existem 12 elementos no tensor `x`.
A menos que especificado de outra forma, um novo tensor
será armazenado na memória principal e designado para computação baseada em CPU.


```{.python .input}
x = np.arange(12)
x
```

```{.python .input}
#@tab pytorch
x = torch.arange(12)
x
```

```{.python .input}
#@tab tensorflow
x = tf.range(12)
x
```

(**Podemos acessar o formato do tensor**) (~~e o número total de elementos~~) (o comprimento em cada coordenada)
inspecionando sua propriedade `shape` .

```{.python .input}
#@tab all
x.shape
```

Se quisermos apenas saber o número total de elementos em um tensor,
ou seja, o produto de todos os *shapes*,
podemos inspecionar seu tamanho.
Porque estamos lidando com um vetor aqui,
o único elemento de seu `shape` é idêntico ao seu tamanho.

```{.python .input}
x.size
```

```{.python .input}
#@tab pytorch
x.numel()
```

```{.python .input}
#@tab tensorflow
tf.size(x)
```

Para [**mudar o *shape* de um tensor sem alterar
o número de elementos ou seus valores**],
podemos invocar a função `reshape`.
Por exemplo, podemos transformar nosso tensor, `x`,
de um vetor linha com forma (12,) para uma matriz com forma (3, 4).
Este novo tensor contém exatamente os mesmos valores,
mas os vê como uma matriz organizada em 3 linhas e 4 colunas.
Para reiterar, embora a forma tenha mudado,
os elementos não.
Observe que o tamanho não é alterado pela remodelagem.


```{.python .input}
#@tab mxnet, pytorch
X = x.reshape(3, 4)
X
```

```{.python .input}
#@tab tensorflow
X = tf.reshape(x, (3, 4))
X
```
A remodelação especificando manualmente todas as dimensões é desnecessária.
Se nossa forma de destino for uma matriz com forma (altura, largura),
então, depois de sabermos a largura, a altura é dada implicitamente.
Por que devemos realizar a divisão nós mesmos?
No exemplo acima, para obter uma matriz com 3 linhas,
especificamos que deve ter 3 linhas e 4 colunas.
Felizmente, os tensores podem calcular automaticamente uma dimensão considerando o resto.
Invocamos esse recurso colocando `-1` para a dimensão
que gostaríamos que os tensores inferissem automaticamente.
No nosso caso, em vez de chamar `x.reshape (3, 4)`,
poderíamos ter chamado equivalentemente `x.reshape (-1, 4)` ou `x.reshape (3, -1)`.

Normalmente, queremos que nossas matrizes sejam inicializadas
seja com zeros, uns, algumas outras constantes,
ou números amostrados aleatoriamente de uma distribuição específica.
[**Podemos criar um tensor representando um tensor com todos os elementos
definido como 0**] (~~ou 1~~)
e uma forma de (2, 3, 4) como a seguir:

```{.python .input}
np.zeros((2, 3, 4))
```

```{.python .input}
#@tab pytorch
torch.zeros((2, 3, 4))
```

```{.python .input}
#@tab tensorflow
tf.zeros((2, 3, 4))
```

Da mesma forma, podemos criar tensores com cada elemento definido como 1 da seguinte maneira:

```{.python .input}
np.ones((2, 3, 4))
```

```{.python .input}
#@tab pytorch
torch.ones((2, 3, 4))
```

```{.python .input}
#@tab tensorflow
tf.ones((2, 3, 4))
```
Frequentemente, queremos [**amostrar aleatoriamente os valores
para cada elemento em um tensor**]
de alguma distribuição de probabilidade.
Por exemplo, quando construímos matrizes para servir
como parâmetros em uma rede neural, vamos
normalmente inicializar seus valores aleatoriamente.
O fragmento a seguir cria um tensor com forma (3, 4).
Cada um de seus elementos é amostrado aleatoriamente
de uma distribuição gaussiana (normal) padrão
com uma média de 0 e um desvio padrão de 1.


```{.python .input}
np.random.normal(0, 1, size=(3, 4))
```

```{.python .input}
#@tab pytorch
torch.randn(3, 4)
```

```{.python .input}
#@tab tensorflow
tf.random.normal(shape=[3, 4])
```
Podemos também [**especificar os valores exatos para cada elemento**] no tensor desejado
fornecendo uma lista Python (ou lista de listas) contendo os valores numéricos.
Aqui, a lista externa corresponde ao eixo 0 e a lista interna ao eixo 1.


```{.python .input}
np.array([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
```

```{.python .input}
#@tab pytorch
torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
```

```{.python .input}
#@tab tensorflow
tf.constant([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
```

## Operações

Este livro não é sobre engenharia de software.
Nossos interesses não se limitam a simplesmente
leitura e gravação de dados de/para matrizes.
Queremos realizar operações matemáticas nessas matrizes.
Algumas das operações mais simples e úteis
são as operações elemento a elemento.
Estes aplicam uma operação escalar padrão
para cada elemento de uma matriz.
Para funções que usam dois arrays como entradas,
as operações elemento a elemento aplicam algum operador binário padrão
em cada par de elementos correspondentes das duas matrizes.
Podemos criar uma função elemento a elemento a partir de qualquer função
que mapeia de um escalar para um escalar.

Em notação matemática, denotaríamos tal
um operador escalar *unário* (tomando uma entrada)
pela assinatura $f: \mathbb{R} \rightarrow \mathbb{R}$.
Isso significa apenas que a função está mapeando
de qualquer número real ($\mathbb{R}$) para outro.
Da mesma forma, denotamos um operador escalar *binário*
(pegando duas entradas reais e produzindo uma saída)
pela assinatura $f: \mathbb{R}, \mathbb{R} \rightarrow \mathbb{R}$.
Dados quaisquer dois vetores $\mathbf{u}$ e $\mathbf{v}$ de mesmo *shape*, 
e um operador binário $f$, podemos produzir um vetor
$\mathbf{c} = F(\mathbf{u},\mathbf{v})$
definindo $c_i \gets f(u_i, v_i)$ para todos $i$,
onde $c_i, u_i$ e $v_i$ são os elementos $i^\mathrm{th}$
dos vetores $\mathbf{c}, \mathbf{u}$, e $\mathbf{v}$.
Aqui, nós produzimos o valor vetorial
$F: \mathbb{R}^d, \mathbb{R}^d \rightarrow \mathbb{R}^d$
*transformando* a função escalar para uma operação de vetor elemento a elemento.

Os operadores aritméticos padrão comuns
(`+`, `-`,` * `,` / `e` ** `)
foram todos transformados em operações elemento a elemento
para quaisquer tensores de formato idêntico de forma arbitrária.
Podemos chamar operações elemento a elemento em quaisquer dois tensores da mesma forma.
No exemplo a seguir, usamos vírgulas para formular uma tupla de 5 elementos,
onde cada elemento é o resultado de uma operação elemento a elemento.

### Operações

[**Os operadores aritméticos padrão comuns
(`+`, `-`,` * `,` / `e` ** `)
foram todos transformados em operações elemento a elemento.**]

```{.python .input}
x = np.array([1, 2, 4, 8])
y = np.array([2, 2, 2, 2])
x + y, x - y, x * y, x / y, x ** y  # O ** é o operador exponenciação
```

```{.python .input}
#@tab pytorch
x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])
x + y, x - y, x * y, x / y, x ** y  #  O ** é o operador exponenciação
```

```{.python .input}
#@tab tensorflow
x = tf.constant([1.0, 2, 4, 8])
y = tf.constant([2.0, 2, 2, 2])
x + y, x - y, x * y, x / y, x ** y  #  O ** é o operador exponenciação
```

Muitos (**mais operações podem ser aplicadas elemento a elemento**),
incluindo operadores unários como exponenciação.

```{.python .input}
np.exp(x)
```

```{.python .input}
#@tab pytorch
torch.exp(x)
```

```{.python .input}
#@tab tensorflow
tf.exp(x)
```

Além de cálculos elemento a elemento,
também podemos realizar operações de álgebra linear,
incluindo produtos escalar de vetor e multiplicação de matrizes.
Explicaremos as partes cruciais da álgebra linear
(sem nenhum conhecimento prévio assumido) em :numref:`sec_linear-algebra`.



Também podemos [***concatenar* vários tensores juntos,**]
empilhando-os ponta a ponta para formar um tensor maior.
Só precisamos fornecer uma lista de tensores
e informar ao sistema ao longo de qual eixo concatenar.
O exemplo abaixo mostra o que acontece quando concatenamos
duas matrizes ao longo das linhas (eixo 0, o primeiro elemento da forma)
vs. colunas (eixo 1, o segundo elemento da forma).
Podemos ver que o comprimento do eixo 0 do primeiro tensor de saída ($6$)
é a soma dos comprimentos do eixo 0 dos dois tensores de entrada ($3 + 3$);
enquanto o comprimento do eixo 1 do segundo tensor de saída ($8$)
é a soma dos comprimentos do eixo 1 dos dois tensores de entrada ($4 + 4$).
```{.python .input}
X = np.arange(12).reshape(3, 4)
Y = np.array([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
np.concatenate([X, Y], axis=0), np.concatenate([X, Y], axis=1)
```

```{.python .input}
#@tab pytorch
X = torch.arange(12, dtype=torch.float32).reshape((3,4))
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
torch.cat((X, Y), dim=0), torch.cat((X, Y), dim=1)
```

```{.python .input}
#@tab tensorflow
X = tf.reshape(tf.range(12, dtype=tf.float32), (3, 4))
Y = tf.constant([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
tf.concat([X, Y], axis=0), tf.concat([X, Y], axis=1)
```

Às vezes, queremos [**construir um tensor binário por meio de *declarações lógicas*.**]
Tome `X == Y` como exemplo.
Para cada posição, se `X` e` Y` forem iguais nessa posição,
a entrada correspondente no novo tensor assume o valor 1,
o que significa que a declaração lógica `X == Y` é verdadeira nessa posição;
caso contrário, essa posição assume 0.

```{.python .input}
#@tab all
X == Y
```
[**Somando todos os elementos no tensor**] resulta em um tensor com apenas um elemento.

```{.python .input}
#@tab mxnet, pytorch
X.sum()
```

```{.python .input}
#@tab tensorflow
tf.reduce_sum(X)
```

## Mecanismo de *Broadcasting* 
:label:`subsec_broadcasting`

Na seção acima, vimos como realizar operações elemento a elemento
em dois tensores da mesma forma. Sob certas condições,
mesmo quando as formas são diferentes, ainda podemos [**realizar operações elementar
invocando o mecanismo de *Broadcasting*.**]
Esse mecanismo funciona da seguinte maneira:
Primeiro, expanda um ou ambos os arrays
copiando elementos de forma adequada
de modo que após esta transformação,
os dois tensores têm a mesma forma.
Em segundo lugar, execute as operações elemento a elemento
nas matrizes resultantes.

Na maioria dos casos, nós transmitimos ao longo de um eixo onde uma matriz
inicialmente tem apenas o comprimento 1, como no exemplo a seguir:

```{.python .input}
a = np.arange(3).reshape(3, 1)
b = np.arange(2).reshape(1, 2)
a, b
```

```{.python .input}
#@tab pytorch
a = torch.arange(3).reshape((3, 1))
b = torch.arange(2).reshape((1, 2))
a, b
```

```{.python .input}
#@tab tensorflow
a = tf.reshape(tf.range(3), (3, 1))
b = tf.reshape(tf.range(2), (1, 2))
a, b
```

Uma vez que `a` e` b` são matrizes $3\times1$ e $1\times2$  respectivamente,
suas formas não correspondem se quisermos adicioná-los.
Nós transmitimos as entradas de ambas as matrizes em uma matriz $3\times2$ maior da seguinte maneira:
para a matriz `a` ele replica as colunas
e para a matriz `b` ele replica as linhas
antes de adicionar ambos os elementos.

```{.python .input}
#@tab all
a + b
```

## Indexação e Fatiamento

Assim como em qualquer outro array Python, os elementos em um tensor podem ser acessados por índice.
Como em qualquer matriz Python, o primeiro elemento tem índice 0
e os intervalos são especificados para incluir o primeiro, mas *antes* do último elemento.
Como nas listas padrão do Python, podemos acessar os elementos
de acordo com sua posição relativa ao final da lista
usando índices negativos.

Assim, [**`[-1]` seleciona o último elemento e `[1: 3]`
seleciona o segundo e o terceiro elementos**] da seguinte forma:

```{.python .input}
#@tab all
X[-1], X[1:3]
```

:begin_tab:`mxnet, pytorch`
Além da leitura, (**também podemos escrever elementos de uma matriz especificando índices.**)
:end_tab:

:begin_tab:`tensorflow`
`Tensors` in TensorFlow are immutable, and cannot be assigned to.
`Variables` in TensorFlow are mutable containers of state that support
assignments. Keep in mind that gradients in TensorFlow do not flow backwards
through `Variable` assignments.
Os `Tensors` no TensorFlow são imutáveis e não podem ser atribuídos a eles.
`Variables` no TensorFlow são contêineres mutáveis de estado que suportam
atribuições. Lembre-se de que gradientes no TensorFlow não fluem para trás
por meio de atribuições `Variable`.

Beyond assigning a value to the entire `Variable`, we can write elements of a
`Variable` by specifying indices.
Além de atribuir um valor a toda a `Variable`, podemos escrever elementos de um
`Variable` especificando índices.
:end_tab:

```{.python .input}
#@tab mxnet, pytorch
X[1, 2] = 9
X
```

```{.python .input}
#@tab tensorflow
X_var = tf.Variable(X)
X_var[1, 2].assign(9)
X_var
```

Se quisermos [**para atribuir a vários elementos o mesmo valor,
simplesmente indexamos todos eles e, em seguida, atribuímos o valor a eles.**]
Por exemplo, `[0: 2,:]` acessa a primeira e a segunda linhas,
onde `:` leva todos os elementos ao longo do eixo 1 (coluna).
Enquanto discutimos a indexação de matrizes,
isso obviamente também funciona para vetores
e para tensores de mais de 2 dimensões.

```{.python .input}
#@tab mxnet, pytorch
X[0:2, :] = 12
X
```

```{.python .input}
#@tab tensorflow
X_var = tf.Variable(X)
X_var[0:2, :].assign(tf.ones(X_var[0:2,:].shape, dtype = tf.float32) * 12)
X_var
```

## Economizando memória

[**As operações em execução podem fazer com que uma nova memória seja
alocado aos resultados do host.**]
Por exemplo, se escrevermos `Y = X + Y`,
vamos desreferenciar o tensor que `Y` costumava apontar para
e, em vez disso, aponte `Y` para a memória recém-alocada.
No exemplo a seguir, demonstramos isso com a função `id ()` do Python,
que nos dá o endereço exato do objeto referenciado na memória.
Depois de executar `Y = Y + X`, descobriremos que` id (Y) `aponta para um local diferente.
Isso ocorre porque o Python primeiro avalia `Y + X`,
alocar nova memória para o resultado e, em seguida, torna `Y`
aponte para este novo local na memória.

```{.python .input}
#@tab all
before = id(Y)
Y = Y + X
id(Y) == before
```

Isso pode ser indesejável por dois motivos.
Em primeiro lugar, não queremos
alocar memória desnecessariamente o tempo todo.
No aprendizado de máquina, podemos ter
centenas de megabytes de parâmetros
e atualizar todos eles várias vezes por segundo.
Normalmente, queremos realizar essas atualizações no local.
Em segundo lugar, podemos apontar os mesmos parâmetros de várias variáveis.
Se não atualizarmos no local, outras referências ainda apontarão para
a localização da memória antiga, tornando possível para partes do nosso código
para referenciar inadvertidamente parâmetros obsoletos.

:begin_tab:`mxnet, pytorch`
Felizmente, (**executar operações no local**) é fácil.
Podemos atribuir o resultado de uma operação
para uma matriz previamente alocada com notação de fatia,
por exemplo, `Y [:] = <expressão>`.
Para ilustrar este conceito, primeiro criamos uma nova matriz `Z`
com a mesma forma de outro `Y`,
usando `zeros_like` para alocar um bloco de $0$ entradas.
: end_tab:

:begin_tab:`tensorflow`
`Variables` são contêineres mutáveis de estado no TensorFlow. Eles providenciam
uma maneira de armazenar os parâmetros do seu modelo.
Podemos atribuir o resultado de uma operação
para uma `Variable` com` assign`.
Para ilustrar este conceito, criamos uma `Variable`` Z`
com a mesma forma de outro tensor `Y`,
usando `zeros_like` para alocar um bloco de $0$ entradas.
:end_tab:

```{.python .input}
Z = np.zeros_like(Y)
print('id(Z):', id(Z))
Z[:] = X + Y
print('id(Z):', id(Z))
```

```{.python .input}
#@tab pytorch
Z = torch.zeros_like(Y)
print('id(Z):', id(Z))
Z[:] = X + Y
print('id(Z):', id(Z))
```

```{.python .input}
#@tab tensorflow
Z = tf.Variable(tf.zeros_like(Y))
print('id(Z):', id(Z))
Z.assign(X + Y)
print('id(Z):', id(Z))
```

:begin_tab:`mxnet, pytorch`
[**Se o valor de `X` não for reutilizado em cálculos subsequentes,
também podemos usar `X [:] = X + Y` ou` X + = Y`
para reduzir a sobrecarga de memória da operação.**]
:end_tab:

:begin_tab:`tensorflow`
Mesmo depois de armazenar o estado persistentemente em uma `Variável`, você
pode querer reduzir ainda mais o uso de memória, evitando o excesso de
alocações para tensores que não são os parâmetros do seu modelo.

Porque os `Tensors`do TensorFlow são imutáveis e gradientes não fluem através de
atribuições de `Variable`, o TensorFlow não fornece uma maneira explícita de executar
uma operação individual no local.

No entanto, o TensorFlow fornece o decorador `tf.function` para encerrar a computação
dentro de um gráfico do TensorFlow que é compilado e otimizado antes da execução.
Isso permite que o TensorFlow remova valores não utilizados e reutilize
alocações anteriores que não são mais necessárias. Isso minimiza a sobrecarga de memória
 de cálculos do TensorFlow.
:end_tab:

```{.python .input}
#@tab mxnet, pytorch
before = id(X)
X += Y
id(X) == before
```

```{.python .input}
#@tab tensorflow
@tf.function
def computation(X, Y):
    Z = tf.zeros_like(Y)  # Este valor não utilizado será esvaziado
    A = X + Y  # Alocações serão reutilizadas quando não mais necessárias
    B = A + Y
    C = B + Y
    return C + Y

computation(X, Y)
```


## Conversão para outros objetos Python

[**Converter para um tensor NumPy**], ou vice-versa, é fácil.
O resultado convertido não compartilha memória.
Este pequeno inconveniente é muito importante:
quando você executa operações na CPU ou GPUs,
você não quer interromper a computação, esperando para ver
se o pacote NumPy do Python deseja fazer outra coisa
com o mesmo pedaço de memória.


```{.python .input}
A = X.asnumpy()
B = np.array(A)
type(A), type(B)
```

```{.python .input}
#@tab pytorch
A = X.numpy()
B = torch.tensor(A)
type(A), type(B)
```

```{.python .input}
#@tab tensorflow
A = X.numpy()
B = tf.constant(A)
type(A), type(B)
```

Para (**converter um tensor de tamanho 1 em um escalar Python**),
podemos invocar a função `item` ou as funções integradas do Python.


```{.python .input}
a = np.array([3.5])
a, a.item(), float(a), int(a)
```

```{.python .input}
#@tab pytorch
a = torch.tensor([3.5])
a, a.item(), float(a), int(a)
```

```{.python .input}
#@tab tensorflow
a = tf.constant([3.5]).numpy()
a, a.item(), float(a), int(a)
```

## Sumário

* A principal interface para armazenar e manipular dados para *Deep Learning* é o tensor (array $n$ -dimensional). Ele fornece uma variedade de funcionalidades, incluindo operações matemáticas básicas, transmissão, indexação, divisão, economia de memória e conversão para outros objetos Python.


## Exercícios


1. Execute o código nesta seção. Altere a declaração condicional `X == Y` nesta seção para` X < Y` ou `X > Y`, e então veja que tipo de tensor você pode obter.
1. Substitua os dois tensores que operam por elemento no mecanismo de transmissão por outras formas, por exemplo, tensores tridimensionais. O resultado é o mesmo que o esperado?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/26)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/27)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/187)
:end_tab:
<!--stackedit_data:
eyJoaXN0b3J5IjpbMTA4ODAyODc4NywxODIxNDIyNDIwLC02Nj
UyNTk0NzYsNzkwOTIwODc3LC0xMzk2MDk1NTcxLC03NTk4MzM3
MywxMTY5Mjg1NTgsLTE2OTYyODE0MTUsLTEzMDQ3MTU0ODBdfQ
==
-->