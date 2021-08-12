# Camadas e Blocos
:label:`sec_model_construction`

Quando introduzimos as redes neurais pela primeira vez,
focamos em modelos lineares com uma única saída.
Aqui, todo o modelo consiste em apenas um único neurônio.
Observe que um único neurônio
(i) leva algum conjunto de entradas;
(ii) gera uma saída escalar correspondente;
e (iii) tem um conjunto de parâmetros associados que podem ser atualizados
para otimizar alguma função objetivo de interesse.
Então, quando começamos a pensar em redes com múltiplas saídas,
nós alavancamos a aritmética vetorizada
para caracterizar uma camada inteira de neurônios.
Assim como os neurônios individuais,
camadas (i) recebem um conjunto de entradas,
(ii) gerar resultados correspondentes,
e (iii) são descritos por um conjunto de parâmetros ajustáveis.
Quando trabalhamos com a regressão softmax,
uma única camada era ela própria o modelo.
No entanto, mesmo quando subsequentemente
introduziu MLPs,
ainda podemos pensar no modelo como
mantendo esta mesma estrutura básica.

Curiosamente, para MLPs,
todo o modelo e suas camadas constituintes
compartilham essa estrutura.
Todo o modelo recebe entradas brutas (os recursos),
gera resultados (as previsões),
e possui parâmetros
(os parâmetros combinados de todas as camadas constituintes).
Da mesma forma, cada camada individual ingere entradas
(fornecido pela camada anterior)
gera saídas (as entradas para a camada subsequente),
e possui um conjunto de parâmetros ajustáveis que são atualizados
de acordo com o sinal que flui para trás
da camada subsequente.

Embora você possa pensar que neurônios, camadas e modelos
dê-nos abstrações suficientes para cuidar de nossos negócios,
Acontece que muitas vezes achamos conveniente
para falar sobre componentes que são
maior do que uma camada individual
mas menor do que o modelo inteiro.
Por exemplo, a arquitetura ResNet-152,
que é muito popular na visão computacional,
possui centenas de camadas.
Essas camadas consistem em padrões repetidos de *grupos de camadas*. Implementar uma camada de rede por vez pode se tornar tedioso.
Essa preocupação não é apenas hipotética --- tal
padrões de projeto são comuns na prática.
A arquitetura ResNet mencionada acima
venceu as competições de visão computacional ImageNet e COCO 2015
para reconhecimento e detecção :cite:`He.Zhang.Ren.ea.2016`
e continua sendo uma arquitetura indispensável para muitas tarefas de visão.
Arquiteturas semelhantes nas quais as camadas são organizadas
em vários padrões repetidos
agora são onipresentes em outros domínios,
incluindo processamento de linguagem natural e fala.

Para implementar essas redes complexas,
introduzimos o conceito de uma rede neural *block*.
Um bloco pode descrever uma única camada,
um componente que consiste em várias camadas,
ou o próprio modelo inteiro!
Uma vantagem de trabalhar com a abstração de bloco
é que eles podem ser combinados em artefatos maiores,
frequentemente recursivamente. Isso é ilustrado em :numref:`fig_blocks`. Definindo o código para gerar blocos
de complexidade arbitrária sob demanda,
podemos escrever código surpreendentemente compacto
e ainda implementar redes neurais complexas.

![Múltiplas camadas são combinadas em blocos, formando padrões repetitivos de um modelo maior.](../img/blocks.svg)
:label:`fig_blocks`

Do ponto de vista da programação, um bloco é representado por uma *classe*.
Qualquer subclasse dele deve definir uma função de propagação direta
que transforma sua entrada em saída
e deve armazenar todos os parâmetros necessários.
Observe que alguns blocos não requerem nenhum parâmetro.
Finalmente, um bloco deve possuir uma função de retropropagação,
para fins de cálculo de gradientes.
Felizmente, devido a alguma magia dos bastidores
fornecido pela diferenciação automática
(introduzido em :numref:`sec_autograd`)
ao definir nosso próprio bloco,
só precisamos nos preocupar com os parâmetros
e a função de propagação direta.

Para começar, revisitamos o código
que usamos para implementar MLPs
(:numref:`sec_mlp_concise`).
O código a seguir gera uma rede
com uma camada oculta totalmente conectada
com 256 unidades e ativação ReLU,
seguido por uma camada de saída totalmente conectada
com 10 unidades (sem função de ativação).

```{.python .input}
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()

net = nn.Sequential()
net.add(nn.Dense(256, activation='relu'))
net.add(nn.Dense(10))
net.initialize()

X = np.random.uniform(size=(2, 20))
net(X)
```

```{.python .input}
#@tab pytorch
import torch
from torch import nn
from torch.nn import functional as F

net = nn.Sequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))

X = torch.rand(2, 20)
net(X)
```

```{.python .input}
#@tab tensorflow
import tensorflow as tf

net = tf.keras.models.Sequential([
    tf.keras.layers.Dense(256, activation=tf.nn.relu),
    tf.keras.layers.Dense(10),
])

X = tf.random.uniform((2, 20))
net(X)
```

:begin_tab:`mxnet`
Neste exemplo, nós construímos
nosso modelo instanciando um `nn.Sequential`,
atribuindo o objeto retornado à variável `net`.
Em seguida, chamamos repetidamente sua função `add`,
anexando camadas no pedido
que eles devem ser executados.
Em suma, `nn.Sequential` define um tipo especial de` Block`,
a classe que apresenta um bloco em Gluon.
Ele mantém uma lista ordenada de `Block` constituintes.
A função `add` simplesmente facilita
a adição de cada `Bloco` sucessivo à lista.
Observe que cada camada é uma instância da classe `Dense`
que é uma subclasse de `Block`.
A função de propagação direta (`forward`) também é notavelmente simples:
ele encadeia cada `Block` na lista,
passando a saída de cada um como entrada para o próximo.
Observe que, até agora, temos invocado nossos modelos
através da construção `net (X)` para obter seus resultados.
Na verdade, isso é apenas um atalho para `net.forward (X)`,
um truque Python habilidoso alcançado via
a função `__call__` da classe` Block`.
:end_tab:

:begin_tab:`pytorch`
Neste exemplo, nós construímos
nosso modelo instanciando um `nn.Sequential`, com camadas na ordem
que eles devem ser executados passados como argumentos.
Em suma, `nn.Sequential` define um tipo especial de `Module`,
a classe que apresenta um bloco em PyTorch.
Ele mantém uma lista ordenada de `Module` constituintes.
Observe que cada uma das duas camadas totalmente conectadas é uma instância da classe `Linear`
que é uma subclasse de `Module`.
A função de propagação direta (`forward`) também é notavelmente simples:
ele encadeia cada bloco da lista,
passando a saída de cada um como entrada para o próximo.
Observe que, até agora, temos invocado nossos modelos
através da construção `net (X)` para obter seus resultados.
Na verdade, isso é apenas um atalho para `net.__call__(X)`.
:end_tab:

:begin_tab:`tensorflow`
Neste exemplo, nós construímos
nosso modelo instanciando um `keras.models.Sequential`, com camadas na ordem
que eles devem ser executados passados como argumentos.
Em suma, `Sequential` define um tipo especial de` keras.Model`,
a classe que apresenta um bloco em Keras.
Ele mantém uma lista ordenada de `Model` constituintes.
Observe que cada uma das duas camadas totalmente conectadas é uma instância da classe `Dense`
que é uma subclasse de `Model`.
A função de propagação direta (`call`) também é extremamente simples:
ele encadeia cada bloco da lista,
passando a saída de cada um como entrada para o próximo.
Observe que, até agora, temos invocado nossos modelos
através da construção `net (X)` para obter seus resultados.
Na verdade, isso é apenas um atalho para `net.call(X)`,
um truque Python habilidoso alcançado via
a função `__call__` da classe Block.
:end_tab:

## Um Bloco Personalizado

Talvez a maneira mais fácil de desenvolver intuição
sobre como funciona um bloco
é implementar um nós mesmos.
Antes de implementar nosso próprio bloco personalizado,
resumimos brevemente a funcionalidade básica
que cada bloco deve fornecer:

1. Ingerir dados de entrada como argumentos para sua função de propagação direta.
1. Gere uma saída fazendo com que a função de propagação direta retorne um valor. Observe que a saída pode ter uma forma diferente da entrada. Por exemplo, a primeira camada totalmente conectada em nosso modelo acima ingere uma entrada de dimensão arbitrária, mas retorna uma saída de dimensão 256.
1. Calcule o gradiente de sua saída em relação à sua entrada, que pode ser acessado por meio de sua função de retropropagação. Normalmente, isso acontece automaticamente.
1. Armazene e forneça acesso aos parâmetros necessários
    para executar o cálculo de propagação direta.
1. Inicialize os parâmetros do modelo conforme necessário.

No seguinte trecho de código,
nós codificamos um bloco do zero
correspondendo a um MLP
com uma camada oculta com 256 unidades ocultas,
e uma camada de saída de 10 dimensões.
Observe que a classe `MLP` abaixo herda a classe que representa um bloco.
Vamos contar muito com as funções da classe pai,
fornecendo apenas nosso próprio construtor (a função `__init__` em Python) e a função de propagação direta.

```{.python .input}
class MLP(nn.Block):
    # Declare uma camada com parâmetros de modelo. 
    # Aqui, nós declaramos duas camadas completamente conectadas
    def __init__(self, **kwargs):
        # Chame o construtor da classe pai `MLP` `Block` para realizar
        # as inicializações necessárias. Desta forma, outros argumentos das funções
        # também podem ser especificados durante a instalação da classe,
        # da mesma forma que os parâmetros do modelo, 'params' (a ser descrito posteriormente)
        super().__init__(**kwargs)
        self.hidden = nn.Dense(256, activation='relu')  # Hidden layer
        self.out = nn.Dense(10)  # Output layer

    # Defina a propagação direta do modelo, ou seja, como retornar 
    # a saída do modelo requirido baseado na entrada 'X'
    def forward(self, X):
        return self.out(self.hidden(X))
```

```{.python .input}
#@tab pytorch
class MLP(nn.Module):
    # Declare uma camada com parâmetros de modelo. Aqui, declaramos duas
    # camadas totalmente conectadas
    def __init__(self):
        # Chame o construtor da classe pai `MLP` `Block` para realizar
        # a inicialização necessária. Desta forma, outros argumentos de função
        # também podem ser especificado durante a instanciação da classe, como 
        # os parametros do modelo, `params` (a ser descritos posteriormente)
        super().__init__()
        self.hidden = nn.Linear(20, 256)  # Hidden layer
        self.out = nn.Linear(256, 10)  # Output layer

    # Defina a propagação direta do modelo, ou seja, como retornar a
    # saída do modelo necessária com base na entrada `X`
    def forward(self, X):
        # Observe aqui que usamos a versão funcional do ReLU definida no
        # módulo nn.functional.
        return self.out(F.relu(self.hidden(X)))
```

```{.python .input}
#@tab tensorflow
class MLP(tf.keras.Model):
    # Declare uma camada com parâmetros de modelo. Aqui, declaramos duas
    # camadas totalmente conectadas
    def __init__(self):
        # Chame o construtor da classe pai `MLP` `Block` para realizar
        # a inicialização necessária. Desta forma, outros argumentos de função
        # também podem ser especificados durante a instanciação da classe, como os 
        # parâmetros do modelo, `params` (a serem descritos mais tarde)
        super().__init__()
        # Camadas escondidas
        self.hidden = tf.keras.layers.Dense(units=256, activation=tf.nn.relu)
        self.out = tf.keras.layers.Dense(units=10)  # Output layer

    # Defina a propagação direta do modelo, ou seja, como retornar a
    # saída do modelo necessária com base na entrada `X`
    def call(self, X):
        return self.out(self.hidden((X)))
```

Vamos primeiro nos concentrar na função de propagação direta.
Observe que leva `X` como entrada,
calcula a representação oculta
com a função de ativação aplicada,
e produz seus *logits*.
Nesta implementação `MLP`,
ambas as camadas são variáveis de instância.
Para ver por que isso é razoável, imagine
instanciando dois MLPs, `net1` e` net2`,
e treiná-los em dados diferentes.
Naturalmente, esperaríamos que eles
para representar dois modelos aprendidos diferentes.

Nós instanciamos as camadas do MLP
no construtor
e posteriormente invocar essas camadas
em cada chamada para a função de propagação direta.
Observe alguns detalhes importantes:
Primeiro, nossa função `__init__` personalizada
invoca a função `__init__` da classe pai
via `super().__ init __()`
poupando-nos da dor de reafirmar o
código padrão aplicável à maioria dos blocos.
Em seguida, instanciamos nossas duas camadas totalmente conectadas,
atribuindo-os a `self.hidden` e` self.out`.
Observe que, a menos que implementemos um novo operador,
não precisamos nos preocupar com a função de *backpropagation*
ou inicialização de parâmetro.
O sistema irá gerar essas funções automaticamente.
Vamos tentar fazer isso.

```{.python .input}
net = MLP()
net.initialize()
net(X)
```

```{.python .input}
#@tab pytorch
net = MLP()
net(X)
```

```{.python .input}
#@tab tensorflow
net = MLP()
net(X)
```

Uma virtude fundamental da abstração em bloco é sua versatilidade.
Podemos criar uma subclasse de um bloco para criar camadas
(como a classe de camada totalmente conectada),
modelos inteiros (como a classe `MLP` acima),
ou vários componentes de complexidade intermediária.
Nós exploramos essa versatilidade
ao longo dos capítulos seguintes,
como ao abordar
redes neurais convolucionais.


## O Bloco Sequencial

Agora podemos dar uma olhada mais de perto
em como a classe `Sequential` funciona.
Lembre-se de que `Sequential` foi projetado
para conectar outros blocos em série.
Para construir nosso próprio `MySequential` simplificado,
só precisamos definir duas funções principais:
1. Uma função para anexar um blocos a uma lista.
2. Uma função de propagação direta para passar uma entrada através da cadeia de blocos, na mesma ordem em que foram acrescentados.

A seguinte classe `MySequential` oferece o mesmo
funcionalidade da classe `Sequential` padrão.

```{.python .input}
class MySequential(nn.Block):
    def add(self, block):
        # Here, `block` is an instance of a `Block` subclass, and we assume 
        #
        # that it has a unique name. We save it in the member variable
        #
        # `_children` of the `Block` class, and its type is OrderedDict. When
        #
        # the `MySequential` instance calls the `initialize` function, the
        #
        # system automatically initializes all members of `_children`
        #
        self._children[block.name] = block

    def forward(self, X):
        # OrderedDict guarantees that members will be traversed in the order
        #
        # they were added
        #
        for block in self._children.values():
            X = block(X)
        return X
```

```{.python .input}
#@tab pytorch
class MySequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        for idx, module in enumerate(args):
            # Here, `module` is an instance of a `Module` subclass. We save it
            #
            # in the member variable `_modules` of the `Module` class, and its
            #
            # type is OrderedDict
            #
            self._modules[str(idx)] = module

    def forward(self, X):
        # OrderedDict guarantees that members will be traversed in the order
        #
        # they were added
        #
        for block in self._modules.values():
            X = block(X)
        return X
```

```{.python .input}
#@tab tensorflow
class MySequential(tf.keras.Model):
    def __init__(self, *args):
        super().__init__()
        self.modules = []
        for block in args:
            # Here, `block` is an instance of a `tf.keras.layers.Layer`
            #
            # subclass
            #
            self.modules.append(block)

    def call(self, X):
        for module in self.modules:
            X = module(X)
        return X
```

:begin_tab:`mxnet`
A função `add` adiciona um único bloco
para o dicionário ordenado `_children`.
Você deve estar se perguntando por que todo bloco de Gluon
possui um atributo `_children`
e por que o usamos em vez de apenas
definir uma lista Python nós mesmos.
Resumindo, a principal vantagem das `_children`
é que durante a inicialização do parâmetro do nosso bloco,
Gluon sabe olhar dentro do dicionário `_children` para encontrar sub-blocos cujo
os parâmetros também precisam ser inicializados.
:end_tab:

:begin_tab:`pytorch`
No método `__init__`, adicionamos todos os módulos
para o dicionário ordenado `_modules` um por um.
Você pode se perguntar por que todo `Module`
possui um atributo `_modules`
e por que o usamos em vez de apenas
definir uma lista Python nós mesmos.
Em suma, a principal vantagem de `_modules`
é que durante a inicialização do parâmetro do nosso módulo,
o sistema sabe olhar dentro do `_modules`
dicionário para encontrar submódulos cujo
os parâmetros também precisam ser inicializados.
:end_tab:

Quando a função de propagação direta de nosso `MySequential` é invocada,
cada bloco adicionado é executado
na ordem em que foram adicionados.
Agora podemos reimplementar um MLP
usando nossa classe `MySequential`.

```{.python .input}
net = MySequential()
net.add(nn.Dense(256, activation='relu'))
net.add(nn.Dense(10))
net.initialize()
net(X)
```

```{.python .input}
#@tab pytorch
net = MySequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
net(X)
```

```{.python .input}
#@tab tensorflow
net = MySequential(
    tf.keras.layers.Dense(units=256, activation=tf.nn.relu),
    tf.keras.layers.Dense(10))
net(X)
```

Observe que este uso de `MySequential`
é idêntico ao código que escrevemos anteriormente
para a classe `Sequential`
(conforme descrito em :numref:`sec_mlp_concise`).

## Execução de Código na Função de Propagação Direta

A classe `Sequential` facilita a construção do modelo,
nos permitindo montar novas arquiteturas
sem ter que definir nossa própria classe.
No entanto, nem todas as arquiteturas são cadeias simples.
Quando uma maior flexibilidade é necessária,
vamos querer definir nossos próprios blocos.
Por exemplo, podemos querer executar o
controle de fluxo do Python dentro da função de propagação direta.
Além disso, podemos querer realizar
operações matemáticas arbitrárias,
não simplesmente depender de camadas de rede neural predefinidas.

Você deve ter notado que até agora,
todas as operações em nossas redes
agiram de acordo com as ativações de nossa rede
e seus parâmetros.
Às vezes, no entanto, podemos querer
incorporar termos
que não são resultado de camadas anteriores
nem parâmetros atualizáveis.
Chamamos isso de *parâmetros constantes*.
Digamos, por exemplo, que queremos uma camada
que calcula a função
$f(\mathbf{x},\mathbf{w}) = c \cdot \mathbf{w}^\top \mathbf{x}$,
onde $\mathbf{x}$ é a entrada, $\mathbf{w}$ é nosso parâmetro,
e $c$ é alguma constante especificada
que não é atualizado durante a otimização.
Portanto, implementamos uma classe `FixedHiddenMLP` como a seguir.

```{.python .input}
class FixedHiddenMLP(nn.Block):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Parâmetros de peso aleatórios criados com a função `get_constant`
        # não são atualizados durante o treinamento (ou seja, parâmetros constantes)
        self.rand_weight = self.params.get_constant(
            'rand_weight', np.random.uniform(size=(20, 20)))
        self.dense = nn.Dense(20, activation='relu')

    def forward(self, X):
        X = self.dense(X)
        # Use os parâmetros constantes criados, bem como as funções `relu` e` dot`
        X = npx.relu(np.dot(X, self.rand_weight.data()) + 1)
        # Reutilize a camada totalmente conectada. Isso é equivalente a compartilhar
        # parâmetros com duas camadas totalmente conectadas
        X = self.dense(X)
        # Control flow
        while np.abs(X).sum() > 1:
            X /= 2
        return X.sum()
```

```{.python .input}
#@tab pytorch
class FixedHiddenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # Parâmetros de peso aleatórios que não computarão gradientes e
        # portanto, mantem-se constante durante o treinamento
        self.rand_weight = torch.rand((20, 20), requires_grad=False)
        self.linear = nn.Linear(20, 20)

    def forward(self, X):
        X = self.linear(X)
        # Use os parâmetros constantes criados, bem como as funções `relu` e` mm`
        X = F.relu(torch.mm(X, self.rand_weight) + 1)
        # Reutilize a camada totalmente conectada. Isso é equivalente a compartilhar
        # parâmetros com duas camadas totalmente conectadas
        X = self.linear(X)
        # Controle de fluxo
        while X.abs().sum() > 1:
            X /= 2
        return X.sum()
```

```{.python .input}
#@tab tensorflow
class FixedHiddenMLP(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.flatten = tf.keras.layers.Flatten()
        # Parâmetros de peso aleatório criados com `tf.constant` não são atualizados
        # durante o treinamento (ou seja, parâmetros constantes)
        self.rand_weight = tf.constant(tf.random.uniform((20, 20)))
        self.dense = tf.keras.layers.Dense(20, activation=tf.nn.relu)

    def call(self, inputs):
        X = self.flatten(inputs)
        # Use os parâmetros constantes criados, bem como as funções `relu` e `matmul`
        X = tf.nn.relu(tf.matmul(X, self.rand_weight) + 1)
        # Reutilize a camada totalmente conectada. Isso é equivalente a compartilhar
        # parâmetros com duas camadas totalmente conectadas
        X = self.dense(X)
        # Control flow
        #
        while tf.reduce_sum(tf.math.abs(X)) > 1:
            X /= 2
        return tf.reduce_sum(X)
```

Neste modelo `FixedHiddenMLP`,
implementamos uma camada oculta cujos pesos
(`self.rand_weight`) são inicializados aleatoriamente
na instanciação e daí em diante constantes.
Este peso não é um parâmetro do modelo
e, portanto, nunca é atualizado por *backpropagation*.
A rede então passa a saída desta camada "fixa"
através de uma camada totalmente conectada.

Observe que antes de retornar a saída,
nosso modelo fez algo incomum.
Executamos um *loop while*, testando
na condição de que sua norma $L_1$ seja maior que $1$,
e dividindo nosso vetor de produção por $2$
até que satisfizesse a condição.
Finalmente, retornamos a soma das entradas em `X`.
Até onde sabemos, nenhuma rede neural padrão
executa esta operação.
Observe que esta operação em particular pode não ser útil
em qualquer tarefa do mundo real.
Nosso objetivo é apenas mostrar como integrar
código arbitrário no fluxo de seu
cálculos de rede neural.

```{.python .input}
net = FixedHiddenMLP()
net.initialize()
net(X)
```

```{.python .input}
#@tab pytorch, tensorflow
net = FixedHiddenMLP()
net(X)
```

Podemos misturar e combinar vários
maneiras de montar blocos juntos.
No exemplo a seguir, aninhamos blocos
de algumas maneiras criativas.

```{.python .input}
class NestMLP(nn.Block):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.net = nn.Sequential()
        self.net.add(nn.Dense(64, activation='relu'),
                     nn.Dense(32, activation='relu'))
        self.dense = nn.Dense(16, activation='relu')

    def forward(self, X):
        return self.dense(self.net(X))

chimera = nn.Sequential()
chimera.add(NestMLP(), nn.Dense(20), FixedHiddenMLP())
chimera.initialize()
chimera(X)
```

```{.python .input}
#@tab pytorch
class NestMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(20, 64), nn.ReLU(),
                                 nn.Linear(64, 32), nn.ReLU())
        self.linear = nn.Linear(32, 16)

    def forward(self, X):
        return self.linear(self.net(X))

chimera = nn.Sequential(NestMLP(), nn.Linear(16, 20), FixedHiddenMLP())
chimera(X)
```

```{.python .input}
#@tab tensorflow
class NestMLP(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.net = tf.keras.Sequential()
        self.net.add(tf.keras.layers.Dense(64, activation=tf.nn.relu))
        self.net.add(tf.keras.layers.Dense(32, activation=tf.nn.relu))
        self.dense = tf.keras.layers.Dense(16, activation=tf.nn.relu)

    def call(self, inputs):
        return self.dense(self.net(inputs))

chimera = tf.keras.Sequential()
chimera.add(NestMLP())
chimera.add(tf.keras.layers.Dense(20))
chimera.add(FixedHiddenMLP())
chimera(X)
```

## Eficiência

:begin_tab:`mxnet`
O leitor ávido pode começar a se preocupar
sobre a eficiência de algumas dessas operações.
Afinal, temos muitas pesquisas de dicionário,
execução de código e muitas outras coisas Pythônicas
ocorrendo no que deveria ser
uma biblioteca de *Deep Learning* de alto desempenho.
Os problemas do [Bloqueio do Interprete Global](https://wiki.python.org/moin/GlobalInterpreterLock) do Python são bem conhecidos.
No contexto de *Deep Learning*,
podemos nos preocupar que nossas GPU(s) extremamente rápidas
pode ter que esperar até uma CPU insignificante
executa o código Python antes de obter outro trabalho para ser executado.
A melhor maneira de acelerar o Python é evitá-lo completamente.

Uma maneira de o Gluon fazer isso é permitindo
*hibridização*, que será descrita mais tarde.
Aqui, o interpretador Python executa um bloco
na primeira vez que é invocado.
O tempo de execução do Gluon registra o que está acontecendo
e, da próxima vez, provoca um curto-circuito nas chamadas para Python.
Isso pode acelerar as coisas consideravelmente em alguns casos
mas é preciso ter cuidado ao controlar o fluxo (como acima)
pois conduz a diferentes ramos em diferentes passagens através da rede.
Recomendamos que o leitor interessado verifique
a seção de hibridização (:numref:`sec_hybridize`)
para aprender sobre a compilação depois de terminar o capítulo atual.
:end_tab:

:begin_tab:`pytorch`
O leitor ávido pode começar a se preocupar
sobre a eficiência de algumas dessas operações.
Afinal, temos muitas pesquisas de dicionário,
execução de código e muitas outras coisas Pythônicas
ocorrendo no que deveria ser
uma biblioteca de *Deep Learning* de alto desempenho.
Os problemas do [bloqueio do interpretador global](https://wiki.python.org/moin/GlobalInterpreterLock) do Python  são bem conhecidos.
No contexto de *Deep Learning*,
podemos nos preocupar que nossas GPU(s) extremamente rápidas
pode ter que esperar até uma CPU insignificante
executa o código Python antes de obter outro trabalho para ser executado.
:end_tab:

:begin_tab:`tensorflow`
O leitor ávido pode começar a se preocupar
sobre a eficiência de algumas dessas operações.
Afinal, temos muitas pesquisas de dicionário,
execução de código e muitas outras coisas Pythônicas
ocorrendo no que deveria ser
uma biblioteca de aprendizado profundo de alto desempenho.
Os problemas do  [bloqueio do interpretador global](https://wiki.python.org/moin/GlobalInterpreterLock) do Python são bem conhecidos.
No contexto de *Deep Learning*,
podemos nos preocupar que nossas GPU(s) extremamente rápidas
pode ter que esperar até uma CPU insignificante
executa o código Python antes de obter outro trabalho para ser executado.
A melhor maneira de acelerar o Python é evitá-lo completamente.
:end_tab:

## Sumário

* Camadas são blocos.
* Muitas camadas podem incluir um bloco.
* Muitos blocos podem incluir um bloco.
* Um bloco pode conter código.
* Os blocos cuidam de muitas tarefas domésticas, incluindo inicialização de parâmetros e *backpropagation*.
* As concatenações sequenciais de camadas e blocos são tratadas pelo bloco `Sequencial`.


## Exercícios

1. Que tipos de problemas ocorrerão se você alterar `MySequential` para armazenar blocos em uma lista Python?
1. Implemente um bloco que tenha dois blocos como argumento, digamos `net1` e `net2` e retorne a saída concatenada de ambas as redes na propagação direta. Isso também é chamado de bloco paralelo.
1. Suponha que você deseja concatenar várias instâncias da mesma rede. Implemente uma função de fábrica que gere várias instâncias do mesmo bloco e construa uma rede maior a partir dele.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/54)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/55)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/264)
:end_tab:
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTE3MjkyMDczNzIsMTQwNjIzOTk0NiwtOD
U2NjE0MzM3LDM1NTY2MzgxMSw4Mzc5MjY4NDcsMTc3MDI4MDk3
NCwtMTYzNzgyOTQ5MF19
-->