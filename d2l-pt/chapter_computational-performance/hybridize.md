# Compiladores e Interpretadores
:label:`sec_hybridize`

Até agora, este livro se concentrou na programação imperativa, que faz uso de instruções como `print`,` + `ou` if` para alterar o estado de um programa. Considere o seguinte exemplo de um programa imperativo simples.

```{.python .input}
#@tab all
def add(a, b):
    return a + b

def fancy_func(a, b, c, d):
    e = add(a, b)
    f = add(c, d)
    g = add(e, f)
    return g

print(fancy_func(1, 2, 3, 4))
```

Python é uma linguagem interpretada. Ao avaliar `fancy_func` ele realiza as operações que compõem o corpo da função *em sequência*. Ou seja, ele avaliará `e = add (a, b)` e armazenará os resultados como a variável `e`, alterando assim o estado do programa. As próximas duas instruções `f = add (c, d)` e `g = add (e, f)` serão executadas de forma semelhante, realizando adições e armazenando os resultados como variáveis.  :numref:`fig_compute_graph` ilustra o fluxo de dados.

![Fluxo de dados em um programa imperativo.](../img/computegraph.svg)
:label:`fig_compute_graph`

Embora a programação imperativa seja conveniente, pode ser ineficiente. Por um lado, mesmo se a função `add` for repetidamente chamada em` fancy_func`, Python executará as três chamadas de função individualmente. Se elas forem executadas, digamos, em uma GPU (ou mesmo em várias GPUs), a sobrecarga decorrente do interpretador Python pode se tornar excessiva. Além disso, ele precisará salvar os valores das variáveis `e` e` f` até que todas as instruções em `fancy_func` tenham sido executadas. Isso ocorre porque não sabemos se as variáveis `e` e` f` serão usadas por outras partes do programa após as instruções `e = add (a, b)` e `f = add (c, d)` serem executadas.

## Programação Simbólica


Considere a alternativa de programação simbólica, em que a computação geralmente é realizada apenas depois que o processo foi totalmente definido. Essa estratégia é usada por vários frameworks de aprendizado profundo, incluindo Theano, Keras e TensorFlow (os dois últimos adquiriram extensões imperativas). Geralmente envolve as seguintes etapas:

1. Definir as operações a serem executadas.
1. Compilar as operações em um programa executável.
1. Fornecer as entradas necessárias e chamar o programa compilado para execução.

Isso permite uma quantidade significativa de otimização. Em primeiro lugar, podemos pular o interpretador Python em muitos casos, removendo assim um gargalo de desempenho que pode se tornar significativo em várias GPUs rápidas emparelhadas com um único thread Python em uma CPU. Em segundo lugar, um compilador pode otimizar e reescrever o código acima em `print ((1 + 2) + (3 + 4))` ou mesmo `print (10)`. Isso é possível porque um compilador consegue ver o código completo antes de transformá-lo em instruções de máquina. Por exemplo, ele pode liberar memória (ou nunca alocá-la) sempre que uma variável não for mais necessária. Ou pode transformar o código inteiramente em uma parte equivalente. Para ter uma ideia melhor, considere a seguinte simulação de programação imperativa (afinal, é Python) abaixo.

```{.python .input}
#@tab all
def add_():
    return '''
def add(a, b):
    return a + b
'''

def fancy_func_():
    return '''
def fancy_func(a, b, c, d):
    e = add(a, b)
    f = add(c, d)
    g = add(e, f)
    return g
'''

def evoke_():
    return add_() + fancy_func_() + 'print(fancy_func(1, 2, 3, 4))'

prog = evoke_()
print(prog)
y = compile(prog, '', 'exec')
exec(y)
```

As diferenças entre a programação imperativa (interpretada) e a programação simbólica são as seguintes:

* A programação imperativa é mais fácil. Quando a programação imperativa é usada em Python, a maior parte do código é direta e fácil de escrever. Também é mais fácil depurar o código de programação imperativo. Isso ocorre porque é mais fácil obter e imprimir todos os valores de variáveis intermediárias relevantes ou usar as ferramentas de depuração integradas do Python.
* A programação simbólica é mais eficiente e fácil de portar. Isso torna mais fácil otimizar o código durante a compilação, além de ter a capacidade de portar o programa para um formato independente do Python. Isso permite que o programa seja executado em um ambiente não-Python, evitando, assim, quaisquer problemas de desempenho em potencial relacionados ao interpretador Python.

## Programação Híbrida


Historicamente, a maioria das estruturas de aprendizagem profunda escolhe entre uma abordagem imperativa ou simbólica. Por exemplo, Theano, TensorFlow (inspirado no último), Keras e CNTK formulam modelos simbolicamente. Por outro lado, Chainer e PyTorch adotam uma abordagem imperativa. Um modo imperativo foi adicionado ao TensorFlow 2.0 (via Eager) e Keras em revisões posteriores.

:begin_tab:`mxnet`
Ao projetar o Gluon, os desenvolvedores consideraram se seria possível combinar os benefícios de ambos os modelos de programação. Isso levou a um modelo híbrido que permite aos usuários desenvolver e depurar usando programação imperativa pura, ao mesmo tempo em que têm a capacidade de converter a maioria dos programas em programas simbólicos a serem executados quando o desempenho e a implantação de computação em nível de produto são necessários.


Na prática, isso significa que construímos modelos usando as classes `HybridBlock` ou` HybridSequential` e `HybridConcurrent`. Por padrão, eles são executados da mesma forma que as classes `Block` ou` Sequential` e `Concurrent` são executadas na programação imperativa. `HybridSequential` é uma subclasse de` HybridBlock` (assim como `Sequential` é subclasse de ` Block`). Quando a função `hybridize`  é chamada, o Gluon compila o modelo na forma usada na programação simbólica. Isso permite otimizar os componentes de computação intensiva sem sacrificar a maneira como um modelo é implementado. Ilustraremos os benefícios abaixo, focalizando apenas modelos sequenciais e blocos (a composição concorrente funciona de forma análoga).
:end_tab:

:begin_tab:`pytorch`
Como mencionado acima, PyTorch é baseado em programação imperativa e usa gráficos de computação dinâmica. Em um esforço para alavancar a portabilidade e eficiência da programação simbólica, os desenvolvedores consideraram se seria possível combinar os benefícios de ambos os modelos de programação. Isso levou a um *torchscript* que permite aos usuários desenvolver e depurar usando programação imperativa pura, ao mesmo tempo em que têm a capacidade de converter a maioria dos programas em programas simbólicos para serem executados quando o desempenho e a implantação de computação em nível de produto forem necessários.
:end_tab:

:begin_tab:`tensorflow`
O paradigma de programação imperativo agora é o padrão no Tensorflow 2, uma mudança acolhedora para aqueles que são novos na linguagem. No entanto, as mesmas técnicas de programação simbólica e gráficos computacionais subsequentes ainda existem no TensorFlow e podem ser acessados pelo decorador `tf.function` fácil de usar. Isso trouxe o paradigma de programação imperativo para o TensorFlow, permitindo que os usuários definissem funções mais intuitivas, depois as envolvessem e compilassem em gráficos computacionais automaticamente usando um recurso que a equipe do TensorFlow chama de [autograph](https://www.tensorflow.org/api_docs/python/tf/autograph).
:end_tab:

## Híbrido-Sequencial

A maneira mais fácil de ter uma ideia de como a hibridização funciona é considerar redes profundas com várias camadas. Convencionalmente, o interpretador Python precisará executar o código para todas as camadas para gerar uma instrução que pode então ser encaminhada para uma CPU ou GPU. Para um único dispositivo de computação (rápido), isso não causa grandes problemas. Por outro lado, se usarmos um servidor avançado de 8 GPUs, como uma instância AWS P3dn.24xlarge, o Python terá dificuldade para manter todas as GPUs ocupadas. O interpretador Python de thread único torna-se o gargalo aqui. Vamos ver como podemos resolver isso para partes significativas do código, substituindo `Sequential` por `HybridSequential`. Começamos definindo um MLP simples.

```{.python .input}
from d2l import mxnet as d2l
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()

# Factory for networks
def get_net():
    net = nn.HybridSequential()  
    net.add(nn.Dense(256, activation='relu'),
            nn.Dense(128, activation='relu'),
            nn.Dense(2))
    net.initialize()
    return net

x = np.random.normal(size=(1, 512))
net = get_net()
net(x)
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn

# Factory for networks
def get_net():
    net = nn.Sequential(nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2))
    return net

x = torch.randn(size=(1, 512))
net = get_net()
net(x)
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
from tensorflow.keras.layers import Dense

# Factory for networks
def get_net():
    net = tf.keras.Sequential()
    net.add(Dense(256, input_shape = (512,), activation = "relu"))
    net.add(Dense(128, activation = "relu"))
    net.add(Dense(2, activation = "linear"))
    return net

x = tf.random.normal([1,512])
net = get_net()
net(x)
```


:begin_tab:`mxnet`
Ao chamar a função `hybridize`, podemos compilar e otimizar o cálculo no MLP. O resultado do cálculo do modelo permanece inalterado.
:end_tab:

:begin_tab: `pytorch`
Ao converter o modelo usando a função `torch.jit.script`, podemos compilar e otimizar a computação no MLP. O resultado do cálculo do modelo permanece inalterado.
:end_tab:

:begin_tab:`tensorflow`
Anteriormente, todas as funções construídas no tensorflow eram construídas como um gráfico computacional e, portanto, JIT compilado por padrão. No entanto, com o lançamento do tensorflow 2.X e tensores *eager*, este não é mais o comportamento padrão.
Podemos reativar essa funcionalidade com tf.function. tf.function é mais comumente usado como um decorador de função, no entanto, é possível chamá-lo diretamente como uma função Python normal, mostrada abaixo. O resultado do cálculo do modelo permanece inalterado.
:end_tab:

```{.python .input}
net.hybridize()
net(x)
```

```{.python .input}
#@tab pytorch
net = torch.jit.script(net)
net(x)
```

```{.python .input}
#@tab tensorflow
net = tf.function(net)
net(x)
```

:begin_tab:`mxnet`
Isso parece bom demais para ser verdade: simplesmente designe um bloco como `HybridSequential`, escreva o mesmo código de antes e invoque `hybridize`. Assim que isso acontecer, a rede estará otimizada (faremos um benchmark do desempenho abaixo). Infelizmente, isso não funciona magicamente para todas as camadas. Dito isso, os blocos fornecidos pelo Gluon são, por padrão, subclasses de `HybridBlock` e, portanto, hibridizáveis. Uma camada não será otimizada se, em vez disso, herdar do `Bloco`.
:end_tab:

:begin_tab:`pytorch`
Convertendo o modelo usando `torch.jit.script` Isso parece quase bom demais para ser verdade: escreva o mesmo código de antes e simplesmente converta o modelo usando` torch.jit.script`. Assim que isso acontecer, a rede estará otimizada (faremos um benchmark do desempenho abaixo).
:end_tab:

:begin_tab:`tensorflow`
Converter o modelo usando `tf.function` nos dá um poder incrível no TensorFlow: escreva o mesmo código de antes e simplesmente converta o modelo usando` tf.function`. Quando isso acontece, a rede é construída como um gráfico computacional na representação intermediária MLIR do TensorFlow e é altamente otimizada no nível do compilador para uma execução rápida (faremos o benchmark do desempenho abaixo).
Adicionar explicitamente a sinalização `jit_compile = True` à chamada `tf.function()` ativa a funcionalidade XLA (Álgebra Linear Acelerada) no TensorFlow. O XLA pode otimizar ainda mais o código compilado JIT em certas instâncias. A execução no modo gráfico é habilitada sem essa definição explícita, no entanto, o XLA pode tornar certas operações de álgebra linear grandes (na veia daquelas que vemos em aplicativos de aprendizado profundo) muito mais rápidas, particularmente em uma GPUenvironment.
:end_tab:

### Aceleração por Hibridização

Para demonstrar a melhoria de desempenho obtida pela compilação, comparamos o tempo necessário para avaliar `net (x)` antes e depois da hibridização. Vamos definir uma função para medir esse tempo primeiro. Será útil ao longo do capítulo à medida que nos propomos a medir (e melhorar) o desempenho.

```{.python .input}
#@tab all
#@save
class Benchmark:
    def __init__(self, description='Done'):
        self.description = description

    def __enter__(self):
        self.timer = d2l.Timer()
        return self

    def __exit__(self, *args):
        print(f'{self.description}: {self.timer.stop():.4f} sec')
```

:begin_tab:`mxnet`
Agora podemos chamar a rede duas vezes, uma com e outra sem hibridização.
:end_tab:

:begin_tab:`pytorch`
Agora podemos chamar a rede duas vezes, uma com e outra sem torchscript.
:end_tab:

:begin_tab:`tensorflow`
Agora podemos invocar a rede três vezes, uma vez executada avidamente, uma vez com execução em modo gráfico e novamente usando XLA compilado por JIT.
:end_tab:

```{.python .input}
net = get_net()
with Benchmark('Without hybridization'):
    for i in range(1000): net(x)
    npx.waitall()

net.hybridize()
with Benchmark('With hybridization'):
    for i in range(1000): net(x)
    npx.waitall()
```

```{.python .input}
#@tab pytorch
net = get_net()
with Benchmark('Without torchscript'):
    for i in range(1000): net(x)

net = torch.jit.script(net)
with Benchmark('With torchscript'):
    for i in range(1000): net(x)
```

```{.python .input}
#@tab tensorflow
net = get_net()
with Benchmark('Eager Mode'):
    for i in range(1000): net(x)

net = tf.function(net)
with Benchmark('Graph Mode'):
    for i in range(1000): net(x)
```


:begin_tab:`mxnet`
Como é observado nos resultados acima, depois que uma instância HybridSequential chama a função `hybridize`, o desempenho da computação é melhorado por meio do uso de programação simbólica.
:end_tab:

:begin_tab:`pytorch`
Conforme observado nos resultados acima, depois que uma instância nn.Sequential é criada com o script da função `torch.jit.script`, o desempenho da computação é aprimorado com o uso de programação simbólica.
:end_tab:

:begin_tab:`tensorflow`
Como é observado nos resultados acima, depois que uma instância tf.keras Sequential é scriptada usando a função `tf.function`, o desempenho da computação é melhorado por meio do uso de programação simbólica por meio da execução em modo gráfico em tensorflow.
:end_tab:

### Serialização

:begin_tab:`mxnet`
Um dos benefícios de compilar os modelos é que podemos serializar (salvar) o modelo e seus parâmetros no disco. Isso nos permite armazenar um modelo de maneira independente da linguagem de front-end de escolha. Isso nos permite implantar modelos treinados em outros dispositivos e usar facilmente outras linguagens de programação front-end. Ao mesmo tempo, o código geralmente é mais rápido do que o que pode ser alcançado na programação imperativa. Vamos ver o método `export` em ação.
:end_tab:

:begin_tab:`pytorch`
Um dos benefícios de compilar os modelos é que podemos serializar (salvar) o modelo e seus parâmetros no disco. Isso nos permite armazenar um modelo de maneira independente da linguagem de front-end de escolha. Isso nos permite implantar modelos treinados em outros dispositivos e usar facilmente outras linguagens de programação front-end. Ao mesmo tempo, o código geralmente é mais rápido do que o que pode ser alcançado na programação imperativa. Vamos ver o método `save` em ação.
:end_tab:

:begin_tab:`tensorflow`
Um dos benefícios de compilar os modelos é que podemos serializar (salvar) o modelo e seus parâmetros no disco. Isso nos permite armazenar um modelo de maneira independente da linguagem de front-end de escolha. Isso nos permite implantar modelos treinados em outros dispositivos e usar facilmente outras linguagens de programação front-end ou executar um modelo treinado em um servidor. Ao mesmo tempo, o código geralmente é mais rápido do que o que pode ser alcançado na programação imperativa.
A API de baixo nível que nos permite salvar em tensorflow é `tf.saved_model`.
Vamos ver a instância `saved_model` em ação.
:end_tab:

```{.python .input}
net.export('my_mlp')
!ls -lh my_mlp*
```

```{.python .input}
#@tab pytorch
net.save('my_mlp')
!ls -lh my_mlp*
```
```{.python .input}
#@tab tensorflow
net = get_net()
tf.saved_model.save(net, 'my_mlp')
!ls -lh my_mlp*
```

:begin_tab:`mxnet`
O modelo é decomposto em um arquivo de parâmetro (binário grande) e uma descrição JSON do programa necessário para executar o cálculo do modelo. Os arquivos podem ser lidos por outras linguagens de front-end suportadas por Python ou MXNet, como C ++, R, Scala e Perl. Vamos dar uma olhada na descrição do modelo.
:end_tab:

```{.python .input}
!head my_mlp-symbol.json
```

:begin_tab:`mxnet`
As coisas são um pouco mais complicadas quando se trata de modelos que se assemelham mais ao código. Basicamente, a hibridização precisa lidar com o fluxo de controle e a sobrecarga do Python de uma maneira muito mais imediata. Além disso,

Ao contrário da instância Block, que precisa usar a função `forward`, para uma instância HybridBlock precisamos usar a função` hybrid_forward`.

Anteriormente, demonstramos que, após chamar a função `hybridize` , o modelo é capaz de atingir desempenho de computação superior e portabilidade. Observe, porém, que a hibridização pode afetar a flexibilidade do modelo, em particular em termos de fluxo de controle. Ilustraremos como projetar modelos mais gerais e também como a compilação removerá elementos Python espúrios.
:end_tab:

```{.python .input}
class HybridNet(nn.HybridBlock):
    def __init__(self, **kwargs):
        super(HybridNet, self).__init__(**kwargs)
        self.hidden = nn.Dense(4)
        self.output = nn.Dense(2)

    def hybrid_forward(self, F, x):
        print('module F: ', F)
        print('value  x: ', x)
        x = F.npx.relu(self.hidden(x))
        print('result  : ', x)
        return self.output(x)
```

:begin_tab:`mxnet`
O código acima implementa uma rede simples com 4 unidades ocultas e 2 saídas. `hybrid_forward` recebe um argumento adicional - o módulo` F`. Isso é necessário porque, dependendo se o código foi hibridizado ou não, ele usará uma biblioteca ligeiramente diferente (`ndarray` ou` símbolo`) para processamento. Ambas as classes executam funções muito semelhantes e o MXNet determina automaticamente o argumento. Para entender o que está acontecendo, imprimimos os argumentos como parte da invocação da função.
:end_tab:

```{.python .input}
net = HybridNet()
net.initialize()
x = np.random.normal(size=(1, 3))
net(x)
```

:begin_tab:`mxnet`
A repetição do cálculo progressivo levará à mesma saída (omitimos os detalhes). Agora vamos ver o que acontece se invocarmos o método `hybridize`.
:end_tab:

```{.python .input}
net.hybridize()
net(x)
```

:begin_tab:`mxnet`
Em vez de usar `ndarray` agora usamos o módulo` symbol` para `F`. Além disso, embora a entrada seja do tipo `ndarray`, os dados que fluem pela rede agora são convertidos para o tipo `symbol` como parte do processo de compilação. Repetir a chamada de função leva a um resultado surpreendente:
:end_tab:

```{.python .input}
net(x)
```

:begin_tab:`mxnet` Isso é bem diferente do que vimos anteriormente. Todas as instruções de impressão, conforme definido em `hybrid_forward` são omitidas. De fato, após a hibridização, a execução de `net (x)` não envolve mais o interpretador Python. Isso significa que qualquer código Python espúrio é omitido (como instruções de impressão) em favor de uma execução muito mais simplificada e melhor desempenho. Em vez disso, o MXNet chama diretamente o back-end C ++. Observe também que algumas funções não são suportadas no módulo `symbol` (como`asnumpy`) e operações no local como `a += b` and `a[:] = a + b` devem ser reescritas como `a = a + b`. No entanto, a compilação de modelos vale o esforço sempre que a velocidade é importante. O benefício pode variar de pequenos pontos percentuais a mais de duas vezes a velocidade, dependendo da complexidade do modelo, da velocidade da CPU e da velocidade e número de GPUs.

## Resumo

* A programação imperativa torna mais fácil projetar novos modelos, pois é possível escrever código com fluxo de controle e a capacidade de usar uma grande parte do ecossistema de *software* Python.
* A programação simbólica requer que especifiquemos o programa e o compilemos antes de executá-lo. O benefício é um desempenho aprimorado.
* MXNet é capaz de combinar as vantagens de ambas as abordagens conforme necessário.
* Modelos construídos pelas classes `HybridSequential` e` HybridBlock` são capazes de converter programas imperativos em programas simbólicos chamando o método `hibridizar`.


## Exercícios

1. Projete uma rede usando a classe `HybridConcurrent`. Como alternativa, olhe em :ref:`sec_googlenet` para uma rede para compor.
1. Adicione `x.asnumpy ()` à primeira linha da função `hybrid_forward` da classe HybridNet nesta seção. Execute o código e observe os erros que encontrar. Por que eles acontecem?
1. O que acontece se adicionarmos o fluxo de controle, ou seja, as instruções Python `if` e` for` na função `hybrid_forward`?
1. Revise os modelos de seu interesse nos capítulos anteriores e use a classe HybridBlock ou HybridSequential para implementá-los.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/360)
:end_tab:
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTU1OTE4NjYzOSwxNTIwNzY2OTg2LDE4Mj
M1MjAyODIsLTEyMzk0Mzc4Nyw2NDU2NDY1NjYsMTgzOTc0NDkz
OCwxMDAxMTk5NDYsMTE4MTM2NjgyOV19
-->