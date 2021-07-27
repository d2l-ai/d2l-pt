# GPUs
:label:`sec_use_gpu`

Em :numref:`tab_intro_decade`, discutimos o rápido crescimento
de computação nas últimas duas décadas.
Em suma, o desempenho da GPU aumentou
por um fator de 1000 a cada década desde 2000.
Isso oferece ótimas oportunidades, mas também sugere
uma necessidade significativa de fornecer tal desempenho.

Nesta seção, começamos a discutir como aproveitar
este desempenho computacional para sua pesquisa.
Primeiro usando GPUs únicas e, posteriormente,
como usar várias GPUs e vários servidores (com várias GPUs).

Especificamente, discutiremos como
para usar uma única GPU NVIDIA para cálculos.
Primeiro, certifique-se de ter pelo menos uma GPU NVIDIA instalada.
Em seguida, baixe o [NVIDIA driver e CUDA](https://developer.nvidia.com/cuda-downloads).
e siga as instruções para definir o caminho apropriado.
Assim que esses preparativos forem concluídos,
o comando `nvidia-smi` pode ser usado
para ver as informações da placa gráfica.

```{.python .input}
#@tab all
!nvidia-smi
```

:begin_tab:`mxnet`
Você deve ter notado que um tensor MXNet
parece quase idêntico a um NumPy `ndarray`.
Mas existem algumas diferenças cruciais.
Um dos principais recursos que distinguem o MXNet
da NumPy é o seu suporte para diversos dispositivos de hardware.

No MXNet, cada array possui um contexto.
Até agora, por padrão, todas as variáveis
e computação associada
foram atribuídos à CPU.
Normalmente, outros contextos podem ser várias GPUs.
As coisas podem ficar ainda mais complicadas quando
nós implantamos trabalhos em vários servidores.
Ao atribuir matrizes a contextos de forma inteligente,
podemos minimizar o tempo gasto
transferência de dados entre dispositivos.
Por exemplo, ao treinar redes neurais em um servidor com uma GPU,
normalmente preferimos que os parâmetros do modelo residam na GPU.

Em seguida, precisamos confirmar que
a versão GPU do MXNet está instalada.
Se uma versão de CPU do MXNet já estiver instalada,
precisamos desinstalá-lo primeiro.
Por exemplo, use o comando `pip uninstall mxnet`,
em seguida, instale a versão MXNet correspondente
de acordo com sua versão CUDA.
Supondo que você tenha o CUDA 10.0 instalado,
você pode instalar a versão MXNet
que suporta CUDA 10.0 via `pip install mxnet-cu100`.
:end_tab:

:begin_tab:`pytorch`
No PyTorch, cada array possui um dispositivo, frequentemente o referimos como um contexto.
Até agora, por padrão, todas as variáveis
e computação associada
foram atribuídos à CPU.
Normalmente, outros contextos podem ser várias GPUs.
As coisas podem ficar ainda mais complicadas quando
nós implantamos trabalhos em vários servidores.
Ao atribuir matrizes a contextos de forma inteligente,
podemos minimizar o tempo gasto
transferência de dados entre dispositivos.
Por exemplo, ao treinar redes neurais em um servidor com uma GPU,
normalmente preferimos que os parâmetros do modelo residam na GPU.

Em seguida, precisamos confirmar que
a versão GPU do PyTorch está instalada.
Se uma versão CPU do PyTorch já estiver instalada,
precisamos desinstalá-lo primeiro.
Por exemplo, use o comando `pip uninstall torch`,
em seguida, instale a versão correspondente do PyTorch
de acordo com sua versão CUDA.
Supondo que você tenha o CUDA 10.0 instalado,
você pode instalar a versão PyTorch
compatível com CUDA 10.0 via `pip install torch-cu100`.
:end_tab:

Para executar os programas desta seção,
você precisa de pelo menos duas GPUs.
Observe que isso pode ser extravagante para a maioria dos computadores desktop
mas está facilmente disponível na nuvem, por exemplo,
usando as instâncias multi-GPU do AWS EC2.
Quase todas as outras seções * não * requerem várias GPUs.
Em vez disso, isso é simplesmente para ilustrar
como os dados fluem entre diferentes dispositivos.

## Dispositivos Computacionais

Podemos especificar dispositivos, como CPUs e GPUs,
para armazenamento e cálculo.
Por padrão, os tensores são criados na memória principal
e, em seguida, use a CPU para calculá-lo.

:begin_tab:`mxnet`
No MXNet, a CPU e a GPU podem ser indicadas por `cpu ()` e `gpu()`.
Deve-se notar que `cpu()`
(ou qualquer número inteiro entre parênteses)
significa todas as CPUs físicas e memória.
Isso significa que os cálculos do MXNet
tentará usar todos os núcleos da CPU.
No entanto, `gpu()` representa apenas uma carta
e a memória correspondente.
Se houver várias GPUs, usamos `gpu(i)`
para representar a $i^\mathrm{th}$ GPU ($i$ começa em 0).
Além disso, `gpu(0)` e `gpu()` são equivalentes.
:end_tab:

:begin_tab:`pytorch`
No PyTorch, a CPU e a GPU podem ser indicadas por `torch.device('cpu')` e `torch.cuda.device('cuda')`.
Deve-se notar que o dispositivo `cpu`
significa todas as CPUs físicas e memória.
Isso significa que os cálculos de PyTorch
tentará usar todos os núcleos da CPU.
No entanto, um dispositivo `gpu` representa apenas uma placa
e a memória correspondente.
Se houver várias GPUs, usamos `torch.cuda.device(f'cuda: {i}')`
para representar a $i^\mathrm{th}$ GPU ($i$ começa em 0).
Além disso, `gpu:0` e `gpu` são equivalentes.
:end_tab:

```{.python .input}
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()

npx.cpu(), npx.gpu(), npx.gpu(1)
```

```{.python .input}
#@tab pytorch
import torch
from torch import nn

torch.device('cpu'), torch.cuda.device('cuda'), torch.cuda.device('cuda:1')
```

```{.python .input}
#@tab tensorflow
import tensorflow as tf

tf.device('/CPU:0'), tf.device('/GPU:0'), tf.device('/GPU:1')
```

Podemos consultar o número de GPUs disponíveis.

```{.python .input}
npx.num_gpus()
```

```{.python .input}
#@tab pytorch
torch.cuda.device_count()
```

```{.python .input}
#@tab tensorflow
len(tf.config.experimental.list_physical_devices('GPU'))
```
Agora definimos duas funções convenientes que nos permitem
para executar o código mesmo que as GPUs solicitadas não existam.

```{.python .input}
def try_gpu(i=0):  #@save
    """Return gpu(i) if exists, otherwise return cpu()."""
    return npx.gpu(i) if npx.num_gpus() >= i + 1 else npx.cpu()

def try_all_gpus():  #@save
    """Return all available GPUs, or [cpu()] if no GPU exists."""
    devices = [npx.gpu(i) for i in range(npx.num_gpus())]
    return devices if devices else [npx.cpu()]

try_gpu(), try_gpu(10), try_all_gpus()
```

```{.python .input}
#@tab pytorch
def try_gpu(i=0):  #@save
    """Return gpu(i) if exists, otherwise return cpu()."""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

def try_all_gpus():  #@save
    """Return all available GPUs, or [cpu(),] if no GPU exists."""
    devices = [torch.device(f'cuda:{i}')
             for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]

try_gpu(), try_gpu(10), try_all_gpus()
```

```{.python .input}
#@tab tensorflow
def try_gpu(i=0):  #@save
    """Return gpu(i) if exists, otherwise return cpu()."""
    if len(tf.config.experimental.list_physical_devices('GPU')) >= i + 1:
        return tf.device(f'/GPU:{i}')
    return tf.device('/CPU:0')

def try_all_gpus():  #@save
    """Return all available GPUs, or [cpu(),] if no GPU exists."""
    num_gpus = len(tf.config.experimental.list_physical_devices('GPU'))
    devices = [tf.device(f'/GPU:{i}') for i in range(num_gpus)]
    return devices if devices else [tf.device('/CPU:0')]

try_gpu(), try_gpu(10), try_all_gpus()
```

## Tensores e GPUs

Por padrão, tensores são criados na CPU.
Podemos consultar o dispositivo onde o tensor está localizado.

```{.python .input}
x = np.array([1, 2, 3])
x.ctx
```

```{.python .input}
#@tab pytorch
x = torch.tensor([1, 2, 3])
x.device
```

```{.python .input}
#@tab tensorflow
x = tf.constant([1, 2, 3])
x.device
```

É importante notar que sempre que quisermos
para operar em vários termos,
eles precisam estar no mesmo dispositivo.
Por exemplo, se somarmos dois tensores,
precisamos ter certeza de que ambos os argumentos
estão no mesmo dispositivo --- caso contrário, a estrutura
não saberia onde armazenar o resultado
ou mesmo como decidir onde realizar o cálculo.

### Armazenamento na GPU

Existem várias maneiras de armazenar um tensor na GPU.
Por exemplo, podemos especificar um dispositivo de armazenamento ao criar um tensor.
A seguir, criamos a variável tensorial `X` no primeiro `gpu`.
O tensor criado em uma GPU consome apenas a memória desta GPU.
Podemos usar o comando `nvidia-smi` para ver o uso de memória da GPU.
Em geral, precisamos ter certeza de não criar dados que excedam o limite de memória da GPU.

```{.python .input}
X = np.ones((2, 3), ctx=try_gpu())
X
```

```{.python .input}
#@tab pytorch
X = torch.ones(2, 3, device=try_gpu())
X
```

```{.python .input}
#@tab tensorflow
with try_gpu():
    X = tf.ones((2, 3))
X
```

Supondo que você tenha pelo menos duas GPUs, o código a seguir criará um tensor aleatório na segunda GPU.

```{.python .input}
Y = np.random.uniform(size=(2, 3), ctx=try_gpu(1))
Y
```

```{.python .input}
#@tab pytorch
Y = torch.rand(2, 3, device=try_gpu(1))
Y
```

```{.python .input}
#@tab tensorflow
with try_gpu(1):
    Y = tf.random.uniform((2, 3))
Y
```

### Copiando

Se quisermos calcular `X + Y`,
precisamos decidir onde realizar esta operação.
Por exemplo, como mostrado em :numref:`fig_copyto`,
podemos transferir `X` para a segunda GPU
e realizar a operação lá.
*Não* simplesmente adicione `X` e` Y`,
pois isso resultará em uma exceção.
O mecanismo de tempo de execução não saberia o que fazer:
ele não consegue encontrar dados no mesmo dispositivo e falha.
Já que `Y` vive na segunda GPU,
precisamos mover `X` para lá antes de podermos adicionar os dois.

![Copiar dados para realizar uma operação no mesmo dispositivo.](../img/copyto.svg)
:label:`fig_copyto`



```{.python .input}
Z = X.copyto(try_gpu(1))
print(X)
print(Z)
```

```{.python .input}
#@tab pytorch
Z = X.cuda(1)
print(X)
print(Z)
```

```{.python .input}
#@tab tensorflow
with try_gpu(1):
    Z = X
print(X)
print(Z)
```

Agora que os dados estão na mesma GPU
(ambos são `Z` e` Y`),
podemos somá-los.

```{.python .input}
#@tab all
Y + Z
```

:begin_tab:`mxnet`
Imagine que sua variável `Z` já esteja em sua segunda GPU.
O que acontece se ainda chamarmos `Z.copyto(gpu(1))`?
Ele fará uma cópia e alocará nova memória,
mesmo que essa variável já resida no dispositivo desejado.
Há momentos em que, dependendo do ambiente em que nosso código está sendo executado,
duas variáveis podem já estar no mesmo dispositivo.
Então, queremos fazer uma cópia apenas se as variáveis
atualmente vivem em dispositivos diferentes.
Nestes casos, podemos chamar `as_in_ctx`.
Se a variável já estiver viva no dispositivo especificado
então este é um ambiente autônomo.
A menos que você queira especificamente fazer uma cópia,
`as_in_ctx` é o método de escolha.
:end_tab:

:begin_tab:`pytorch`
Imagine que sua variável `Z` já esteja em sua segunda GPU.
O que acontece se ainda chamarmos `Z.cuda(1)`?
Ele retornará `Z` em vez de fazer uma cópia e alocar nova memória.
:end_tab:

:begin_tab:`tensorflow`
Imagine que sua variável `Z` já esteja em sua segunda GPU.
O que acontece se ainda chamarmos `Z2 = Z` no mesmo escopo de dispositivo?
Ele retornará `Z` em vez de fazer uma cópia e alocar nova memória.
:end_tab:

```{.python .input}
Z.as_in_ctx(try_gpu(1)) is Z
```

```{.python .input}
#@tab pytorch
Z.cuda(1) is Z
```

```{.python .input}
#@tab tensorflow
with try_gpu(1):
    Z2 = Z
Z2 is Z
```

### Informações extra

As pessoas usam GPUs para fazer aprendizado de máquina
porque eles esperam que ela seja rápida.
Mas a transferência de variáveis entre dispositivos é lenta.
Então, queremos que você tenha 100% de certeza
que você deseja fazer algo lento antes de deixá-lo fazer.
Se a estrutura de *Deep Learning* apenas fizesse a cópia automaticamente
sem bater, então você pode não perceber
que você escreveu algum código lento.

Além disso, a transferência de dados entre dispositivos (CPU, GPUs e outras máquinas)
é algo muito mais lento do que a computação.
Também torna a paralelização muito mais difícil,
já que temos que esperar que os dados sejam enviados (ou melhor, para serem recebidos)
antes de prosseguirmos com mais operações.
É por isso que as operações de cópia devem ser realizadas com muito cuidado.
Como regra geral, muitas pequenas operações
são muito piores do que uma grande operação.
Além disso, várias operações ao mesmo tempo
são muito melhores do que muitas operações simples intercaladas no código
a menos que você saiba o que está fazendo.
Este é o caso, uma vez que tais operações podem bloquear se um dispositivo
tem que esperar pelo outro antes de fazer outra coisa.
É um pouco como pedir seu café em uma fila
em vez de pré-encomendá-lo por telefone
e descobrir que ele está pronto quando você estiver.

Por último, quando imprimimos tensores ou convertemos tensores para o formato NumPy,
se os dados não estiverem na memória principal,
o framework irá copiá-lo para a memória principal primeiro,
resultando em sobrecarga de transmissão adicional.
Pior ainda, agora está sujeito ao temido bloqueio de intérprete global
isso faz tudo esperar que o Python seja concluído.


## Redes Neurais e GPUs

Da mesma forma, um modelo de rede neural pode especificar dispositivos.
O código a seguir coloca os parâmetros do modelo na GPU.

```{.python .input}
net = nn.Sequential()
net.add(nn.Dense(1))
net.initialize(ctx=try_gpu())
```

```{.python .input}
#@tab pytorch
net = nn.Sequential(nn.Linear(3, 1))
net = net.to(device=try_gpu())
```

```{.python .input}
#@tab tensorflow
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    net = tf.keras.models.Sequential([
        tf.keras.layers.Dense(1)])
```

Veremos muitos mais exemplos de
como executar modelos em GPUs nos capítulos seguintes,
simplesmente porque eles se tornarão um pouco mais intensivos em termos de computação.

Quando a entrada é um tensor na GPU, o modelo calculará o resultado na mesma GPU.

```{.python .input}
#@tab all
net(X)
```
Vamos confirmar se os parâmetros do modelo estão armazenados na mesma GPU.

```{.python .input}
net[0].weight.data().ctx
```

```{.python .input}
#@tab pytorch
net[0].weight.data.device
```

```{.python .input}
#@tab tensorflow
net.layers[0].weights[0].device, net.layers[0].weights[1].device
```

Resumindo, contanto que todos os dados e parâmetros estejam no mesmo dispositivo, podemos aprender modelos com eficiência. Nos próximos capítulos, veremos vários desses exemplos.

## Sumário

* Podemos especificar dispositivos para armazenamento e cálculo, como CPU ou GPU.
   Por padrão, os dados são criados na memória principal
   e então usa-se a CPU para cálculos.
* A estrutura de *Deep Learning* requer todos os dados de entrada para cálculo
   estar no mesmo dispositivo,
   seja CPU ou a mesma GPU.
* Você pode perder um desempenho significativo movendo dados sem cuidado.
   Um erro típico é o seguinte: calcular a perda
   para cada minibatch na GPU e relatando de volta
   para o usuário na linha de comando (ou registrando-o em um NumPy `ndarray`)
   irá disparar um bloqueio global do interpretador que paralisa todas as GPUs.
   É muito melhor alocar memória
   para registrar dentro da GPU e apenas mover registros maiores.

## Exercícios

1. Tente uma tarefa de computação maior, como a multiplicação de grandes matrizes,
    e veja a diferença de velocidade entre a CPU e a GPU.
    Que tal uma tarefa com uma pequena quantidade de cálculos?
1. Como devemos ler e escrever os parâmetros do modelo na GPU?
1. Meça o tempo que leva para calcular 1000
    multiplicações matriz-matriz de $100 \times 100$ matrizes
    e registrar a norma de Frobenius da matriz de saída, um resultado de cada vez
    vs. manter um registro na GPU e transferir apenas o resultado final.
1. Meça quanto tempo leva para realizar duas multiplicações matriz-matriz
    em duas GPUs ao mesmo tempo vs. em sequência
    em uma GPU. Dica: você deve ver uma escala quase linear.
    
:begin_tab:`mxnet`
[Discussão](https://discuss.d2l.ai/t/62)
:end_tab:

:begin_tab:`pytorch`
[Discussão](https://discuss.d2l.ai/t/63)
:end_tab:

:begin_tab:`tensorflow`
[Discussão](https://discuss.d2l.ai/t/270)
:end_tab:
<!--stackedit_data:
eyJoaXN0b3J5IjpbMTM1NzUzMTk4OCwtMTA5NzI3MTAzNiw4MD
c2MDQzNTksLTY3MjIyODU4NF19
-->