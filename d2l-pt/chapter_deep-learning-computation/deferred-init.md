# Deferred Initialization
:label:`sec_deferred_init`

Até agora, pode parecer que escapamos
por ser descuidados na configuração de nossas redes.
Especificamente, fizemos as seguintes coisas não intuitivas,
que podem não parecer que deveriam funcionar:

* Definimos as arquiteturas de rede
   sem especificar a dimensionalidade de entrada.
* Adicionamos camadas sem especificar
   a dimensão de saída da camada anterior.
* Nós até "inicializamos" esses parâmetros
   antes de fornecer informações suficientes para determinar
   quantos parâmetros nossos modelos devem conter.

Você pode se surpreender com o fato de nosso código ser executado.
Afinal, não há como o *framework* de *Deep Learning*
poderia dizer qual seria a dimensionalidade de entrada de uma rede.
O truque aqui é que o *framework* adia a inicialização,
esperando até a primeira vez que passamos os dados pelo modelo,
para inferir os tamanhos de cada camada na hora.

Mais tarde, ao trabalhar com redes neurais convolucionais,
esta técnica se tornará ainda mais conveniente
desde a dimensionalidade de entrada
(ou seja, a resolução de uma imagem)
afetará a dimensionalidade
de cada camada subsequente.
Consequentemente, a capacidade de definir parâmetros
sem a necessidade de saber,
no momento de escrever o código,
qual é a dimensionalidade
pode simplificar muito a tarefa de especificar
e subsequentemente modificando nossos modelos.
A seguir, vamos nos aprofundar na mecânica da inicialização.


## Instanciando a Rede

Para começar, vamos instanciar um MLP.

```{.python .input}
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()

def get_net():
    net = nn.Sequential()
    net.add(nn.Dense(256, activation='relu'))
    net.add(nn.Dense(10))
    return net

net = get_net()
```

```{.python .input}
#@tab tensorflow
import tensorflow as tf

net = tf.keras.models.Sequential([
    tf.keras.layers.Dense(256, activation=tf.nn.relu),
    tf.keras.layers.Dense(10),
])
```

Neste ponto, a rede não pode saber
as dimensões dos pesos da camada de entrada
porque a dimensão de entrada permanece desconhecida.
Consequentemente, a estrutura ainda não inicializou nenhum parâmetro.
Confirmamos tentando acessar os parâmetros abaixo.

```{.python .input}
print(net.collect_params)
print(net.collect_params())
```

```{.python .input}
#@tab tensorflow
[net.layers[i].get_weights() for i in range(len(net.layers))]
```

:begin_tab:`mxnet`
Observe que, embora os objetos de parâmetro existam,
a dimensão de entrada para cada camada é listada como -1.
MXNet usa o valor especial -1 para indicar
que a dimensão do parâmetro permanece desconhecida.
Neste ponto, tenta acessar `net [0].weight.data()`
desencadearia um erro de tempo de execução informando que a rede
deve ser inicializado antes que os parâmetros possam ser acessados.
Agora vamos ver o que acontece quando tentamos inicializar
parâmetros por meio da função `initialize`.
:end_tab:

:begin_tab:`tensorflow`
Observe que cada objeto de camada existe, mas os pesos estão vazios.
Usar `net.get_weights()` geraria um erro, uma vez que os pesos
ainda não foram inicializados.
:end_tab:

```{.python .input}
net.initialize()
net.collect_params()
```

:begin_tab:`mxnet`
Como podemos ver, nada mudou.
Quando as dimensões de entrada são desconhecidas,
chamadas para inicializar não inicializam corretamente os parâmetros.
Em vez disso, esta chamada se registra no MXNet que desejamos
(e opcionalmente, de acordo com qual distribuição)
para inicializar os parâmetros.
:end_tab:

Em seguida, vamos passar os dados pela rede
para fazer o *framework* finalmente inicializar os parâmetros.

```{.python .input}
X = np.random.uniform(size=(2, 20))
net(X)

net.collect_params()
```

```{.python .input}
#@tab tensorflow
X = tf.random.uniform((2, 20))
net(X)
[w.shape for w in net.get_weights()]
```

Assim que sabemos a dimensionalidade da entrada,
20,
a estrutura pode identificar a forma da matriz de peso da primeira camada conectando o valor de 20.
Tendo reconhecido a forma da primeira camada, a estrutura prossegue
para a segunda camada,
e assim por diante através do grafo computacional
até que todas as formas sejam conhecidas.
Observe que, neste caso,
apenas a primeira camada requer inicialização adiada,
mas a estrutura inicializa sequencialmente.
Uma vez que todas as formas dos parâmetros são conhecidas,
a estrutura pode finalmente inicializar os parâmetros.

## Sumário

* A inicialização adiada pode ser conveniente, permitindo ao *framework* inferir formas de parâmetros automaticamente, facilitando a modificação de arquiteturas e eliminando uma fonte comum de erros.
* Podemos passar dados através do modelo para fazer o *framework* finalmente inicializar os parâmetros.


## Exercícios

1. O que acontece se você especificar as dimensões de entrada para a primeira camada, mas não para as camadas subsequentes? Você consegue inicialização imediata?
1. O que acontece se você especificar dimensões incompatíveis?
1. O que você precisa fazer se tiver dados de dimensionalidade variável? Dica: observe a vinculação de parâmetros.

:begin_tab:`mxnet`
[Discussão](https://discuss.d2l.ai/t/280)
:end_tab:

:begin_tab:`tensorflow`
[Discussão](https://discuss.d2l.ai/t/281)
:end_tab:
<!--stackedit_data:
eyJoaXN0b3J5IjpbNzExNDA4MzAxLC0zODc0Mjc4NTEsNjQ1Nz
g1NDQyLDExMzU1ODY3NzRdfQ==
-->