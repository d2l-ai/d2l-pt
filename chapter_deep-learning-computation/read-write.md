# Entrada e Saída de Arquivos

Até agora, discutimos como processar dados e como
para construir, treinar e testar modelos de *Deep Learning*.
No entanto, em algum momento, esperamos ser felizes o suficiente
com os modelos aprendidos que queremos
para salvar os resultados para uso posterior em vários contextos
(talvez até mesmo para fazer previsões na implantação).
Além disso, ao executar um longo processo de treinamento,
a prática recomendada é salvar resultados intermediários periodicamente (pontos de verificação)
para garantir que não perdemos vários dias de computação
se tropeçarmos no cabo de alimentação do nosso servidor.
Portanto, é hora de aprender como carregar e armazenar
ambos os vetores de peso individuais e modelos inteiros.
Esta seção aborda ambos os problemas.

## Carregando e Salvando Tensores

Para tensores individuais, podemos diretamente
invocar as funções `load` e `save`
para ler e escrever respectivamente.
Ambas as funções exigem que forneçamos um nome,
e `save` requer como entrada a variável a ser salva.

```{.python .input}
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()

x = np.arange(4)
npx.save('x-file', x)
```

```{.python .input}
#@tab pytorch
import torch
from torch import nn
from torch.nn import functional as F

x = torch.arange(4)
torch.save(x, 'x-file')
```

```{.python .input}
#@tab tensorflow
import tensorflow as tf
import numpy as np

x = tf.range(4)
np.save("x-file.npy", x)
```

Agora podemos ler os dados do arquivo armazenado de volta na memória.

```{.python .input}
x2 = npx.load('x-file')
x2
```

```{.python .input}
#@tab pytorch
x2 = torch.load("x-file")
x2
```

```{.python .input}
#@tab tensorflow
x2 = np.load('x-file.npy', allow_pickle=True)
x2
```
Podemos armazenar uma lista de tensores e lê-los de volta na memória.
```{.python .input}
y = np.zeros(4)
npx.save('x-files', [x, y])
x2, y2 = npx.load('x-files')
(x2, y2)
```

```{.python .input}
#@tab pytorch
y = torch.zeros(4)
torch.save([x, y],'x-files')
x2, y2 = torch.load('x-files')
(x2, y2)
```

```{.python .input}
#@tab tensorflow
y = tf.zeros(4)
np.save('xy-files.npy', [x, y])
x2, y2 = np.load('xy-files.npy', allow_pickle=True)
(x2, y2)
```
Podemos até escrever e ler um dicionário que mapeia
de *strings* a tensores.
Isso é conveniente quando queremos 
ler ou escrever todos os pesos em um modelo.

```{.python .input}
mydict = {'x': x, 'y': y}
npx.save('mydict', mydict)
mydict2 = npx.load('mydict')
mydict2
```

```{.python .input}
#@tab pytorch
mydict = {'x': x, 'y': y}
torch.save(mydict, 'mydict')
mydict2 = torch.load('mydict')
mydict2
```

```{.python .input}
#@tab tensorflow
mydict = {'x': x, 'y': y}
np.save('mydict.npy', mydict)
mydict2 = np.load('mydict.npy', allow_pickle=True)
mydict2
```

## Carregando e Salvando Parâmetros de Modelos

Salvar vetores de peso individuais (ou outros tensores) é útil,
mas fica muito tedioso se quisermos salvar
(e depois carregar) um modelo inteiro.
Afinal, podemos ter centenas de
grupos de parâmetros espalhados por toda parte.
Por esta razão, a estrutura de *Deep Learning* fornece funcionalidades integradas
para carregar e salvar redes inteiras.
Um detalhe importante a notar é que este
salva o modelo *parâmetros* e não o modelo inteiro.
Por exemplo, se tivermos um MLP de 3 camadas,
precisamos especificar a arquitetura separadamente.
A razão para isso é que os próprios modelos podem conter código arbitrário,
portanto, eles não podem ser serializados naturalmente.
Assim, para restabelecer um modelo, precisamos
para gerar a arquitetura em código
e carregue os parâmetros do disco.
Vamos começar com nosso MLP familiar.

```{.python .input}
class MLP(nn.Block):
    def __init__(self, **kwargs):
        super(MLP, self).__init__(**kwargs)
        self.hidden = nn.Dense(256, activation='relu')
        self.output = nn.Dense(10)

    def forward(self, x):
        return self.output(self.hidden(x))

net = MLP()
net.initialize()
X = np.random.uniform(size=(2, 20))
Y = net(X)
```

```{.python .input}
#@tab pytorch
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.output = nn.Linear(256, 10)

    def forward(self, x):
        return self.output(F.relu(self.hidden(x)))

net = MLP()
X = torch.randn(size=(2, 20))
Y = net(X)
```

```{.python .input}
#@tab tensorflow
class MLP(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.flatten = tf.keras.layers.Flatten()
        self.hidden = tf.keras.layers.Dense(units=256, activation=tf.nn.relu)
        self.out = tf.keras.layers.Dense(units=10)

    def call(self, inputs):
        x = self.flatten(inputs)
        x = self.hidden(x)
        return self.out(x)

net = MLP()
X = tf.random.uniform((2, 20))
Y = net(X)
```
A seguir, armazenamos os parâmetros do modelo como um arquivo com o nome "mlp.params".

```{.python .input}
net.save_parameters('mlp.params')
```

```{.python .input}
#@tab pytorch
torch.save(net.state_dict(), 'mlp.params')
```

```{.python .input}
#@tab tensorflow
net.save_weights('mlp.params')
```

Para recuperar o modelo, instanciamos um clone
do modelo MLP original.
Em vez de inicializar aleatoriamente os parâmetros do modelo,
lemos os parâmetros armazenados no arquivo diretamente.

```{.python .input}
clone = MLP()
clone.load_parameters('mlp.params')
```

```{.python .input}
#@tab pytorch
clone = MLP()
clone.load_state_dict(torch.load("mlp.params"))
clone.eval()
```

```{.python .input}
#@tab tensorflow
clone = MLP()
clone.load_weights("mlp.params")
```

Uma vez que ambas as instâncias têm os mesmos parâmetros de modelo,
o resultado computacional da mesma entrada `X` deve ser o mesmo.
Deixe-nos verificar isso.

```{.python .input}
Y_clone = clone(X)
Y_clone == Y
```

```{.python .input}
#@tab pytorch
Y_clone = clone(X)
Y_clone == Y
```

```{.python .input}
#@tab tensorflow
Y_clone = clone(X)
Y_clone == Y
```

## Sumário

* As funções `save` e `load` podem ser usadas para executar E/S de arquivo para objetos tensores.
* Podemos salvar e carregar todos os conjuntos de parâmetros de uma rede por meio de um dicionário de parâmetros.
* Salvar a arquitetura deve ser feito em código e não em parâmetros.

## Exercícios

1. Mesmo se não houver necessidade de implantar modelos treinados em um dispositivo diferente, quais são os benefícios práticos de armazenar parâmetros de modelo?
1. Suponha que desejamos reutilizar apenas partes de uma rede para serem incorporadas a uma rede de arquitetura diferente. Como você usaria, digamos, as duas primeiras camadas de uma rede anterior em uma nova rede?
1. Como você salvaria a arquitetura e os parâmetros da rede? Que restrições você imporia à arquitetura?

:begin_tab:`mxnet`
[Discussão](https://discuss.d2l.ai/t/60)
:end_tab:

:begin_tab:`pytorch`
[Discussão](https://discuss.d2l.ai/t/61)
:end_tab:

:begin_tab:`tensorflow`
[Discussão](https://discuss.d2l.ai/t/327)
:end_tab:
<!--stackedit_data:
eyJoaXN0b3J5IjpbMzc3NjU2MDM4XX0=
-->