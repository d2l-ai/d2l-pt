#  Detecção *Single Shot Multibox* (SSD)

Nas poucas seções anteriores, apresentamos caixas delimitadoras, caixas de âncora, detecção de objetos multiescala e conjuntos de dados. Agora, usaremos esse conhecimento prévio para construir um modelo de detecção de objetos: detecção multibox de disparo único [*Single Shot Multibox Detection*] (SSD) :cite:`Liu.Anguelov.Erhan.ea.2016`.  Este modelo rápido e fácil já é amplamente utilizado. Alguns dos conceitos de design e detalhes de implementação deste modelo também são aplicáveis a outros modelos de detecção de objetos.


## Modelo

:numref:`fig_ssd` mostra o design de um modelo SSD. Os principais componentes do modelo são um bloco de rede básico e vários blocos de recursos multiescala conectados em série. Aqui, o bloco de rede de base é usado para as características extras de imagens originais e geralmente assumem a forma de uma rede neural convolucional profunda. O artigo sobre SSDs opta por colocar um VGG truncado antes do
camada de classificação :cite:`Liu.Anguelov.Erhan.ea.2016`, mas agora é comumente substituído pelo ResNet. Podemos projetar uma rede de base para que ela produza alturas e larguras maiores. Desta forma, mais caixas de âncora são geradas com base neste mapa de características, permitindo-nos detectar objetos menores. Em seguida, cada bloco de feições multiescala reduz a altura e largura do mapa de feições fornecidas pela camada anterior (por exemplo, pode reduzir os tamanhos pela metade). Os blocos então usam cada elemento no mapa de recursos para expandir o campo receptivo na imagem de entrada. Desta forma, quanto mais próximo um bloco de feições multiescala estiver do topo de :numref:`fig_ssd` menor será o mapa de feições de saída e menos caixas de âncora são geradas com base no mapa de feições. Além disso, quanto mais próximo um bloco de recursos estiver do topo, maior será o campo receptivo de cada elemento no mapa de recursos e mais adequado será para detectar objetos maiores. Como o SSD gera diferentes números de caixas de âncora de tamanhos diferentes com base no bloco de rede de base e cada bloco de recursos multiescala e, em seguida, prevê como categorias e deslocamentos (ou seja, caixas delimitadoras previsão) das caixas de âncora para detectar objetos de tamanhos diferentes, SSD é um modelo de detecção de objetos multiescala.

![O SSD é composto de um bloco de rede base e vários blocos de recursos multiescala conectados em série. ](../img/ssd.svg)
:label:`fig_ssd`


A seguir, descreveremos a implementação dos módulos em :numref:`fig_ssd`. Primeiro, precisamos discutir a implementação da previsão da categoria e da previsão da caixa delimitadora.

### Camada de Previsão da Categoria

Defina o número de categorias de objeto como $q$. Nesse caso, o número de categorias de caixa de âncora é $q+1$, com 0 indicando uma caixa de âncora que contém apenas o fundo. Para uma determinada escala, defina a altura e a largura do mapa de feições para $h$ e $w$, respectivamente. Se usarmos cada elemento como o centro para gerar $a$
caixas de âncora, precisamos classificar um total de $hwa$ caixas de âncora. Se usarmos uma camada totalmente conectada (FCN) para a saída, isso provavelmente resultará em um número excessivo de parâmetros do modelo. Lembre-se de como usamos canais de camada convolucional para gerar previsões de categoria em :numref:`sec_nin`. O SSD usa o mesmo método para reduzir a complexidade do modelo.

Especificamente, a camada de predição de categoria usa uma camada convolucional que mantém a altura e largura de entrada. Assim, a saída e a entrada têm uma correspondência de um para um com as coordenadas espaciais ao longo da largura e altura do mapa de características. Supondo que a saída e a entrada tenham as mesmas
coordenadas $(x, y)$, o canal para as coordenadas $(x, y)$ no mapa de feição de saída contém as previsões de categoria para todas as caixas âncora geradas usando as coordenadas do mapa de feição de entrada $(x, y)$ como o Centro. Portanto, existem $a(q+1)$ canais de saída, com os canais de saída indexados como $i(q+1)+j$
($0 \leq j \leq q$) representando as previsões do índice de categoria $j$ para o índice de caixa de âncora $i$.

Agora, vamos definir uma camada de predição de categoria deste tipo. Depois de especificar os parâmetros $a$ e $q$, ele usa uma camada convolucional $3\times3$ com um preenchimento de 1. As alturas e larguras de entrada e saída dessa camada convolucional permanecem inalteradas.

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, gluon, image, init, np, npx
from mxnet.gluon import nn

npx.set_np()

def cls_predictor(num_anchors, num_classes):
    return nn.Conv2D(num_anchors * (num_classes + 1), kernel_size=3,
                     padding=1)
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
import torchvision
from torch import nn
from torch.nn import functional as F

def cls_predictor(num_inputs, num_anchors, num_classes):
    return nn.Conv2d(num_inputs, num_anchors * (num_classes + 1),
                     kernel_size=3, padding=1)
```

### Camada de Previsão de Caixa Delimitadora

O design da camada de previsão da caixa delimitadora é semelhante ao da camada de previsão da categoria. A única diferença é que, aqui, precisamos prever 4 deslocamentos para cada caixa de âncora, em vez de categorias $q+1$.

```{.python .input}
def bbox_predictor(num_anchors):
    return nn.Conv2D(num_anchors * 4, kernel_size=3, padding=1)
```

```{.python .input}
#@tab pytorch
def bbox_predictor(num_inputs, num_anchors):
    return nn.Conv2d(num_inputs, num_anchors * 4, kernel_size=3, padding=1)
```

### Concatenando Previsões para Múltiplas Escalas


Como mencionamos, o SSD usa mapas de recursos com base em várias escalas para gerar caixas de âncora e prever suas categorias e deslocamentos. Como as formas e o número de caixas de âncora centradas no mesmo elemento diferem para os mapas de recursos de escalas diferentes, as saídas de predição em escalas diferentes podem ter formas diferentes.

No exemplo a seguir, usamos o mesmo lote de dados para construir mapas de características de duas escalas diferentes, `Y1` e `Y2`. Aqui, `Y2` tem metade da altura e metade da largura de `Y1`. Usando a previsão de categoria como exemplo, assumimos que cada elemento nos mapas de características `Y1` e` Y2` gera cinco (Y1) ou três (Y2) caixas de âncora. Quando há 10 categorias de objeto, o número de canais de saída de predição de categoria é $5\times(10+1)=55$ ou $3\times(10+1)=33$. O formato da saída de previsão é (tamanho do lote, número de canais, altura, largura). Como você pode ver, exceto pelo tamanho do lote, os tamanhos das outras dimensões são diferentes. Portanto, devemos transformá-los em um formato consistente e concatenar as previsões das várias escalas para facilitar o cálculo subsequente.

```{.python .input}
def forward(x, block):
    block.initialize()
    return block(x)

Y1 = forward(np.zeros((2, 8, 20, 20)), cls_predictor(5, 10))
Y2 = forward(np.zeros((2, 16, 10, 10)), cls_predictor(3, 10))
(Y1.shape, Y2.shape)
```

```{.python .input}
#@tab pytorch
def forward(x, block):
    return block(x)

Y1 = forward(torch.zeros((2, 8, 20, 20)), cls_predictor(8, 5, 10))
Y2 = forward(torch.zeros((2, 16, 10, 10)), cls_predictor(16, 3, 10))
(Y1.shape, Y2.shape)
```

A dimensão do canal contém as previsões para todas as caixas de âncora com o mesmo centro. Primeiro movemos a dimensão do canal para a dimensão final. Como o tamanho do lote é o mesmo para todas as escalas, podemos converter os resultados da previsão para o formato binário (tamanho do lote, altura $\times$ largura $\times$ número de canais) para facilitar a concatenação subsequente no $1^{\mathrm{st}}$ dimensão.

```{.python .input}
def flatten_pred(pred):
    return npx.batch_flatten(pred.transpose(0, 2, 3, 1))

def concat_preds(preds):
    return np.concatenate([flatten_pred(p) for p in preds], axis=1)
```

```{.python .input}
#@tab pytorch
def flatten_pred(pred):
    return torch.flatten(pred.permute(0, 2, 3, 1), start_dim=1)

def concat_preds(preds):
    return torch.cat([flatten_pred(p) for p in preds], dim=1)
```

Assim, independentemente das diferentes formas de `Y1` e` Y2`, ainda podemos concatenar os resultados da previsão para as duas escalas diferentes do mesmo lote.

```{.python .input}
#@tab all
concat_preds([Y1, Y2]).shape
```

### Bloco de Redução de Amostragem de Altura e Largura

Para detecção de objetos multiescala, definimos o seguinte bloco `down_sample_blk`, que reduz a altura e largura em 50%. Este bloco consiste em duas camadas convolucionais $3\times3$ com um preenchimento de 1 e uma camada de *pooling* máximo $2\times2$ com uma distância de 2 conectadas em uma série. Como sabemos, $3\times3$ camadas convolucionais com um preenchimento de 1 não alteram a forma dos mapas de características. No entanto, a camada de agrupamento subsequente reduz diretamente o tamanho do mapa de feições pela metade. Como $1\times 2+(3-1)+(3-1)=6$, cada elemento no mapa de recursos de saída tem um campo receptivo no mapa de recursos de entrada da forma $6\times6$.  Como você pode ver, o bloco de redução de altura e largura aumenta o campo receptivo de cada elemento no mapa de recursos de saída.

```{.python .input}
def down_sample_blk(num_channels):
    blk = nn.Sequential()
    for _ in range(2):
        blk.add(nn.Conv2D(num_channels, kernel_size=3, padding=1),
                nn.BatchNorm(in_channels=num_channels),
                nn.Activation('relu'))
    blk.add(nn.MaxPool2D(2))
    return blk
```

```{.python .input}
#@tab pytorch
def down_sample_blk(in_channels, out_channels):
    blk = []
    for _ in range(2):
        blk.append(nn.Conv2d(in_channels, out_channels,
                             kernel_size=3, padding=1))
        blk.append(nn.BatchNorm2d(out_channels))
        blk.append(nn.ReLU())
        in_channels = out_channels
    blk.append(nn.MaxPool2d(2))
    return nn.Sequential(*blk)
```

Ao testar a computação direta no bloco de redução de altura e largura, podemos ver que ele altera o número de canais de entrada e divide a altura e a largura pela metade.

```{.python .input}
forward(np.zeros((2, 3, 20, 20)), down_sample_blk(10)).shape
```

```{.python .input}
#@tab pytorch
forward(torch.zeros((2, 3, 20, 20)), down_sample_blk(3, 10)).shape
```

### Bloco de Rede Base

O bloco de rede básico é usado para extrair recursos das imagens originais. Para simplificar o cálculo, construiremos uma pequena rede de base. Essa rede consiste em três blocos de *downsample* de altura e largura conectados em série, portanto, dobra o número de canais em cada etapa. Quando inserimos uma imagem original com a forma $256\times256$, o bloco de rede base produz um mapa de características com a forma $32 \times 32$.

```{.python .input}
def base_net():
    blk = nn.Sequential()
    for num_filters in [16, 32, 64]:
        blk.add(down_sample_blk(num_filters))
    return blk

forward(np.zeros((2, 3, 256, 256)), base_net()).shape
```

```{.python .input}
#@tab pytorch
def base_net():
    blk = []
    num_filters = [3, 16, 32, 64]
    for i in range(len(num_filters) - 1):
        blk.append(down_sample_blk(num_filters[i], num_filters[i+1]))
    return nn.Sequential(*blk)

forward(torch.zeros((2, 3, 256, 256)), base_net()).shape
```

### O Modelo Completo

O modelo SSD contém um total de cinco módulos. Cada módulo produz um mapa de recursos usado para gerar caixas de âncora e prever as categorias e deslocamentos dessas caixas de âncora. O primeiro módulo é o bloco de rede base, os módulos de dois a quatro são blocos de redução de amostragem de altura e largura e o quinto módulo é um bloco global
camada de pooling máxima que reduz a altura e largura para 1. Portanto, os módulos dois a cinco são todos blocos de recursos multiescala mostrados em :numref:`fig_ssd`.
```{.python .input}
def get_blk(i):
    if i == 0:
        blk = base_net()
    elif i == 4:
        blk = nn.GlobalMaxPool2D()
    else:
        blk = down_sample_blk(128)
    return blk
```

```{.python .input}
#@tab pytorch
def get_blk(i):
    if i == 0:
        blk = base_net()
    elif i == 1:
        blk = down_sample_blk(64, 128)
    elif i == 4:
        blk = nn.AdaptiveMaxPool2d((1,1))
    else:
        blk = down_sample_blk(128, 128)
    return blk
```

Agora, vamos definir o processo de computação progressiva para cada módulo. Em contraste com as redes neurais convolucionais descritas anteriormente, este módulo não só retorna a saída do mapa de características `Y` por computação convolucional, mas também as caixas de âncora da escala atual gerada a partir de` Y` e suas categorias e deslocamentos previstos.

```{.python .input}
def blk_forward(X, blk, size, ratio, cls_predictor, bbox_predictor):
    Y = blk(X)
    anchors = d2l.multibox_prior(Y, sizes=size, ratios=ratio)
    cls_preds = cls_predictor(Y)
    bbox_preds = bbox_predictor(Y)
    return (Y, anchors, cls_preds, bbox_preds)
```

```{.python .input}
#@tab pytorch
def blk_forward(X, blk, size, ratio, cls_predictor, bbox_predictor):
    Y = blk(X)
    anchors = d2l.multibox_prior(Y, sizes=size, ratios=ratio)
    cls_preds = cls_predictor(Y)
    bbox_preds = bbox_predictor(Y)
    return (Y, anchors, cls_preds, bbox_preds)
```

Como mencionamos, quanto mais próximo um bloco de recursos multiescala está do topo em :numref:`fig_ssd`, maiores são os objetos que ele detecta e maiores são as caixas de âncora que deve gerar. Aqui, primeiro dividimos o intervalo de 0,2 a 1,05 em cinco partes iguais para determinar os tamanhos das caixas de âncora menores em escalas diferentes: 0,2, 0,37, 0,54, etc. Então, de acordo com $\sqrt{0.2 \times 0.37} = 0.272$,  $\sqrt{0.37 \times 0.54} = 0.447$, e fórmulas semelhantes, determinamos os tamanhos de caixas de âncora maiores em escalas diferentes.

```{.python .input}
#@tab all
sizes = [[0.2, 0.272], [0.37, 0.447], [0.54, 0.619], [0.71, 0.79],
         [0.88, 0.961]]
ratios = [[1, 2, 0.5]] * 5
num_anchors = len(sizes[0]) + len(ratios[0]) - 1
```

Agora, podemos definir o modelo completo, `TinySSD`.

```{.python .input}
class TinySSD(nn.Block):
    def __init__(self, num_classes, **kwargs):
        super(TinySSD, self).__init__(**kwargs)
        self.num_classes = num_classes
        for i in range(5):
            # The assignment statement is self.blk_i = get_blk(i)
            setattr(self, f'blk_{i}', get_blk(i))
            setattr(self, f'cls_{i}', cls_predictor(num_anchors, num_classes))
            setattr(self, f'bbox_{i}', bbox_predictor(num_anchors))

    def forward(self, X):
        anchors, cls_preds, bbox_preds = [None] * 5, [None] * 5, [None] * 5
        for i in range(5):
            # getattr(self, 'blk_%d' % i) accesses self.blk_i
            X, anchors[i], cls_preds[i], bbox_preds[i] = blk_forward(
                X, getattr(self, f'blk_{i}'), sizes[i], ratios[i],
                getattr(self, f'cls_{i}'), getattr(self, f'bbox_{i}'))
        # In the reshape function, 0 indicates that the batch size remains
        # unchanged
        anchors = np.concatenate(anchors, axis=1)
        cls_preds = concat_preds(cls_preds)
        cls_preds = cls_preds.reshape(
            cls_preds.shape[0], -1, self.num_classes + 1)
        bbox_preds = concat_preds(bbox_preds)
        return anchors, cls_preds, bbox_preds
```

```{.python .input}
#@tab pytorch
class TinySSD(nn.Module):
    def __init__(self, num_classes, **kwargs):
        super(TinySSD, self).__init__(**kwargs)
        self.num_classes = num_classes
        idx_to_in_channels = [64, 128, 128, 128, 128]
        for i in range(5):
            # The assignment statement is self.blk_i = get_blk(i)
            setattr(self, f'blk_{i}', get_blk(i))
            setattr(self, f'cls_{i}', cls_predictor(idx_to_in_channels[i],
                                                    num_anchors, num_classes))
            setattr(self, f'bbox_{i}', bbox_predictor(idx_to_in_channels[i],
                                                      num_anchors))

    def forward(self, X):
        anchors, cls_preds, bbox_preds = [None] * 5, [None] * 5, [None] * 5
        for i in range(5):
            # getattr(self, 'blk_%d' % i) accesses self.blk_i
            X, anchors[i], cls_preds[i], bbox_preds[i] = blk_forward(
                X, getattr(self, f'blk_{i}'), sizes[i], ratios[i],
                getattr(self, f'cls_{i}'), getattr(self, f'bbox_{i}'))
        # In the reshape function, 0 indicates that the batch size remains
        # unchanged
        anchors = torch.cat(anchors, dim=1)
        cls_preds = concat_preds(cls_preds)
        cls_preds = cls_preds.reshape(
            cls_preds.shape[0], -1, self.num_classes + 1)
        bbox_preds = concat_preds(bbox_preds)
        return anchors, cls_preds, bbox_preds
```

Agora criamos uma instância de modelo SSD e a usamos para realizar cálculos avançados no minibatch de imagem `X`, que tem uma altura e largura de 256 pixels. Como verificamos anteriormente, o primeiro módulo gera um mapa de recursos com a forma $32 \times 32$. Como os módulos dois a quatro são blocos de redução de altura e largura, o módulo cinco é uma camada de agrupamento global e cada elemento no mapa de recursos é usado como o centro para 4 caixas de âncora, um total de $(32^2 + 16^2 + 8^2 + 4^2 + 1)\times 4 = 5444$ caixas de âncora são geradas para cada imagem nas cinco escalas.

```{.python .input}
net = TinySSD(num_classes=1)
net.initialize()
X = np.zeros((32, 3, 256, 256))
anchors, cls_preds, bbox_preds = net(X)

print('output anchors:', anchors.shape)
print('output class preds:', cls_preds.shape)
print('output bbox preds:', bbox_preds.shape)
```

```{.python .input}
#@tab pytorch
net = TinySSD(num_classes=1)
X = torch.zeros((32, 3, 256, 256))
anchors, cls_preds, bbox_preds = net(X)

print('output anchors:', anchors.shape)
print('output class preds:', cls_preds.shape)
print('output bbox preds:', bbox_preds.shape)
```

## Treinamento

Agora, vamos explicar, passo a passo, como treinar o modelo SSD para detecção de objetos.

### Leitura e Inicialização de Dados

Lemos o conjunto de dados de detecção de banana que criamos na seção anterior.

```{.python .input}
#@tab all
batch_size = 32
train_iter, _ = d2l.load_data_bananas(batch_size)
```

Existe 1 categoria no conjunto de dados de detecção de banana. Depois de definir o módulo, precisamos inicializar os parâmetros do modelo e definir o algoritmo de otimização.

```{.python .input}
device, net = d2l.try_gpu(), TinySSD(num_classes=1)
net.initialize(init=init.Xavier(), ctx=device)
trainer = gluon.Trainer(net.collect_params(), 'sgd',
                        {'learning_rate': 0.2, 'wd': 5e-4})
```

```{.python .input}
#@tab pytorch
device, net = d2l.try_gpu(), TinySSD(num_classes=1)
trainer = torch.optim.SGD(net.parameters(), lr=0.2, weight_decay=5e-4)
```

### Definindo Funções de Perda e Avaliação

A detecção de objetos está sujeita a dois tipos de perdas. a primeira é a perda da categoria da caixa de âncora. Para isso, podemos simplesmente reutilizar a função de perda de entropia cruzada que usamos na classificação de imagens. A segunda perda é a perda de deslocamento da caixa de âncora positiva. A previsão de deslocamento é um problema de normalização. No entanto, aqui, não usamos a perda quadrática introduzida anteriormente. Em vez disso, usamos a perda de norma $L_1$, que é o valor absoluto da diferença entre o valor previsto e o valor verdadeiro. A variável de máscara `bbox_masks` remove caixas de âncora negativas e caixas de âncora de preenchimento do cálculo de perda. Finalmente, adicionamos a categoria de caixa de âncora e compensamos as perdas para encontrar a função de perda final para o modelo.

```{.python .input}
cls_loss = gluon.loss.SoftmaxCrossEntropyLoss()
bbox_loss = gluon.loss.L1Loss()

def calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks):
    cls = cls_loss(cls_preds, cls_labels)
    bbox = bbox_loss(bbox_preds * bbox_masks, bbox_labels * bbox_masks)
    return cls + bbox
```

```{.python .input}
#@tab pytorch
cls_loss = nn.CrossEntropyLoss(reduction='none')
bbox_loss = nn.L1Loss(reduction='none')

def calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks):
    batch_size, num_classes = cls_preds.shape[0], cls_preds.shape[2]
    cls = cls_loss(cls_preds.reshape(-1, num_classes),
                   cls_labels.reshape(-1)).reshape(batch_size, -1).mean(dim=1)
    bbox = bbox_loss(bbox_preds * bbox_masks,
                     bbox_labels * bbox_masks).mean(dim=1)
    return cls + bbox
```

Podemos usar a taxa de precisão para avaliar os resultados da classificação. Como usamos a perda de norma $L_1$, usaremos o erro absoluto médio para avaliar os resultados da previsão da caixa delimitadora.

```{.python .input}
def cls_eval(cls_preds, cls_labels):
    # Because the category prediction results are placed in the final
    # dimension, argmax must specify this dimension
    return float((cls_preds.argmax(axis=-1).astype(
        cls_labels.dtype) == cls_labels).sum())

def bbox_eval(bbox_preds, bbox_labels, bbox_masks):
    return float((np.abs((bbox_labels - bbox_preds) * bbox_masks)).sum())
```

```{.python .input}
#@tab pytorch
def cls_eval(cls_preds, cls_labels):
    # Because the category prediction results are placed in the final
    # dimension, argmax must specify this dimension
    return float((cls_preds.argmax(dim=-1).type(
        cls_labels.dtype) == cls_labels).sum())

def bbox_eval(bbox_preds, bbox_labels, bbox_masks):
    return float((torch.abs((bbox_labels - bbox_preds) * bbox_masks)).sum())
```

### Treinando o Modelo

Durante o treinamento do modelo, devemos gerar caixas de âncora multiescala (`âncoras`) no processo de computação direta do modelo e prever a categoria (`cls_preds`) e o deslocamento (`bbox_preds`) para cada caixa de âncora. Depois, rotulamos a categoria (`cls_labels`) e o deslocamento (`bbox_labels`) de cada caixa de âncora gerada com base nas informações do rótulo `Y`. Finalmente, calculamos a função de perda usando a categoria predita e rotulada e os valores de compensação. Para simplificar o código, não avaliamos o conjunto de dados de treinamento aqui.

```{.python .input}
num_epochs, timer = 20, d2l.Timer()
animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                        legend=['class error', 'bbox mae'])
for epoch in range(num_epochs):
    # accuracy_sum, mae_sum, num_examples, num_labels
    metric = d2l.Accumulator(4)
    for features, target in train_iter:
        timer.start()
        X = features.as_in_ctx(device)
        Y = target.as_in_ctx(device)
        with autograd.record():
            # Generate multiscale anchor boxes and predict the category and
            # offset of each
            anchors, cls_preds, bbox_preds = net(X)
            # Label the category and offset of each anchor box
            bbox_labels, bbox_masks, cls_labels = d2l.multibox_target(anchors,
                                                                      Y)
            # Calculate the loss function using the predicted and labeled
            # category and offset values
            l = calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels,
                          bbox_masks)
        l.backward()
        trainer.step(batch_size)
        metric.add(cls_eval(cls_preds, cls_labels), cls_labels.size,
                   bbox_eval(bbox_preds, bbox_labels, bbox_masks),
                   bbox_labels.size)
    cls_err, bbox_mae = 1 - metric[0] / metric[1], metric[2] / metric[3]
    animator.add(epoch + 1, (cls_err, bbox_mae))
print(f'class err {cls_err:.2e}, bbox mae {bbox_mae:.2e}')
print(f'{len(train_iter._dataset) / timer.stop():.1f} examples/sec on '
      f'{str(device)}')
```

```{.python .input}
#@tab pytorch
num_epochs, timer = 20, d2l.Timer()
animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                        legend=['class error', 'bbox mae'])
net = net.to(device)
for epoch in range(num_epochs):
    # accuracy_sum, mae_sum, num_examples, num_labels
    metric = d2l.Accumulator(4)
    net.train()
    for features, target in train_iter:
        timer.start()
        trainer.zero_grad()
        X, Y = features.to(device), target.to(device)
        # Generate multiscale anchor boxes and predict the category and
        # offset of each
        anchors, cls_preds, bbox_preds = net(X)
        # Label the category and offset of each anchor box
        bbox_labels, bbox_masks, cls_labels = d2l.multibox_target(anchors, Y)
        # Calculate the loss function using the predicted and labeled
        # category and offset values
        l = calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels,
                      bbox_masks)
        l.mean().backward()
        trainer.step()
        metric.add(cls_eval(cls_preds, cls_labels), cls_labels.numel(),
                   bbox_eval(bbox_preds, bbox_labels, bbox_masks),
                   bbox_labels.numel())
    cls_err, bbox_mae = 1 - metric[0] / metric[1], metric[2] / metric[3]
    animator.add(epoch + 1, (cls_err, bbox_mae))
print(f'class err {cls_err:.2e}, bbox mae {bbox_mae:.2e}')
print(f'{len(train_iter.dataset) / timer.stop():.1f} examples/sec on '
      f'{str(device)}')
```

## Predição

Na fase de previsão, queremos detectar todos os objetos de interesse na imagem. Abaixo, lemos a imagem de teste e transformamos seu tamanho. Então, nós o convertemos para o formato quadridimensional exigido pela camada convolucional.

```{.python .input}
img = image.imread('../img/banana.jpg')
feature = image.imresize(img, 256, 256).astype('float32')
X = np.expand_dims(feature.transpose(2, 0, 1), axis=0)
```

```{.python .input}
#@tab pytorch
X = torchvision.io.read_image('../img/banana.jpg').unsqueeze(0).float()
img = X.squeeze(0).permute(1,2,0).long()
```

Usando a função `multibox_detection`, prevemos as caixas delimitadoras com base nas caixas de âncora e seus deslocamentos previstos. Em seguida, usamos a supressão não máxima para remover caixas delimitadoras semelhantes.

```{.python .input}
def predict(X):
    anchors, cls_preds, bbox_preds = net(X.as_in_ctx(device))
    cls_probs = npx.softmax(cls_preds).transpose(0, 2, 1)
    output = d2l.multibox_detection(cls_probs, bbox_preds, anchors)
    idx = [i for i, row in enumerate(output[0]) if row[0] != -1]
    return output[0, idx]

output = predict(X)
```

```{.python .input}
#@tab pytorch
def predict(X):
    net.eval()
    anchors, cls_preds, bbox_preds = net(X.to(device))
    cls_probs = F.softmax(cls_preds, dim=2).permute(0, 2, 1)
    output = d2l.multibox_detection(cls_probs, bbox_preds, anchors)
    idx = [i for i, row in enumerate(output[0]) if row[0] != -1]
    return output[0, idx]

output = predict(X)
```

Por fim, pegamos todas as caixas delimitadoras com um nível de confiança de pelo menos 0,9 e as exibimos como a saída final.

```{.python .input}
def display(img, output, threshold):
    d2l.set_figsize((5, 5))
    fig = d2l.plt.imshow(img.asnumpy())
    for row in output:
        score = float(row[1])
        if score < threshold:
            continue
        h, w = img.shape[0:2]
        bbox = [row[2:6] * np.array((w, h, w, h), ctx=row.ctx)]
        d2l.show_bboxes(fig.axes, bbox, '%.2f' % score, 'w')

display(img, output, threshold=0.9)
```

```{.python .input}
#@tab pytorch
def display(img, output, threshold):
    d2l.set_figsize((5, 5))
    fig = d2l.plt.imshow(img)
    for row in output:
        score = float(row[1])
        if score < threshold:
            continue
        h, w = img.shape[0:2]
        bbox = [row[2:6] * torch.tensor((w, h, w, h), device=row.device)]
        d2l.show_bboxes(fig.axes, bbox, '%.2f' % score, 'w')

display(img, output.cpu(), threshold=0.9)
```

## Resumo

* SSD é um modelo de detecção de objetos multiescala. Este modelo gera diferentes números de caixas de âncora de tamanhos diferentes com base no bloco de rede de base e cada bloco de recursos multiescala e prevê as categorias e deslocamentos das caixas de âncora para detectar objetos de tamanhos diferentes.
* Durante o treinamento do modelo SSD, a função de perda é calculada usando a categoria prevista e rotulada e os valores de deslocamento.



## Exercícios

1. Devido a limitações de espaço, ignoramos alguns dos detalhes de implementação do modelo SSD neste experimento. Você pode melhorar ainda mais o modelo nas seguintes áreas?


### Função de Perda

A. Para as compensações previstas, substitua $L_1$ perda de norma por $L_1$ de perda de regularização. Esta função de perda usa uma função quadrada em torno de zero para maior suavidade. Esta é a área regularizada controlada pelo hiperparâmetro $\sigma$:

$$
f(x) =
    \begin{cases}
    (\sigma x)^2/2,& \text{if }|x| < 1/\sigma^2\\
    |x|-0.5/\sigma^2,& \text{otherwise}
    \end{cases}
$$

Quando $\sigma$ é grande, essa perda é semelhante à perda normal de $L_1$. Quando o valor é pequeno, a função de perda é mais suave.

```{.python .input}
sigmas = [10, 1, 0.5]
lines = ['-', '--', '-.']
x = np.arange(-2, 2, 0.1)
d2l.set_figsize()

for l, s in zip(lines, sigmas):
    y = npx.smooth_l1(x, scalar=s)
    d2l.plt.plot(x.asnumpy(), y.asnumpy(), l, label='sigma=%.1f' % s)
d2l.plt.legend();
```

```{.python .input}
#@tab pytorch
def smooth_l1(data, scalar):
    out = []
    for i in data:
        if abs(i) < 1 / (scalar ** 2):
            out.append(((scalar * i) ** 2) / 2)
        else:
            out.append(abs(i) - 0.5 / (scalar ** 2))
    return torch.tensor(out)

sigmas = [10, 1, 0.5]
lines = ['-', '--', '-.']
x = torch.arange(-2, 2, 0.1)
d2l.set_figsize()

for l, s in zip(lines, sigmas):
    y = smooth_l1(x, scalar=s)
    d2l.plt.plot(x, y, l, label='sigma=%.1f' % s)
d2l.plt.legend();
```

No experimento, usamos a perda de entropia cruzada para a previsão da categoria. Agora, assuma que a probabilidade de predição da categoria real $j$ é $p_j$ e a perda de entropia cruzada é $-\log p_j$. Também podemos usar a perda focal :cite:`Lin.Goyal.Girshick.ea.2017`. Dados os hiperparâmetros positivos $\gamma$ e $\alpha$, essa perda é definida como:

$$ - \alpha (1-p_j)^{\gamma} \log p_j.$$

Como você pode ver, ao aumentar $\gamma$, podemos efetivamente reduzir a perda quando a probabilidade de prever a categoria correta for alta.

```{.python .input}
def focal_loss(gamma, x):
    return -(1 - x) ** gamma * np.log(x)

x = np.arange(0.01, 1, 0.01)
for l, gamma in zip(lines, [0, 1, 5]):
    y = d2l.plt.plot(x.asnumpy(), focal_loss(gamma, x).asnumpy(), l,
                     label='gamma=%.1f' % gamma)
d2l.plt.legend();
```

```{.python .input}
#@tab pytorch
def focal_loss(gamma, x):
    return -(1 - x) ** gamma * torch.log(x)

x = torch.arange(0.01, 1, 0.01)
for l, gamma in zip(lines, [0, 1, 5]):
    y = d2l.plt.plot(x, focal_loss(gamma, x), l, label='gamma=%.1f' % gamma)
d2l.plt.legend();
```

### Treinamento e Previsão


B. Quando um objeto é relativamente grande em comparação com a imagem, o modelo normalmente adota um tamanho de imagem de entrada maior.

C. Isso geralmente produz um grande número de caixas de âncora negativas ao rotular as categorias da caixa de âncora. Podemos amostrar as caixas de âncora negativas para equilibrar melhor as categorias de dados. Para fazer isso, podemos definir um parâmetro `negative_mining_ratio` na função `multibox_target`.

D. Atribuir hiperparâmetros com pesos diferentes para a perda de categoria da caixa de âncora e a perda de deslocamento da caixa de âncora positiva na função de perda.

E. Consulte o documento SSD. Quais métodos podem ser usados para avaliar a precisão dos modelos de detecção de objetos :cite:`Liu.Anguelov.Erhan.ea.2016`?

:begin_tab:`mxnet`
[Discussões](https://discuss.d2l.ai/t/373)
:end_tab:

:begin_tab:`pytorch`
[Discussões](https://discuss.d2l.ai/t/1604)
:end_tab:
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTczMDA0Njg4OSwtMTkwMzAyNjc2NCw0Mz
MxMDMwNzAsNzQ0NjI0NDEwLDE2NzYyMjE0OTIsLTEzNzQ2Nzc5
NzUsLTIxNDM2NzY5NzcsMjAwNTYxMDkyMiwzNzM1NTgzNCwzMT
AzNTU1NTJdfQ==
-->