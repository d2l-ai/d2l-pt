# Redes Totalmente Convolucionais (*Fully Convolutional Networks*, FCN)
:label:`sec_fcn`


Discutimos anteriormente a segmentação semântica usando cada pixel em uma imagem para
previsão de categoria. Uma rede totalmente convolucional (FCN)
:cite:`Long.Shelhamer.Darrell.2015` usa uma rede neural convolucional para
transformar os pixels da imagem em categorias de pixels. Ao contrário das redes neurais convolucionais
previamente introduzidas, uma FCN transforma a altura e largura do
mapa de recurso da camada intermediária de volta ao tamanho da imagem de entrada por meio do
camada de convolução transposta, de modo que as previsões tenham uma
correspondência com a imagem de entrada em dimensão espacial (altura e largura). Dado uma
posição na dimensão espacial, a saída da dimensão do canal será uma
previsão de categoria do pixel correspondente ao local.

Primeiro importaremos o pacote ou módulo necessário para o experimento e depois
explicaremos a camada de convolução transposta.

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import gluon, image, init, np, npx
from mxnet.gluon import nn

npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
import torchvision
from torch import nn
from torch.nn import functional as F
```

## Construindo um Modelo

Aqui, demonstramos o projeto mais básico de um modelo de rede totalmente convolucional. Conforme mostrado em :numref:`fig_fcn`, a rede totalmente convolucional primeiro usa a rede neural convolucional para extrair características da imagem, então transforma o número de canais no número de categorias através da camada de convolução $1\times 1$ e, finalmente, transforma a altura e largura do mapa de recursos para o tamanho da imagem de entrada usando a camada de convolução transposta :numref:`sec_transposed_conv`. A saída do modelo tem a mesma altura e largura da imagem de entrada e uma correspondência de um para um nas posições espaciais. O canal de saída final contém a previsão da categoria do pixel da posição espacial correspondente.

![Rede totalmente convolucional.](../img/fcn.svg)
:label:`fig_fcn`

Abaixo, usamos um modelo ResNet-18 pré-treinado no conjunto de dados ImageNet para extrair recursos de imagem e registrar a instância de rede como `pretrained_net`. Como você pode ver, as duas últimas camadas da variável membro do modelo `features` são a camada de agrupamento global médio` GlobalAvgPool2D` e a camada de nivelamento de exemplo `Flatten` O módulo `output` contém a camada totalmente conectada usada para saída. Essas camadas não são necessárias para uma rede totalmente convolucional.

```{.python .input}
pretrained_net = gluon.model_zoo.vision.resnet18_v2(pretrained=True)
pretrained_net.features[-4:], pretrained_net.output
```

```{.python .input}
#@tab pytorch
pretrained_net = torchvision.models.resnet18(pretrained=True)
pretrained_net.layer4[1], pretrained_net.avgpool, pretrained_net.fc
```

Em seguida, criamos a instância de rede totalmente convolucional `net`. Ela duplica todas as camadas neurais, exceto as duas últimas camadas da variável membro de instância `features` de `pretrained_net` e os parâmetros do modelo obtidos após o pré-treinamento.

```{.python .input}
net = nn.HybridSequential()
for layer in pretrained_net.features[:-2]:
    net.add(layer)
```

```{.python .input}
#@tab pytorch
net = nn.Sequential(*list(pretrained_net.children())[:-2])
```

Dada uma entrada de altura e largura de 320 e 480 respectivamente, o cálculo direto de `net` reduzirá a altura e largura da entrada para $1/32$ do original, ou seja, 10 e 15.

```{.python .input}
X = np.random.uniform(size=(1, 3, 320, 480))
net(X).shape
```

```{.python .input}
#@tab pytorch
X = torch.rand(size=(1, 3, 320, 480))
net(X).shape
```

Em seguida, transformamos o número de canais de saída para o número de categorias de Pascal VOC2012 (21) por meio da camada de convolução $1\times 1$. Finalmente, precisamos ampliar a altura e largura do mapa de feições por um fator de 32 para alterá-los de volta para a altura e largura da imagem de entrada. Lembre-se do cálculo
método para a forma de saída da camada de convolução descrita em
:numref:`sec_padding`. Porque $(320-64+16\times2+32)/32=10$ e $(480-64+16\times2+32)/32=15$, construímos uma camada de convolução transposta com uma distância de 32 e definimos a altura e largura do *kernel* de convolução para 64 e o preenchimento para 16. Não é difícil ver que, se o passo for $s$, o preenchimento é $s/2$  (assumindo que $s/2$ é um inteiro ), e a altura e largura do *kernel* de convolução são $2s$, o *kernel* de convolução transposto aumentará a altura e a largura da entrada por um fator de $s$.

```{.python .input}
num_classes = 21
net.add(nn.Conv2D(num_classes, kernel_size=1),
        nn.Conv2DTranspose(
            num_classes, kernel_size=64, padding=16, strides=32))
```

```{.python .input}
#@tab pytorch
num_classes = 21
net.add_module('final_conv', nn.Conv2d(512, num_classes, kernel_size=1))
net.add_module('transpose_conv', nn.ConvTranspose2d(num_classes, num_classes,
                                    kernel_size=64, padding=16, stride=32))
```

## Inicializando a Camada de Convolução Transposta
Já sabemos que a camada de convolução transposta pode ampliar um mapa de feições. No processamento de imagem, às vezes precisamos ampliar a imagem, ou seja, *upsampling*. Existem muitos métodos para aumentar a amostragem e um método comum é a interpolação bilinear. Simplesmente falando, para obter o pixel da imagem de saída nas coordenadas $(x, y)$, as coordenadas são primeiro mapeadas para as coordenadas da imagem de entrada $(x ', y')$. Isso pode ser feito com base na proporção do tamanho de três entradas em relação ao tamanho da saída. Os valores mapeados $x'$ e $y'$ são geralmente números reais. Então, encontramos os quatro pixels mais próximos da coordenada $(x ', y')$ na imagem de entrada. Finalmente, os pixels da imagem de saída nas coordenadas $(x, y)$ são calculados com base nesses quatro pixels na imagem de entrada e suas distâncias relativas a $(x ', y')$. O *upsampling* por interpolação bilinear pode ser implementado pela camada de convolução transposta do *kernel* de convolução construído usando a seguinte função `bilinear_kernel`. Devido a limitações de espaço, fornecemos apenas a implementação da função `bilinear_kernel` e não discutiremos os princípios do algoritmo.

```{.python .input}
def bilinear_kernel(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = (np.arange(kernel_size).reshape(-1, 1),
          np.arange(kernel_size).reshape(1, -1))
    filt = (1 - np.abs(og[0] - center) / factor) * \
           (1 - np.abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size))
    weight[range(in_channels), range(out_channels), :, :] = filt
    return np.array(weight)
```

```{.python .input}
#@tab pytorch
def bilinear_kernel(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = (torch.arange(kernel_size).reshape(-1, 1),
          torch.arange(kernel_size).reshape(1, -1))
    filt = (1 - torch.abs(og[0] - center) / factor) * \
           (1 - torch.abs(og[1] - center) / factor)
    weight = torch.zeros((in_channels, out_channels,
                          kernel_size, kernel_size))
    weight[range(in_channels), range(out_channels), :, :] = filt
    return weight
```

Agora, vamos experimentar com upsampling de interpolação bilinear implementado por camadas de convolução transpostas. Construa uma camada de convolução transposta que amplie a altura e a largura da entrada por um fator de 2 e inicialize seu kernel de convolução com a função `bilinear_kernel`.

```{.python .input}
conv_trans = nn.Conv2DTranspose(3, kernel_size=4, padding=1, strides=2)
conv_trans.initialize(init.Constant(bilinear_kernel(3, 3, 4)))
```

```{.python .input}
#@tab pytorch
conv_trans = nn.ConvTranspose2d(3, 3, kernel_size=4, padding=1, stride=2,
                                bias=False)
conv_trans.weight.data.copy_(bilinear_kernel(3, 3, 4));
```

Leia a imagem `X` e registre o resultado do upsampling como` Y`. Para imprimir a imagem, precisamos ajustar a posição da dimensão do canal.

```{.python .input}
img = image.imread('../img/catdog.jpg')
X = np.expand_dims(img.astype('float32').transpose(2, 0, 1), axis=0) / 255
Y = conv_trans(X)
out_img = Y[0].transpose(1, 2, 0)
```

```{.python .input}
#@tab pytorch
img = torchvision.transforms.ToTensor()(d2l.Image.open('../img/catdog.jpg'))
X = img.unsqueeze(0)
Y = conv_trans(X)
out_img = Y[0].permute(1, 2, 0).detach()
```

Como você pode ver, a camada de convolução transposta amplia a altura e largura da imagem em um fator de 2. Vale ressaltar que, além da diferença na escala de coordenadas, a imagem ampliada por interpolação bilinear e a imagem original impressa em :numref:`sec_bbox` tem a mesma aparência.

```{.python .input}
d2l.set_figsize()
print('input image shape:', img.shape)
d2l.plt.imshow(img.asnumpy());
print('output image shape:', out_img.shape)
d2l.plt.imshow(out_img.asnumpy());
```

```{.python .input}
#@tab pytorch
d2l.set_figsize()
print('input image shape:', img.permute(1, 2, 0).shape)
d2l.plt.imshow(img.permute(1, 2, 0));
print('output image shape:', out_img.shape)
d2l.plt.imshow(out_img);
```

Em uma rede totalmente convolucional, inicializamos a camada de convolução transposta para interpolação bilinear com upsampled. Para uma camada de convolução $1\times 1$, usamos o Xavier para inicialização aleatória.

```{.python .input}
W = bilinear_kernel(num_classes, num_classes, 64)
net[-1].initialize(init.Constant(W))
net[-2].initialize(init=init.Xavier())
```

```{.python .input}
#@tab pytorch
W = bilinear_kernel(num_classes, num_classes, 64)
net.transpose_conv.weight.data.copy_(W);
```

## Lendo o *Dataset*

Lemos o *dataset* usando o método descrito na seção anterior. Aqui, especificamos a forma da imagem de saída cortada aleatoriamente como $320\times 480$, portanto, a altura e a largura são divisíveis por 32.

```{.python .input}
#@tab all
batch_size, crop_size = 32, (320, 480)
train_iter, test_iter = d2l.load_data_voc(batch_size, crop_size)
```

## Treinamento

Agora podemos começar a treinar o modelo. A função de perda e o cálculo de precisão aqui não são substancialmente diferentes daqueles usados na classificação de imagens. Como usamos o canal da camada de convolução transposta para prever as categorias de pixels, a opção `axis = 1` (dimensão do canal) é especificada em `SoftmaxCrossEntropyLoss`. Além disso, o modelo calcula a precisão com base em se a categoria de previsão de cada pixel está correta.

```{.python .input}
num_epochs, lr, wd, devices = 5, 0.1, 1e-3, d2l.try_all_gpus()
loss = gluon.loss.SoftmaxCrossEntropyLoss(axis=1)
net.collect_params().reset_ctx(devices)
trainer = gluon.Trainer(net.collect_params(), 'sgd',
                        {'learning_rate': lr, 'wd': wd})
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)
```

```{.python .input}
#@tab pytorch
def loss(inputs, targets):
    return F.cross_entropy(inputs, targets, reduction='none').mean(1).mean(1)

num_epochs, lr, wd, devices = 5, 0.001, 1e-3, d2l.try_all_gpus()
trainer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=wd)
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)
```

## Predição

Durante a previsão, precisamos padronizar a imagem de entrada em cada canal e transformá-los no formato de entrada quadridimensional exigido pela rede neural convolucional.

```{.python .input}
def predict(img):
    X = test_iter._dataset.normalize_image(img)
    X = np.expand_dims(X.transpose(2, 0, 1), axis=0)
    pred = net(X.as_in_ctx(devices[0])).argmax(axis=1)
    return pred.reshape(pred.shape[1], pred.shape[2])
```

```{.python .input}
#@tab pytorch
def predict(img):
    X = test_iter.dataset.normalize_image(img).unsqueeze(0)
    pred = net(X.to(devices[0])).argmax(dim=1)
    return pred.reshape(pred.shape[1], pred.shape[2])
```

Para visualizar as categorias previstas para cada pixel, mapeamos as categorias previstas de volta às suas cores rotuladas no conjunto de dados.

```{.python .input}
def label2image(pred):
    colormap = np.array(d2l.VOC_COLORMAP, ctx=devices[0], dtype='uint8')
    X = pred.astype('int32')
    return colormap[X, :]
```

```{.python .input}
#@tab pytorch
def label2image(pred):
    colormap = torch.tensor(d2l.VOC_COLORMAP, device=devices[0])
    X = pred.long()
    return colormap[X, :]
```


O tamanho e a forma das imagens no conjunto de dados de teste variam. Como o modelo usa uma camada de convolução transposta com uma distância de 32, quando a altura ou largura da imagem de entrada não é divisível por 32, a altura ou largura da saída da camada de convolução transposta se desvia do tamanho da imagem de entrada. Para resolver esse problema, podemos recortar várias áreas retangulares na imagem com alturas e larguras como múltiplos inteiros de 32 e, em seguida, realizar cálculos para a frente nos pixels nessas áreas. Quando combinadas, essas áreas devem cobrir completamente a imagem de entrada. Quando um pixel é coberto por várias áreas, a média da saída da camada de convolução transposta no cálculo direto das diferentes áreas pode ser usada como uma entrada para a operação softmax para prever a categoria.

Para simplificar, lemos apenas algumas imagens de teste grandes e recortamos uma área com um formato de $320\times480$ no canto superior esquerdo da imagem. Apenas esta área é usada para previsão. Para a imagem de entrada, imprimimos primeiro a área cortada, depois imprimimos o resultado previsto e, por fim, imprimimos a categoria rotulada.

```{.python .input}
voc_dir = d2l.download_extract('voc2012', 'VOCdevkit/VOC2012')
test_images, test_labels = d2l.read_voc_images(voc_dir, False)
n, imgs = 4, []
for i in range(n):
    crop_rect = (0, 0, 480, 320)
    X = image.fixed_crop(test_images[i], *crop_rect)
    pred = label2image(predict(X))
    imgs += [X, pred, image.fixed_crop(test_labels[i], *crop_rect)]
d2l.show_images(imgs[::3] + imgs[1::3] + imgs[2::3], 3, n, scale=2);
```

```{.python .input}
#@tab pytorch
voc_dir = d2l.download_extract('voc2012', 'VOCdevkit/VOC2012')
test_images, test_labels = d2l.read_voc_images(voc_dir, False)
n, imgs = 4, []
for i in range(n):
    crop_rect = (0, 0, 320, 480)
    X = torchvision.transforms.functional.crop(test_images[i], *crop_rect)
    pred = label2image(predict(X))
    imgs += [X.permute(1,2,0), pred.cpu(),
             torchvision.transforms.functional.crop(
                 test_labels[i], *crop_rect).permute(1,2,0)]
d2l.show_images(imgs[::3] + imgs[1::3] + imgs[2::3], 3, n, scale=2);
```

## Resumo

* A rede totalmente convolucional primeiro usa a rede neural convolucional para extrair características da imagem, depois transforma o número de canais no número de categorias por meio da camada de convolução $1\times 1$ e, finalmente, transforma a altura e largura do mapa de características para o tamanho da imagem de entrada usando a camada de convolução transposta para produzir a categoria de cada pixel.
* Em uma rede totalmente convolucional, inicializamos a camada de convolução transposta para interpolação bilinear com *upsampling*.


## Exercícios

1. Se usarmos Xavier para inicializar aleatoriamente a camada de convolução transposta, o que acontecerá com o resultado?
1. Você pode melhorar ainda mais a precisão do modelo ajustando os hiperparâmetros?
1. Preveja as categorias de todos os pixels na imagem de teste.
1. As saídas de algumas camadas intermediárias da rede neural convolucional também são usadas no artigo sobre redes totalmente convolucionais :cite:`Long.Shelhamer.Darrell.2015`. Tente implementar essa ideia.

:begin_tab:`mxnet`
[Discussões](https://discuss.d2l.ai/t/377)
:end_tab:

:begin_tab:`pytorch`
[Discussões](https://discuss.d2l.ai/t/1582)
:end_tab:
<!--stackedit_data:
eyJoaXN0b3J5IjpbNzA2ODU4NzIwLDEyNTQzNDE4NDQsNjE0NT
Y1NDkyLC03NTc0OTA3MDIsLTE0MzcxNjcwNDJdfQ==
-->