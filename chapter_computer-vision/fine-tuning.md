# Ajustes
:label:`sec_fine_tuning`


Nos capítulos anteriores, discutimos como treinar modelos no conjunto de dados de treinamento Fashion-MNIST, que tem apenas 60.000 imagens. Também descrevemos o ImageNet, o conjunto de dados de imagens em grande escala mais usado no mundo acadêmico, com mais de 10 milhões de imagens e objetos de mais de 1000 categorias. No entanto, o tamanho dos conjuntos de dados com os quais frequentemente lidamos é geralmente maior do que o primeiro, mas menor do que o segundo.

Suponha que queremos identificar diferentes tipos de cadeiras nas imagens e, em seguida, enviar o link de compra para o usuário. Um método possível é primeiro encontrar cem cadeiras comuns, obter mil imagens diferentes com diferentes ângulos para cada cadeira e, em seguida, treinar um modelo de classificação no conjunto de dados de imagens coletado. Embora esse conjunto de dados possa ser maior do que o Fashion-MNIST, o número de exemplos ainda é menor que um décimo do ImageNet. Isso pode resultar em sobreajuste do modelo complicado aplicável ao ImageNet neste conjunto de dados. Ao mesmo tempo, devido à quantidade limitada de dados, a precisão do modelo final treinado pode não atender aos requisitos práticos.


Para lidar com os problemas acima, uma solução óbvia é coletar mais dados. No entanto, coletar e rotular dados pode consumir muito tempo e dinheiro. Por exemplo, para coletar os conjuntos de dados ImageNet, os pesquisadores gastaram milhões de dólares em financiamento de pesquisa. Embora, recentemente, os custos de coleta de dados tenham caído significativamente, os custos ainda não podem ser ignorados.

Outra solução é aplicar o aprendizado de transferência para migrar o conhecimento aprendido do conjunto de dados de origem para o conjunto de dados de destino. Por exemplo, embora as imagens no ImageNet não tenham relação com cadeiras, os modelos treinados neste conjunto de dados podem extrair recursos de imagem mais gerais que podem ajudar a identificar bordas, texturas, formas e composição de objetos. Esses recursos semelhantes podem ser igualmente eficazes para o reconhecimento de uma cadeira.

Nesta seção, apresentamos uma técnica comum no aprendizado por transferência: o ajuste fino. Conforme mostrado em :numref:`fig_finetune, o ajuste fino consiste nas quatro etapas a seguir:

1. Pré-treine um modelo de rede neural, ou seja, o modelo de origem, em um conjunto de dados de origem (por exemplo, o conjunto de dados ImageNet).
2. Crie um novo modelo de rede neural, ou seja, o modelo de destino. Isso replica todos os designs de modelo e seus parâmetros no modelo de origem, exceto a camada de saída. Assumimos que esses parâmetros do modelo contêm o conhecimento aprendido com o conjunto de dados de origem e esse conhecimento será igualmente aplicável ao conjunto de dados de destino. Também assumimos que a camada de saída do modelo de origem está intimamente relacionada aos rótulos do conjunto de dados de origem e, portanto, não é usada no modelo de destino.
3. Adicione uma camada de saída cujo tamanho de saída é o número de categorias de conjunto de dados de destino ao modelo de destino e inicialize aleatoriamente os parâmetros do modelo desta camada.
4. Treine o modelo de destino em um conjunto de dados de destino, como um conjunto de dados de cadeira. Vamos treinar a camada de saída do zero, enquanto os parâmetros de todas as camadas restantes são ajustados com base nos parâmetros do modelo de origem.

![Fine tuning. ](../img/finetune.svg)
:label:`fig_finetune`


## Reconhecimento de Cachorro-quente


A seguir, usaremos um exemplo específico para prática: reconhecimento de cachorro-quente. Faremos o ajuste fino do modelo ResNet treinado no conjunto de dados ImageNet com base em um pequeno conjunto de dados. Este pequeno conjunto de dados contém milhares de imagens, algumas das quais contêm cachorros-quentes. Usaremos o modelo obtido pelo ajuste fino para identificar se uma imagem contém cachorro-quente.

Primeiro, importe os pacotes e módulos necessários para o experimento. O pacote `model_zoo` do Gluon fornece um modelo comum pré-treinado. Se você deseja obter mais modelos pré-treinados para visão computacional, você pode usar o [GluonCV Toolkit](https://gluon-cv.mxnet.io).

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import gluon, init, np, npx
from mxnet.gluon import nn
import os

npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
from torch import nn
import torch
import torchvision
import os
```

### Obtendo o *Dataset*

O conjunto de dados de cachorro-quente que usamos foi obtido de imagens online e contém $1.400$ imagens positivas contendo cachorros-quentes e o mesmo número de imagens negativas contendo outros alimentos. $1,000$ imagens de várias classes são usadas para treinamento e o resto é usado para teste.

Primeiro, baixamos o conjunto de dados compactado e obtemos duas pastas `hotdog/train` e `hotdog/test`. Ambas as pastas têm subpastas das categorias `hotdog` e `not-hotdog`, cada uma com arquivos de imagem correspondentes.

```{.python .input}
#@tab all
#@save
d2l.DATA_HUB['hotdog'] = (d2l.DATA_URL+'hotdog.zip', 
                         'fba480ffa8aa7e0febbb511d181409f899b9baa5')

data_dir = d2l.download_extract('hotdog')
```

Criamos duas instâncias `ImageFolderDataset` para ler todos os arquivos de imagem no conjunto de dados de treinamento e teste, respectivamente.

```{.python .input}
train_imgs = gluon.data.vision.ImageFolderDataset(
    os.path.join(data_dir, 'train'))
test_imgs = gluon.data.vision.ImageFolderDataset(
    os.path.join(data_dir, 'test'))
```

```{.python .input}
#@tab pytorch
train_imgs = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'train'))
test_imgs = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'test'))
```

Os primeiros 8 exemplos positivos e as últimas 8 imagens negativas são mostrados abaixo. Como você pode ver, as imagens variam em tamanho e proporção.

```{.python .input}
#@tab all
hotdogs = [train_imgs[i][0] for i in range(8)]
not_hotdogs = [train_imgs[-i - 1][0] for i in range(8)]
d2l.show_images(hotdogs + not_hotdogs, 2, 8, scale=1.4);
```

Durante o treinamento, primeiro recortamos uma área aleatória com tamanho e proporção aleatória da imagem e, em seguida, dimensionamos a área para uma entrada com altura e largura de 224 pixels. Durante o teste, dimensionamos a altura e a largura das imagens para 256 pixels e, em seguida, recortamos a área central com altura e largura de 224 pixels para usar como entrada. Além disso, normalizamos os valores dos três canais de cores RGB (vermelho, verde e azul). A média de todos os valores do canal é subtraída de cada valor e o resultado é dividido pelo desvio padrão de todos os valores do canal para produzir a saída.

```{.python .input}
# We specify the mean and variance of the three RGB channels to normalize the
# image channel
normalize = gluon.data.vision.transforms.Normalize(
    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

train_augs = gluon.data.vision.transforms.Compose([
    gluon.data.vision.transforms.RandomResizedCrop(224),
    gluon.data.vision.transforms.RandomFlipLeftRight(),
    gluon.data.vision.transforms.ToTensor(),
    normalize])

test_augs = gluon.data.vision.transforms.Compose([
    gluon.data.vision.transforms.Resize(256),
    gluon.data.vision.transforms.CenterCrop(224),
    gluon.data.vision.transforms.ToTensor(),
    normalize])
```

```{.python .input}
#@tab pytorch
# We specify the mean and variance of the three RGB channels to normalize the
# image channel
normalize = torchvision.transforms.Normalize(
    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

train_augs = torchvision.transforms.Compose([
    torchvision.transforms.RandomResizedCrop(224),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    normalize])

test_augs = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    normalize])
```

### Definindo e Inicializando o Modelo

Usamos o ResNet-18, que foi pré-treinado no conjunto de dados ImageNet, como modelo de origem. Aqui, especificamos `pretrained = True` para baixar e carregar automaticamente os parâmetros do modelo pré-treinado. Na primeira vez em que são usados, os parâmetros do modelo precisam ser baixados da Internet.

```{.python .input}
pretrained_net = gluon.model_zoo.vision.resnet18_v2(pretrained=True)
```

```{.python .input}
#@tab pytorch
pretrained_net = torchvision.models.resnet18(pretrained=True)
```

A instância do modelo de origem pré-treinada contém duas variáveis de membro: `features` e `output`. O primeiro contém todas as camadas do modelo, exceto a camada de saída, e o último é a camada de saída do modelo. O principal objetivo desta divisão é facilitar o ajuste fino dos parâmetros do modelo de todas as camadas, exceto a camada de saída. A variável membro `output` do modelo de origem é fornecida abaixo. Como uma camada totalmente conectada, ele transforma a saída final da camada de agrupamento média global do ResNet em uma saída de 1000 classes no conjunto de dados ImageNet.

```{.python .input}
pretrained_net.output
```

```{.python .input}
#@tab pytorch
pretrained_net.fc
```

Em seguida, construímos uma nova rede neural para usar como modelo-alvo. Ele é definido da mesma forma que o modelo de origem pré-treinado, mas o número final de saídas é igual ao número de categorias no conjunto de dados de destino. No código abaixo, os parâmetros do modelo na variável membro `features` da instância do modelo de destino `finetune_net` são inicializados para modelar os parâmetros da camada correspondente do modelo de origem. Como os parâmetros do modelo em `features` são obtidos por pré-treinamento no conjunto de dados ImageNet, é bom o suficiente. Portanto, geralmente só precisamos usar pequenas taxas de aprendizado para "ajustar" esses parâmetros. Em contraste, os parâmetros do modelo na variável membro `output` são inicializados aleatoriamente e geralmente requerem uma taxa de aprendizado maior para aprender do zero. Suponha que a taxa de aprendizado na instância `Trainer` seja $\eta$ e use uma taxa de aprendizado de $10\eta$ para atualizar os parâmetros do modelo na variável membro `output`.

```{.python .input}
finetune_net = gluon.model_zoo.vision.resnet18_v2(classes=2)
finetune_net.features = pretrained_net.features
finetune_net.output.initialize(init.Xavier())
# The model parameters in output will be updated using a learning rate ten
# times greater
finetune_net.output.collect_params().setattr('lr_mult', 10)
```

```{.python .input}
#@tab pytorch
finetune_net = torchvision.models.resnet18(pretrained=True)
finetune_net.fc = nn.Linear(finetune_net.fc.in_features, 2)
nn.init.xavier_uniform_(finetune_net.fc.weight);
# If `param_group=True`, the model parameters in fc layer will be updated 
# using a learning rate ten times greater, defined in the trainer.
```

### Ajustando o Modelo

Primeiro definimos uma função de treinamento `train_fine_tuning` que usa ajuste fino para que possa ser chamada várias vezes.

```{.python .input}
def train_fine_tuning(net, learning_rate, batch_size=128, num_epochs=5):
    train_iter = gluon.data.DataLoader(
        train_imgs.transform_first(train_augs), batch_size, shuffle=True)
    test_iter = gluon.data.DataLoader(
        test_imgs.transform_first(test_augs), batch_size)
    devices = d2l.try_all_gpus()
    net.collect_params().reset_ctx(devices)
    net.hybridize()
    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {
        'learning_rate': learning_rate, 'wd': 0.001})
    d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs,
                   devices)
```

```{.python .input}
#@tab pytorch
def train_fine_tuning(net, learning_rate, batch_size=128, num_epochs=5,
                      param_group=True):
    train_iter = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(
        os.path.join(data_dir, 'train'), transform=train_augs),
        batch_size=batch_size, shuffle=True)
    test_iter = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(
        os.path.join(data_dir, 'test'), transform=test_augs),
        batch_size=batch_size)
    devices = d2l.try_all_gpus()
    loss = nn.CrossEntropyLoss(reduction="none")
    if param_group:
        params_1x = [param for name, param in net.named_parameters()
             if name not in ["fc.weight", "fc.bias"]]
        trainer = torch.optim.SGD([{'params': params_1x},
                                   {'params': net.fc.parameters(),
                                    'lr': learning_rate * 10}],
                                lr=learning_rate, weight_decay=0.001)
    else:
        trainer = torch.optim.SGD(net.parameters(), lr=learning_rate,
                                  weight_decay=0.001)    
    d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs,
                   devices)
```

Definimos a taxa de aprendizado na instância `Trainer` para um valor menor, como 0,01, a fim de ajustar os parâmetros do modelo obtidos no pré-treinamento. Com base nas configurações anteriores, treinaremos os parâmetros da camada de saída do modelo de destino do zero, usando uma taxa de aprendizado dez vezes maior.

```{.python .input}
train_fine_tuning(finetune_net, 0.01)
```

```{.python .input}
#@tab pytorch
train_fine_tuning(finetune_net, 5e-5)
```

Para comparação, definimos um modelo idêntico, mas inicializamos todos os seus parâmetros de modelo para valores aleatórios. Como todo o modelo precisa ser treinado do zero, podemos usar uma taxa de aprendizado maior.

```{.python .input}
scratch_net = gluon.model_zoo.vision.resnet18_v2(classes=2)
scratch_net.initialize(init=init.Xavier())
train_fine_tuning(scratch_net, 0.1)
```

```{.python .input}
#@tab pytorch
scratch_net = torchvision.models.resnet18()
scratch_net.fc = nn.Linear(scratch_net.fc.in_features, 2)
train_fine_tuning(scratch_net, 5e-4, param_group=False)
```

Como você pode ver, o modelo ajustado tende a obter maior precisão na mesma época porque os valores iniciais dos parâmetros são melhores.

## Resumo


* A aprendizagem de transferência migra o conhecimento aprendido do conjunto de dados de origem para o conjunto de dados de destino. O ajuste fino é uma técnica comum para a aprendizagem por transferência.
* O modelo de destino replica todos os designs de modelo e seus parâmetros no modelo de origem, exceto a camada de saída, e ajusta esses parâmetros com base no conjunto de dados de destino. Em contraste, a camada de saída do modelo de destino precisa ser treinada do zero.
* Geralmente, os parâmetros de ajuste fino usam uma taxa de aprendizado menor, enquanto o treinamento da camada de saída do zero pode usar uma taxa de aprendizado maior.


## Exercícios

1. Continue aumentando a taxa de aprendizado de `finetune_net`. Como a precisão do modelo muda?
2. Ajuste ainda mais os hiperparâmetros de `finetune_net` e `scratch_net` no experimento comparativo. Eles ainda têm precisões diferentes?
3. Defina os parâmetros em `finetune_net.features` para os parâmetros do modelo de origem e não os atualize durante o treinamento. O que vai acontecer? Você pode usar o seguinte código.

```{.python .input}
finetune_net.features.collect_params().setattr('grad_req', 'null')
```

```{.python .input}
#@tab pytorch
for param in finetune_net.parameters():
    param.requires_grad = False
```

4. Na verdade, também existe uma classe "hotdog" no conjunto de dados `ImageNet`. Seu parâmetro de peso correspondente na camada de saída pode ser obtido usando o código a seguir. Como podemos usar este parâmetro?

```{.python .input}
weight = pretrained_net.output.weight
hotdog_w = np.split(weight.data(), 1000, axis=0)[713]
hotdog_w.shape
```

```{.python .input}
#@tab pytorch
weight = pretrained_net.fc.weight
hotdog_w = torch.split(weight.data, 1, dim=0)[713]
hotdog_w.shape
```

:begin_tab:`mxnet`
[Discussões](https://discuss.d2l.ai/t/368)
:end_tab:

:begin_tab:`pytorch`
[Discussões](https://discuss.d2l.ai/t/1439)
:end_tab:
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTQ0MzI0MDg2NSwtMTA0ODc0NTA1XX0=
-->