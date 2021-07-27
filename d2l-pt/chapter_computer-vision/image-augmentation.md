# Aumento de Imagem
:label:`sec_image_augmentation`



Mencionamos que conjuntos de dados em grande escala são pré-requisitos para a aplicação bem-sucedida de redes neurais profundas em :numref:`sec_alexnet`. A tecnologia de aumento de imagem expande a escala dos conjuntos de dados de treinamento, fazendo uma série de alterações aleatórias nas imagens de treinamento para produzir exemplos de treinamento semelhantes, mas diferentes. Outra maneira de explicar o aumento de imagem é
que exemplos de treinamento que mudam aleatoriamente podem reduzir a dependência de um modelo em certas propriedades, melhorando assim sua capacidade de generalização. Por exemplo, podemos recortar as imagens de diferentes maneiras, para que os objetos de interesse apareçam em diferentes posições, reduzindo a dependência do modelo da posição onde os objetos aparecem. Também podemos ajustar o brilho, a cor e outros fatores para reduzir a sensibilidade do modelo à cor. Pode-se dizer que a tecnologia de aumento de imagem contribuiu muito para o sucesso do AlexNet. Nesta seção, discutiremos essa tecnologia, que é amplamente usada na visão computacional.

Primeiro, importe os pacotes ou módulos necessários para o experimento nesta seção.

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, gluon, image, init, np, npx
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
```

## Método Comum de Aumento de Imagem

Neste experimento, usaremos uma imagem com um formato de $400\times 500$ como exemplo.

```{.python .input}
d2l.set_figsize()
img = image.imread('../img/cat1.jpg')
d2l.plt.imshow(img.asnumpy());
```

```{.python .input}
#@tab pytorch
d2l.set_figsize()
img = d2l.Image.open('../img/cat1.jpg')
d2l.plt.imshow(img);
```

A maioria dos métodos de aumento de imagem tem um certo grau de aleatoriedade. Para facilitar a observação do efeito do aumento da imagem, definimos a seguir a função auxiliar `aplicar`. Esta função executa o método de aumento de imagem `aug` várias vezes na imagem de entrada` img` e mostra todos os resultados.

```{.python .input}
#@tab all
def apply(img, aug, num_rows=2, num_cols=4, scale=1.5):
    Y = [aug(img) for _ in range(num_rows * num_cols)]
    d2l.show_images(Y, num_rows, num_cols, scale=scale)
```

### Invertendo e Recortando

Virar a imagem para a esquerda e para a direita geralmente não altera a categoria do objeto. Este é um dos métodos mais antigos e mais amplamente usados de aumento de imagem. Em seguida, usamos o módulo `transforms` para criar a instância `RandomFlipLeftRight`, que apresenta uma chance de 50% de que a imagem seja virada para a esquerda e para a direita.

```{.python .input}
apply(img, gluon.data.vision.transforms.RandomFlipLeftRight())
```

```{.python .input}
#@tab pytorch
apply(img, torchvision.transforms.RandomHorizontalFlip())
```

Virar para cima e para baixo não é tão comumente usado como girar para a esquerda e para a direita. No entanto, pelo menos para esta imagem de exemplo, virar para cima e para baixo não impede o reconhecimento. Em seguida, criamos uma instância `RandomFlipTopBottom` para uma chance de 50% de virar a imagem para cima e para baixo.

```{.python .input}
apply(img, gluon.data.vision.transforms.RandomFlipTopBottom())
```

```{.python .input}
#@tab pytorch
apply(img, torchvision.transforms.RandomVerticalFlip())
```


Na imagem de exemplo que usamos, o gato está no meio da imagem, mas pode não ser o caso para todas as imagens. Em
:numref:`sec_pooling`, explicamos que a camada de pooling pode reduzir a sensibilidade da camada convolucional ao local de destino. Além disso, podemos fazer os objetos aparecerem em diferentes posições na imagem em diferentes proporções aleatoriamente
recortando a imagem. Isso também pode reduzir a sensibilidade do modelo à posição de destino.

No código a seguir, recortamos aleatoriamente uma região com uma área de 10% a 100% da área original, e a proporção entre largura e altura da região é selecionada aleatoriamente entre 0,5 e 2. Em seguida, a largura e a altura de as regiões são dimensionadas para 200 pixels. Salvo indicação em contrário, o número aleatório entre $a$ e $b$ nesta seção refere-se a um valor contínuo obtido por amostragem uniforme no intervalo $[a, b]$.

```{.python .input}
shape_aug = gluon.data.vision.transforms.RandomResizedCrop(
    (200, 200), scale=(0.1, 1), ratio=(0.5, 2))
apply(img, shape_aug)
```

```{.python .input}
#@tab pytorch
shape_aug = torchvision.transforms.RandomResizedCrop(
    (200, 200), scale=(0.1, 1), ratio=(0.5, 2))
apply(img, shape_aug)
```

### Mudando a Cor

Outro método de aumento é mudar as cores. Podemos alterar quatro aspectos da cor da imagem: brilho, contraste, saturação e matiz. No exemplo abaixo, alteramos aleatoriamente o brilho da imagem para um valor entre 50% ($1-0.5$) e 150% ($1+0.5$) da imagem original.

```{.python .input}
apply(img, gluon.data.vision.transforms.RandomBrightness(0.5))
```

```{.python .input}
#@tab pytorch
apply(img, torchvision.transforms.ColorJitter(
    brightness=0.5, contrast=0, saturation=0, hue=0))
```

Da mesma forma, podemos alterar aleatoriamente o matiz da imagem.

```{.python .input}
apply(img, gluon.data.vision.transforms.RandomHue(0.5))
```

```{.python .input}
#@tab pytorch
apply(img, torchvision.transforms.ColorJitter(
    brightness=0, contrast=0, saturation=0, hue=0.5))
```

Também podemos criar uma instância `RandomColorJitter` e definir como alterar aleatoriamente o `brightness`, `contrast`, `saturation`, e `hue` da imagem ao mesmo tempo.

```{.python .input}
color_aug = gluon.data.vision.transforms.RandomColorJitter(
    brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
apply(img, color_aug)
```

```{.python .input}
#@tab pytorch
color_aug = torchvision.transforms.ColorJitter(
    brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
apply(img, color_aug)
```

### Métodos de Aumento de Imagem Múltipla Sobreposta

Na prática, iremos sobrepor vários métodos de aumento de imagem. Podemos sobrepor os diferentes métodos de aumento de imagem definidos acima e aplicá-los a cada imagem usando uma instância `Compose`.

```{.python .input}
augs = gluon.data.vision.transforms.Compose([
    gluon.data.vision.transforms.RandomFlipLeftRight(), color_aug, shape_aug])
apply(img, augs)
```

```{.python .input}
#@tab pytorch
augs = torchvision.transforms.Compose([
    torchvision.transforms.RandomHorizontalFlip(), color_aug, shape_aug])
apply(img, augs)
```

## Usando um Modelo de Treinamento de Aumento de Imagem

A seguir, veremos como aplicar o aumento de imagem no treinamento real. Aqui, usamos o conjunto de dados CIFAR-10, em vez do conjunto de dados Fashion-MNIST que usamos. Isso ocorre porque a posição e o tamanho dos objetos no conjunto de dados Fashion-MNIST foram normalizados, e as diferenças de cor e tamanho dos objetos no conjunto de dados CIFAR-10 são mais significativas. As primeiras 32 imagens de treinamento no conjunto de dados CIFAR-10 são mostradas abaixo.

```{.python .input}
d2l.show_images(gluon.data.vision.CIFAR10(
    train=True)[0:32][0], 4, 8, scale=0.8);
```

```{.python .input}
#@tab pytorch
all_images = torchvision.datasets.CIFAR10(train=True, root="../data",
                                          download=True)
d2l.show_images([all_images[i][0] for i in range(32)], 4, 8, scale=0.8);
```

Para obter resultados definitivos durante a previsão, geralmente aplicamos apenas o aumento da imagem ao exemplo de treinamento e não usamos o aumento da imagem com operações aleatórias durante a previsão. Aqui, usamos apenas o método de inversão aleatório da esquerda para a direita mais simples. Além disso, usamos uma instância `ToTensor` para converter imagens de minibatch no formato exigido pelo MXNet, ou seja, números de ponto flutuante de 32 bits com a forma de (tamanho do lote, número de canais, altura, largura) e intervalo de valores entre 0 e 1.

```{.python .input}
train_augs = gluon.data.vision.transforms.Compose([
    gluon.data.vision.transforms.RandomFlipLeftRight(),
    gluon.data.vision.transforms.ToTensor()])

test_augs = gluon.data.vision.transforms.Compose([
    gluon.data.vision.transforms.ToTensor()])
```

```{.python .input}
#@tab pytorch
train_augs = torchvision.transforms.Compose([
     torchvision.transforms.RandomHorizontalFlip(),
     torchvision.transforms.ToTensor()])

test_augs = torchvision.transforms.Compose([
     torchvision.transforms.ToTensor()])
```

A seguir, definimos uma função auxiliar para facilitar a leitura da imagem e aplicar o aumento da imagem. A função `transform_first` fornecida pelo conjunto de dados do Gluon aplica o aumento da imagem ao primeiro elemento de cada exemplo de treinamento (imagem e rótulo), ou seja, o elemento na parte superior da imagem. Para descrições detalhadas de `DataLoader`, consulte :numref:`sec_fashion_mnist`.

```{.python .input}
def load_cifar10(is_train, augs, batch_size):
    return gluon.data.DataLoader(
        gluon.data.vision.CIFAR10(train=is_train).transform_first(augs),
        batch_size=batch_size, shuffle=is_train,
        num_workers=d2l.get_dataloader_workers())
```

```{.python .input}
#@tab pytorch
def load_cifar10(is_train, augs, batch_size):
    dataset = torchvision.datasets.CIFAR10(root="../data", train=is_train,
                                           transform=augs, download=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                    shuffle=is_train, num_workers=d2l.get_dataloader_workers())
    return dataloader
```

### Usando um Modelo de Treinamento Multi-GPU


Treinamos o modelo ResNet-18 descrito em: numref: `sec_resnet`
no conjunto de dados CIFAR-10. Também aplicaremos os métodos descritos em :numref:`sec_multi_gpu_concise` e usaremos um modelo de treinamento multi-GPU.

Em seguida, definimos a função de treinamento para treinar e avaliar o modelo usando várias GPUs.

```{.python .input}
#@save
def train_batch_ch13(net, features, labels, loss, trainer, devices,
                     split_f=d2l.split_batch):
    X_shards, y_shards = split_f(features, labels, devices)
    with autograd.record():
        pred_shards = [net(X_shard) for X_shard in X_shards]
        ls = [loss(pred_shard, y_shard) for pred_shard, y_shard
              in zip(pred_shards, y_shards)]
    for l in ls:
        l.backward()
    # The True flag allows parameters with stale gradients, which is useful
    # later (e.g., in fine-tuning BERT)
    trainer.step(labels.shape[0], ignore_stale_grad=True)
    train_loss_sum = sum([float(l.sum()) for l in ls])
    train_acc_sum = sum(d2l.accuracy(pred_shard, y_shard)
                        for pred_shard, y_shard in zip(pred_shards, y_shards))
    return train_loss_sum, train_acc_sum
```

```{.python .input}
#@tab pytorch
#@save
def train_batch_ch13(net, X, y, loss, trainer, devices):
    if isinstance(X, list):
        # Required for BERT Fine-tuning (to be covered later)
        X = [x.to(devices[0]) for x in X]
    else:
        X = X.to(devices[0])
    y = y.to(devices[0])
    net.train()
    trainer.zero_grad()
    pred = net(X)
    l = loss(pred, y)
    l.sum().backward()
    trainer.step()
    train_loss_sum = l.sum()
    train_acc_sum = d2l.accuracy(pred, y)
    return train_loss_sum, train_acc_sum
```

```{.python .input}
#@save
def train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs,
               devices=d2l.try_all_gpus(), split_f=d2l.split_batch):
    timer, num_batches = d2l.Timer(), len(train_iter)
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 1],
                            legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        # Store training_loss, training_accuracy, num_examples, num_features
        metric = d2l.Accumulator(4)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            l, acc = train_batch_ch13(
                net, features, labels, loss, trainer, devices, split_f)
            metric.add(l, acc, labels.shape[0], labels.size)
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[2], metric[1] / metric[3],
                              None))
        test_acc = d2l.evaluate_accuracy_gpus(net, test_iter, split_f)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {metric[0] / metric[2]:.3f}, train acc '
          f'{metric[1] / metric[3]:.3f}, test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec on '
          f'{str(devices)}')
```

```{.python .input}
#@tab pytorch
#@save
def train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs,
               devices=d2l.try_all_gpus()):
    timer, num_batches = d2l.Timer(), len(train_iter)
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 1],
                            legend=['train loss', 'train acc', 'test acc'])
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    for epoch in range(num_epochs):
        # Store training_loss, training_accuracy, num_examples, num_features
        metric = d2l.Accumulator(4)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            l, acc = train_batch_ch13(
                net, features, labels, loss, trainer, devices)
            metric.add(l, acc, labels.shape[0], labels.numel())
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[2], metric[1] / metric[3],
                              None))
        test_acc = d2l.evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {metric[0] / metric[2]:.3f}, train acc '
          f'{metric[1] / metric[3]:.3f}, test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec on '
          f'{str(devices)}')
```

Agora, podemos definir a função `train_with_data_aug` para usar o aumento da imagem para treinar o modelo. Esta função obtém todas as GPUs disponíveis e usa Adam como o algoritmo de otimização para o treinamento. Em seguida, ele aplica o aumento da imagem ao conjunto de dados de treinamento e, finalmente, chama a função `train_ch13` definida para treinar e avaliar o modelo.

```{.python .input}
batch_size, devices, net = 256, d2l.try_all_gpus(), d2l.resnet18(10)
net.initialize(init=init.Xavier(), ctx=devices)

def train_with_data_aug(train_augs, test_augs, net, lr=0.001):
    train_iter = load_cifar10(True, train_augs, batch_size)
    test_iter = load_cifar10(False, test_augs, batch_size)
    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net.collect_params(), 'adam',
                            {'learning_rate': lr})
    train_ch13(net, train_iter, test_iter, loss, trainer, 10, devices)
```

```{.python .input}
#@tab pytorch
batch_size, devices, net = 256, d2l.try_all_gpus(), d2l.resnet18(10, 3)

def init_weights(m):
    if type(m) in [nn.Linear, nn.Conv2d]:
        nn.init.xavier_uniform_(m.weight)

net.apply(init_weights)

def train_with_data_aug(train_augs, test_augs, net, lr=0.001):
    train_iter = load_cifar10(True, train_augs, batch_size)
    test_iter = load_cifar10(False, test_augs, batch_size)
    loss = nn.CrossEntropyLoss(reduction="none")
    trainer = torch.optim.Adam(net.parameters(), lr=lr)
    train_ch13(net, train_iter, test_iter, loss, trainer, 10, devices)
```

Now we train the model using image augmentation of random flipping left and right.

```{.python .input}
#@tab all
train_with_data_aug(train_augs, test_augs, net)
```

## Resumo

* O aumento de imagem gera imagens aleatórias com base nos dados de treinamento existentes para lidar com o sobreajuste.
* Para obter resultados definitivos durante a previsão, geralmente aplicamos apenas o aumento da imagem ao exemplo de treinamento e não usamos o aumento da imagem com operações aleatórias durante a previsão.
* Podemos obter classes relacionadas ao aumento de imagem do módulo `transforma` do Gluon.

## Exercícios

1. Treine o modelo sem usar aumento de imagem: `train_with_data_aug (no_aug, no_aug)`. Compare a precisão do treinamento e do teste ao usar e não usar o aumento de imagem. Este experimento comparativo pode apoiar o argumento de que o aumento da imagem pode mitigar o sobreajuste? Por quê?
1. Adicione diferentes métodos de aumento de imagem no treinamento do modelo com base no *dataset* CIFAR-10. Observe os resultados da implementação.
1. Com referência à documentação do MXNet, que outros métodos de aumento de imagem são fornecidos no módulo `transforms` do Gluon?

:begin_tab:`mxnet`
[Discussões](https://discuss.d2l.ai/t/367)
:end_tab:

:begin_tab:`pytorch`
[Discussões](https://discuss.d2l.ai/t/1404)
:end_tab:
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTUyMzI0OTc3OSwtMTY5MjkyMDA3MCw2OD
Y5OTQ3NzAsMTEzNTA2OTY0LDE3MDMwNzI5NzldfQ==
-->