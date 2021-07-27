# Detecção de Objetos e Caixas Delimitadoras
:label:`sec_bbox`



Na seção anterior, apresentamos muitos modelos para classificação de imagens. Nas tarefas de classificação de imagens, presumimos que haja apenas uma característica principal na imagem e nos concentramos apenas em como identificar a categoria de destino. No entanto, em muitas situações, existem várias características na imagem que nos interessam. Não queremos apenas classificá-las, mas também queremos obter suas posições específicas na imagem. Na visão computacional, nos referimos a tarefas como detecção de objetos (ou reconhecimento de objetos).

A detecção de objetos é amplamente usada em muitos campos. Por exemplo, na tecnologia de direção autônoma, precisamos planejar rotas identificando a localização de veículos, pedestres, estradas e obstáculos na imagem de vídeo capturada. Os robôs geralmente realizam esse tipo de tarefa para detectar alvos de interesse. Os sistemas no campo da segurança precisam detectar alvos anormais, como intrusos ou bombas.

Nas próximas seções, apresentaremos vários modelos de aprendizado profundo usados ​​para detecção de objetos. Antes disso, devemos discutir o conceito de localização de destino. Primeiro, importe os pacotes e módulos necessários para o experimento.


```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import image, npx, np

npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
```

A seguir, carregaremos as imagens de amostra que serão usadas nesta seção. Podemos ver que há um cachorro no lado esquerdo da imagem e um gato no lado direito. Eles são os dois alvos principais desta imagem.

```{.python .input}
d2l.set_figsize()
img = image.imread('../img/catdog.jpg').asnumpy()
d2l.plt.imshow(img);
```

```{.python .input}
#@tab pytorch, tensorflow
d2l.set_figsize()
img = d2l.plt.imread('../img/catdog.jpg')
d2l.plt.imshow(img);
```

## Caixa Delimitadora

Na detecção de objetos, geralmente usamos uma caixa delimitadora para descrever o local de destino.
A caixa delimitadora é uma caixa retangular que pode ser determinada pelas coordenadas dos eixos $x$ e $y$ no canto superior esquerdo e pelas coordenadas dos eixos $x$ e $y$ no canto inferior direito do retângulo.
Outra representação de caixa delimitadora comumente usada são as coordenadas dos eixos $x$ e $y$ do centro da caixa delimitadora e sua largura e altura.
Aqui definimos funções para converter entre esses dois
representações, `box_corner_to_center` converte da representação de dois cantos para a apresentação centro-largura-altura,
e vice-versa `box_center_to_corner`.
O argumento de entrada `boxes` pode ter um tensor de comprimento $4$,
ou um tensor $(N, 4)$ 2-dimensional.

```{.python .input}
#@tab all
#@save
def box_corner_to_center(boxes):
    """Convert from (upper_left, bottom_right) to (center, width, height)"""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    boxes = d2l.stack((cx, cy, w, h), axis=-1)
    return boxes

#@save
def box_center_to_corner(boxes):
    """Convert from (center, width, height) to (upper_left, bottom_right)"""
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    boxes = d2l.stack((x1, y1, x2, y2), axis=-1)
    return boxes
```

Vamos definir as caixas delimitadoras do cão e do gato na imagem baseada
nas informações de coordenadas. A origem das coordenadas na imagem
é o canto superior esquerdo da imagem, e para a direita e para baixo estão as
direções positivas do eixo $x$ e do eixo $y$, respectivamente.

```{.python .input}
#@tab all
# bbox is the abbreviation for bounding box
dog_bbox, cat_bbox = [60.0, 45.0, 378.0, 516.0], [400.0, 112.0, 655.0, 493.0]
```

Podemos verificar a exatidão das funções de conversão da caixa convertendo duas vezes.

```{.python .input}
#@tab all
boxes = d2l.tensor((dog_bbox, cat_bbox))
box_center_to_corner(box_corner_to_center(boxes)) - boxes
```

Podemos desenhar a caixa delimitadora na imagem para verificar se ela é precisa. Antes de desenhar a caixa, definiremos uma função auxiliar `bbox_to_rect`. Ele representa a caixa delimitadora no formato de caixa delimitadora de `matplotlib`.

```{.python .input}
#@tab all
#@save
def bbox_to_rect(bbox, color):
    """Convert bounding box to matplotlib format."""
    # Convert the bounding box (top-left x, top-left y, bottom-right x,
    # bottom-right y) format to matplotlib format: ((upper-left x,
    # upper-left y), width, height)
    return d2l.plt.Rectangle(
        xy=(bbox[0], bbox[1]), width=bbox[2]-bbox[0], height=bbox[3]-bbox[1],
        fill=False, edgecolor=color, linewidth=2)
```

Depois de carregar a caixa delimitadora na imagem, podemos ver que o contorno principal do alvo está basicamente dentro da caixa.

```{.python .input}
#@tab all
fig = d2l.plt.imshow(img)
fig.axes.add_patch(bbox_to_rect(dog_bbox, 'blue'))
fig.axes.add_patch(bbox_to_rect(cat_bbox, 'red'));
```

## Resumo

* Na detecção de objetos, não precisamos apenas identificar todos os objetos de interesse na imagem, mas também suas posições. As posições são geralmente representadas por uma caixa delimitadora retangular.

## Exercícios

1. Encontre algumas imagens e tente rotular uma caixa delimitadora que contém o alvo. Compare a diferença entre o tempo que leva para rotular a caixa delimitadora e rotular a categoria.

:begin_tab:`mxnet`
[Discussões](https://discuss.d2l.ai/t/369)
:end_tab:

:begin_tab:`pytorch`
[Discussões](https://discuss.d2l.ai/t/1527)
:end_tab:
<!--stackedit_data:
eyJoaXN0b3J5IjpbNjI2MTI5NDMzLDc5NDU2NjI2NywtNTE3Nj
U3NzJdfQ==
-->