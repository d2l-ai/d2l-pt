# Detecção de Objetos Multiescala


Em :numref:`sec_anchor`, geramos várias caixas de âncora centralizadas em cada pixel da imagem de entrada. Essas caixas de âncora são usadas para amostrar diferentes regiões da imagem de entrada. No entanto, se as caixas de âncora forem geradas centralizadas em cada pixel da imagem, logo haverá muitas caixas de âncora para calcularmos. Por exemplo, assumimos que a imagem de entrada tem uma altura e uma largura de 561 e 728 pixels, respectivamente. Se cinco formas diferentes de caixas de âncora são geradas centralizadas em cada pixel, mais de dois milhões de caixas de âncora ($561 \times 728 \times 5$) precisam ser previstas e rotuladas na imagem.

Não é difícil reduzir o número de caixas de âncora. Uma maneira fácil é aplicar amostragem uniforme em uma pequena parte dos pixels da imagem de entrada e gerar caixas de âncora centralizadas nos pixels amostrados. Além disso, podemos gerar caixas de âncora de números e tamanhos variados em várias escalas. Observe que é mais provável que objetos menores sejam posicionados na imagem do que objetos maiores. Aqui, usaremos um exemplo simples: Objetos com formas de $1 \times 1$, $1 \times 2$, e $2 \times 2$ podem ter 4, 2 e 1 posição(ões) possível(is) em uma imagem com a forma $2 \times 2$.. Portanto, ao usar caixas de âncora menores para detectar objetos menores, podemos amostrar mais regiões; ao usar caixas de âncora maiores para detectar objetos maiores, podemos amostrar menos regiões.

Para demonstrar como gerar caixas de âncora em várias escalas, vamos ler uma imagem primeiro. Tem uma altura e largura de $561 \times 728$ pixels.

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import image, np, npx

npx.set_np()

img = image.imread('../img/catdog.jpg')
h, w = img.shape[0:2]
h, w
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch

img = d2l.plt.imread('../img/catdog.jpg')
h, w = img.shape[0:2]
h, w
```


Em :numref:`sec_conv_layer`, a saída da matriz 2D da rede neural convolucional (CNN) é chamada
um mapa de recursos. Podemos determinar os pontos médios de caixas de âncora uniformemente amostradas
em qualquer imagem, definindo a forma do mapa de feições.

A função `display_anchors` é definida abaixo. Vamos gerar caixas de âncora `anchors` centradas em cada unidade (pixel) no mapa de feições `fmap`. Uma vez que as coordenadas dos eixos $x$ e $y$ nas caixas de âncora `anchors` foram divididas pela largura e altura do mapa de feições `fmap`, valores entre 0 e 1 podem ser usados ​​para representar as posições relativas das caixas de âncora em o mapa de recursos. Uma vez que os pontos médios das "âncoras" das caixas de âncora se sobrepõem a todas as unidades no mapa de características "fmap", as posições espaciais relativas dos pontos médios das "âncoras" em qualquer imagem devem ter uma distribuição uniforme. Especificamente, quando a largura e a altura do mapa de feições são definidas para `fmap_w` e` fmap_h` respectivamente, a função irá conduzir uma amostragem uniforme para linhas `fmap_h` e colunas de pixels` fmap_w` e usá-los como pontos médios para gerar caixas de âncora com tamanho `s` (assumimos que o comprimento da lista `s` é 1) e diferentes proporções (`ratios`).

```{.python .input}
def display_anchors(fmap_w, fmap_h, s):
    d2l.set_figsize()
    # The values from the first two dimensions will not affect the output
    fmap = np.zeros((1, 10, fmap_h, fmap_w))
    anchors = npx.multibox_prior(fmap, sizes=s, ratios=[1, 2, 0.5])
    bbox_scale = np.array((w, h, w, h))
    d2l.show_bboxes(d2l.plt.imshow(img.asnumpy()).axes,
                    anchors[0] * bbox_scale)
```

```{.python .input}
#@tab pytorch
def display_anchors(fmap_w, fmap_h, s):
    d2l.set_figsize()
    # The values from the first two dimensions will not affect the output
    fmap = d2l.zeros((1, 10, fmap_h, fmap_w))
    anchors = d2l.multibox_prior(fmap, sizes=s, ratios=[1, 2, 0.5])
    bbox_scale = d2l.tensor((w, h, w, h))
    d2l.show_bboxes(d2l.plt.imshow(img).axes,
                    anchors[0] * bbox_scale)
```

Primeiro, vamos nos concentrar na detecção de pequenos objetos. A fim de tornar mais fácil distinguir na exibição, as caixas de âncora com diferentes pontos médios aqui não se sobrepõem. Assumimos que o tamanho das caixas de âncora é 0,15 e a altura e largura do mapa de feições são 4. Podemos ver que os pontos médios das caixas de âncora das 4 linhas e 4 colunas da imagem estão uniformemente distribuídos.

```{.python .input}
#@tab all
display_anchors(fmap_w=4, fmap_h=4, s=[0.15])
```

Vamos reduzir a altura e a largura do mapa de feições pela metade e usar uma caixa de âncora maior para detectar objetos maiores. Quando o tamanho é definido como 0,4, ocorrerão sobreposições entre as regiões de algumas caixas de âncora.

```{.python .input}
#@tab all
display_anchors(fmap_w=2, fmap_h=2, s=[0.4])
```

Finalmente, vamos reduzir a altura e a largura do mapa de feições pela metade e aumentar o tamanho da caixa de âncora para 0,8. Agora, o ponto médio da caixa de âncora é o centro da imagem.

```{.python .input}
#@tab all
display_anchors(fmap_w=1, fmap_h=1, s=[0.8])
```


Como geramos caixas de âncora de tamanhos diferentes em escalas múltiplas, vamos usá-las para detectar objetos de vários tamanhos em escalas diferentes. Agora vamos apresentar um método baseado em redes neurais convolucionais (CNNs).

Em uma determinada escala, suponha que geramos $h \times w$ conjuntos de caixas de âncora com diferentes pontos médios baseados em $c_i$ mapas de feições com a forma $h \times w$ e o número de caixas de âncora em cada conjunto é $a$ . Por exemplo, para a primeira escala do experimento, geramos 16 conjuntos de caixas de âncora com diferentes pontos médios com base em 10 (número de canais) mapas de recursos com uma forma de $4 \times 4$, e cada conjunto contém 3 caixas de âncora.
A seguir, cada caixa de âncora é rotulada com uma categoria e deslocamento com base na classificação e posição da caixa delimitadora de verdade. Na escala atual, o modelo de detecção de objeto precisa prever a categoria e o offset de $h \times w$ conjuntos de caixas de âncora com diferentes pontos médios com base na imagem de entrada.

Assumimos que os mapas de características $c_i$ são a saída intermediária da CNN com base na imagem de entrada. Uma vez que cada mapa de características tem $h \times w$ posições espaciais diferentes, a mesma posição terá $c_i$ unidades. De acordo com a definição de campo receptivo em :numref:`sec_conv_layer`, as unidades $c_i$ do mapa de feições na mesma posição espacial têm o mesmo campo receptivo na imagem de entrada. Assim, elas representam a
informação da imagem de entrada neste mesmo campo receptivo. Portanto, podemos transformar as unidades $c_i$ do mapa de feições na mesma posição espacial nas categorias e deslocamentos das $a$ caixas de âncora geradas usando essa posição como um ponto médio. Não é difícil perceber que, em essência, usamos as informações da imagem de entrada em um determinado campo receptivo para prever a categoria e
deslocamento das caixas de âncora perto do campo na imagem de entrada.


Quando os mapas de recursos de camadas diferentes têm campos receptivos de tamanhos diferentes na imagem de entrada, eles são usados para detectar objetos de tamanhos diferentes. Por exemplo, podemos projetar uma rede para ter um campo receptivo mais amplo para cada unidade no mapa de recursos que está mais perto da camada de saída, para detectar objetos com tamanhos maiores na imagem de entrada.

Implementaremos um modelo de detecção de objetos multiescala na seção seguinte.


## Resumo

* Podemos gerar caixas de âncora com diferentes números e tamanhos em várias escalas para detectar objetos de diferentes tamanhos em várias escalas.
* A forma do mapa de feições pode ser usada para determinar o ponto médio das caixas de âncora que amostram uniformemente qualquer imagem.
* Usamos as informações da imagem de entrada de um determinado campo receptivo para prever a categoria e o deslocamento das caixas de âncora próximas a esse campo na imagem.


## Exercícios

1. Dada uma imagem de entrada, suponha que $1 \times c_i \times h \times w$ seja a forma do mapa de características, enquanto $c_i, h, w$ são o número, altura e largura do mapa de características. Em quais métodos você pode pensar para converter essa variável na categoria e deslocamento da caixa de âncora? Qual é o formato da saída?

:begin_tab:`mxnet`
[Discussões](https://discuss.d2l.ai/t/371)
:end_tab:

:begin_tab:`pytorch`
[Discussões](https://discuss.d2l.ai/t/1607)
:end_tab:
<!--stackedit_data:
eyJoaXN0b3J5IjpbMTc3NzAwODA1OSwtMjAxMzI3NTgyNSwxOD
U1NjY2NzY3LDIwNTY0MDk1NzAsLTE3ODMzMjAzMF19
-->