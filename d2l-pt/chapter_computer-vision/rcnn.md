# Region-based CNNs (R-CNNs)


Redes neurais convolucionais baseadas em regiões ou regiões com recursos CNN (R-CNNs) são uma abordagem pioneira que aplica modelos profundos para detecção de objetos :cite:`Girshick.Donahue.Darrell.ea.2014`. Nesta seção, discutiremos R-CNNs e uma série de melhorias feitas a eles: Fast R-CNN :cite:`Girshick.2015`, Faster R-CNN :cite:`Ren.He.Girshick.ea.2015` e Mask R-CNN
 :cite:`He.Gkioxari.Dollar.ea.2017`. Devido às limitações de espaço, limitaremos nossa discussão aos designs desses modelos.


## R-CNNs

Os modelos R-CNN primeiro selecionam várias regiões propostas de uma imagem (por exemplo, as caixas de âncora são um tipo de método de seleção) e, em seguida, rotulam suas categorias e caixas delimitadoras (por exemplo, deslocamentos). Em seguida, eles usam uma CNN para realizar cálculos avançados para extrair recursos de cada área proposta. Depois, usamos os recursos de cada região proposta para prever suas categorias e caixas delimitadoras. :numref:`fig_r-cnn` mostra um modelo R-CNN.

![Modelo R-CNN. ](../img/r-cnn.svg)
:label:`fig_r-cnn`

Especificamente, os R-CNNs são compostos por quatro partes principais:

1. A pesquisa seletiva é realizada na imagem de entrada para selecionar várias
   regiões propostas de alta qualidade
   :cite:`Uijlings.Van-De-Sande.Gevers.ea.2013`. Essas regiões propostas são
   geralmente selecionados em várias escalas e têm diferentes formas e
   tamanhos. A categoria e a caixa delimitadora da verdade fundamental de cada região proposta é
   etiquetada.
1. Um CNN pré-treinado é selecionado e colocado, de forma truncada, antes da
   camada de saída. Ele transforma cada região proposta nas dimensões de entrada
   exigido pela rede e usa computação direta para gerar os recursos
   extraídos das regiões propostas.
1. Os recursos e a categoria rotulada de cada região proposta são combinados como um
   exemplo para treinar várias máquinas de vetores de suporte para classificação de objeto.
   Aqui, cada máquina de vetor de suporte é usada para determinar
   se um exemplo pertence a uma determinada categoria.
1. Os recursos e a caixa delimitadora rotulada de cada região proposta são combinados como
   um exemplo para treinar um modelo de regressão linear para a predição da caixa delimitadora de verdade básica.

Embora os modelos R-CNN usem CNNs pré-treinados para extrair recursos de imagem com eficácia, a principal desvantagem é a velocidade lenta. Como você pode imaginar, podemos selecionar milhares de regiões propostas a partir de uma única imagem, exigindo milhares de cálculos diretos da CNN para realizar a detecção de objetos. Essa enorme carga de computação significa que os R-CNNs não são amplamente usados em aplicativos reais.


## Fast R-CNN

O principal gargalo de desempenho de um modelo R-CNN é a necessidade de extrair recursos de forma independente para cada região proposta. Como essas regiões têm um alto grau de sobreposição, a extração de recursos independentes resulta em um alto volume de cálculos repetitivos. Fast R-CNN melhora o R-CNN por apenas executar CNN
computação progressiva na imagem como um todo.

![Modelo Fast R-CNN.](../img/fast-rcnn.svg)
:label:`fig_fast_r-cnn`


:numref:`fig_fast_r-cnn` mostra um modelo Fast R-CNN. Suas principais etapas de computação são descritas abaixo:

1. Comparado a um modelo R-CNN, um modelo Fast R-CNN usa a imagem inteira como o
   Entrada da CNN para extração de recursos, em vez de cada região proposta. Além disso,
   esta rede é geralmente treinada para atualizar os parâmetros do modelo. Enquanto o
   a entrada é uma imagem inteira, a forma de saída do CNN é $1 \times c \times h_1
   \times w_1$.
1. Supondo que a pesquisa seletiva gere $n$ regiões propostas, seus diferentes
   formas indicam regiões de interesses (RoIs) de diferentes formas na CNN
   resultado. Características das mesmas formas devem ser extraídas dessas RoIs (aqui
   assumimos que a altura é $h_2$ e a largura é $w_2$). R-CNN rápido
   apresenta o pool de RoI, que usa a saída CNN e RoIs como entrada para saída
   uma concatenação dos recursos extraídos de cada região proposta com o
   forma $n \times c \times h_2 \times w_2$.
1. Uma camada totalmente conectada é usada para transformar a forma de saída em $n \times d$, onde $d$ é determinado pelo design do modelo.
1. Durante a previsão da categoria, a forma da saída da camada totalmente conectada é
   novamente transformada em $n \times q$ e usamos a regressão softmax ($q$ é o
   número de categorias). Durante a previsão da caixa delimitadora, a forma do
   a saída da camada conectada é novamente transformada em $n \times 4$. Isso significa que
   prevemos a categoria e a caixa delimitadora para cada região proposta.

A camada de pooling de RoI no Fast R-CNN é um pouco diferente das camadas de pool que discutimos antes. Em uma camada de pooling normal, definimos a janela de pool, preenchimento e passo para controlar a forma de saída. Em uma camada de pooling de RoI, podemos especificar diretamente a forma de saída de cada região, como especificar a altura e a largura de cada região como $h_2, w_2$. Supondo que o
altura e largura da janela RoI são $h$ e $w$, esta janela é dividida em uma grade de subjanelas com a forma h_2 \times w_2$. O tamanho de cada subjanela é de cerca de $(h/h_2) \times (w/w_2)$. A altura e largura da subjanela devem ser sempre inteiros e o maior elemento é usado como saída para um
determinada subjanela. Isso permite que a camada de pooling de RoI extraia recursos do mesmo formato de RoIs de formatos diferentes.

Em :numref:`fig_roi`, selecionamos uma região $3\times 3$ como um RoI da entrada $4 \times 4$. Para este RoI, usamos uma camada de pool de $2\times 2$ RoI para obter uma única saída $2\times 2$. Quando dividimos a região em quatro subjanelas, elas contêm respectivamente os elementos 0, 1, 4 e 5 (5 é o maior); 2 e 6 (6 é o maior); 8 e 9 (9 é o maior); e 10.

![Camada de *pooling* RoI $2 \times 2$ .](../img/roi.svg)
:label:`fig_roi`

:begin_tab:`mxnet`
Usamos a função `ROIPooling` para demonstrar a computação da camada de pooling RoI. Suponha que a CNN extraia o elemento `X` com altura e largura 4 e apenas um único canal.
:end_tab:

:begin_tab:`pytorch`
Usamos a função `roi_pool` de` torchvision` para demonstrar a computação da camada de pooling RoI. Suponha que a CNN extraia o elemento `X` com altura e largura 4 e apenas um único canal.
:end_tab:

```{.python .input}
from mxnet import np, npx

npx.set_np()

X = np.arange(16).reshape(1, 1, 4, 4)
X
```

```{.python .input}
#@tab pytorch
import torch
import torchvision

X = torch.arange(16.).reshape(1, 1, 4, 4)
X
```

Suponha que a altura e a largura da imagem sejam de 40 pixels e que a busca seletiva gere duas regiões propostas na imagem. Cada região é expressa como cinco elementos: a categoria de objeto da região e as coordenadas $x, y$ de seus cantos superior esquerdo e inferior direito.

```{.python .input}
rois = np.array([[0, 0, 0, 20, 20], [0, 0, 10, 30, 30]])
```

```{.python .input}
#@tab pytorch
rois = torch.Tensor([[0, 0, 0, 20, 20], [0, 0, 10, 30, 30]])
```

Como a altura e largura de `X` são $1/10$ da altura e largura da imagem, as coordenadas das duas regiões propostas são multiplicadas por 0,1 de acordo com `escala_espacial`, e então os RoIs são rotulados em `X` como `X[:,:, 0: 3, 0: 3]` e `X[:,:, 1: 4, 0: 4]`, respectivamente. Por fim, dividimos os dois RoIs em uma grade de subjanela e extraímos recursos com altura e largura 2.

```{.python .input}
npx.roi_pooling(X, rois, pooled_size=(2, 2), spatial_scale=0.1)
```

```{.python .input}
#@tab pytorch
torchvision.ops.roi_pool(X, rois, output_size=(2, 2), spatial_scale=0.1)
```

## R-CNN Mais Rápido

A fim de obter resultados precisos de detecção de objeto, Fast R-CNN geralmente requer que muitas regiões propostas sejam geradas em busca seletiva. O Faster R-CNN substitui a pesquisa seletiva por uma rede de proposta regional. Isso reduz o número de regiões propostas geradas, garantindo a detecção precisa do objeto.


![Modelo R-CNN mais rápido.](../img/faster-rcnn.svg)
:label:`fig_faster_r-cnn`


:numref:`fig_faster_r-cnn` mostra um modelo Faster R-CNN. Comparado ao Fast R-CNN, o Faster R-CNN apenas muda o método para gerar regiões propostas de pesquisa seletiva para rede de proposta de região. As outras partes do modelo permanecem inalteradas. O processo de computação de rede de proposta de região detalhada é descrito abaixo:

1. Usamos uma camada convolucional $3\times 3$ com um preenchimento de 1 para transformar a
    saída CNN e definir o número de canais de saída para $c$. Assim, cada
    elemento no mapa de recursos que a CNN extrai da imagem é um novo recurso
    com um comprimento de $c$.
1. Usamos cada elemento no mapa de recursos como um centro para gerar várias
    caixas de âncora de diferentes tamanhos e proporções de aspecto e, em seguida, etiquetá-las.
1. Usamos os recursos dos elementos de comprimento $c$ no centro da âncora
    caixas para prever a categoria binária (objeto ou fundo) e caixa delimitadora
    para suas respectivas caixas de âncora.
1. Em seguida, usamos a supressão não máxima para remover resultados semelhantes da caixa delimitadora
    que correspondem às previsões da categoria de "objeto". Finalmente, nós produzimos as
    caixas delimitadoras previstas como as regiões propostas exigidas pelo *pooling* de RoI
    camada.


É importante notar que, como parte do modelo R-CNN mais rápido, a rede proposta da região é treinada em conjunto com o resto do modelo. Além disso, as funções de objeto do Faster R-CNN incluem as predições de categoria e caixa delimitadora na detecção de objetos, bem como a categoria binária e predições de caixa delimitadora para as caixas de âncora na rede de proposta da região. Finalmente, a rede proposta de região pode aprender como gerar regiões propostas de alta qualidade, o que reduz o número de regiões propostas, enquanto mantém a precisão da detecção de objetos.


## Máscara R-CNN

Se os dados de treinamento forem rotulados com as posições de nível de pixel de cada objeto em uma imagem, um modelo Mask R-CNN pode usar efetivamente esses rótulos detalhados para melhorar ainda mais a precisão da detecção de objeto.

![Modelo de máscara R-CNN. ](../img/mask-rcnn.svg)
:label:`fig_mask_r-cnn`

Conforme mostrado em :numref:`fig_mask_r-cnn`, Mask R-CNN é uma modificação do modelo Faster R-CNN. Os modelos de máscara R-CNN substituem a camada de pooling RoI por uma camada de alinhamento RoI. Isso permite o uso de interpolação bilinear para reter informações espaciais em mapas de características, tornando o Mask R-CNN mais adequado para previsões em nível de pixel. A camada de alinhamento de RoI produz mapas de recursos do mesmo formato para todos os RoIs. Isso não apenas prevê as categorias e caixas delimitadoras de RoIs, mas nos permite usar uma rede totalmente convolucional adicional para prever as posições de objetos em nível de pixel. Descreveremos como usar redes totalmente convolucionais para prever a semântica em nível de pixel em imagens posteriormente neste capítulo.



## Resumo

* Um modelo R-CNN seleciona várias regiões propostas e usa um CNN para realizar computação direta e extrair os recursos de cada região proposta. Em seguida, usa esses recursos para prever as categorias e caixas delimitadoras das regiões propostas.
* Fast R-CNN melhora o R-CNN realizando apenas cálculos de encaminhamento de CNN na imagem como um todo. Ele apresenta uma camada de pooling de RoI para extrair recursos do mesmo formato de RoIs de formatos diferentes.
* O R-CNN mais rápido substitui a pesquisa seletiva usada no Fast R-CNN por uma rede de proposta de região. Isso reduz o número de regiões propostas geradas, garantindo a detecção precisa do objeto.
* Mask R-CNN usa a mesma estrutura básica que R-CNN mais rápido, mas adiciona uma camada de convolução completa para ajudar a localizar objetos no nível de pixel e melhorar ainda mais a precisão da detecção de objetos.


## Exercícios

1. Estude a implementação de cada modelo no [kit de ferramentas GluonCV](https://github.com/dmlc/gluon-cv/) relacionado a esta seção.

:begin_tab:`mxnet`
[Discussões](https://discuss.d2l.ai/t/374)
:end_tab:

:begin_tab:`pytorch`
[Discussões](https://discuss.d2l.ai/t/1409)
:end_tab:
<!--stackedit_data:
eyJoaXN0b3J5IjpbMTk5NDAwMjg5OCwtMTI0MTMwNDAyMSwtMT
M0MTM1ODA3NCwxMTY5MjE5MjAyLDE2MDQwMDkxMiwtMTQ5MzQw
MDY4NF19
-->