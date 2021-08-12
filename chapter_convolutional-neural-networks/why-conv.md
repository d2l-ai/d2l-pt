# De Camadas Totalmente Conectadas às Convoluções
:label:`sec_why-conv`


Até hoje,
os modelos que discutimos até agora
permanecem opções apropriadas
quando estamos lidando com dados tabulares.
Por tabular, queremos dizer que os dados consistem
de linhas correspondentes a exemplos
e colunas correspondentes a *features*.
Com dados tabulares, podemos antecipar
que os padrões que buscamos podem envolver
interações entre as características,
mas não assumimos nenhuma estrutura *a priori*
sobre como as características interagem.

Às vezes, realmente não temos conhecimento para orientar
a construção de arquiteturas mais artesanais.
Nestes casos, um MLP
pode ser o melhor que podemos fazer.
No entanto, para dados perceptivos de alta dimensão,
essas redes sem estrutura podem se tornar difíceis de manejar.

Por exemplo, vamos voltar ao nosso exemplo de execução
de distinguir gatos de cães.
Digamos que fazemos um trabalho completo na coleta de dados,
coletando um conjunto de dados anotado de fotografias de um megapixel.
Isso significa que cada entrada na rede tem um milhão de dimensões.
De acordo com nossas discussões sobre custo de parametrização
de camadas totalmente conectadas em :numref:`subsec_parameterization-cost-fc-layers`,
até mesmo uma redução agressiva para mil dimensões ocultas
exigiria uma camada totalmente conectada
caracterizada por $10^6 \times 10^3 = 10^9$ parâmetros.
A menos que tenhamos muitas GPUs, um talento
para otimização distribuída,
e uma quantidade extraordinária de paciência,
aprender os parâmetros desta rede
pode acabar sendo inviável.

Um leitor cuidadoso pode objetar a este argumento
na base de que a resolução de um megapixel pode não ser necessária.
No entanto, embora possamos ser capazes
de escapar com apenas cem mil pixels,
nossa camada oculta de tamanho 1000 subestima grosseiramente
o número de unidades ocultas que leva
para aprender boas representações de imagens,
portanto, um sistema prático ainda exigirá bilhões de parâmetros.
Além disso, aprender um classificador ajustando tantos parâmetros
pode exigir a coleta de um enorme conjunto de dados.
E ainda hoje tanto os humanos quanto os computadores são capazes
de distinguir gatos de cães muito bem,
aparentemente contradizendo essas intuições.
Isso ocorre porque as imagens exibem uma estrutura rica
que pode ser explorada por humanos
e modelos de aprendizado de máquina semelhantes.
Redes neurais convolucionais (CNNs) são uma forma criativa
que o *machine learning* adotou para explorar
algumas das estruturas conhecidas em imagens naturais.


## Invariância

Imagine que você deseja detectar um objeto em uma imagem.
Parece razoável que qualquer método
que usamos para reconhecer objetos não deveria se preocupar demais
com a localização precisa do objeto na imagem.
Idealmente, nosso sistema deve explorar esse conhecimento.
Os porcos geralmente não voam e os aviões geralmente não nadam.
No entanto, devemos ainda reconhecer
um porco era aquele que aparecia no topo da imagem.
Podemos tirar alguma inspiração aqui
do jogo infantil "Cadê o Wally"
(representado em :numref:`img_waldo`).
O jogo consiste em várias cenas caóticas
repletas de atividades.
Wally aparece em algum lugar em cada uma,
normalmente à espreita em algum local improvável.
O objetivo do leitor é localizá-lo.
Apesar de sua roupa característica,
isso pode ser surpreendentemente difícil,
devido ao grande número de distrações.
No entanto, *a aparência do Wally*
não depende de *onde o Wally está localizado*.
Poderíamos varrer a imagem com um detector Wally
que poderia atribuir uma pontuação a cada *patch*,
indicando a probabilidade de o *patch* conter Wally.
CNNs sistematizam essa ideia de *invariância espacial*,
explorando para aprender representações úteis
com menos parâmetros.

![Uma imagem do jogo "Onde está Wally".](../img/where-wally-walker-books.jpg)
:width:`400px`
:label:`img_waldo`



Agora podemos tornar essas intuições mais concretas
enumerando alguns desideratos para orientar nosso design
de uma arquitetura de rede neural adequada para visão computacional:

1. Nas primeiras camadas, nossa rede
     deve responder de forma semelhante ao mesmo *patch*,
     independentemente de onde aparece na imagem. Este princípio é denominado *invariância da tradução*.
1. As primeiras camadas da rede devem se concentrar nas regiões locais,
    sem levar em conta o conteúdo da imagem em regiões distantes. Este é o princípio de *localidade*.
    Eventualmente, essas representações locais podem ser agregadas
    para fazer previsões em todo o nível da imagem.

Vamos ver como isso se traduz em matemática.



## Restringindo o MLP


Para começar, podemos considerar um MLP
com imagens bidimensionais $\mathbf{X}$ como entradas
e suas representações ocultas imediatas
$\mathbf{H}$ similarmente representadas como matrizes em matemática e como tensores bidimensionais em código, onde $\mathbf{X}$ e $\mathbf{H}$ têm a mesma forma.
Deixe isso penetrar.
Agora concebemos não apenas as entradas, mas
também as representações ocultas como possuidoras de estrutura espacial.

Deixe $[\mathbf{X}]_{i, j}$ e $[\mathbf{H}]_{i, j}$ denotarem o pixel
no local ($i$, $j$)
na imagem de entrada e representação oculta, respectivamente.
Consequentemente, para que cada uma das unidades ocultas
receba entrada de cada um dos pixels de entrada,
nós deixaríamos de usar matrizes de peso
(como fizemos anteriormente em MLPs)
para representar nossos parâmetros
como tensores de peso de quarta ordem $\mathsf{W}$.
Suponha que $\mathbf{U}$ contenha *bias*,
poderíamos expressar formalmente a camada totalmente conectada como

$$\begin{aligned} \left[\mathbf{H}\right]_{i, j} &= [\mathbf{U}]_{i, j} + \sum_k \sum_l[\mathsf{W}]_{i, j, k, l}  [\mathbf{X}]_{k, l}\\ &=  [\mathbf{U}]_{i, j} +
\sum_a \sum_b [\mathsf{V}]_{i, j, a, b}  [\mathbf{X}]_{i+a, j+b}.\end{aligned},$$

onde a mudança de $\mathsf{W}$ para $\mathsf{V}$ é inteiramente cosmética por enquanto
uma vez que existe uma correspondência um-para-um
entre coeficientes em ambos os tensores de quarta ordem.
Nós simplesmente reindexamos os subscritos $(k, l)$
de modo que $k = i+a$ and $l = j+b$.
Em outras palavras, definimos $[\mathsf{V}]_{i, j, a, b} = [\mathsf{W}]_{i, j, i+a, j+b}$.
Os índices $a$ e $b$ ultrapassam os deslocamentos positivos e negativos,
cobrindo toda a imagem.
Para qualquer localização dada ($i$, $j$) na representação oculta $[\mathbf{H}]_{i, j}$,
calculamos seu valor somando os pixels em $x$,
centralizado em torno de $(i, j)$ e ponderado por $[\mathsf{V}]_{i, j, a, b}$.

### Invariância de Tradução

Agora vamos invocar o primeiro princípio
estabelecido acima: invariância de tradução.
Isso implica que uma mudança na entrada $\mathbf{X}$
deve simplesmente levar a uma mudança na representação oculta $\mathbf{H}$.
Isso só é possível se $\mathsf{V}$ e $\mathbf{U}$ não dependem realmente de $(i, j)$,
ou seja, temos $[\mathsf{V}]_{i, j, a, b} = [\mathbf{V}]_{a, b}$ e $\mathbf{U}$$ é uma constante, digamos $u$.
Como resultado, podemos simplificar a definição de $\mathbf{H}$:

$$[\mathbf{H}]_{i, j} = u + \sum_a\sum_b [\mathbf{V}]_{a, b}  [\mathbf{X}]_{i+a, j+b}.$$


Esta é uma *convolução*!
Estamos efetivamente ponderando pixels em $(i+a, j+b)$
nas proximidades da localização $(i, j)$ com coeficientes$[\mathbf{V}]_{a, b}$
para obter o valor $[\mathbf{H}]_{i, j}$.
Observe que $[\mathbf{V}]_{a, b}$ precisa de muito menos coeficientes do que $[\mathsf{V}]_{i, j, a, b}, pois ele
não depende mais da localização na imagem.
Fizemos um progresso significativo!

###  Localidade

Agora, vamos invocar o segundo princípio: localidade.
Conforme motivado acima, acreditamos que não devemos ter
parecer muito longe do local $(i, j)$
a fim de coletar informações relevantes
para avaliar o que está acontecendo em $[\mathbf{H}]_{i, j}$.
Isso significa que fora de algum intervalo $|a|> \Delta$ or $|b| > \Delta$,
devemos definir $[\mathbf{V}]_{a, b} = 0$.
Equivalentemente, podemos reescrever $[\mathbf{H}]_{i, j}$ como

$$[\mathbf{H}]_{i, j} = u + \sum_{a = -\Delta}^{\Delta} \sum_{b = -\Delta}^{\Delta} [\mathbf{V}]_{a, b}  [\mathbf{X}]_{i+a, j+b}.$$
:eqlabel:`eq_conv-layer`

Observe que :eqref:`eq_conv-layer`, em poucas palavras, é uma *camada convolucional*.
*Redes neurais convolucionais* (CNNs[^1])
são uma família especial de redes neurais que contêm camadas convolucionais.
Na comunidade de pesquisa de *deep learning*,
$\mathbf{V}$ é referido como um *kernel de convolução*,
um *filtro*, ou simplesmente os *pesos* da camada que são parâmetros frequentemente aprendíveis.
Quando a região local é pequena,
a diferença em comparação com uma rede totalmente conectada pode ser dramática.
Embora anteriormente, pudéssemos ter exigido bilhões de parâmetros
para representar apenas uma única camada em uma rede de processamento de imagem,
agora precisamos de apenas algumas centenas, sem
alterar a dimensionalidade de qualquer
as entradas ou as representações ocultas.
O preço pago por esta redução drástica de parâmetros
é que nossos recursos agora são invariantes de tradução
e que nossa camada só pode incorporar informações locais,
ao determinar o valor de cada ativação oculta.
Todo aprendizado depende da imposição de *bias* indutivos.
Quando esses *bias* concordam com a realidade,
obtemos modelos com amostras eficientes
que generalizam bem para dados invisíveis.
Mas é claro, se esses *bias* não concordam com a realidade,
por exemplo, se as imagens acabassem não sendo invariantes à tradução,
nossos modelos podem ter dificuldade até mesmo para se ajustar aos nossos dados de treinamento.

[^1]: *Convolutional Neural Networks.*

## Convoluções


Antes de prosseguir, devemos revisar brevemente
porque a operação acima é chamada de convolução.
Em matemática, a *convolução* entre duas funções,
digamos que $f, g: \mathbb{R}^d \to \mathbb{R}$ é definida como

$$(f * g)(\mathbf{x}) = \int f(\mathbf{z}) g(\mathbf{x}-\mathbf{z}) d\mathbf{z}.$$

Ou seja, medimos a sobreposição entre $f$ e $g$
quando uma função é "invertida" e deslocada por $\mathbf{x}$.
Sempre que temos objetos discretos, a integral se transforma em uma soma.
Por exemplo, para vetores do conjunto de vetores dimensionais infinitos somados ao quadrado
com o índice acima de $\mathbb{Z}$, obtemos a seguinte definição:

$$(f * g)(i) = \sum_a f(a) g(i-a).$$

Para tensores bidimensionais, temos uma soma correspondente
com índices $(a, b)$ para $f$ e $(i-a, j-b)$ para $g$, respectivamente:

$$(f * g)(i, j) = \sum_a\sum_b f(a, b) g(i-a, j-b).$$
:eqlabel:`eq_2d-conv-discrete`

Isso é semelhante a :eqref:`eq_conv-layer`, com uma grande diferença.
Em vez de usar $(i+a, j+b)$, estamos usando a diferença.
Observe, porém, que esta distinção é principalmente cosmética
uma vez que sempre podemos combinar a notação entre
:eqref:`eq_conv-layer` e :eqref:`eq_2d-conv-discrete`.
Nossa definição original em :eqref:`eq_conv-layer` mais apropriadamente
descreve uma *correlação cruzada*.
Voltaremos a isso na seção seguinte.




## "Onde está Wally" Revisitado

Voltando ao nosso detector Wally, vamos ver como é.
A camada convolucional escolhe janelas de um determinado tamanho
e pesa as intensidades de acordo com o filtro $\mathsf{V}$, conforme demonstrado em :numref:`fig_waldo_mask`.
Podemos ter como objetivo aprender um modelo para que
onde quer que a "Wallyneza" seja mais alta,
devemos encontrar um pico nas representações das camadas ocultas.

![Detectar Wally.](../img/waldo-mask.jpg)
:width:`400px`
:label:`fig_waldo_mask`


### Canais
:label:`subsec_why-conv-channels`

Existe apenas um problema com essa abordagem.
Até agora, felizmente ignoramos que as imagens consistem
de 3 canais: vermelho, verde e azul.
Na realidade, as imagens não são objetos bidimensionais
mas sim tensores de terceira ordem,
caracterizados por uma altura, largura e canal,
por exemplo, com forma $1024 \times 1024 \times 3$ pixels.
Enquanto os dois primeiros desses eixos dizem respeito às relações espaciais,
o terceiro pode ser considerado como atribuição
uma representação multidimensional para cada localização de pixel.
Assim, indexamos $\mathsf{X}$ como $[\mathsf{X}]_{i, j, k}$.
O filtro convolucional deve se adaptar em conformidade.
Em vez de $[\mathbf{V}]_{a,b}$, agora temos $[\mathsf{V}]_{a,b,c}$.

Além disso, assim como nossa entrada consiste em um tensor de terceira ordem,
é uma boa ideia formular de forma semelhante
nossas representações ocultas como tensores de terceira ordem $\mathsf{H}$.
Em outras palavras, em vez de apenas ter uma única representação oculta
correspondendo a cada localização espacial,
queremos todo um vetor de representações ocultas
correspondente a cada localização espacial.
Poderíamos pensar nas representações ocultas como abrangendo
várias grades bidimensionais empilhadas umas sobre as outras.
Como nas entradas, às vezes são chamados de *canais*.
Eles também são chamados de *mapas de características*,
já que cada um fornece um conjunto espacializado
de recursos aprendidos para a camada subsequente.
Intuitivamente, você pode imaginar que nas camadas inferiores que estão mais próximas das entradas,
alguns canais podem se tornar especializados para reconhecer bordas enquanto
outros podem reconhecer texturas.


Para suportar canais múltiplos em ambas as entradas ($\mathsf{X}$) e representações ocultas ($\mathsf{H}$),
podemos adicionar uma quarta coordenada a  $\mathsf{V}$: $[\mathsf{V}]_{a, b, c, d}$.
Juntando tudo, temos:

$$[\mathsf{H}]_{i,j,d} = \sum_{a = -\Delta}^{\Delta} \sum_{b = -\Delta}^{\Delta} \sum_c [\mathsf{V}]_{a, b, c, d} [\mathsf{X}]_{i+a, j+b, c},$$
:eqlabel:`eq_conv-layer-channels`


onde $d$ indexa os canais de saída nas representações ocultas $\mathsf{H}$. A camada convolucional subsequente irá tomar um tensor de terceira ordem, $\mathsf{H}$, como entrada.
Sendo mais geral,
:eqref:`eq_conv-layer-channels` é
a definição de uma camada convolucional para canais múltiplos, onde $\mathsf{V}$ é um *kernel* ou filtro da camada.

Ainda existem muitas operações que precisamos abordar.
Por exemplo, precisamos descobrir como combinar todas as representações ocultas
para uma única saída, por exemplo, se há um Wally *em qualquer lugar* da imagem.
Também precisamos decidir como computar as coisas de forma eficiente,
como combinar várias camadas,
funções de ativação apropriadas,
e como fazer escolhas de design razoáveis
para produzir redes eficazes na prática.
Voltaremos a essas questões no restante do capítulo.

## Resumo

* A invariância da tradução nas imagens implica que todas as manchas de uma imagem serão tratadas da mesma maneira.
* Localidade significa que apenas uma pequena vizinhança de pixels será usada para calcular as representações ocultas correspondentes.
* No processamento de imagem, as camadas convolucionais geralmente requerem muito menos parâmetros do que as camadas totalmente conectadas.
* CNNS são uma família especial de redes neurais que contêm camadas convolucionais.
* Os canais de entrada e saída permitem que nosso modelo capture vários aspectos de uma imagem em cada localização espacial.

## Exercícios

1. Suponha que o tamanho do *kernel* de convolução seja $\Delta = 0$.
    Mostre que, neste caso, o *kernel* de convolução
    implementa um MLP independentemente para cada conjunto de canais.
1. Por que a invariância da tradução pode não ser uma boa ideia, afinal?
1. Com quais problemas devemos lidar ao decidir como tratar representações ocultas correspondentes a localizações de pixels
    na fronteira de uma imagem?
1. Descreva uma camada convolucional análoga para áudio.
1. Você acha que as camadas convolucionais também podem ser aplicáveis para dados de texto?
    Por que ou por que não?
1. Prove que $f * g = g * f$.

[Discussions](https://discuss.d2l.ai/t/64)
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTE5NzY4MTMyNzYsMjAwOTM3MTA1MCw5NT
cwMjU3MDUsLTE5MjE4MTMwODgsMjQyMjkwODk3LC0xNTg0MzI4
NzgwLC0xMTIyNTk3ODYzLC0xNjU3MzY2MjUwLDE2NzM2MjU2MT
AsLTEyOTkyNDE5NjRdfQ==
-->