# Regressão *Softmax*
:label:`sec_softmax`


Em :numref:`sec_linear_regression`, introduzimos a regressão linear,
trabalhando através de implementações do zero em :numref:`sec_linear_scratch`
e novamente usando APIs de alto nível de uma estrutura de *deep learning*
em :numref:`sec_linear_concise` para fazer o trabalho pesado.

A regressão é o martelo que procuramos quando
queremos responder a perguntas*quanto?* ou *quantas?*.
Se você deseja prever o número de dólares (preço)
a que uma casa será vendida,
ou o número de vitórias que um time de beisebol pode ter,
ou o número de dias que um paciente permanecerá hospitalizado antes de receber alta,
então provavelmente você está procurando um modelo de regressão.

Na prática, estamos mais frequentemente interessados ​​na *classificação*:
perguntando não "quanto", mas "qual":

* Este e-mail pertence à pasta de spam ou à caixa de entrada?
* É mais provável que este cliente *se inscreva* ou *não se inscreva* em um serviço de assinatura?
* Esta imagem retrata um burro, um cachorro, um gato ou um galo?
* Qual filme Aston tem mais probabilidade de assistir a seguir?

Coloquialmente, praticantes de *machine learning*
sobrecarregam a palavra *classificação*
para descrever dois problemas sutilmente diferentes:
(i) aqueles em que estamos interessados ​​apenas em
atribuições difíceis de exemplos a categorias (classes);
e (ii) aqueles em que desejamos fazer atribuições leves,
ou seja, para avaliar a probabilidade de que cada categoria se aplica.
A distinção tende a ficar confusa, em parte,
porque muitas vezes, mesmo quando nos preocupamos apenas com tarefas difíceis,
ainda usamos modelos que fazem atribuições suaves.


## Problema de Classificação
:label:`subsec_classification-problem`


Para molhar nossos pés, vamos começar com
um problema simples de classificação de imagens.
Aqui, cada entrada consiste em uma imagem em tons de cinza $2\times 2$.
Podemos representar cada valor de pixel com um único escalar,
dando-nos quatro características $x_1, x_2, x_3, x_4$.
Além disso, vamos supor que cada imagem pertence a uma
entre as categorias "gato", "frango" e "cachorro".

A seguir, temos que escolher como representar os *labels*.
Temos duas escolhas óbvias.
Talvez o impulso mais natural seja escolher $y\in \{1, 2, 3 \}$,
onde os inteiros representam $\{\text{cachorro}, \text{gato}, \text{frango}\}$ respectivamente.
Esta é uma ótima maneira de *armazenar* essas informações em um computador.
Se as categorias tivessem alguma ordem natural entre elas,
digamos se estivéssemos tentando prever $\{\text {bebê}, \text{criança}, \text{adolescente}, \text{jovem adulto}, \text{adulto}, \text{idoso}\}$,
então pode até fazer sentido lançar este problema como uma regressão
e manter os rótulos neste formato.

Mas os problemas gerais de classificação não vêm com ordenações naturais entre as classes.
Felizmente, os estatísticos há muito tempo inventaram uma maneira simples
para representar dados categóricos: a *codificação one-hot*.
Uma codificação *one-hot* é um vetor com tantos componentes quantas categorias temos.
O componente correspondente à categoria da instância em particular é definido como 1
e todos os outros componentes são definidos como 0.
Em nosso caso, um rótulo $y$ seria um vetor tridimensional,
com $(1, 0, 0)$ correspondendo a "gato", $(0, 1, 0)$ a "galinha",
e $(0, 0, 1)$ para "cachorro":

$$y \in \{(1, 0, 0), (0, 1, 0), (0, 0, 1)\}.$$

## Arquitetura de Rede

A fim de estimar as probabilidades condicionais associadas a todas as classes possíveis,
precisamos de um modelo com várias saídas, uma por classe.
Para abordar a classificação com modelos lineares,
precisaremos de tantas funções afins quantas forem as saídas.
Cada saída corresponderá a sua própria função afim.
No nosso caso, uma vez que temos 4 *features* e 3 categorias de saída possíveis,
precisaremos de 12 escalares para representar os pesos ($w$ com subscritos),
e 3 escalares para representar os *offsets* ($b$ com subscritos).
Calculamos esses três *logits*, $o_1, o_2$, and $o_3$, para cada entrada:

$$
\begin{aligned}
o_1 &= x_1 w_{11} + x_2 w_{12} + x_3 w_{13} + x_4 w_{14} + b_1,\\
o_2 &= x_1 w_{21} + x_2 w_{22} + x_3 w_{23} + x_4 w_{24} + b_2,\\
o_3 &= x_1 w_{31} + x_2 w_{32} + x_3 w_{33} + x_4 w_{34} + b_3.
\end{aligned}
$$

Podemos representar esse cálculo com o diagrama da rede neural mostrado em :numref:`fig_softmaxreg`.
Assim como na regressão linear, a regressão *softmax* também é uma rede neural de camada única.
E desde o cálculo de cada saída, $o_1, o_2$, e $o_3$,
depende de todas as entradas, $x_1$, $x_2$, $x_3$, e $x_4$,
a camada de saída da regressão *softmax* também pode ser descrita como uma camada totalmente conectada.

![Softmax regression is a single-layer neural network.](../img/softmaxreg.svg)
:label:`fig_softmaxreg`

Para expressar o modelo de forma mais compacta, podemos usar a notação de álgebra linear.
Na forma vetorial, chegamos a
$\mathbf{o} = \mathbf{W} \mathbf{x} + \mathbf{b}$,
uma forma mais adequada tanto para matemática quanto para escrever código.
Observe que reunimos todos os nossos pesos em uma matriz $3 \times 4$ 
e que para características de um dado exemplo de dados $\mathbf{x}$,
nossas saídas são dadas por um produto vetor-matriz de nossos pesos por nossos recursos de entrada
mais nossos *offsets* $\mathbf{b}$.


## Custo de Parametrização de Camadas Totalmente Conectadas
:label:`subsec_parameterization-cost-fc-layers`

Como veremos nos capítulos subsequentes,
camadas totalmente conectadas são onipresentes no *deep learning*.
No entanto, como o nome sugere,
camadas totalmente conectadas são *totalmente* conectadas
com muitos parâmetros potencialmente aprendíveis.
Especificamente,
para qualquer camada totalmente conectada
com $d$ entradas e $q$ saídas,
o custo de parametrização é $\mathcal{O}(dq)$,
que pode ser proibitivamente alto na prática.
Felizmente,
este custo
de transformar $d$ entradas em $q$ saídas
pode ser reduzido a $\mathcal{O}(\frac{dq}{n})$,
onde o hiperparâmetro $n$ pode ser especificado de maneira flexível
por nós para equilibrar entre o salvamento de parâmetros e a eficácia do modelo em aplicações do mundo real :cite:`Zhang.Tay.Zhang.ea.2021`.




## Operação do *Softmax* 
:label:`subsec_softmax_operation`


A abordagem principal que vamos adotar aqui
é interpretar as saídas de nosso modelo como probabilidades.
Vamos otimizar nossos parâmetros para produzir probabilidades
que maximizam a probabilidade dos dados observados.
Então, para gerar previsões, vamos definir um limite,
por exemplo, escolhendo o *label* com as probabilidades máximas previstas.

Colocado formalmente, gostaríamos de qualquer saída $\hat{y}_j$
fosse interpretada como a probabilidade
que um determinado item pertence à classe $j$.
Então podemos escolher a classe com o maior valor de saída
como nossa previsão $\operatorname*{argmax}_j y_j$.
Por exemplo, se $\hat{y}_1$, $\hat{y}_2$, and $\hat{y}_3$
são 0,1, 0,8 e 0,1, respectivamente,
então, prevemos a categoria 2, que (em nosso exemplo) representa "frango".

Você pode ficar tentado a sugerir que interpretemos
os *logits* $o$ diretamente como nossas saídas de interesse.
No entanto, existem alguns problemas com 
interpretação direta da saída da camada linear como uma probabilidade.
Por um lado,
nada restringe esses números a somarem 1.
Por outro lado, dependendo das entradas, podem assumir valores negativos.
Estes violam axiomas básicos de probabilidade apresentados em :numref:`sec_prob`

Para interpretar nossos resultados como probabilidades,
devemos garantir que (mesmo em novos dados),
eles serão não negativos e somam 1.
Além disso, precisamos de um objetivo de treinamento que incentive
o modelo para estimar probabilidades com fidelidade.
De todas as instâncias quando um classificador produz 0,5,
esperamos que metade desses exemplos
realmente pertenceram à classe prevista.
Esta é uma propriedade chamada *calibração*.

A *função softmax*, inventada em 1959 pelo cientista social
R. Duncan Luce no contexto de *modelos de escolha*,
faz exatamente isso.
Para transformar nossos *logits* de modo que eles se tornem não negativos e somem 1,
ao mesmo tempo em que exigimos que o modelo permaneça diferenciável,
primeiro exponenciamos cada *logit* (garantindo a não negatividade)
e, em seguida, dividimos pela soma (garantindo que somem 1):

$$\hat{\mathbf{y}} = \mathrm{softmax}(\mathbf{o})\quad \text{where}\quad \hat{y}_j = \frac{\exp(o_j)}{\sum_k \exp(o_k)}. $$
:eqlabel:`eq_softmax_y_and_o`

É fácil ver $\hat{y}_1 + \hat{y}_2 + \hat{y}_3 = 1$
with $0 \leq \hat{y}_j \leq 1$ para todo $j$.
Assim, $\hat{\mathbf{y}}$ é uma distribuição de probabilidade adequada
cujos valores de elementos podem ser interpretados em conformidade.
Observe que a operação *softmax* não muda a ordem entre os *logits* $\mathbf{o}$,
que são simplesmente os valores pré-*softmax*
que determinam as probabilidades atribuídas a cada classe.
Portanto, durante a previsão, ainda podemos escolher a classe mais provável por

$$
\operatorname*{argmax}_j \hat y_j = \operatorname*{argmax}_j o_j.
$$

Embora *softmax* seja uma função não linear,
as saídas da regressão *softmax* ainda são *determinadas* por
uma transformação afim de recursos de entrada;
portanto, a regressão *softmax* é um modelo linear.



## Vetorização para *Minibatches*
:label:`subsec_softmax_vectorization`

Para melhorar a eficiência computacional e aproveitar as vantagens das GPUs,
normalmente realizamos cálculos vetoriais para *minibatches* de dados.
Suponha que recebemos um *minibatch* $\mathbf{X}$ de exemplos
com dimensionalidade do recurso (número de entradas) $d$ e tamanho do lote $n$.
Além disso, suponha que temos $q$ categorias na saída.
Então os *features* de *minibatch* $\mathbf{X}$ estão em $\mathbb{R}^{n \times d}$,
pesos $\mathbf{W} \in \mathbb{R}^{d \times q}$,
e o *bias* satisfaz $\mathbf{b} \in \mathbb{R}^{1\times q}$.

$$ \begin{aligned} \mathbf{O} &= \mathbf{X} \mathbf{W} + \mathbf{b}, \\ \hat{\mathbf{Y}} & = \mathrm{softmax}(\mathbf{O}). \end{aligned} $$
:eqlabel:`eq_minibatch_softmax_reg`

Isso acelera a operação dominante em
um produto matriz-matriz $\mathbf{X} \mathbf{W}$
vs. os produtos de vetor-matriz que estaríamos executando
se processamos um exemplo de cada vez.
Uma vez que cada linha em $\mathbf{X}$ representa um exemplo de dados,
a própria operação *softmax* pode ser calculada *rowwise* (através das colunas):
para cada linha de $\mathbf{O}$, exponenciando todas as entradas e depois normalizando-as pela soma.
Disparando a transmissão durante a soma $\mathbf{X} \mathbf{W} + \mathbf{b}$ in :eqref:`eq_minibatch_softmax_reg`,
o *minibatch* registra $\mathbf{O}$ e as probabilidades de saída $\hat{\mathbf{Y}}$
são matrizes $n \times q$.


## Função de Perda

Em seguida, precisamos de uma função de perda para medir
a qualidade de nossas probabilidades previstas.
Contaremos com a estimativa de probabilidade máxima,
o mesmo conceito que encontramos
ao fornecer uma justificativa probabilística
para o objetivo de erro quadrático médio na regressão linear
(:numref:`subsec_normal_distribution_and_squared_loss`).


### *Log-Likelihood*

A função *softmax* nos dá um vetor $\hat{\mathbf{y}}$,
que podemos interpretar como probabilidades condicionais estimadas
de cada classe dada qualquer entrada $\mathbf{x}$,, por exemplo,
$\hat{y}_1$ = $P(y=\text{cat} \mid \mathbf{x})$.
Suponha que todo o conjunto de dados $\{\mathbf{X}, \mathbf{Y}\}$  tenha $n$ exemplos,
onde o exemplo indexado por $i$
consiste em um vetor de característica $\mathbf{x}^{(i)}$ e um vetor de rótulo único $\mathbf{y}^{(i)}$.
Podemos comparar as estimativas com a realidade
verificando quão prováveis as classes reais são
de acordo com nosso modelo, dadas as características:

$$
P(\mathbf{Y} \mid \mathbf{X}) = \prod_{i=1}^n P(\mathbf{y}^{(i)} \mid \mathbf{x}^{(i)}).
$$

De acordo com a estimativa de máxima *likelihood*,
maximizamos $P(\mathbf{Y} \mid \mathbf{X})$,
que é
equivalente a minimizar a probabilidade de *log-likelihood* negativo:

$$
-\log P(\mathbf{Y} \mid \mathbf{X}) = \sum_{i=1}^n -\log P(\mathbf{y}^{(i)} \mid \mathbf{x}^{(i)})
= \sum_{i=1}^n l(\mathbf{y}^{(i)}, \hat{\mathbf{y}}^{(i)}),
$$

onde para qualquer par de rótulo $\mathbf{y}$ e predição de modelo $\hat{\mathbf{y}}$ sobre $q$ classes,
a função de perda $l$ é

$$ l(\mathbf{y}, \hat{\mathbf{y}}) = - \sum_{j=1}^q y_j \log \hat{y}_j. $$
:eqlabel:`eq_l_cross_entropy`

Por razões explicadas mais tarde, a função de perda em :eqref:`eq_l_cross_entropy`
é comumente chamada de *perda de entropia cruzada*.
Uma vez que $\mathbf{y}$ é um vetor *one-hot* de comprimento $q$,
a soma de todas as suas coordenadas $j$ desaparece para todos, exceto um termo.
Uma vez que todos os $\hat{y}_j$ são probabilidades previstas,
seu logaritmo nunca é maior que $0$.
Consequentemente, a função de perda não pode ser minimizada mais,
se predizermos corretamente o rótulo real com *certeza*,
ou seja, se a probabilidade prevista $P(\mathbf{y} \mid \mathbf{x}) = 1$  for o *label* real $\mathbf{y}$.
Observe que isso geralmente é impossível.
Por exemplo, pode haver ruído de *label* no *dataset*
(alguns exemplos podem estar classificados incorretamente).
Também pode não ser possível quando os recursos de entrada
não são suficientemente informativos
para classificar todos os exemplos perfeitamente.

### *Softmax*  e Derivadas
:label:`subsec_softmax_and_derivatives`

Uma vez que o *softmax* e a perda correspondente são tão comuns,
vale a pena entender um pouco melhor como ele é calculado.
Conectando :eqref:`eq_softmax_y_and_o` na definição da perda
em :eqref:`eq_l_cross_entropy`
e usando a definição do *softmax* obtemos:

$$
\begin{aligned}
l(\mathbf{y}, \hat{\mathbf{y}}) &=  - \sum_{j=1}^q y_j \log \frac{\exp(o_j)}{\sum_{k=1}^q \exp(o_k)} \\
&= \sum_{j=1}^q y_j \log \sum_{k=1}^q \exp(o_k) - \sum_{j=1}^q y_j o_j\\
&= \log \sum_{k=1}^q \exp(o_k) - \sum_{j=1}^q y_j o_j.
\end{aligned}
$$

Para entender um pouco melhor o que está acontecendo,
considere a derivada com respeito a qualquer *logit* $o_j$. Nós temos
$$
\partial_{o_j} l(\mathbf{y}, \hat{\mathbf{y}}) = \frac{\exp(o_j)}{\sum_{k=1}^q \exp(o_k)} - y_j = \mathrm{softmax}(\mathbf{o})_j - y_j.
$$

Em outras palavras, a derivada é a diferença
entre a probabilidade atribuída pelo nosso modelo,
conforme expresso pela operação *softmax*,
e o que realmente aconteceu, conforme expresso por elementos no vetor *one-hot* de *labels*.
Nesse sentido, é muito semelhante ao que vimos na regressão,
onde o gradiente era a diferença
entre a observação $y$ e a estimativa $\hat{y}$.
Isso não é coincidência.
Em qualquer família exponencial (veja o modelo no
[apêndice online sobre distribuições](https://d2l.ai/chapter_apencha-mathematics-for-deep-learning/distributions.html)),
os gradientes da probabilidade logarítmica são dados precisamente por esse termo.
Esse fato torna a computação dos gradientes fáceis na prática.

### Perda de Entropia Cruzada

Agora considere o caso em que observamos não apenas um único resultado
mas toda uma distribuição de resultados.
Podemos usar a mesma representação de antes para o rótulo $\mathbf{y}$.
A única diferença é que, em vez de um vetor contendo apenas entradas binárias,
digamos $(0, 0, 1)$, agora temos um vetor de probabilidade genérico, digamos $(0.1, 0.2, 0.7)$.
A matemática que usamos anteriormente para definir a perda $l$
em :eqref:`eq_l_cross_entropy`
ainda funciona bem,
apenas que a interpretação é um pouco mais geral.
É o valor esperado da perda de uma distribuição nos *labels*.
Esta perda é chamada de *perda de entropia cruzada* e é
uma das perdas mais comumente usadas para problemas de classificação.
Podemos desmistificar o nome apresentando apenas os fundamentos da teoria da informação.
Se você deseja entender mais detalhes da teoria da informação,
você também pode consultar o [apêndice online sobre teoria da informação](https://d2l.ai/chapter_apencha-mathematics-for-deep-learning/information-theory.html).



## Fundamentos da Teoria da Informação
:label:`subsec_info_theory_basics`

*Teoria da informação* lida com o problema de codificação, decodificação, transmissão,
e manipulação informações (também conhecidas como dados) da forma mais concisa possível.


### Entropia

A ideia central na teoria da informação é quantificar o conteúdo da informação nos dados.
Essa quantidade impõe um limite rígido à nossa capacidade de compactar os dados.
Na teoria da informação, essa quantidade é chamada de *entropia* de uma distribuição $P$,
e é definida pela seguinte equação:

$$H[P] = \sum_j - P(j) \log P(j).$$
:eqlabel:`eq_softmax_reg_entropy`

Um dos teoremas fundamentais da teoria da informação
afirma que, a fim de codificar dados retirados aleatoriamente da distribuição $P$,
precisamos de pelo menos $H[P]$ "*nats*" para codificá-lo.
Se você quer saber o que é um "*nat*", é o equivalente a bit
mas ao usar um código com base $e$ em vez de um com base 2.
Assim, um *nat* é $\frac{1}{\log(2)} \approx 1.44$ bit.


### Surpresa


Você pode estar se perguntando o que a compressão tem a ver com a predição.
Imagine que temos um fluxo de dados que queremos compactar.
Se sempre for fácil para nós prevermos o próximo *token*,
então esses dados são fáceis de compactar!
Veja o exemplo extremo em que cada token no fluxo sempre leva o mesmo valor.
Esse é um fluxo de dados muito chato!
E não só é chato, mas também é fácil de prever.
Por serem sempre iguais, não precisamos transmitir nenhuma informação
para comunicar o conteúdo do fluxo.
Fácil de prever, fácil de compactar.

No entanto, se não podemos prever perfeitamente todos os eventos,
então às vezes podemos ficar surpresos.
Nossa surpresa é maior quando atribuímos uma probabilidade menor a um evento.
Claude Shannon estabeleceu $\log \frac{1}{P(j)} = -\log P(j)$
para quantificar a *surpresa* de alguém ao observar um evento $j$
tendo-lhe atribuído uma probabilidade (subjetiva) $P(j)$.
A entropia definida em :eqref:`eq_softmax_reg_entropy` é então a *surpresa esperada*
quando alguém atribuiu as probabilidades corretas
que realmente correspondem ao processo de geração de dados.


### Entropia Cruzada Revisitada


Então, se a entropia é o nível de surpresa experimentado
por alguém que conhece a verdadeira probabilidade,
então você deve estar se perguntando, o que é entropia cruzada?
A entropia cruzada *de* $P$ *a* $Q$, denotada $H(P, Q)$,
é a surpresa esperada de um observador com probabilidades subjetivas $Q$
ao ver os dados que realmente foram gerados de acordo com as probabilidades $P$.
A menor entropia cruzada possível é alcançada quando $P=Q$.
Nesse caso, a entropia cruzada de $P$ a $Q$ é $H(P, P)= H(P)$.

Em suma, podemos pensar no objetivo da classificação de entropia cruzada
de duas maneiras: (i) maximizando a probabilidade dos dados observados;
e (ii) minimizando nossa surpresa (e, portanto, o número de bits)
necessário para comunicar os rótulos.

## Predição do Modelo e Avaliação

Depois de treinar o modelo de regressão *softmax*, dados quaisquer recursos de exemplo,
podemos prever a probabilidade de cada classe de saída.
Normalmente, usamos a classe com a maior probabilidade prevista como a classe de saída.
A previsão está correta se for consistente com a classe real (*label*).
Na próxima parte do experimento,
usaremos *exatidão* para avaliar o desempenho do modelo.
Isso é igual à razão entre o número de previsões corretas e o número total de previsões.


## Resumo

* A operação *softmax* pega um vetor e o mapeia em probabilidades.
* A regressão *Softmax* se aplica a problemas de classificação. Ela usa a distribuição de probabilidade da classe de saída na operação *softmax*.
* A entropia cruzada é uma boa medida da diferença entre duas distribuições de probabilidade. Ela mede o número de bits necessários para codificar os dados de nosso modelo.

## Exercícios

1. Podemos explorar a conexão entre as famílias exponenciais e o *softmax* com um pouco mais de profundidade.
    1. Calcule a segunda derivada da perda de entropia cruzada $l(\mathbf{y},\hat{\mathbf{y}})$ para o *softmax*.
    1. Calcule a variância da distribuição dada por $\mathrm{softmax}(\mathbf{o})$ e mostre que ela corresponde à segunda derivada calculada acima.
1. Suponha que temos três classes que ocorrem com probabilidade igual, ou seja, o vetor de probabilidade é $(\frac{1}{3}, \frac{1}{3}, \frac{1}{3})$.
    1. Qual é o problema se tentarmos projetar um código binário para ele?
    1. Você pode criar um código melhor? Dica: o que acontece se tentarmos codificar duas observações independentes? E se codificarmos $n$ observações em conjunto?
1. *Softmax* é um nome impróprio para o mapeamento apresentado acima (mas todos no aprendizado profundo o usam). O *softmax* real é definido como $\mathrm{RealSoftMax}(a, b) = \log (\exp(a) + \exp(b))$.
    1. Prove que $\mathrm{RealSoftMax}(a, b) > \mathrm{max}(a, b)$.
    1. Prove que isso vale para$\lambda^{-1} \mathrm{RealSoftMax}(\lambda a, \lambda b)$, desde que $\lambda > 0$.
    1. Mostre que para $\lambda \to \infty$ temos $\lambda^{-1} \mathrm{RealSoftMax}(\lambda a, \lambda b) \to \mathrm{max}(a, b)$.
    1. Qual é a aparência do soft-min?
    1. Estenda isso para mais de dois números.

[Discussions](https://discuss.d2l.ai/t/46)
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTEyOTQ4NzMwNDEsLTIwNDM2Mjc4OTIsLT
g2NTE4MTkxOCwtMTMwMjM4MDE4OSwtMTQxNDIzMjY5NSwzODQ3
OTU5MzUsMTcyOTA2MDk4MywtMTI5MTE0NzU5NiwxMDY2OTU2MD
kzLDE4NDMzNzUwOTQsMTM5NDcyMzQ2MF19
-->