# Propagação Direta, Propagação Reversa e Gráficos Computacionais
:label:`sec_backprop`


Até agora, treinamos nossos modelos
com gradiente descendente estocástico de *minibatch*.
No entanto, quando implementamos o algoritmo,
nós apenas nos preocupamos com os cálculos envolvidos
em *propagação direta* através do modelo.
Quando chegou a hora de calcular os gradientes,
acabamos de invocar a função *backpropagation* (propagação reversa) fornecida pela estrutura de *deep learning*.

O cálculo automático de gradientes (diferenciação automática) simplifica profundamente
a implementação de algoritmos de *deep learning*.
Antes da diferenciação automática,
mesmo pequenas mudanças em modelos complicados exigiam
recalcular derivadas complicadas manualmente.
Surpreendentemente, muitas vezes, os trabalhos acadêmicos tiveram que alocar
várias páginas para derivar regras de atualização.
Embora devamos continuar a confiar na diferenciação automática
para que possamos nos concentrar nas partes interessantes,
você deve saber como esses gradientes
são calculados sob o capô
se você quiser ir além de uma rasa
compreensão da aprendizagem profunda.

Nesta seção, fazemos um mergulho profundo
nos detalhes de *propagação para trás*
(mais comumente chamado de *backpropagation*).
Para transmitir alguns *insights* para ambas as
técnicas e suas implementações,
contamos com alguma matemática básica e gráficos computacionais.
Para começar, focamos nossa exposição em
um MLP de uma camada oculta
com *weight decay* (regularização$L_2$).

## Propagação Direta


*Propagação direta* (ou *passagem direta*) refere-se ao cálculo e armazenamento
de variáveis intermediárias (incluindo saídas)
para uma rede neural em ordem
da camada de entrada para a camada de saída.
Agora trabalhamos passo a passo com a mecânica
de uma rede neural com uma camada oculta.
Isso pode parecer tedioso, mas nas palavras eternas
do virtuoso do funk James Brown,
você deve "pagar o custo para ser o chefe".


Por uma questão de simplicidade, vamos assumir
que o exemplo de entrada é  $\mathbf{x}\in \mathbb{R}^d$
e que nossa camada oculta não inclui um termo de *bias*.
Aqui, a variável intermediária é:

$$\mathbf{z}= \mathbf{W}^{(1)} \mathbf{x},$$

onde $\mathbf{W}^{(1)} \in \mathbb{R}^{h \times d}$
é o parâmetro de peso da camada oculta.
Depois de executar a variável intermediária
$\mathbf{z}\in \mathbb{R}^h$ através da
função de ativação $\phi$
obtemos nosso vetor de ativação oculto de comprimento $h$,

$$\mathbf{h}= \phi (\mathbf{z}).$$

A variável oculta $\mathbf{h}$
também é uma variável intermediária.
Supondo que os parâmetros da camada de saída
só possuem um peso de
$\mathbf{W}^{(2)} \in \mathbb{R}^{q \times h}$,
podemos obter uma variável de camada de saída
com um vetor de comprimento $q$:

$$\mathbf{o}= \mathbf{W}^{(2)} \mathbf{h}.$$

Supondo que a função de perda seja $l$
e o *label* de exemplo é $y$,
podemos então calcular o prazo de perda
para um único exemplo de dados,

$$L = l(\mathbf{o}, y).$$

De acordo com a definição de regularização de $L_2$
dado o hiperparâmetro $\lambda$,
o prazo de regularização é

$$s = \frac{\lambda}{2} \left(\|\mathbf{W}^{(1)}\|_F^2 + \|\mathbf{W}^{(2)}\|_F^2\right),$$
:eqlabel:`eq_forward-s`

onde a norma Frobenius da matriz
é simplesmente a norma $L_2$ aplicada
depois de achatar a matriz em um vetor.
Por fim, a perda regularizada do modelo
em um dado exemplo de dados é:

$$J = L + s.$$

Referimo-nos a $J$ como a *função objetivo*
na discussão a seguir.


## Gráfico Computatcional de Propagação Direta

Traçar *gráficos computacionais* nos ajuda a visualizar
as dependências dos operadores
e variáveis dentro do cálculo.
:numref:`fig_forward` contém o gráfico associado
com a rede simples descrita acima,
onde quadrados denotam variáveis e círculos denotam operadores.
O canto inferior esquerdo significa a entrada
e o canto superior direito é a saída.
Observe que as direções das setas
(que ilustram o fluxo de dados)
são principalmente para a direita e para cima.

![Gráfico computacional de propagação direta.](../img/forward.svg)
:label:`fig_forward`

## Propagação Reversa

*Propagação reversa* (*retropropagação*) refere-se ao método de cálculo do gradiente dos parâmetros da rede neural.
Em suma, o método atravessa a rede na ordem inversa,
da saída para a camada de entrada,
de acordo com a *regra da cadeia* do cálculo.
O algoritmo armazena quaisquer variáveis intermediárias
(derivadas parciais)
necessárias ao calcular o gradiente
com relação a alguns parâmetros.
Suponha que temos funções
$\mathsf{Y}=f(\mathsf{X})$
e $\mathsf{Z}=g(\mathsf{Y})$,
em que a entrada e a saída
$\mathsf{X}, \mathsf{Y}, \mathsf{Z}$
são tensores de formas arbitrárias.
Usando a regra da cadeia,
podemos calcular a derivada
de $\mathsf{Z}$ with respect to $\mathsf{X}$ via

$$\frac{\partial \mathsf{Z}}{\partial \mathsf{X}} = \text{prod}\left(\frac{\partial \mathsf{Z}}{\partial \mathsf{Y}}, \frac{\partial \mathsf{Y}}{\partial \mathsf{X}}\right).$$


Aqui usamos o operador $\text{prod}$ para
multiplicar seus argumentos
depois que as operações necessárias,
como transposição e troca de posições de entrada,
foram realizadas.
Para vetores, isso é simples:
é simplesmente multiplicação matriz-matriz.
Para tensores dimensionais superiores,
usamos a contraparte apropriada.
O operador $\text{prod}$ esconde todo o *overhead* de notação.

Lembre-se disso
os parâmetros da rede simples com uma camada oculta,
cujo gráfico computacional está em :numref:`fig_forward`,
são $\mathbf{W}^{(1)}$ e $\mathbf{W}^{(2)}$.
O objetivo da retropropagação é
calcular os gradientes $\partial J/\partial \mathbf{W}^{(1)}$
e $\partial J/\partial \mathbf{W}^{(2)}$.
Para conseguir isso, aplicamos a regra da cadeia
e calcular, por sua vez, o gradiente de
cada variável e parâmetro intermediário.
A ordem dos cálculos é invertida
em relação àquelas realizadas na propagação direta,
uma vez que precisamos começar com o resultado do gráfico computacional
e trabalhar nosso caminho em direção aos parâmetros.
O primeiro passo é calcular os gradientes
da função objetivo $J=L+s$
com relação ao prazo de perda $L$
e o prazo de regularização $s$.

$$\frac{\partial J}{\partial L} = 1 \; \text{e} \; \frac{\partial J}{\partial s} = 1.$$

Em seguida, calculamos o gradiente da função objetivo
em relação à variável da camada de saída $\mathbf{o}$
de acordo com a regra da cadeia:

$$
\frac{\partial J}{\partial \mathbf{o}}
= \text{prod}\left(\frac{\partial J}{\partial L}, \frac{\partial L}{\partial \mathbf{o}}\right)
= \frac{\partial L}{\partial \mathbf{o}}
\in \mathbb{R}^q.
$$

Em seguida, calculamos os gradientes
do termo de regularização
com respeito a ambos os parâmetros:

$$\frac{\partial s}{\partial \mathbf{W}^{(1)}} = \lambda \mathbf{W}^{(1)}
\; \text{and} \;
\frac{\partial s}{\partial \mathbf{W}^{(2)}} = \lambda \mathbf{W}^{(2)}.$$

Agora podemos calcular o gradiente
$\partial J/\partial \mathbf{W}^{(2)} \in \mathbb{R}^{q \times h}$
dos parâmetros do modelo mais próximos da camada de saída.
Usar a regra da cadeia produz:

$$\frac{\partial J}{\partial \mathbf{W}^{(2)}}= \text{prod}\left(\frac{\partial J}{\partial \mathbf{o}}, \frac{\partial \mathbf{o}}{\partial \mathbf{W}^{(2)}}\right) + \text{prod}\left(\frac{\partial J}{\partial s}, \frac{\partial s}{\partial \mathbf{W}^{(2)}}\right)= \frac{\partial J}{\partial \mathbf{o}} \mathbf{h}^\top + \lambda \mathbf{W}^{(2)}.$$
:eqlabel:`eq_backprop-J-h`

Para obter o gradiente em relação a $\mathbf{W}^{(1)}$
precisamos continuar a retropropagação
ao longo da camada de saída para a camada oculta.
O gradiente em relação às saídas da camada oculta
$\partial J/\partial \mathbf{h} \in \mathbb{R}^h$ é dado por


$$
\frac{\partial J}{\partial \mathbf{h}}
= \text{prod}\left(\frac{\partial J}{\partial \mathbf{o}}, \frac{\partial \mathbf{o}}{\partial \mathbf{h}}\right)
= {\mathbf{W}^{(2)}}^\top \frac{\partial J}{\partial \mathbf{o}}.
$$

Uma vez que a função de ativação $\phi$ se aplica aos elementos,
calculando o gradiente $\partial J/\partial \mathbf{z} \in \mathbb{R}^h$
da variável intermediária $\mathbf{z}$
requer que usemos o operador de multiplicação elemento a elemento,
que denotamos por $\odot$:

$$
\frac{\partial J}{\partial \mathbf{z}}
= \text{prod}\left(\frac{\partial J}{\partial \mathbf{h}}, \frac{\partial \mathbf{h}}{\partial \mathbf{z}}\right)
= \frac{\partial J}{\partial \mathbf{h}} \odot \phi'\left(\mathbf{z}\right).
$$

Finalmente, podemos obter o gradiente
$\partial J/\partial \mathbf{W}^{(1)} \in \mathbb{R}^{h \times d}$
dos parâmetros do modelo mais próximos da camada de entrada.
De acordo com a regra da cadeia, obtemos

$$
\frac{\partial J}{\partial \mathbf{W}^{(1)}}
= \text{prod}\left(\frac{\partial J}{\partial \mathbf{z}}, \frac{\partial \mathbf{z}}{\partial \mathbf{W}^{(1)}}\right) + \text{prod}\left(\frac{\partial J}{\partial s}, \frac{\partial s}{\partial \mathbf{W}^{(1)}}\right)
= \frac{\partial J}{\partial \mathbf{z}} \mathbf{x}^\top + \lambda \mathbf{W}^{(1)}.
$$



## Treinando Redes Neurais


Ao treinar redes neurais,
a propagação direta e reversa dependem uma da outra.
Em particular, para propagação direta,
atravessamos o gráfico computacional na direção das dependências
e calculamos todas as variáveis em seu caminho.
Eles são então usadas para retropropagação
onde a ordem de computação no gráfico é invertida.

Tome a rede simples mencionada acima como um exemplo para ilustrar.
Por um lado,
calcular o termo de regularização :eqref:`eq_forward-s`
durante a propagação para a frente
depende dos valores atuais dos parâmetros do modelo $\mathbf{W}^{(1)}$ and $\mathbf{W}^{(2)}$.
Eles são fornecidos pelo algoritmo de otimização de acordo com a retropropagação na iteração mais recente.
Por outro lado,
o cálculo do gradiente para o parâmetro
:eqref:`eq_backprop-J-h` durante a retropropagação
depende do valor atual da variável oculta $\mathbf{h}$,
que é fornecido por propagação direta.


Portanto, ao treinar redes neurais, após os parâmetros do modelo serem inicializados,
alternamos a propagação direta com a retropropagação,
atualizando os parâmetros do modelo usando gradientes fornecidos por retropropagação.
Observe que a retropropagação reutiliza os valores intermediários armazenados da propagação direta para evitar cálculos duplicados.
Uma das consequências é que precisamos reter
os valores intermediários até que a retropropagação seja concluída.
Esta é também uma das razões pelas quais o treinamento
requer muito mais memória do que a previsão simples.
Além disso, o tamanho de tais valores intermediários é aproximadamente
proporcional ao número de camadas de rede e ao tamanho do lote.
Por isso,
treinar redes mais profundas usando tamanhos de lote maiores
mais facilmente leva a erros de *falta de memória*.

## Resumo

* A propagação direta calcula e armazena sequencialmente variáveis intermediárias no gráfico computacional definido pela rede neural. Ela prossegue da camada de entrada para a camada de saída.
* A retropropagação calcula e armazena sequencialmente os gradientes de variáveis e parâmetros intermediários na rede neural na ordem inversa.
* Ao treinar modelos de *deep learning*, a propagação direta e reversa são interdependentes.
* O treinamento requer muito mais memória do que a previsão.


## Exercícios

1. Suponha que as entradas $\mathbf{X}$ para alguma função escalar $f$ sejam matrizes $n \times m$. Qual é a dimensionalidade do gradiente de $f$ em relação a $\mathbf{X}$?
1. Adicione um *bias* à camada oculta do modelo descrito nesta seção (você não precisa incluir um *bias* no termo de regularização).
     1. Desenhe o gráfico computacional correspondente.
     1. Derive as equações de propagação direta e reversa.
1. Calcule a pegada de memória para treinamento e predição no modelo descrito nesta seção.
1. Suponha que você deseja calcular derivadas secundárias. O que acontece com o gráfico computacional? Quanto tempo você espera que o cálculo demore?
1. Suponha que o gráfico computacional seja muito grande para sua GPU.
     1. Você pode particioná-lo em mais de uma GPU?
     2. Quais são as vantagens e desvantagens em relação ao treinamento em um *minibatch* menor?

[Discussions](https://discuss.d2l.ai/t/102)
<!--stackedit_data:
eyJoaXN0b3J5IjpbMjY5MzQyNTI1LDE4MDQ5MTM2ODYsLTk1OD
AxMDM5NiwtODMwNzQ0MzMyLC0xOTA2NDU3ODI2LDg1NzQ2OTgz
NiwxMDExMTQzMzM3LDEzNTc3MjA0MjldfQ==
-->