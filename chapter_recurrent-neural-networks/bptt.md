# Retropropagação ao Longo do Tempo
:label:`sec_bptt`


Até agora, temos repetidamente aludido a coisas como
*gradientes explosivos*,
*gradientes de desaparecimento*,
e a necessidade de
*destacar o gradiente* para RNNs.
Por exemplo, em :numref:`sec_rnn_scratch`
invocamos a função `detach` na sequência.
Nada disso foi realmente completamente
explicado, no interesse de ser capaz de construir um modelo rapidamente e
para ver como funciona.
Nesta secção,
vamos nos aprofundar um pouco mais
nos detalhes de retropropagação para modelos de sequência e por que (e como) a matemática funciona.

Encontramos alguns dos efeitos da explosão de gradiente quando primeiro
RNNs implementados (:numref:`sec_rnn_scratch`).
No
especial,
se você resolveu os exercícios,
você poderia
ter visto que o corte de gradiente é vital para garantir
convergência.
Para fornecer uma melhor compreensão deste problema, esta
seção irá rever como os gradientes são calculados para modelos de sequência.
Observe
que não há nada conceitualmente novo em como funciona. Afinal, ainda estamos apenas aplicando a regra da cadeia para calcular gradientes. No entanto,
vale a pena revisar a retropropagação (:numref:`sec_backprop`) novamente.


Descrevemos propagações para frente e para trás
e gráficos computacionais
em MLPs em :numref:`sec_backprop`.
A propagação direta em uma RNN é relativamente
para a frente.
*Retropropagação através do tempo* é, na verdade, uma aplicação específica de retropropagação
em RNNs :cite:`Werbos.1990`.
Isto
exige que expandamos o
gráfico computacional de uma RNN
um passo de cada vez para
obter as dependências
entre variáveis ​​e parâmetros do modelo.
Então,
com base na regra da cadeia,
aplicamos retropropagação para calcular e
gradientes de loja.
Uma vez que as sequências podem ser bastante longas, a dependência pode ser bastante longa.
Por exemplo, para uma sequência de 1000 caracteres,
o primeiro token pode ter uma influência significativa sobre o token na posição final.
Isso não é realmente viável computacionalmente
(leva muito tempo e requer muita memória) e requer mais de 1000 produtos de matriz antes de chegarmos a esse gradiente muito indescritível.
Este é um processo repleto de incertezas computacionais e estatísticas.
A seguir iremos elucidar o que acontece
e como resolver isso na prática.

## Análise de Gradientes em RNNs
:label:`subsec_bptt_analysis`


Começamos com um modelo simplificado de como funciona uma RNN.
Este modelo ignora detalhes sobre as especificações do estado oculto e como ele é atualizado.
A notação matemática aqui
não distingue explicitamente
escalares, vetores e matrizes como costumava fazer.
Esses detalhes são irrelevantes para a análise
e serviriam apenas para bagunçar a notação
nesta subseção.

Neste modelo simplificado,
denotamos $h_t$ como o estado oculto,
$x_t$ como a entrada e $o_t$ como a saída
no passo de tempo $t$.
Lembre-se de nossas discussões em
:numref:`subsec_rnn_w_hidden_states`
que a entrada e o estado oculto
podem ser concatenados ao
serem multiplicados por uma variável de peso na camada oculta.
Assim, usamos $w_h$ e $w_o$ para
indicar os pesos da camada oculta e da camada de saída, respectivamente.
Como resultado, os estados ocultos e saídas em cada etapa de tempo podem ser explicados como

$$\begin{aligned}h_t &= f(x_t, h_{t-1}, w_h),\\o_t &= g(h_t, w_o),\end{aligned}$$
:eqlabel:`eq_bptt_ht_ot`

onde $f$ e $g$ são transformações
da camada oculta e da camada de saída, respectivamente.
Portanto, temos uma cadeia de valores $\{\ldots, (x_{t-1}, h_{t-1}, o_{t-1}), (x_{t}, h_{t}, o_t), \ldots\}$ que dependem uns dos outros por meio de computação recorrente.
A propagação direta é bastante direta.
Tudo o que precisamos é percorrer as triplas $(x_t, h_t, o_t)$ um passo de tempo de cada vez.
A discrepância entre a saída $o_t$ e o rótulo desejado $y_t$ é então avaliada por uma função objetivo
em todas as etapas de tempo $T$
como

$$L(x_1, \ldots, x_T, y_1, \ldots, y_T, w_h, w_o) = \frac{1}{T}\sum_{t=1}^T l(y_t, o_t).$$



Para retropropagação, as coisas são um pouco mais complicadas, especialmente quando calculamos os gradientes em relação aos parâmetros $w_h$ da função objetivo $L$. Para ser específico, pela regra da cadeia,

$$\begin{aligned}\frac{\partial L}{\partial w_h}  & = \frac{1}{T}\sum_{t=1}^T \frac{\partial l(y_t, o_t)}{\partial w_h}  \\& = \frac{1}{T}\sum_{t=1}^T \frac{\partial l(y_t, o_t)}{\partial o_t} \frac{\partial g(h_t, w_o)}{\partial h_t}  \frac{\partial h_t}{\partial w_h}.\end{aligned}$$
:eqlabel:`eq_bptt_partial_L_wh`

O primeiro e o segundo fatores do
produto em :eqref:`eq_bptt_partial_L_wh`
são fáceis de calcular.
O terceiro fator $\partial h_t/\partial w_h$ é onde as coisas ficam complicadas, já que precisamos calcular recorrentemente o efeito do parâmetro $w_h$ em $h_t$.
De acordo com o cálculo recorrente
em :eqref:`eq_bptt_ht_ot`,
$h_t$ depende de $h_{t-1}$ e $w_h$,
onde cálculo de $h_{t-1}$
também depende de $w_h$.
Assim,
usando a regra da cadeia temos

$$\frac{\partial h_t}{\partial w_h}= \frac{\partial f(x_{t},h_{t-1},w_h)}{\partial w_h} +\frac{\partial f(x_{t},h_{t-1},w_h)}{\partial h_{t-1}} \frac{\partial h_{t-1}}{\partial w_h}.$$
:eqlabel:`eq_bptt_partial_ht_wh_recur`


Para derivar o gradiente acima, suponha que temos três sequências $\{a_{t}\},\{b_{t}\},\{c_{t}\}$ satisfatória
$a_{0}=0$ and $a_{t}=b_{t}+c_{t}a_{t-1}$ for $t=1, 2,\ldots$.
Então, para $t\geq 1$, é fácil mostrar

$$a_{t}=b_{t}+\sum_{i=1}^{t-1}\left(\prod_{j=i+1}^{t}c_{j}\right)b_{i}.$$
:eqlabel:`eq_bptt_at`

Substituindo $$a_t$, $b_t$, e $c_t$
de acordo com

$$\begin{aligned}a_t &= \frac{\partial h_t}{\partial w_h},\\
b_t &= \frac{\partial f(x_{t},h_{t-1},w_h)}{\partial w_h}, \\
c_t &= \frac{\partial f(x_{t},h_{t-1},w_h)}{\partial h_{t-1}},\end{aligned}$$

o cálculo do gradiente em: eqref: `eq_bptt_partial_ht_wh_recur` satisfaz
$a_{t}=b_{t}+c_{t}a_{t-1}$.
Assim,
por :eqref:`eq_bptt_at`,
podemos remover o cálculo recorrente em :eqref:`eq_bptt_partial_ht_wh_recur`
com

$$\frac{\partial h_t}{\partial w_h}=\frac{\partial f(x_{t},h_{t-1},w_h)}{\partial w_h}+\sum_{i=1}^{t-1}\left(\prod_{j=i+1}^{t} \frac{\partial f(x_{j},h_{j-1},w_h)}{\partial h_{j-1}} \right) \frac{\partial f(x_{i},h_{i-1},w_h)}{\partial w_h}.$$
:eqlabel:`eq_bptt_partial_ht_wh_gen`

Embora possamos usar a regra da cadeia para calcular \partial h_t/\partial w_h$ recursivamente, esta cadeia pode ficar muito longa sempre que $t$ for grande. Vamos discutir uma série de estratégias para lidar com esse problema.

### Computação Completa

Obviamente,
podemos apenas calcular a soma total em
:eqref:`eq_bptt_partial_ht_wh_gen`.
Porém,
isso é muito lento e os gradientes podem explodir,
uma vez que mudanças sutis nas condições iniciais podem afetar muito o resultado.
Ou seja, poderíamos ver coisas semelhantes ao efeito borboleta, em que mudanças mínimas nas condições iniciais levam a mudanças desproporcionais no resultado.
Na verdade, isso é bastante indesejável em termos do modelo que queremos estimar.
Afinal, estamos procurando estimadores robustos que generalizem bem. Portanto, essa estratégia quase nunca é usada na prática.

### Truncamento de Etapas de Tempo

Alternativamente,
podemos truncar a soma em
:eqref:`eq_bptt_partial_ht_wh_gen`
após $\tau$  passos.
Isso é o que estivemos discutindo até agora,
como quando separamos os gradientes em :numref:`sec_rnn_scratch`. 
Isso leva a uma *aproximação* do gradiente verdadeiro, simplesmente terminando a soma em
$\partial h_{t-\tau}/\partial w_h$. 
Na prática, isso funciona muito bem. É o que é comumente referido como retropropagação truncada ao longo do tempo :cite:`Jaeger.2002`.
Uma das consequências disso é que o modelo se concentra principalmente na influência de curto prazo, e não nas consequências de longo prazo. Na verdade, isso é *desejável*, pois inclina a estimativa para modelos mais simples e estáveis.

### Truncamento Randomizado ### 

Por último, podemos substituir $\partial h_t/\partial w_h$
por uma variável aleatória que está correta na expectativa, mas trunca a sequência.
Isso é conseguido usando uma sequência de $\xi_t$
com $0 \leq \pi_t \leq 1$ predefinido,
onde $P(\xi_t = 0) = 1-\pi_t$ e  $P(\xi_t = \pi_t^{-1}) = \pi_t$,  portanto  $E[\xi_t] = 1$.
Usamos isso para substituir o gradiente
$\partial h_t/\partial w_h$
em :eqref:`eq_bptt_partial_ht_wh_recur`
com

$$z_t= \frac{\partial f(x_{t},h_{t-1},w_h)}{\partial w_h} +\xi_t \frac{\partial f(x_{t},h_{t-1},w_h)}{\partial h_{t-1}} \frac{\partial h_{t-1}}{\partial w_h}.$$


Segue da definição de $\xi_t$ that $E[z_t] = \partial h_t/\partial w_h$.
Sempre que $\xi_t = 0$ o cálculo recorrente
termina nesse momento no passo $t$.
Isso leva a uma soma ponderada de sequências de comprimentos variados, em que sequências longas são raras, mas apropriadamente sobrecarregadas.
Esta ideia foi proposta por Tallec e Ollivier
:cite:`Tallec.Ollivier.2017`.

### Comparando Estratégias

![Comparando estratégias para computar gradientes em RNNs. De cima para baixo: truncamento aleatório, truncamento regular e computação completa.](../img/truncated-bptt.svg)
:label:`fig_truncated_bptt`



:numref:`fig_truncated_bptt` ilustra as três estratégias ao analisar os primeiros caracteres do livro *The Time Machine* usando retropropagação através do tempo para RNNs:

* A primeira linha é o truncamento aleatório que divide o texto em segmentos de comprimentos variados.
* A segunda linha é o truncamento regular que divide o texto em subsequências do mesmo comprimento. Isso é o que temos feito em experimentos RNN.
* A terceira linha é a retropropagação completa ao longo do tempo que leva a uma expressão computacionalmente inviável.


Infelizmente, embora seja atraente em teoria, o truncamento aleatório não funciona muito melhor do que o truncamento regular, provavelmente devido a uma série de fatores.
Primeiro, o efeito de uma observação após várias etapas de retropropagação no passado é suficiente para capturar dependências na prática.
Segundo, o aumento da variância neutraliza o fato de que o gradiente é mais preciso com mais etapas.
Terceiro, nós realmente *queremos* modelos que tenham apenas um curto intervalo de interações. Conseqüentemente, a retropropagação regularmente truncada ao longo do tempo tem um leve efeito de regularização que pode ser desejável.

## Retropropagação ao Longo do Tempo em Detalhes

Depois de discutir o princípio geral,
vamos discutir a retropropagação ao longo do tempo em detalhes.
Diferente da análise em
:numref:`subsec_bptt_analysis`,
na sequência
vamos mostrar
como calcular
os gradientes da função objetivo
com respeito a todos os parâmetros do modelo decomposto.
Para manter as coisas simples, consideramos
uma RNN sem parâmetros de polarização,
cuja função de ativação na camada oculta
usa o mapeamento de identidade ($\phi(x)=x$).
Para a etapa de tempo $t$,
deixe a entrada de exemplo único e o rótulo ser
$$\mathbf{x}_t \in \mathbb{R}^d$ and $y_t$, respectivamente.
O estado oculto $\mathbf{h}_t \in \mathbb{R}^h$ 
e a saída $\mathbf{o}_t \in \mathbb{R}^q$
são computados como

$$\begin{aligned}\mathbf{h}_t &= \mathbf{W}_{hx} \mathbf{x}_t + \mathbf{W}_{hh} \mathbf{h}_{t-1},\\
\mathbf{o}_t &= \mathbf{W}_{qh} \mathbf{h}_{t},\end{aligned}$$

onde $\mathbf{W}_{hx} \in \mathbb{R}^{h \times d}$,  $\mathbf{W}_{hh} \in \mathbb{R}^{h \times h}$, e
$\mathbf{W}_{qh} \in \mathbb{R}^{q \times h}$
são os parâmetros de peso.
Denotar por $l(\mathbf{o}_t, y_t)$
a perda na etapa de tempo $t$.
Nossa função objetivo,
a perda em etapas de tempo de $T$
desde o início da sequência
é assim

$$L = \frac{1}{T} \sum_{t=1}^T l(\mathbf{o}_t, y_t).$$


A fim de visualizar as dependências entre
variáveis e parâmetros do modelo durante o cálculo
do RNN,
podemos desenhar um gráfico computacional para o modelo,
como mostrado em :numref:`fig_rnn_bptt`.
Por exemplo, o cálculo dos estados ocultos do passo de tempo 3, $\mathbf{h}_3$, depende dos parâmetros do modelo $\mathbf{W}_{hx}$ e $\mathbf{W}_{hh}$,
o estado oculto da última etapa de tempo $\mathbf{h}_2$,
e a entrada do intervalo de tempo atual $\mathbf{x}_3$.

![Gráfico computacional mostrando dependências para um modelo RNN com três intervalos de tempo. Caixas representam variáveis (não sombreadas) ou parâmetros (sombreados) e círculos representam operadores.](../img/rnn-bptt.svg)
:label:`fig_rnn_bptt`

Como acabamos de mencionar, os parâmetros do modelo em :numref:`fig_rnn_bptt` são $\mathbf{W}_{hx}$, $\mathbf{W}_{hh}$, e $\mathbf{W}_{qh}$. 
Geralmente, treinar este modelo
requer cálculo de gradiente em relação a esses parâmetros
$\partial L/\partial \mathbf{W}_{hx}$, $\partial L/\partial \mathbf{W}_{hh}$, e $\partial L/\partial \mathbf{W}_{qh}$.
De acordo com as dependências em :numref:`fig_rnn_bptt`,
nós podemos atravessar
na direção oposta das setas
para calcular e armazenar os gradientes por sua vez.
Para expressar de forma flexível a multiplicação
de matrizes, vetores e escalares de diferentes formas
na regra da cadeia,
nós continuamos a usar
o operador $\text{prod}$ conforme descrito em
:numref:`sec_backprop`.

Em primeiro lugar,
diferenciando a função objetivo
com relação à saída do modelo
a qualquer momento, etapa $t$
é bastante simples:

$$\frac{\partial L}{\partial \mathbf{o}_t} =  \frac{\partial l (\mathbf{o}_t, y_t)}{T \cdot \partial \mathbf{o}_t} \in \mathbb{R}^q.$$
:eqlabel:`eq_bptt_partial_L_ot`

Agora, podemos calcular o gradiente da função objetivo
em relação ao parâmetro $\mathbf{W}_{qh}$
na camada de saída:
$\partial L/\partial \mathbf{W}_{qh} \in \mathbb{R}^{q \times h}$. Com base em :numref:`fig_rnn_bptt`, 
a função objetivo
$L$ depende de $\mathbf{W}_{qh}$ via $\mathbf{o}_1, \ldots, \mathbf{o}_T$. Usar a regra da cadeia produz

$$
\frac{\partial L}{\partial \mathbf{W}_{qh}}
= \sum_{t=1}^T \text{prod}\left(\frac{\partial L}{\partial \mathbf{o}_t}, \frac{\partial \mathbf{o}_t}{\partial \mathbf{W}_{qh}}\right)
= \sum_{t=1}^T \frac{\partial L}{\partial \mathbf{o}_t} \mathbf{h}_t^\top,
$$


onde $\partial L/\partial \mathbf{o}_t$
é fornecido por :eqref:`eq_bptt_partial_L_ot`.

A seguir, conforme mostrado em :numref:`fig_rnn_bptt`,
no tempo final, passo $T$
a função objetivo
$L$ depende do estado oculto $\mathbf{h}_T$ apenas via $\mathbf{o}_T$.
Portanto, podemos facilmente encontrar
o gradiente
$\partial L/\partial \mathbf{h}_T \in \mathbb{R}^h$
usando a regra da cadeia:

$$\frac{\partial L}{\partial \mathbf{h}_T} = \text{prod}\left(\frac{\partial L}{\partial \mathbf{o}_T}, \frac{\partial \mathbf{o}_T}{\partial \mathbf{h}_T} \right) = \mathbf{W}_{qh}^\top \frac{\partial L}{\partial \mathbf{o}_T}.$$
:eqlabel:`eq_bptt_partial_L_hT_final_step`

Fica mais complicado para qualquer passo de tempo $t < T$,
onde a função objetivo $L$ depende de $\mathbf{h}_t$ via $\mathbf{h}_{t+1}$ e $\mathbf{o}_t$.
De acordo com a regra da cadeia,
o gradiente do estado oculto
$\partial L/\partial \mathbf{h}_t \in \mathbb{R}^h$
a qualquer momento, o passo $t<T$ pode ser calculado recorrentemente como:


$$\frac{\partial L}{\partial \mathbf{h}_t} = \text{prod}\left(\frac{\partial L}{\partial \mathbf{h}_{t+1}}, \frac{\partial \mathbf{h}_{t+1}}{\partial \mathbf{h}_t} \right) + \text{prod}\left(\frac{\partial L}{\partial \mathbf{o}_t}, \frac{\partial \mathbf{o}_t}{\partial \mathbf{h}_t} \right) = \mathbf{W}_{hh}^\top \frac{\partial L}{\partial \mathbf{h}_{t+1}} + \mathbf{W}_{qh}^\top \frac{\partial L}{\partial \mathbf{o}_t}.$$
:eqlabel:`eq_bptt_partial_L_ht_recur`

Para análise,
expandindo a computação recorrente
para qualquer etapa de tempo $1 \leq t \leq T$
dá

$$\frac{\partial L}{\partial \mathbf{h}_t}= \sum_{i=t}^T {\left(\mathbf{W}_{hh}^\top\right)}^{T-i} \mathbf{W}_{qh}^\top \frac{\partial L}{\partial \mathbf{o}_{T+t-i}}.$$
:eqlabel:`eq_bptt_partial_L_ht`


Podemos ver em :eqref:`eq_bptt_partial_L_ht` que
este exemplo linear simples já
exibe alguns problemas-chave de modelos de sequência longa: envolve potências potencialmente muito grandes de $\mathbf{W}_{hh}^\top$.
Nele, autovalores menores que 1 desaparecem
e os autovalores maiores que 1 divergem.
Isso é numericamente instável,
que se manifesta na forma de desaparecimento
e gradientes explosivos.
Uma maneira de resolver isso é truncar as etapas de tempo
em um tamanho computacionalmente conveniente conforme discutido em :numref:`subsec_bptt_analysis`. 
Na prática, esse truncamento é efetuado destacando-se o gradiente após um determinado número de etapas de tempo.
Mais tarde
veremos como modelos de sequência mais sofisticados, como a memória de curto prazo longa, podem aliviar ainda mais isso.

Finalmente,
:numref:`fig_rnn_bptt` mostra que
a função objetivo
$L$ depende dos parâmetros do modelo
$\mathbf{W}_{hx}$ e $\mathbf{W}_{hh}$
na camada oculta
via estados ocultos
$\mathbf{h}_1, \ldots, \mathbf{h}_T$.
Para calcular gradientes
com respeito a tais parâmetros
$\partial L / \partial \mathbf{W}_{hx} \in \mathbb{R}^{h \times d}$ e $\partial L / \partial \mathbf{W}_{hh} \in \mathbb{R}^{h \times h}$,
aplicamos a regra da cadeia que dá

$$
\begin{aligned}
\frac{\partial L}{\partial \mathbf{W}_{hx}}
&= \sum_{t=1}^T \text{prod}\left(\frac{\partial L}{\partial \mathbf{h}_t}, \frac{\partial \mathbf{h}_t}{\partial \mathbf{W}_{hx}}\right)
= \sum_{t=1}^T \frac{\partial L}{\partial \mathbf{h}_t} \mathbf{x}_t^\top,\\
\frac{\partial L}{\partial \mathbf{W}_{hh}}
&= \sum_{t=1}^T \text{prod}\left(\frac{\partial L}{\partial \mathbf{h}_t}, \frac{\partial \mathbf{h}_t}{\partial \mathbf{W}_{hh}}\right)
= \sum_{t=1}^T \frac{\partial L}{\partial \mathbf{h}_t} \mathbf{h}_{t-1}^\top,
\end{aligned}
$$


Onde
$\partial L/\partial \mathbf{h}_t$
que é calculado recorrentemente por
:eqref:`eq_bptt_partial_L_hT_final_step`
e
:eqref:`eq_bptt_partial_L_ht_recur`
é a quantidade chave
que afeta a estabilidade numérica.



Como a retropropagação através do tempo
é a aplicação de retropropagação em RNNs,
como explicamos em :numref:`sec_backprop`,
o treinamento de RNNs
alterna a propagação direta com
retropropagação através do tempo.
Além do mais,
retropropagação através do tempo
calcula e armazena os gradientes acima
por sua vez.
Especificamente,
valores intermediários armazenados
são reutilizados
para evitar cálculos duplicados,
como armazenar
$\partial L/\partial \mathbf{h}_t$
para ser usado no cálculo de  $\partial L / \partial \mathbf{W}_{hx}$ e $\partial L / \partial \mathbf{W}_{hh}$.


## Resumo

* A retropropagação através do tempo é meramente uma aplicação da retropropagação para sequenciar modelos com um estado oculto.
* O truncamento é necessário para conveniência computacional e estabilidade numérica, como truncamento regular e truncamento aleatório.
* Altos poderes de matrizes podem levar a autovalores divergentes ou desaparecendo. Isso se manifesta na forma de gradientes explodindo ou desaparecendo.
* Para computação eficiente, os valores intermediários são armazenados em cache durante a retropropagação ao longo do tempo.



## Exercícios

1. Suponha que temos uma matriz simétrica $\mathbf{M} \in \mathbb{R}^{n \times n}$ with eigenvalues $\lambda_i$ cujos autovetores correspondentes são $\mathbf{v}_i$ ($i = 1, \ldots, n$). Sem perda de generalidade, assuma que eles estão ordenados na ordem $|\lambda_i| \geq |\lambda_{i+1}|$. 
    1. Mostre que $\mathbf{M}^k$ tem autovalores $\lambda_i^k$.
    1. Prove que para um vetor aleatório $\mathbf{x} \in \mathbb{R}^n$, com alta probabilidade $\mathbf{M}^k \mathbf{x}$  estará muito alinhado com o autovetor $\mathbf{v}_1$ de $\mathbf{M}$. Formalize esta declaração.
    1. O que o resultado acima significa para gradientes em RNNs?
1. Além do recorte de gradiente, você consegue pensar em outros métodos para lidar com a explosão de gradiente em redes neurais recorrentes?

[Discussions](https://discuss.d2l.ai/t/334)
<!--stackedit_data:
eyJoaXN0b3J5IjpbODMxNDY5NTkwLDU4MDc2MjQ2NywtMTQzNT
gwMTMzMCwtMTYzNTEyOTY0MCwtODc1MDA3ODk0LDE5MDU0NzE5
NTMsNTAwNDY0NTMyLC0xMTAyNzY5NDA0XX0=
-->