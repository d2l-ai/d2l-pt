# Pesquisa de feixe
:label:`sec_beam-search`

Em :numref:`sec_seq2seq`,
previmos a sequência de saída token por token
até o final da sequênceia especial "&lt;eos&gt;" token
é predito.
Nesta secção,
começaremos formalizando essa estratégia de *busca gananciosa*
e explorando problemas com isso,
em seguida, comparamos essa estratégia com outras alternativas:
* pesquisa exaustiva* e *pesquisa por feixe*.

Antes de uma introdução formal à busca gananciosa,
vamos formalizar o problema de pesquisa
usando
a mesma notação matemática de :numref:`sec_seq2seq`.
A qualquer momento, passo $t'$,
a probabilidade de saída do decodificador $y_{t '}$
é condicional
na subsequência de saída
$y_1, \ldots, y_{t'-1}$ antes de $t'$ e
a variável de contexto $\mathbf{c}$ que
codifica as informações da sequência de entrada.
Para quantificar o custo computacional,
denotar por
$\mathcal{Y}$ (contém "&lt;eos&gt;")
o vocabulário de saída.
Portanto, a cardinalidade $\left|\mathcal{Y}\right|$ deste conjunto de vocabulário
é o tamanho do vocabulário.
Vamos também especificar o número máximo de tokens
de uma sequência de saída como $T'$.
Como resultado,
nosso objetivo é procurar um resultado ideal
de todo o
$\mathcal{O}(\left|\mathcal{Y}\right|^{T'})$
possíveis sequências de saída.
Claro,
para todas essas sequências de saída,
porções incluindo e após "&lt;eos&gt;" será descartado
na saída real.

## Busca Gulosa

Primeiro, vamos dar uma olhada em
uma estratégia simples: *busca gananciosa*.
Esta estratégia foi usada para prever sequências em :numref:`sec_seq2seq`.
Em busca gananciosa,
a qualquer momento, etapa $t'$ da sequência de saída,
nós procuramos pelo token
com a maior probabilidade condicional de $\mathcal{Y}$, ou seja,

$$y_{t'} = \operatorname*{argmax}_{y \in \mathcal{Y}} P(y \mid y_1, \ldots, y_{t'-1}, \mathbf{c}),$$

como a saída.
Uma vez que "&lt;eos&gt;" é emitida ou a sequência de saída atingiu seu comprimento máximo $T'$, a sequência de saída é concluída.

Então, o que pode dar errado com a busca gananciosa?
Na verdade,
a *sequência ideal*
deve ser a sequência de saída
com o máximo
$\prod_{t'=1}^{T'} P(y_{t'} \mid y_1, \ldots, y_{t'-1}, \mathbf{c})$,
qual é
a probabilidade condicional de gerar uma sequência de saída com base na sequência de entrada.
Infelizmente, não há garantia
que a sequência ótima será obtida
por busca gananciosa.

![At each time step, greedy search selects the token with the highest conditional probability.](../img/s2s-prob1.svg)
:label:`fig_s2s-prob1`

Vamos ilustrar com um exemplo.
Suponha que existam quatro tokens
"A", "B", "C" e "&lt;eos&gt;" no dicionário de saída.
Em: numref: `fig_s2s-prob1`,
os quatro números em cada etapa de tempo representam as probabilidades condicionais de gerar "A", "B", "C" e "&lt;eos&gt;" nessa etapa de tempo, respectivamente.
Em cada etapa de tempo,
a pesquisa gananciosa seleciona o token com a probabilidade condicional mais alta.
Portanto, a sequência de saída "A", "B", "C" e "&lt;eos&gt;" será previsto
in :numref:`fig_s2s-prob1`.
A probabilidade condicional desta sequência de saída é $0.5\times0.4\times0.4\times0.6 = 0.048$.

![Os quatro números em cada etapa de tempo representam as probabilidades condicionais de gerar "A", "B", "C" e "&lt;eos&gt;" nessa etapa de tempo. Na etapa de tempo 2, o token "C", que tem a segunda maior probabilidade condicional, é selecionado.](../img/s2s-prob2.svg)
:label:`fig_s2s-prob2`

A seguir, vejamos outro exemplo
in :numref:`fig_s2s-prob2`.
Ao contrário de :numref:`fig_s2s-prob1`,
no passo de tempo 2
selecionamos o token "C"
in :numref:`fig_s2s-prob2`,
que tem a *segunda* probabilidade condicional mais alta.
Uma vez que as subseqüências de saída nas etapas de tempo 1 e 2,
em que a etapa de tempo 3 se baseia,
mudaram de "A" e "B" em :numref:`fig_s2s-prob1` para"A" e "C" em :numref:`fig_s2s-prob2`,
a probabilidade condicional de cada token
na etapa 3 também mudou em :numref:`fig_s2s-prob2`.
Suponha que escolhemos o token "B" na etapa de tempo 3.
Agora, a etapa 4 está condicionada a
a subseqüência de saída nas três primeiras etapas de tempo
"A", "C" e "B",
que é diferente de "A", "B" e "C" em :numref:`fig_s2s-prob1`.
Portanto, a probabilidade condicional de gerar cada token na etapa de tempo 4 em :numref:`fig_s2s-prob2` também é diferente daquela em :numref:`fig_s2s-prob1`.
Como resultado,
a probabilidade condicional da sequência de saída "A", "C", "B" e "&lt;eos&gt;"
in :numref:`fig_s2s-prob2`
é $0.5\times0.3 \times0.6\times0.6=0.054$, 
que é maior do que a busca gananciosa em :numref:`fig_s2s-prob1`.
Neste exemplo,
a sequência de saída "A", "B", "C" e "&lt;eos&gt;" obtida pela busca gananciosa não é uma sequência ótima.

## Busca Exaustiva 

Se o objetivo é obter a sequência ideal, podemos considerar o uso de *pesquisa exaustiva*:
enumerar exaustivamente todas as sequências de saída possíveis com suas probabilidades condicionais,
em seguida, envie o um
com a probabilidade condicional mais alta.

Embora possamos usar uma pesquisa exaustiva para obter a sequência ideal,
seu custo computacional $\mathcal{O}(\left|\mathcal{Y}\right|^{T'})$ é provavelmente excessivamente alto.
Por exemplo, quando $|\mathcal{Y}|=10000$ e $T'=10$, precisaremos avaliar $10000^{10} = 10^{40}$ sequências. Isso é quase impossível!
Por outro lado,
o custo computacional da busca gananciosa é
$\mathcal{O}(\left|\mathcal{Y}\right|T')$: 
geralmente é significativamente menor do que
o da pesquisa exaustiva. Por exemplo, quando $|\mathcal{Y}|=10000$ e$T'=10$, só precisamos avaliar $10000\times10=10^5$ sequências.

## Busca de Feixe

Decisões sobre estratégias de busca de sequência
mentem em um espectro,
com perguntas fáceis em qualquer um dos extremos.
E se apenas a precisão importasse?
Obviamente, pesquisa exaustiva.
E se apenas o custo computacional importa?
Claramente, busca gananciosa.
Um aplicativo do mundo real geralmente pergunta
uma pergunta complicada,
em algum lugar entre esses dois extremos.

*Pesquisa de feixe* é uma versão aprimorada da pesquisa gananciosa. Ele tem um hiperparâmetro denominado *tamanho do feixe*, $k$.
Na etapa de tempo 1,
selecionamos $k$ tokens com as probabilidades condicionais mais altas.
Cada um deles será o primeiro símbolo de
$k$ sequências de saída candidatas, respectivamente.
Em cada etapa de tempo subsequente,
com base nas sequências de saída do candidato $k$
na etapa de tempo anterior,
continuamos a selecionar sequências de saída candidatas a $k$
com as maiores probabilidades condicionais
de $k\left|\mathcal{Y}\right|$ escolhas possíveis.

![O processo de busca do feixe (tamanho do feixe: 2, comprimento máximo de uma sequência de saída: 3). As sequências de saída candidatas são $A$, $C$, $AB$, $CE$, $ABD$, e$CED$.](../img/beam-search.svg)
:label:`fig_beam-search`

: numref: `fig_beam-search` demonstra o
processo de pesquisa de feixe com um exemplo.
Suponha que o vocabulário de saída
contém apenas cinco elementos:
$\mathcal{Y} = \{A, B, C, D, E\}$, 
onde um deles é “&lt;eos&gt;”. 
Deixe o tamanho do feixe ser 2 e
o comprimento máximo de uma sequência de saída é 3.
Na etapa de tempo 1,
suponha que os tokens com as probabilidades condicionais mais altas $P(y_1 \mid \mathbf{c})$ sejam $A$ e $C$ No passo de tempo 2, para todos os $y_2 \in \mathcal{Y},$ calculamos

$$\begin{aligned}P(A, y_2 \mid \mathbf{c}) = P(A \mid \mathbf{c})P(y_2 \mid A, \mathbf{c}),\\ P(C, y_2 \mid \mathbf{c}) = P(C \mid \mathbf{c})P(y_2 \mid C, \mathbf{c}),\end{aligned}$$  

e escolha os dois maiores entre esses dez valores, digamos
$P(A, B \mid \mathbf{c})$ e$P(C, E \mid \mathbf{c})$.
Depois para o passo de tempo 3, para todos $y_3 \in \mathcal{Y}$, nós computamos

$$\begin{aligned}P(A, B, y_3 \mid \mathbf{c}) = P(A, B \mid \mathbf{c})P(y_3 \mid A, B, \mathbf{c}),\\P(C, E, y_3 \mid \mathbf{c}) = P(C, E \mid \mathbf{c})P(y_3 \mid C, E, \mathbf{c}),\end{aligned}$$ 

e escolha os dois maiores entre esses dez valores, digamos
$P(A, B, D \mid \mathbf{c})$  e $P(C, E, D \mid  \mathbf{c}).$
Como resultado, obtemos seis sequências de saída de candidatos:  (i) $A$; (ii) $C$; (iii) $A$, $B$; (iv) $C$, $E$; (v) $A$, $B$, $D$; e (vi) $C$, $E$, $D$. 

No final, obtemos o conjunto de sequências de saída candidatas finais com base nessas seis sequências (por exemplo, descarte porções incluindo e após “&lt;eos&gt;”).
Então
escolhemos a sequência com a maior das seguintes pontuações como a sequência de saída:

$$ \frac{1}{L^\alpha} \log P(y_1, \ldots, y_{L}) = \frac{1}{L^\alpha} \sum_{t'=1}^L \log P(y_{t'} \mid y_1, \ldots, y_{t'-1}, \mathbf{c}),$$
:eqlabel:`eq_beam-search-score`

onde $L$ é o comprimento da sequência candidata final e $\alpha$ é geralmente definido como 0,75.
Uma vez que uma sequência mais longa tem mais termos logarítmicos na soma de :eqref:`eq_beam-search-score`,
o termo $L^\alpha$ no denominador penaliza
longas sequências.

O custo computacional da pesquisa do feixe é $\mathcal{O}(k\left|\mathcal{Y}\right|T')$.
Esse resultado está entre o da busca gananciosa e o da busca exaustiva. Na verdade, a pesquisa gananciosa pode ser tratada como um tipo especial de pesquisa de feixe com
um tamanho de feixe de 1.
Com uma escolha flexível do tamanho do feixe,
pesquisa de feixe fornece uma compensação entre
precisão versus custo computacional.



## Sumário

* As estratégias de busca de sequência incluem busca gananciosa, busca exaustiva e busca de feixe.
* A pesquisa de feixe oferece uma compensação entre precisão e custo computacional por meio de sua escolha flexível do tamanho do feixe.

## Exercícios

1. Podemos tratar a pesquisa exaustiva como um tipo especial de pesquisa de feixe? Por que ou por que não?
2. Aplique a pesquisa de feixe no problema de tradução automática em :numref:`sec_seq2seq`. Como o tamanho do feixe afeta os resultados da tradução e a velocidade de previsão?
3. Usamos modelagem de linguagem para gerar texto seguindo prefixos fornecidos pelo usuário em :numref:`sec_rnn_scratch`. Que tipo de estratégia de pesquisa ele usa? Você pode melhorar isso?

[Discussão](https://discuss.d2l.ai/t/338)
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTEwNzcwOTI4MjYsMjUyMDc2NDc5XX0=
-->