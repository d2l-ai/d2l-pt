# Incorporação de palavras com vetores globais (GloVe)
:label:`sec_glove`

Primeiro, devemos revisar o modelo skip-gram no word2vec. A probabilidade condicional $P(w_j\mid w_i)$ expressa no modelo skip-gram usando a operação softmax será registrada como $q_{ij}$, ou seja:

$$q_{ij}=\frac{\exp(\mathbf{u}_j^\top \mathbf{v}_i)}{ \sum_{k \in \mathcal{V}} \text{exp}(\mathbf{u}_k^\top \mathbf{v}_i)},$$

onde $\mathbf{v}_i$ e $\mathbf{u}_i$ são as representações vetoriais da palavra $w_i$ do índice $i$ como a palavra central e a palavra de contexto, respectivamente, e $\mathcal{V} = \{0, 1, \ldots, |\mathcal{V}|-1\}$ é o conjunto de índices de vocabulário.

Para a palavra $w_i$, ela pode aparecer no conjunto de dados várias vezes. Coletamos todas as palavras de contexto sempre que $w_i$ é uma palavra central e mantemos duplicatas, denotadas como multiset $\mathcal{C}_i$. O número de um elemento em um multiconjunto é chamado de multiplicidade do elemento. Por exemplo, suponha que a palavra $w_i$ apareça duas vezes no conjunto de dados: as janelas de contexto quando essas duas $w_i$ se tornam palavras centrais na sequência de texto contêm índices de palavras de contexto $2, 1, 5, 2$ e $2, 3, 2, 1$. Então, multiset $\mathcal{C}_i = \{1, 1, 2, 2, 2, 2, 3, 5\}$, onde a multiplicidade do elemento 1 é 2, a multiplicidade do elemento 2 é 4 e multiplicidades de os elementos 3 e 5 são 1. Denote a multiplicidade do elemento $j$ no multiset $\mathcal{C}_i$ as $x_{ij}$: é o número da palavra $w_j$ em todas as janelas de contexto para a palavra central $w_i$ em todo o conjunto de dados. Como resultado, a função de perda do modelo skip-gram pode ser expressa de uma maneira diferente:

$$-\sum_{i\in\mathcal{V}}\sum_{j\in\mathcal{V}} x_{ij} \log\,q_{ij}.$$

Adicionamos o número de todas as palavras de contexto para a palavra alvo central $w_i$ para obter $x_i$, e registramos a probabilidade condicional $x_{ij}/x_i$ para gerar a palavra de contexto $w_j$ com base na palavra alvo central $w_i$ como $p_{ij}$. Podemos reescrever a função de perda do modelo skip-gram como

$$-\sum_{i\in\mathcal{V}} x_i \sum_{j\in\mathcal{V}} p_{ij} \log\,q_{ij}.$$

Na fórmula acima, $\sum_{j\in\mathcal{V}} p_{ij} \log\,q_{ij}$ calcula a distribuição de probabilidade condicional $p_{ij}$ para geração de palavras de contexto com base na central palavra-alvo $w_i$ e a entropia cruzada da distribuição de probabilidade condicional $q_{ij}$ prevista pelo modelo. A função de perda é ponderada usando a soma do número de palavras de contexto com a palavra alvo central $w_i$. Se minimizarmos a função de perda da fórmula acima, seremos capazes de permitir que a distribuição de probabilidade condicional prevista se aproxime o mais próximo possível da verdadeira distribuição de probabilidade condicional.

No entanto, embora o tipo mais comum de função de perda, a perda de entropia cruzada
função às vezes não é uma boa escolha. Por um lado, como mencionamos em
:numref:`sec_approx_train`
o custo de deixar o
a previsão do modelo $q_{ij}$ torna-se a distribuição de probabilidade legal tem a soma
de todos os itens em todo o dicionário em seu denominador. Isso pode facilmente levar
a sobrecarga computacional excessiva. Por outro lado, costuma haver muitos
palavras incomuns no dicionário e raramente aparecem no conjunto de dados. No
função de perda de entropia cruzada, a previsão final da probabilidade condicional
a distribuição em um grande número de palavras incomuns provavelmente será imprecisa.



## O modelo GloVe

Para resolver isso, GloVe :cite:`Pennington.Socher.Manning.2014`, um modelo de incorporação de palavras que veio depois de word2vec, adota
perda quadrada e faz três alterações no modelo de grama de salto com base nessa perda.

1. Aqui, usamos as variáveis de distribuição não probabilística $p'_{ij}=x_{ij}$ e $q'_{ij}=\exp(\mathbf{u}_j^\top \mathbf{v}_i)$ e pegue seus logs. Portanto, obtemos a perda quadrada $\left(\log\,p'_{ij} - \log\,q'_{ij}\right)^2 = \left(\mathbf{u}_j^\top \mathbf{v}_i - \log\,x_{ij}\right)^2$.
2. Adicionamos dois parâmetros do modelo escalar para cada palavra $w_i$: os termos de polarização $b_i$ (para palavras-alvo centrais) e $c_i$ (para palavras de contexto).
3. Substitua o peso de cada perda pela função $h(x_{ij})$. A função de peso $h(x)$ é uma função monótona crescente com o intervalo $[0, 1]$.

Portanto, o objetivo do GloVe é minimizar a função de perda.

$$\sum_{i\in\mathcal{V}} \sum_{j\in\mathcal{V}} h(x_{ij}) \left(\mathbf{u}_j^\top \mathbf{v}_i + b_i + c_j - \log\,x_{ij}\right)^2.$$

Aqui, temos uma sugestão para a escolha da função de peso $h(x)$: quando $x<c$ (por exemplo, $c=100$), faça $h(x) = (x/c) ^\alpha$ (por exemplo, $\alpha = 0.75$), caso contrário, faça $h(x) = 1$. Como $h(0)=0$, o termo de perda ao quadrado de $x_{ij}=0$ pode ser simplesmente ignorado. Quando usamos o minibatch SGD para treinamento, conduzimos uma amostragem aleatória para obter um minibatch diferente de zero $x_{ij}$ de cada intervalo de tempo e calculamos o gradiente para atualizar os parâmetros do modelo. Esses $x_{ij}$ diferentes de zero são calculados antecipadamente com base em todo o conjunto de dados e contêm estatísticas globais para o conjunto de dados. Portanto, o nome GloVe é retirado de "Vetores globais".

Observe que se a palavra $w_i$ aparecer na janela de contexto da palavra $w_j$, então a palavra $w_j$ também aparecerá na janela de contexto da palavra $w_i$. Portanto, $x_{ij}=x_{ji}$. Ao contrário de word2vec, GloVe ajusta o simétrico $\log\, x_{ij}$ no lugar da probabilidade condicional assimétrica $p_{ij}$. Portanto, o vetor de palavra alvo central e o vetor de palavra de contexto de qualquer palavra são equivalentes no GloVe. No entanto, os dois conjuntos de vetores de palavras que são aprendidos pela mesma palavra podem ser diferentes no final devido a valores de inicialização diferentes. Depois de aprender todos os vetores de palavras, o GloVe usará a soma do vetor da palavra-alvo central e do vetor da palavra de contexto como o vetor da palavra final para a palavra.


## Compreendendo o GloVe a partir das razões de probabilidade condicionais

Também podemos tentar entender a incorporação de palavras GloVe de outra perspectiva. Continuaremos a usar os símbolos anteriores nesta seção, $P(w_j \mid w_i)$ representa a probabilidade condicional de gerar a palavra de contexto $w_j$ com a palavra alvo central $w_i$ no conjunto de dados, e será registrado como $p_{ij}$. A partir de um exemplo real de um grande corpus, temos aqui os seguintes dois conjuntos de probabilidades condicionais com "gelo" e "vapor" como palavras-alvo centrais e a proporção entre elas:

|$w_k$=|solid|gas|water|fashion|
|--:|:-:|:-:|:-:|:-:|
|$p_1=P(w_k\mid \text{ice})$|0.00019|0.000066|0.003|0.000017|
|$p_2=P(w_k\mid\text{steam})$|0.000022|0.00078|0.0022|0.000018|
|$p_1/p_2$|8.9|0.085|1.36|0.96|

Seremos capazes de observar fenômenos como:

* Para uma palavra $w_k$ que está relacionada a "gelo", mas não a "vapor", como $w_k=\text{solid}$, esperaríamos uma razão de probabilidade condicional maior, como o valor 8,9 na última linha da tabela acima.
* Para uma palavra $w_k$ que está relacionada a "vapor", mas não a "gelo", como $w_k=\text{gas}$, esperaríamos uma razão de probabilidade condicional menor, como o valor 0,085 na última linha da tabela acima.
* Para uma palavra $w_k$ que está relacionada a "gelo" e "vapor", como $w_k=\text{agua}$, esperaríamos uma razão de probabilidade condicional próxima de 1, como o valor 1,36 no último linha da tabela acima.
* Para uma palavra $w_k$ que não está relacionada a "gelo" ou "vapor", como $w_k=\text{fashion}$, esperaríamos uma razão de probabilidade condicional próxima de 1, como o valor 0,96 no último linha da tabela acima.

Podemos ver que a razão de probabilidade condicional pode representar a relação entre diferentes palavras de forma mais intuitiva. Podemos construir uma função de vetor de palavras para ajustar a razão de probabilidade condicional de forma mais eficaz. Como sabemos, para obter qualquer razão deste tipo são necessárias três palavras $w_i$, $w_j$, e $w_k$. A razão de probabilidade condicional com $w_i$ como palavra alvo central é ${p_{ij}}/{p_{ik}}$. Podemos encontrar uma função que usa vetores de palavras para ajustar essa razão de probabilidade condicional.

$$f(\mathbf{u}_j, \mathbf{u}_k, {\mathbf{v}}_i) \approx \frac{p_{ij}}{p_{ik}}.$$

O projeto possível da função $f$ aqui não será exclusivo. Precisamos apenas considerar uma possibilidade mais razoável. Observe que a razão de probabilidade condicional é escalar, podemos limitar $f$ para ser uma função escalar: $f(\mathbf{u}_j, \mathbf{u}_k, {\mathbf{v}}_i) = f\left((\mathbf{u}_j - \mathbf{u}_k)^\top {\mathbf{v}}_i\right)$. Depois de trocar o índice $j$ por $k$, seremos capazes de ver que a função $f$ satisfaz a condição $f(x)f(-x)=1$, então uma possibilidade poderia ser $f(x)=\exp(x)$. Assim: 

$$f(\mathbf{u}_j, \mathbf{u}_k, {\mathbf{v}}_i) = \frac{\exp\left(\mathbf{u}_j^\top {\mathbf{v}}_i\right)}{\exp\left(\mathbf{u}_k^\top {\mathbf{v}}_i\right)} \approx \frac{p_{ij}}{p_{ik}}.$$

Uma possibilidade que satisfaz o lado direito do sinal de aproximação é $\exp\left(\mathbf{u}_j^\top {\mathbf{v}}_i\right) \approx \alpha p_{ij}$, onde $\alpha$ é uma constante. Considerando que $p_{ij}=x_{ij}/x_i$, após tomar o logaritmo, obtemos $\mathbf{u}_j^\top {\mathbf{v}}_i \approx \log\,\alpha + \log\,x_{ij} - \log\,x_i$. Usamos termos de polarização adicionais para ajustar $- \log\, \alpha + \log\, x_i$, como o termo de polarização de palavra-alvo central $b_i$ e o termo de polarização de palavra de contexto $c_j$:

$$\mathbf{u}_j^\top \mathbf{v}_i + b_i + c_j \approx \log(x_{ij}).$$

Pegando o erro quadrado e ponderando os lados esquerdo e direito da fórmula acima, podemos obter a função de perda de GloVe.


## Sumário

* Em alguns casos, a função de perda de entropia cruzada pode ter uma desvantagem. O GloVe usa a perda quadrática e o vetor de palavras para ajustar as estatísticas globais calculadas antecipadamente com base em todo o conjunto de dados.
* O vetor da palavra alvo central e o vetor da palavra de contexto de qualquer palavra são equivalentes no GloVe.

## Exercícios

1. Se uma palavra aparecer na janela de contexto de outra palavra, como podemos usar o
   distância entre eles na sequência de texto para redesenhar o método para
   calculando a probabilidade condicional $p_{ij}$? Dica: Consulte a seção 4.2 do
   paper GloVe :cite:`Pennington.Socher.Manning.2014`.
1. Para qualquer palavra, o termo de polarização da palavra-alvo central e o termo de polarização da palavra de contexto serão equivalentes entre si no GloVe? Por quê?

[Discussão](https://discuss.d2l.ai/t/385)
<!--stackedit_data:
eyJoaXN0b3J5IjpbMTI0MDIzMjg5MSwtMTAxNzg3NjI4MV19
-->