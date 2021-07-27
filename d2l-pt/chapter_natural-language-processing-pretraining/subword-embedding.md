# Incorporação de subpalavra
:label:`sec_fasttext`

As palavras em inglês geralmente têm estruturas internas e métodos de formação. Por exemplo, podemos deduzir a relação entre "dog", "dogs" e "dogcatcher" por sua grafia. Todas essas palavras têm a mesma raiz, "cachorro", mas usam sufixos diferentes para mudar o significado da palavra. Além disso, essa associação pode ser estendida a outras palavras. Por exemplo, a relação entre "cachorro" e "cachorros" é exatamente como a relação entre "cat" e "cats". A relação entre "boy" e "boyfriend" é igual à relação entre "girl" e "girlfriend". Essa característica não é exclusiva do inglês. Em francês e espanhol, muitos verbos podem ter mais de 40 formas diferentes, dependendo do contexto. Em finlandês, um substantivo pode ter mais de 15 formas. Na verdade, a morfologia, que é um importante ramo da linguística, estuda a estrutura interna e a formação das palavras.

## fastText

No word2vec, não usamos informações morfológicas diretamente. Em ambos os
modelo skip-gram e modelo de saco de palavras contínuo, usamos diferentes vetores para
representam palavras com diferentes formas. Por exemplo, "cachorro" e "cachorros" são
representado por dois vetores diferentes, enquanto a relação entre esses dois
vetores não é representado diretamente no modelo. Em vista disso, fastText :cite:`Bojanowski.Grave.Joulin.ea.2017`
propõe o método de incorporação de subpalavra, tentando assim introduzir
informação morfológica no modelo skip-gram em word2vec.

Em fastText, cada palavra central é representada como uma coleção de subpalavras. A seguir, usamos a palavra "onde" como exemplo para entender como as subpalavras são formadas. Primeiro, adicionamos os caracteres especiais “&lt;” e “&gt;” no início e no final da palavra para distinguir as subpalavras usadas como prefixos e sufixos. Em seguida, tratamos a palavra como uma sequência de caracteres para extrair os $n$-gramas. Por exemplo, quando $n=3$, podemos obter todas as subpalavras com um comprimento de $3$:

$$\textrm{"<wh"}, \ \textrm{"whe"}, \ \textrm{"her"}, \ \textrm{"ere"}, \ \textrm{"re>"},$$

e a subpalavra especial $\textrm{"<where>"}$.

Em fastText, para uma palavra $w$, registramos a união de todas as suas subpalavras com comprimento de $3$ a $6$ e as subpalavras especiais como $\mathcal{G}_w$. Assim, o dicionário é a união da coleção de subpalavras de todas as palavras. Suponha que o vetor da subpalavra $g$ no dicionário seja $\mathbf{z}_g$. Então, o vetor de palavra central $\mathbf{u}_w$ para a palavra $w$ no modelo de grama de salto pode ser expresso como

$$\mathbf{u}_w = \sum_{g\in\mathcal{G}_w} \mathbf{z}_g.$$

O resto do processo fastText é consistente com o modelo skip-gram, portanto, não é repetido aqui. Como podemos ver, em comparação com o modelo skip-gram, o dicionário em fastText é maior, resultando em mais parâmetros do modelo. Além disso, o vetor de uma palavra requer a soma de todos os vetores de subpalavra, o que resulta em maior complexidade de computação. No entanto, podemos obter vetores melhores para palavras complexas mais incomuns, mesmo palavras que não existem no dicionário, olhando para outras palavras com estruturas semelhantes.


## Codificação de par de bytes
:label:`subsec_Byte_Pair_Encoding`

Em fastText, todas as subpalavras extraídas devem ter os comprimentos especificados, como $3$ a $6$, portanto, o tamanho do vocabulário não pode ser predefinido.
Para permitir subpalavras de comprimento variável em um vocabulário de tamanho fixo,
podemos aplicar um algoritmo de compressão
chamado de *codificação de par de bytes* (BPE) para extrair subpalavras :cite:`Sennrich.Haddow.Birch.2015`.

A codificação de pares de bytes realiza uma análise estatística do conjunto de dados de treinamento para descobrir símbolos comuns em uma palavra,
como caracteres consecutivos de comprimento arbitrário.
Começando com símbolos de comprimento $1$,
a codificação de pares de bytes mescla iterativamente o par mais frequente de símbolos consecutivos para produzir novos símbolos mais longos.
Observe que, para eficiência, os pares que cruzam os limites das palavras não são considerados.
No final, podemos usar esses símbolos como subpalavras para segmentar palavras.
A codificação de pares de bytes e suas variantes foram usadas para representações de entrada em modelos populares de pré-treinamento de processamento de linguagem natural, como GPT-2 :cite:`Radford.Wu.Child.ea.2019` e RoBERTa :cite:`Liu.Ott.Goyal.ea.2019`.
A seguir, ilustraremos como funciona a codificação de pares de bytes.

Primeiro, inicializamos o vocabulário de símbolos como todos os caracteres minúsculos do inglês, um símbolo especial de fim de palavra `'_'` e um símbolo especial desconhecido`' [UNK] '`.

```{.python .input}
#@tab all
import collections

symbols = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
           'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
           '_', '[UNK]']
```

Uma vez que não consideramos pares de símbolos que cruzam os limites das palavras,
precisamos apenas de um dicionário `raw_token_freqs` que mapeia as palavras para suas frequências (número de ocorrências)
em um conjunto de dados.
Observe que o símbolo especial `'_'` é anexado a cada palavra para que
podemos facilmente recuperar uma sequência de palavras (por exemplo, "um homem mais alto")
a partir de uma sequência de símbolos de saída (por exemplo, "a_ tall er_ man").
Uma vez que iniciamos o processo de fusão a partir de um vocabulário de apenas caracteres únicos e símbolos especiais, o espaço é inserido entre cada par de caracteres consecutivos dentro de cada palavra (chaves do dicionário `token_freqs`).
Em outras palavras, o espaço é o delimitador entre os símbolos de uma palavra.

```{.python .input}
#@tab all
raw_token_freqs = {'fast_': 4, 'faster_': 3, 'tall_': 5, 'taller_': 4}
token_freqs = {}
for token, freq in raw_token_freqs.items():
    token_freqs[' '.join(list(token))] = raw_token_freqs[token]
token_freqs
```

Definimos a seguinte função `get_max_freq_pair` que
retorna o par mais frequente de símbolos consecutivos em uma palavra,
onde as palavras vêm de chaves do dicionário de entrada `token_freqs`.

```{.python .input}
#@tab all
def get_max_freq_pair(token_freqs):
    pairs = collections.defaultdict(int)
    for token, freq in token_freqs.items():
        symbols = token.split()
        for i in range(len(symbols) - 1):
            # Key of `pairs` is a tuple of two consecutive symbols
            pairs[symbols[i], symbols[i + 1]] += freq
    return max(pairs, key=pairs.get)  # Key of `pairs` with the max value
```

Como uma abordagem gananciosa com base na frequência de símbolos consecutivos,
a codificação de pares de bytes usará a seguinte função `merge_symbols` para mesclar o par mais frequente de símbolos consecutivos para produzir novos símbolos.

```{.python .input}
#@tab all
def merge_symbols(max_freq_pair, token_freqs, symbols):
    symbols.append(''.join(max_freq_pair))
    new_token_freqs = dict()
    for token, freq in token_freqs.items():
        new_token = token.replace(' '.join(max_freq_pair),
                                  ''.join(max_freq_pair))
        new_token_freqs[new_token] = token_freqs[token]
    return new_token_freqs
```

Agora realizamos iterativamente o algoritmo de codificação de par de bytes sobre as chaves do dicionário `token_freqs`. Na primeira iteração, o par mais frequente de símbolos consecutivos são `'t'` e`' a'`, portanto, a codificação de pares de bytes os mescla para produzir um novo símbolo `'ta'`. Na segunda iteração, a codificação do par de bytes continua a mesclar `'ta'` e`' l'` para resultar em outro novo símbolo `'tal'`.

```{.python .input}
#@tab all
num_merges = 10
for i in range(num_merges):
    max_freq_pair = get_max_freq_pair(token_freqs)
    token_freqs = merge_symbols(max_freq_pair, token_freqs, symbols)
    print(f'merge #{i + 1}:', max_freq_pair)
```

Após 10 iterações de codificação de par de bytes, podemos ver que a lista de "símbolos" agora contém mais 10 símbolos que são mesclados iterativamente de outros símbolos.

```{.python .input}
#@tab all
print(symbols)
```

Para o mesmo conjunto de dados especificado nas chaves do dicionário `raw_token_freqs`,
cada palavra no conjunto de dados agora é segmentada por subpalavras "fast_", "fast", "er_", "tall_" e "tall"
como resultado do algoritmo de codificação de par de bytes.
Por exemplo, as palavras "faster_" e "taller_" são segmentadas como "fast er_" e "tall er_", respectivamente.

```{.python .input}
#@tab all
print(list(token_freqs.keys()))
```

Observe que o resultado da codificação do par de bytes depende do conjunto de dados que está sendo usado.
Também podemos usar as sub-palavras aprendidas com um conjunto de dados
para segmentar palavras de outro conjunto de dados.
Como uma abordagem gananciosa, a seguinte função `segment_BPE` tenta dividir as palavras nas subpalavras mais longas possíveis a partir dos `symbols` do argumento de entrada.

```{.python .input}
#@tab all
def segment_BPE(tokens, symbols):
    outputs = []
    for token in tokens:
        start, end = 0, len(token)
        cur_output = []
        # Segment token with the longest possible subwords from symbols
        while start < len(token) and start < end:
            if token[start: end] in symbols:
                cur_output.append(token[start: end])
                start = end
                end = len(token)
            else:
                end -= 1
        if start < len(token):
            cur_output.append('[UNK]')
        outputs.append(' '.join(cur_output))
    return outputs
```

A seguir, usamos as sub-palavras em 'symbols' da lista, que é aprendido com o conjunto de dados acima mencionado,
para segmentar `tokens` que representam outro conjunto de dados.

```{.python .input}
#@tab all
tokens = ['tallest_', 'fatter_']
print(segment_BPE(tokens, symbols))
```

## Sumário

* FastText propõe um método de incorporação de subpalavra. Com base no modelo skip-gram em word2vec, ele representa o vetor de palavra central como a soma dos vetores de subpalavra da palavra.
* A incorporação de subpalavra utiliza os princípios da morfologia, o que geralmente melhora a qualidade das representações de palavras incomuns.
* A codificação de pares de bytes realiza uma análise estatística do conjunto de dados de treinamento para descobrir símbolos comuns em uma palavra. Como uma abordagem gananciosa, a codificação de pares de bytes mescla iterativamente o par mais frequente de símbolos consecutivos.


## Exercícios

1. Quando há muitas subpalavras (por exemplo, 6 palavras em inglês resultam em cerca de $3\times 10^8$ combinações), quais são os problemas? Você consegue pensar em algum método para resolvê-los? Dica: consulte o final da seção 3.2 do artigo fastText :cite:`Bojanowski.Grave.Joulin.ea.2017`.
1. Como você pode projetar um modelo de incorporação de subpalavra com base no modelo de saco de palavras contínuo?
1. Para obter um vocabulário de tamanho $m$, quantas operações de fusão são necessárias quando o tamanho inicial do vocabulário de símbolos é $n$?
1. Como podemos estender a ideia de codificação de par de bytes para extrair frases?

:begin_tab:`mxnet`
[Discussão](https://discuss.d2l.ai/t/386)
:end_tab:
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTEzMzk5MDY3ODYsLTEzMjIwOTgxOTFdfQ
==
-->