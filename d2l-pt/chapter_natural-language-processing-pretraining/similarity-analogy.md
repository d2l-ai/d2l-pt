# Encontrando sinônimos e analogias
:label:`sec_synonyms`

Em :numref:`sec_word2vec_pretraining` treinamos um modelo de incorporação de palavras word2vec
em um conjunto de dados de pequena escala e procurou por sinônimos usando a similaridade de cosseno
de vetores de palavras. Na prática, vetores de palavras pré-treinados em um corpus de grande escala
muitas vezes pode ser aplicado a tarefas de processamento de linguagem natural downstream. Esta
seção irá demonstrar como usar esses vetores de palavras pré-treinados para encontrar
sinônimos e analogias. Continuaremos a aplicar vetores de palavras pré-treinados em
seções subsequentes.

```{.python .input}
from d2l import mxnet as d2l
from mxnet import np, npx
import os

npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
import os
```

## Usando vetores de palavras pré-treinados

Abaixo lista os embeddings GloVe pré-treinados de dimensões 50, 100 e 300,
que pode ser baixado do [site do GloVe](https://nlp.stanford.edu/projects/glove/).
Os fastText pré-treinados embarcados estão disponíveis em vários idiomas.
Aqui, consideramos uma versão em inglês ("wiki.en" 300-dimensional) que pode ser baixada do
[site fastText](https://fasttext.cc/).

```{.python .input}
#@tab all
#@save
d2l.DATA_HUB['glove.6b.50d'] = (d2l.DATA_URL + 'glove.6B.50d.zip',
                                '0b8703943ccdb6eb788e6f091b8946e82231bc4d')

#@save
d2l.DATA_HUB['glove.6b.100d'] = (d2l.DATA_URL + 'glove.6B.100d.zip',
                                 'cd43bfb07e44e6f27cbcc7bc9ae3d80284fdaf5a')

#@save
d2l.DATA_HUB['glove.42b.300d'] = (d2l.DATA_URL + 'glove.42B.300d.zip',
                                  'b5116e234e9eb9076672cfeabf5469f3eec904fa')

#@save
d2l.DATA_HUB['wiki.en'] = (d2l.DATA_URL + 'wiki.en.zip',
                           'c1816da3821ae9f43899be655002f6c723e91b88')
```

Definimos a seguinte classe `TokenEmbedding` para carregar os embeddings pré-treinados Glove e fastText acima.

```{.python .input}
#@tab all
#@save
class TokenEmbedding:
    """Token Embedding."""
    def __init__(self, embedding_name):
        self.idx_to_token, self.idx_to_vec = self._load_embedding(
            embedding_name)
        self.unknown_idx = 0
        self.token_to_idx = {token: idx for idx, token in
                             enumerate(self.idx_to_token)}

    def _load_embedding(self, embedding_name):
        idx_to_token, idx_to_vec = ['<unk>'], []
        data_dir = d2l.download_extract(embedding_name)
        # GloVe website: https://nlp.stanford.edu/projects/glove/
        # fastText website: https://fasttext.cc/
        with open(os.path.join(data_dir, 'vec.txt'), 'r') as f:
            for line in f:
                elems = line.rstrip().split(' ')
                token, elems = elems[0], [float(elem) for elem in elems[1:]]
                # Skip header information, such as the top row in fastText
                if len(elems) > 1:
                    idx_to_token.append(token)
                    idx_to_vec.append(elems)
        idx_to_vec = [[0] * len(idx_to_vec[0])] + idx_to_vec
        return idx_to_token, d2l.tensor(idx_to_vec)

    def __getitem__(self, tokens):
        indices = [self.token_to_idx.get(token, self.unknown_idx)
                   for token in tokens]
        vecs = self.idx_to_vec[d2l.tensor(indices)]
        return vecs

    def __len__(self):
        return len(self.idx_to_token)
```

Em seguida, usamos embeddings GloVe de 50 dimensões pré-treinados em um subconjunto da Wikipedia. A incorporação de palavras correspondente é baixada automaticamente na primeira vez que criamos uma instância de incorporação de palavras pré-treinada.

```{.python .input}
#@tab all
glove_6b50d = TokenEmbedding('glove.6b.50d')
```

Produza o tamanho do dicionário. O dicionário contém $400.000$ palavras e um token especial desconhecido.

```{.python .input}
#@tab all
len(glove_6b50d)
```

Podemos usar uma palavra para obter seu índice no dicionário ou podemos obter a palavra de seu índice.

```{.python .input}
#@tab all
glove_6b50d.token_to_idx['beautiful'], glove_6b50d.idx_to_token[3367]
```

## Aplicação de vetores de palavras pré-treinados

Abaixo, demonstramos a aplicação de vetores de palavras pré-treinados, usando GloVe como exemplo.

### Encontrando sinônimos

Aqui, reimplementamos o algoritmo usado para pesquisar sinônimos por cosseno
similaridade introduzida em :numref:`sec_word2vec`

A fim de reutilizar a lógica para buscar os $k$ vizinhos mais próximos quando
buscando analogias, encapsulamos esta parte da lógica separadamente no `knn`
função ($k$-vizinhos mais próximos).

```{.python .input}
def knn(W, x, k):
    # The added 1e-9 is for numerical stability
    cos = np.dot(W, x.reshape(-1,)) / (
        np.sqrt(np.sum(W * W, axis=1) + 1e-9) * np.sqrt((x * x).sum()))
    topk = npx.topk(cos, k=k, ret_typ='indices')
    return topk, [cos[int(i)] for i in topk]
```

```{.python .input}
#@tab pytorch
def knn(W, x, k):
    # The added 1e-9 is for numerical stability
    cos = torch.mv(W, x.reshape(-1,)) / (
        torch.sqrt(torch.sum(W * W, axis=1) + 1e-9) *
        torch.sqrt((x * x).sum()))
    _, topk = torch.topk(cos, k=k)
    return topk, [cos[int(i)] for i in topk]
```

Em seguida, buscamos sinônimos pré-treinando a instância do vetor de palavras `embed`.

```{.python .input}
#@tab all
def get_similar_tokens(query_token, k, embed):
    topk, cos = knn(embed.idx_to_vec, embed[[query_token]], k + 1)
    for i, c in zip(topk[1:], cos[1:]):  # Remove input words
        print(f'cosine sim={float(c):.3f}: {embed.idx_to_token[int(i)]}')
```

O dicionário de instância de vetor de palavras pré-treinadas `glove_6b50d` já criado contém 400.000 palavras e um token especial desconhecido. Excluindo palavras de entrada e palavras desconhecidas, procuramos as três palavras que têm o significado mais semelhante a "chip".

```{.python .input}
#@tab all
get_similar_tokens('chip', 3, glove_6b50d)
```

A seguir, procuramos os sinônimos de "baby" e "beautiful".

```{.python .input}
#@tab all
get_similar_tokens('baby', 3, glove_6b50d)
```

```{.python .input}
#@tab all
get_similar_tokens('beautiful', 3, glove_6b50d)
```

### Encontrando Analogias

Além de buscar sinônimos, também podemos usar o vetor de palavras pré-treinadas para buscar analogias entre palavras. Por exemplo, “man”:“woman”::“son”:“daughter” é um exemplo de analogia, “man” está para “woman” como “son” está para “daughter”. O problema de buscar analogias pode ser definido da seguinte forma: para quatro palavras na relação analógica $a : b :: c : d$, dadas as três primeiras palavras, $a$, $b$ e $c$, queremos encontre $d$. Suponha que a palavra vetor para a palavra $w$ seja $\text{vec}(w)$. Para resolver o problema de analogia, precisamos encontrar o vetor de palavras que é mais semelhante ao vetor de resultado de $\text{vec}(c)+\text{vec}(b)-\text{vec}(a)$.

```{.python .input}
#@tab all
def get_analogy(token_a, token_b, token_c, embed):
    vecs = embed[[token_a, token_b, token_c]]
    x = vecs[1] - vecs[0] + vecs[2]
    topk, cos = knn(embed.idx_to_vec, x, 1)
    return embed.idx_to_token[int(topk[0])]  # Remove unknown words
```

Verifique a analogia "male-female".

```{.python .input}
#@tab all
get_analogy('man', 'woman', 'son', glove_6b50d)
```

Analogia de “país-capital”: "beijing" é para "china" como "tokyo" é para quê? A resposta deve ser "japão".

```{.python .input}
#@tab all
get_analogy('beijing', 'china', 'tokyo', glove_6b50d)
```

Analogia do "adjetivo-adjetivo superlativo": "ruim" está para o "pior", assim como "grande" está para o quê? A resposta deve ser "maior".

```{.python .input}
#@tab all
get_analogy('bad', 'worst', 'big', glove_6b50d)
```

Analogia do "verbo presente-verbo no pretérito": "do" é "did" assim como "go" é para quê? A resposta deve ser "went".

```{.python .input}
#@tab all
get_analogy('do', 'did', 'go', glove_6b50d)
```

## Sumário

* Vetores de palavras pré-treinados em um corpus de grande escala podem frequentemente ser aplicados a tarefas de processamento de linguagem natural downstream.
* Podemos usar vetores de palavras pré-treinados para buscar sinônimos e analogias.


## Exercícios

1. Teste os resultados do fastText usando `TokenEmbedding ('wiki.en')`.
1. Se o dicionário for extremamente grande, como podemos acelerar a localização de sinônimos e analogias?

:begin_tab:`mxnet`
[Discussão](https://discuss.d2l.ai/t/387)
:end_tab:

:begin_tab:`pytorch`
[Discussão](https://discuss.d2l.ai/t/1336)
:end_tab:
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTEzNzYyMDg0OCw1MTk1NDA4MTFdfQ==
-->