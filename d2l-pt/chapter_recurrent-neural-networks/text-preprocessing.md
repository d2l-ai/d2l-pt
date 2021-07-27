# Preprocessamento de Texto
:label:`sec_text_preprocessing`


Nós revisamos e avaliamos
ferramentas estatísticas
e desafios de previsão
para dados de sequência.
Esses dados podem assumir várias formas.
Especificamente,
como vamos nos concentrar em
em muitos capítulos do livro,
text é um dos exemplos mais populares de dados de sequência.
Por exemplo,
um artigo pode ser visto simplesmente como uma sequência de palavras ou mesmo uma sequência de caracteres.
Para facilitar nossos experimentos futuros
com dados de sequência,
vamos dedicar esta seção
para explicar as etapas comuns de pré-processamento para texto.
Normalmente, essas etapas são:

1. Carregar o texto como strings na memória.
1. Dividir as strings em tokens (por exemplo, palavras e caracteres).
1. Construir uma tabela de vocabulário para mapear os tokens divididos em índices numéricos.
1. Converter o texto em sequências de índices numéricos para que possam ser facilmente manipulados por modelos.

```{.python .input}
import collections
from d2l import mxnet as d2l
import re
```

```{.python .input}
#@tab pytorch
import collections
from d2l import torch as d2l
import re
```

```{.python .input}
#@tab tensorflow
import collections
from d2l import tensorflow as d2l
import re
```

## Lendo o Dataset

Para começar, carregamos o texto de H. G. Wells '[*The Time Machine*] (http://www.gutenberg.org/ebooks/35).
Este é um corpus bastante pequeno de pouco mais de 30000 palavras, mas para o propósito do que queremos ilustrar, está tudo bem.
Coleções de documentos mais realistas contêm muitos bilhões de palavras.
A função a seguir lê o conjunto de dados em uma lista de linhas de texto, onde cada linha é uma *string*.
Para simplificar, aqui ignoramos a pontuação e a capitalização.

```{.python .input}
#@tab all
#@save
d2l.DATA_HUB['time_machine'] = (d2l.DATA_URL + 'timemachine.txt',
                                '090b5e7e70c295757f55df93cb0a180b9691891a')

def read_time_machine():  #@save
    """Load the time machine dataset into a list of text lines."""
    with open(d2l.download('time_machine'), 'r') as f:
        lines = f.readlines()
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]

lines = read_time_machine()
print(f'# text lines: {len(lines)}')
print(lines[0])
print(lines[10])
```

## Tokenização

A seguinte função `tokenize`
recebe uma lista (`lines`) como entrada,
onde cada lista é uma sequência de texto (por exemplo, uma linha de texto).
Cada sequência de texto é dividida em uma lista de tokens.
Um *token* é a unidade básica no texto.
No fim,
uma lista de listas de tokens é retornada,
onde cada token é uma string.

```{.python .input}
#@tab all
def tokenize(lines, token='word'):  #@save
    """Split text lines into word or character tokens."""
    if token == 'word':
        return [line.split() for line in lines]
    elif token == 'char':
        return [list(line) for line in lines]
    else:
        print('ERROR: unknown token type: ' + token)

tokens = tokenize(lines)
for i in range(11):
    print(tokens[i])
```

## Vocabulário

O tipo de string do token é inconveniente para ser usado por modelos, que usam entradas numéricas.
Agora, vamos construir um dicionário, também chamado de *vocabulário*, para mapear tokens de string em índices numéricos começando em 0.
Para fazer isso, primeiro contamos os tokens exclusivos em todos os documentos do conjunto de treinamento,
ou seja, um *corpus*,
e, em seguida, atribua um índice numérico a cada token exclusivo de acordo com sua frequência.
Os tokens raramente exibidos são frequentemente removidos para reduzir a complexidade.
Qualquer token que não exista no corpus ou que tenha sido removido é mapeado em um token especial desconhecido “&lt;unk&gt;”.
Opcionalmente, adicionamos uma lista de tokens reservados, como
“&lt;pad&gt;” para preenchimento,
“&lt;bos&gt;” para apresentar o início de uma sequência, e “&lt;eos&gt;”” para o final de uma sequência.

```{.python .input}
#@tab all
class Vocab:  #@save
    """Vocabulary for text."""
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = [] 
        # Sort according to frequencies
        counter = count_corpus(tokens)
        self.token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                  reverse=True)
        # The index for the unknown token is 0
        self.unk, uniq_tokens = 0, ['<unk>'] + reserved_tokens
        uniq_tokens += [token for token, freq in self.token_freqs
                        if freq >= min_freq and token not in uniq_tokens]
        self.idx_to_token, self.token_to_idx = [], dict()
        for token in uniq_tokens:
            self.idx_to_token.append(token)
            self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

def count_corpus(tokens):  #@save
    """Count token frequencies."""
    # Here `tokens` is a 1D list or 2D list
    if len(tokens) == 0 or isinstance(tokens[0], list):
        # Flatten a list of token lists into a list of tokens
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)
```

Construímos um vocabulário usando o conjunto de dados da máquina do tempo como corpus.
Em seguida, imprimimos os primeiros tokens frequentes com seus índices.

```{.python .input}
#@tab all
vocab = Vocab(tokens)
print(list(vocab.token_to_idx.items())[:10])
```

Agora podemos converter cada linha de texto em uma lista de índices numéricos.

```{.python .input}
#@tab all
for i in [0, 10]:
    print('words:', tokens[i])
    print('indices:', vocab[tokens[i]])
```

## Juntando Todas as Coisas

Usando as funções acima, empacotamos tudo na função `load_corpus_time_machine`, que retorna` corpus`, uma lista de índices de token, e `vocabulário`, o vocabulário do corpus da máquina do tempo.
As modificações que fizemos aqui são:
i) simbolizamos o texto em caracteres, não em palavras, para simplificar o treinamento em seções posteriores;
ii) `corpus` é uma lista única, não uma lista de listas de tokens, uma vez que cada linha de texto no conjunto de dados da máquina do tempo não é necessariamente uma frase ou um parágrafo.

```{.python .input}
#@tab all
def load_corpus_time_machine(max_tokens=-1):  #@save
    """Return token indices and the vocabulary of the time machine dataset."""
    lines = read_time_machine()
    tokens = tokenize(lines, 'char')
    vocab = Vocab(tokens)
    # Since each text line in the time machine dataset is not necessarily a
    # sentence or a paragraph, flatten all the text lines into a single list
    corpus = [vocab[token] for line in tokens for token in line]
    if max_tokens > 0:
        corpus = corpus[:max_tokens]
    return corpus, vocab

corpus, vocab = load_corpus_time_machine()
len(corpus), len(vocab)
```

## Resumo

* O texto é uma forma importante de dados de sequência.
* Para pré-processar o texto, geralmente dividimos o texto em tokens, construímos um vocabulário para mapear strings de token em índices numéricos e convertemos dados de texto em índices de token para os modelos manipularem.


## Exercícios

1. A tokenização é uma etapa chave de pré-processamento. Isso varia para diferentes idiomas. Tente encontrar outros três métodos comumente usados para tokenizar texto.
1. No experimento desta seção, tokenize o texto em palavras e varie os argumentos `min_freq` da instância` Vocab`. Como isso afeta o tamanho do vocabulário?

[Discussions](https://discuss.d2l.ai/t/115)
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTEwMjkxNDYyMTksLTExNTcwMTczOTRdfQ
==
-->