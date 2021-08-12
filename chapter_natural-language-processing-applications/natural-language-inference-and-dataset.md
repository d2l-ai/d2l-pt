# Inferência de Linguagem Natural e o *Dataset*
:label:`sec_natural-language-inference-and-dataset`

Em :numref:`sec_sentiment`, discutimos o problema da análise de sentimento.
Esta tarefa visa classificar uma única sequência de texto em categorias predefinidas, como um conjunto de polaridades de sentimento.
No entanto, quando há a necessidade de decidir se uma frase pode ser inferida de outra ou eliminar a redundância identificando frases semanticamente equivalentes,
saber classificar uma sequência de texto é insuficiente.
Em vez disso, precisamos ser capazes de raciocinar sobre pares de sequências de texto.


## Inferência de Linguagem Natural


*Inferência de linguagem natural* estuda se uma *hipótese*
pode ser inferida de uma *premissa*, onde ambas são uma sequência de texto.
Em outras palavras, a inferência de linguagem natural determina a relação lógica entre um par de sequências de texto.
Esses relacionamentos geralmente se enquadram em três tipos:

* *Implicação*: a hipótese pode ser inferida a partir da premissa.
* *Contradição*: a negação da hipótese pode ser inferida a partir da premissa.
* *Neutro*: todos os outros casos.

A inferência de linguagem natural também é conhecida como a tarefa de reconhecimento de vinculação textual.
Por exemplo, o par a seguir será rotulado como *implicação* porque "mostrar afeto" na hipótese pode ser inferido de "abraçar um ao outro" na premissa.


>Premissa: Duas mulheres estão se abraçando.

>Hipótese: Duas mulheres estão demonstrando afeto.


A seguir está um exemplo de *contradição*, pois "executando o exemplo de codificação" indica "não dormindo" em vez de "dormindo".

> Premissa: Um homem está executando o exemplo de codificação do *Dive into Deep Learning*.

> Hipótese: O homem está dormindo.


O terceiro exemplo mostra uma relação de *neutralidade* porque nem "famoso" nem "não famoso" podem ser inferidos do fato de que "estão se apresentando para nós".

> Premissa: Os músicos estão se apresentando para nós.

> Hipótese: Os músicos são famosos.

A inferência da linguagem natural tem sido um tópico central para a compreensão da linguagem natural.
Desfruta de uma ampla gama de aplicações, desde
recuperação de informações para resposta a perguntas de domínio aberto.
Para estudar esse problema, começaremos investigando um popular conjunto de dados de referência de inferência em linguagem natural.


## Conjunto de dados Stanford Natural Language Inference (SNLI)

Stanford Natural Language Inference (SNLI) Corpus é uma coleção de mais de $500.000$ pares de frases em inglês rotulados :cite:`Bowman.Angeli.Potts.ea.2015`.
Baixamos e armazenamos o conjunto de dados SNLI extraído no caminho `../data/snli_1.0`.

```{.python .input}
from d2l import mxnet as d2l
from mxnet import gluon, np, npx
import os
import re

npx.set_np()

#@save
d2l.DATA_HUB['SNLI'] = (
    'https://nlp.stanford.edu/projects/snli/snli_1.0.zip',
    '9fcde07509c7e87ec61c640c1b2753d9041758e4')

data_dir = d2l.download_extract('SNLI')
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
import os
import re

#@save
d2l.DATA_HUB['SNLI'] = (
    'https://nlp.stanford.edu/projects/snli/snli_1.0.zip',
    '9fcde07509c7e87ec61c640c1b2753d9041758e4')

data_dir = d2l.download_extract('SNLI')
```

### Lendo o *Dataset*

O conjunto de dados SNLI original contém informações muito mais ricas do que realmente precisamos em nossos experimentos. Assim, definimos uma função `read_snli` para extrair apenas parte do conjunto de dados e, em seguida, retornar listas de premissas, hipóteses e seus rótulos.

```{.python .input}
#@tab all
#@save
def read_snli(data_dir, is_train):
    """Read the SNLI dataset into premises, hypotheses, and labels."""
    def extract_text(s):
        # Remove information that will not be used by us
        s = re.sub('\\(', '', s) 
        s = re.sub('\\)', '', s)
        # Substitute two or more consecutive whitespace with space
        s = re.sub('\\s{2,}', ' ', s)
        return s.strip()
    label_set = {'entailment': 0, 'contradiction': 1, 'neutral': 2}
    file_name = os.path.join(data_dir, 'snli_1.0_train.txt'
                             if is_train else 'snli_1.0_test.txt')
    with open(file_name, 'r') as f:
        rows = [row.split('\t') for row in f.readlines()[1:]]
    premises = [extract_text(row[1]) for row in rows if row[0] in label_set]
    hypotheses = [extract_text(row[2]) for row in rows if row[0] in label_set]
    labels = [label_set[row[0]] for row in rows if row[0] in label_set]
    return premises, hypotheses, labels
```

Agora, vamos imprimir os primeiros $3$ pares de premissa e hipótese, bem como seus rótulos ("0", "1" e "2" correspondem a "implicação", "contradição" e "neutro", respectivamente).

```{.python .input}
#@tab all
train_data = read_snli(data_dir, is_train=True)
for x0, x1, y in zip(train_data[0][:3], train_data[1][:3], train_data[2][:3]):
    print('premise:', x0)
    print('hypothesis:', x1)
    print('label:', y)
```

O conjunto de treinamento tem cerca de $550.000$ pares,
e o conjunto de teste tem cerca de $10.000$ pares.
O seguinte mostra que
os três rótulos "implicação", "contradição" e "neutro" são equilibrados em
o conjunto de treinamento e o conjunto de teste.

```{.python .input}
#@tab all
test_data = read_snli(data_dir, is_train=False)
for data in [train_data, test_data]:
    print([[row for row in data[2]].count(i) for i in range(3)])
```

### Definindo uma Classe para Carregar o *Dataset*

Abaixo, definimos uma classe para carregar o dataset SNLI herdando da classe `Dataset` no Gluon. O argumento `num_steps` no construtor de classe especifica o comprimento de uma sequência de texto para que cada minibatch de sequências tenha a mesma forma.
Em outras palavras,
tokens após os primeiros `num_steps` em uma sequência mais longa são cortados, enquanto tokens especiais “& lt; pad & gt;” serão anexados a sequências mais curtas até que seu comprimento se torne `num_steps`.
Implementando a função `__getitem__`, podemos acessar arbitrariamente a premissa, hipótese e rótulo com o índice `idx`.

```{.python .input}
#@save
class SNLIDataset(gluon.data.Dataset):
    """A customized dataset to load the SNLI dataset."""
    def __init__(self, dataset, num_steps, vocab=None):
        self.num_steps = num_steps
        all_premise_tokens = d2l.tokenize(dataset[0])
        all_hypothesis_tokens = d2l.tokenize(dataset[1])
        if vocab is None:
            self.vocab = d2l.Vocab(all_premise_tokens + all_hypothesis_tokens,
                                   min_freq=5, reserved_tokens=['<pad>'])
        else:
            self.vocab = vocab
        self.premises = self._pad(all_premise_tokens)
        self.hypotheses = self._pad(all_hypothesis_tokens)
        self.labels = np.array(dataset[2])
        print('read ' + str(len(self.premises)) + ' examples')

    def _pad(self, lines):
        return np.array([d2l.truncate_pad(
            self.vocab[line], self.num_steps, self.vocab['<pad>'])
                         for line in lines])

    def __getitem__(self, idx):
        return (self.premises[idx], self.hypotheses[idx]), self.labels[idx]

    def __len__(self):
        return len(self.premises)
```

```{.python .input}
#@tab pytorch
#@save
class SNLIDataset(torch.utils.data.Dataset):
    """A customized dataset to load the SNLI dataset."""
    def __init__(self, dataset, num_steps, vocab=None):
        self.num_steps = num_steps
        all_premise_tokens = d2l.tokenize(dataset[0])
        all_hypothesis_tokens = d2l.tokenize(dataset[1])
        if vocab is None:
            self.vocab = d2l.Vocab(all_premise_tokens + all_hypothesis_tokens,
                                   min_freq=5, reserved_tokens=['<pad>'])
        else:
            self.vocab = vocab
        self.premises = self._pad(all_premise_tokens)
        self.hypotheses = self._pad(all_hypothesis_tokens)
        self.labels = torch.tensor(dataset[2])
        print('read ' + str(len(self.premises)) + ' examples')

    def _pad(self, lines):
        return torch.tensor([d2l.truncate_pad(
            self.vocab[line], self.num_steps, self.vocab['<pad>'])
                         for line in lines])

    def __getitem__(self, idx):
        return (self.premises[idx], self.hypotheses[idx]), self.labels[idx]

    def __len__(self):
        return len(self.premises)
```

### Juntando Tudo

Agora podemos invocar a função `read_snli` e a classe `SNLIDataset` para baixar o conjunto de dados SNLI e retornar instâncias de `DataLoader` para ambos os conjuntos de treinamento e teste, junto com o vocabulário do conjunto de treinamento.
Vale ressaltar que devemos utilizar o vocabulário construído a partir do conjunto de treinamento
como aquele do conjunto de teste.
Como resultado, qualquer novo *token* do conjunto de teste será desconhecido para o modelo treinado no conjunto de treinamento.

```{.python .input}
#@save
def load_data_snli(batch_size, num_steps=50):
    """Download the SNLI dataset and return data iterators and vocabulary."""
    num_workers = d2l.get_dataloader_workers()
    data_dir = d2l.download_extract('SNLI')
    train_data = read_snli(data_dir, True)
    test_data = read_snli(data_dir, False)
    train_set = SNLIDataset(train_data, num_steps)
    test_set = SNLIDataset(test_data, num_steps, train_set.vocab)
    train_iter = gluon.data.DataLoader(train_set, batch_size, shuffle=True,
                                       num_workers=num_workers)
    test_iter = gluon.data.DataLoader(test_set, batch_size, shuffle=False,
                                      num_workers=num_workers)
    return train_iter, test_iter, train_set.vocab
```

```{.python .input}
#@tab pytorch
#@save
def load_data_snli(batch_size, num_steps=50):
    """Download the SNLI dataset and return data iterators and vocabulary."""
    num_workers = d2l.get_dataloader_workers()
    data_dir = d2l.download_extract('SNLI')
    train_data = read_snli(data_dir, True)
    test_data = read_snli(data_dir, False)
    train_set = SNLIDataset(train_data, num_steps)
    test_set = SNLIDataset(test_data, num_steps, train_set.vocab)
    train_iter = torch.utils.data.DataLoader(train_set, batch_size,
                                             shuffle=True,
                                             num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(test_set, batch_size,
                                            shuffle=False,
                                            num_workers=num_workers)
    return train_iter, test_iter, train_set.vocab
```

Aqui, definimos o tamanho do lote em $128$ e o comprimento da sequência em $50$,
e invoque a função `load_data_snli` para obter os iteradores de dados e vocabulário.
Em seguida, imprimimos o tamanho do vocabulário.

```{.python .input}
#@tab all
train_iter, test_iter, vocab = load_data_snli(128, 50)
len(vocab)
```

Agora imprimimos a forma do primeiro minibatch.
Ao contrário da análise de sentimento,
temos $2$ entradas `X[0]` e `X[1]` representando pares de premissas e hipóteses.

```{.python .input}
#@tab all
for X, Y in train_iter:
    print(X[0].shape)
    print(X[1].shape)
    print(Y.shape)
    break
```

## Resumo

* A inferência em linguagem natural estuda se uma hipótese pode ser inferida de uma premissa, onde ambas são uma sequência de texto.
* Na inferência em linguagem natural, as relações entre premissas e hipóteses incluem implicação, contradição e neutro.
* *Stanford Natural Language Inference* (SNLI) Corpus é um popular *dataset* de referência de inferência em linguagem natural.


## Exercícios

1. A tradução automática há muito é avaliada com base na correspondência superficial de $n$-grama entre uma tradução de saída e uma tradução de verdade. Você pode criar uma medida para avaliar os resultados da tradução automática usando a inferência de linguagem natural?
1. Como podemos alterar os hiperparâmetros para reduzir o tamanho do vocabulário?

:begin_tab:`mxnet`
[Discussões](https://discuss.d2l.ai/t/394)
:end_tab:

:begin_tab:`pytorch`
[Discussões](https://discuss.d2l.ai/t/1388)
:end_tab:
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTEyNjA2NzAwNiw2NTg2MzEyMjEsLTE2MD
QzODUxODAsMTE4NTgyNjY4NCwtMTk4ODk5OTU5LC0xMjg0MTIz
MjE5XX0=
-->