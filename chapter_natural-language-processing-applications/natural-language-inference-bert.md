# Inferência de Linguagem Natural: Ajuste Fino do BERT
:label:`sec_natural-language-inference-bert`


Nas seções anteriores deste capítulo,
nós projetamos uma arquitetura baseada na atenção
(em :numref:`sec_natural-language-inference-attention`)
para a tarefa de inferência de linguagem natural
no conjunto de dados SNLI (conforme descrito em :numref:`sec_natural-language-inference-and-dataset`).
Agora, revisitamos essa tarefa fazendo o ajuste fino do BERT.
Conforme discutido em :numref:`sec_finetuning-bert`,
a inferência de linguagem natural é um problema de classificação de pares de texto em nível de sequência,
e o ajuste fino de BERT requer apenas uma arquitetura adicional baseada em MLP,
conforme ilustrado em :numref:`fig_nlp-map-nli-bert`.

![Esta seção alimenta BERT pré-treinado para uma arquitetura baseada em MLP para inferência de linguagem natural.](../img/nlp-map-nli-bert.svg)
:label:`fig_nlp-map-nli-bert`

Nesta secção,
vamos baixar uma versão pequena pré-treinada de BERT,
então ajuste-o
para inferência de linguagem natural no conjunto de dados SNLI.

```{.python .input}
from d2l import mxnet as d2l
import json
import multiprocessing
from mxnet import gluon, np, npx
from mxnet.gluon import nn
import os

npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import json
import multiprocessing
import torch
from torch import nn
import os
```

## Carregando o BERT Pré-treinado

Explicamos como pré-treinar BERT no conjunto de dados WikiText-2 em
:numref:`sec_bert-dataset` e :numref:`sec_bert-pretraining`
(observe que o modelo BERT original é pré-treinado em corpora muito maiores).
Conforme discutido em :numref:`sec_bert-pretraining`,
o modelo BERT original tem centenas de milhões de parâmetros.
Na sequência,
nós fornecemos duas versões de BERT pré-treinados:
"bert.base" é quase tão grande quanto o modelo de base BERT original, que requer muitos recursos computacionais para o ajuste fino,
enquanto "bert.small" é uma versão pequena para facilitar a demonstração.

```{.python .input}
d2l.DATA_HUB['bert.base'] = (d2l.DATA_URL + 'bert.base.zip',
                             '7b3820b35da691042e5d34c0971ac3edbd80d3f4')
d2l.DATA_HUB['bert.small'] = (d2l.DATA_URL + 'bert.small.zip',
                              'a4e718a47137ccd1809c9107ab4f5edd317bae2c')
```

```{.python .input}
#@tab pytorch
d2l.DATA_HUB['bert.base'] = (d2l.DATA_URL + 'bert.base.torch.zip',
                             '225d66f04cae318b841a13d32af3acc165f253ac')
d2l.DATA_HUB['bert.small'] = (d2l.DATA_URL + 'bert.small.torch.zip',
                              'c72329e68a732bef0452e4b96a1c341c8910f81f')
```

Qualquer um dos modelos BERT pré-treinados contém um arquivo "vocab.json" que define o conjunto de vocabulário
e um arquivo "pretrained.params" dos parâmetros pré-treinados.
Implementamos a seguinte função `load_pretrained_model` para carregar os parâmetros BERT pré-treinados.

```{.python .input}
def load_pretrained_model(pretrained_model, num_hiddens, ffn_num_hiddens,
                          num_heads, num_layers, dropout, max_len, devices):
    data_dir = d2l.download_extract(pretrained_model)
    # Define an empty vocabulary to load the predefined vocabulary
    vocab = d2l.Vocab()
    vocab.idx_to_token = json.load(open(os.path.join(data_dir, 'vocab.json')))
    vocab.token_to_idx = {token: idx for idx, token in enumerate(
        vocab.idx_to_token)}
    bert = d2l.BERTModel(len(vocab), num_hiddens, ffn_num_hiddens, num_heads, 
                         num_layers, dropout, max_len)
    # Load pretrained BERT parameters
    bert.load_parameters(os.path.join(data_dir, 'pretrained.params'),
                         ctx=devices)
    return bert, vocab
```

```{.python .input}
#@tab pytorch
def load_pretrained_model(pretrained_model, num_hiddens, ffn_num_hiddens,
                          num_heads, num_layers, dropout, max_len, devices):
    data_dir = d2l.download_extract(pretrained_model)
    # Define an empty vocabulary to load the predefined vocabulary
    vocab = d2l.Vocab()
    vocab.idx_to_token = json.load(open(os.path.join(data_dir, 'vocab.json')))
    vocab.token_to_idx = {token: idx for idx, token in enumerate(
        vocab.idx_to_token)}
    bert = d2l.BERTModel(len(vocab), num_hiddens, norm_shape=[256],
                         ffn_num_input=256, ffn_num_hiddens=ffn_num_hiddens,
                         num_heads=4, num_layers=2, dropout=0.2,
                         max_len=max_len, key_size=256, query_size=256,
                         value_size=256, hid_in_features=256,
                         mlm_in_features=256, nsp_in_features=256)
    # Load pretrained BERT parameters
    bert.load_state_dict(torch.load(os.path.join(data_dir,
                                                 'pretrained.params')))
    return bert, vocab
```

Para facilitar a demonstração na maioria das máquinas,
vamos carregar e ajustar a versão pequena ("bert.small") do BERT pré-treinado nesta seção.
No exercício, mostraremos como ajustar o "bert.base" muito maior para melhorar significativamente a precisão do teste.

```{.python .input}
#@tab all
devices = d2l.try_all_gpus()
bert, vocab = load_pretrained_model(
    'bert.small', num_hiddens=256, ffn_num_hiddens=512, num_heads=4,
    num_layers=2, dropout=0.1, max_len=512, devices=devices)
```

## O Conjunto de Dados para Ajuste Fino do BERT

Para a inferência de linguagem natural da tarefa *downstream* no conjunto de dados SNLI,
definimos uma classe de conjunto de dados customizada `SNLIBERTDataset`.
Em cada exemplo,
a premissa e a hipótese formam um par de sequência de texto
e são compactados em uma sequência de entrada de BERT conforme descrito em :numref:`fig_bert-two-seqs`.
Lembre-se :numref:`subsec_bert_input_rep` que IDs de segmento
são usados para distinguir a premissa e a hipótese em uma sequência de entrada do BERT.
Com o comprimento máximo predefinido de uma sequência de entrada de BERT (`max_len`),
o último *token* do mais longo do par de texto de entrada continua sendo removido até
`max_len` é atendido.
Para acelerar a geração do conjunto de dados SNLI
para o ajuste fino de BERT,
usamos 4 processos de trabalho para gerar exemplos de treinamento ou teste em paralelo.

```{.python .input}
class SNLIBERTDataset(gluon.data.Dataset):
    def __init__(self, dataset, max_len, vocab=None):
        all_premise_hypothesis_tokens = [[
            p_tokens, h_tokens] for p_tokens, h_tokens in zip(
            *[d2l.tokenize([s.lower() for s in sentences])
              for sentences in dataset[:2]])]
        
        self.labels = np.array(dataset[2])
        self.vocab = vocab
        self.max_len = max_len
        (self.all_token_ids, self.all_segments,
         self.valid_lens) = self._preprocess(all_premise_hypothesis_tokens)
        print('read ' + str(len(self.all_token_ids)) + ' examples')

    def _preprocess(self, all_premise_hypothesis_tokens):
        pool = multiprocessing.Pool(4)  # Use 4 worker processes
        out = pool.map(self._mp_worker, all_premise_hypothesis_tokens)
        all_token_ids = [
            token_ids for token_ids, segments, valid_len in out]
        all_segments = [segments for token_ids, segments, valid_len in out]
        valid_lens = [valid_len for token_ids, segments, valid_len in out]
        return (np.array(all_token_ids, dtype='int32'),
                np.array(all_segments, dtype='int32'), 
                np.array(valid_lens))

    def _mp_worker(self, premise_hypothesis_tokens):
        p_tokens, h_tokens = premise_hypothesis_tokens
        self._truncate_pair_of_tokens(p_tokens, h_tokens)
        tokens, segments = d2l.get_tokens_and_segments(p_tokens, h_tokens)
        token_ids = self.vocab[tokens] + [self.vocab['<pad>']] \
                             * (self.max_len - len(tokens))
        segments = segments + [0] * (self.max_len - len(segments))
        valid_len = len(tokens)
        return token_ids, segments, valid_len

    def _truncate_pair_of_tokens(self, p_tokens, h_tokens):
        # Reserve slots for '<CLS>', '<SEP>', and '<SEP>' tokens for the BERT
        # input
        while len(p_tokens) + len(h_tokens) > self.max_len - 3:
            if len(p_tokens) > len(h_tokens):
                p_tokens.pop()
            else:
                h_tokens.pop()

    def __getitem__(self, idx):
        return (self.all_token_ids[idx], self.all_segments[idx],
                self.valid_lens[idx]), self.labels[idx]

    def __len__(self):
        return len(self.all_token_ids)
```

```{.python .input}
#@tab pytorch
class SNLIBERTDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, max_len, vocab=None):
        all_premise_hypothesis_tokens = [[
            p_tokens, h_tokens] for p_tokens, h_tokens in zip(
            *[d2l.tokenize([s.lower() for s in sentences])
              for sentences in dataset[:2]])]
        
        self.labels = torch.tensor(dataset[2])
        self.vocab = vocab
        self.max_len = max_len
        (self.all_token_ids, self.all_segments,
         self.valid_lens) = self._preprocess(all_premise_hypothesis_tokens)
        print('read ' + str(len(self.all_token_ids)) + ' examples')

    def _preprocess(self, all_premise_hypothesis_tokens):
        pool = multiprocessing.Pool(4)  # Use 4 worker processes
        out = pool.map(self._mp_worker, all_premise_hypothesis_tokens)
        all_token_ids = [
            token_ids for token_ids, segments, valid_len in out]
        all_segments = [segments for token_ids, segments, valid_len in out]
        valid_lens = [valid_len for token_ids, segments, valid_len in out]
        return (torch.tensor(all_token_ids, dtype=torch.long),
                torch.tensor(all_segments, dtype=torch.long), 
                torch.tensor(valid_lens))

    def _mp_worker(self, premise_hypothesis_tokens):
        p_tokens, h_tokens = premise_hypothesis_tokens
        self._truncate_pair_of_tokens(p_tokens, h_tokens)
        tokens, segments = d2l.get_tokens_and_segments(p_tokens, h_tokens)
        token_ids = self.vocab[tokens] + [self.vocab['<pad>']] \
                             * (self.max_len - len(tokens))
        segments = segments + [0] * (self.max_len - len(segments))
        valid_len = len(tokens)
        return token_ids, segments, valid_len

    def _truncate_pair_of_tokens(self, p_tokens, h_tokens):
        # Reserve slots for '<CLS>', '<SEP>', and '<SEP>' tokens for the BERT
        # input
        while len(p_tokens) + len(h_tokens) > self.max_len - 3:
            if len(p_tokens) > len(h_tokens):
                p_tokens.pop()
            else:
                h_tokens.pop()

    def __getitem__(self, idx):
        return (self.all_token_ids[idx], self.all_segments[idx],
                self.valid_lens[idx]), self.labels[idx]

    def __len__(self):
        return len(self.all_token_ids)
```

Depois de baixar o conjunto de dados SNLI,
geramos exemplos de treinamento e teste
instanciando a classe `SNLIBERTDataset`.
Esses exemplos serão lidos em minibatches durante o treinamento e teste
de inferência de linguagem natural.

```{.python .input}
# Reduce `batch_size` if there is an out of memory error. In the original BERT
# model, `max_len` = 512
batch_size, max_len, num_workers = 512, 128, d2l.get_dataloader_workers()
data_dir = d2l.download_extract('SNLI')
train_set = SNLIBERTDataset(d2l.read_snli(data_dir, True), max_len, vocab)
test_set = SNLIBERTDataset(d2l.read_snli(data_dir, False), max_len, vocab)
train_iter = gluon.data.DataLoader(train_set, batch_size, shuffle=True,
                                   num_workers=num_workers)
test_iter = gluon.data.DataLoader(test_set, batch_size,
                                  num_workers=num_workers)
```

```{.python .input}
#@tab pytorch
# Reduce `batch_size` if there is an out of memory error. In the original BERT
# model, `max_len` = 512
batch_size, max_len, num_workers = 512, 128, d2l.get_dataloader_workers()
data_dir = d2l.download_extract('SNLI')
train_set = SNLIBERTDataset(d2l.read_snli(data_dir, True), max_len, vocab)
test_set = SNLIBERTDataset(d2l.read_snli(data_dir, False), max_len, vocab)
train_iter = torch.utils.data.DataLoader(train_set, batch_size, shuffle=True,
                                   num_workers=num_workers)
test_iter = torch.utils.data.DataLoader(test_set, batch_size,
                                  num_workers=num_workers)
```

## Ajuste Fino do BERT

Como :numref:`fig_bert-two-seqs` indica,
ajuste fino do BERT para inferência de linguagem natural
requer apenas um MLP extra que consiste em duas camadas totalmente conectadas
(veja `self.hidden` e` self.output` na seguinte classe `BERTClassifier`).
Este MLP transforma o
Representação de BERT do *token* especial “&lt;cls&gt;”,
que codifica as informações tanto da premissa quanto da hipótese,
em três resultados de inferência de linguagem natural:
implicação, contradição e neutro.

```{.python .input}
class BERTClassifier(nn.Block):
    def __init__(self, bert):
        super(BERTClassifier, self).__init__()
        self.encoder = bert.encoder
        self.hidden = bert.hidden
        self.output = nn.Dense(3)

    def forward(self, inputs):
        tokens_X, segments_X, valid_lens_x = inputs
        encoded_X = self.encoder(tokens_X, segments_X, valid_lens_x)
        return self.output(self.hidden(encoded_X[:, 0, :]))
```

```{.python .input}
#@tab pytorch
class BERTClassifier(nn.Module):
    def __init__(self, bert):
        super(BERTClassifier, self).__init__()
        self.encoder = bert.encoder
        self.hidden = bert.hidden
        self.output = nn.Linear(256, 3)

    def forward(self, inputs):
        tokens_X, segments_X, valid_lens_x = inputs
        encoded_X = self.encoder(tokens_X, segments_X, valid_lens_x)
        return self.output(self.hidden(encoded_X[:, 0, :]))
```

Na sequência,
o modelo BERT pré-treinado `bert` é alimentado na instância` BERTClassifier` `net` para
a aplicação *downstream*.
Em implementações comuns de ajuste fino de BERT,
apenas os parâmetros da camada de saída do MLP adicional (`net.output`) serão aprendidos do zero.
Todos os parâmetros do codificador BERT pré-treinado (`net.encoder`) e a camada oculta do MLP adicional (`net.hidden`) serão ajustados.

```{.python .input}
net = BERTClassifier(bert)
net.output.initialize(ctx=devices)
```

```{.python .input}
#@tab pytorch
net = BERTClassifier(bert)
```


Lembre-se disso
in :numref:`sec_bert`
ambas as classes `MaskLM` e` NextSentencePred`
têm parâmetros em suas MLPs empregadas.
Esses parâmetros são parte daqueles no modelo BERT pré-treinado
`bert`, e, portanto, parte dos parâmetros em `net`.
No entanto, esses parâmetros são apenas para computação
a perda de modelagem de linguagem mascarada
e a perda de previsão da próxima frase
durante o pré-treinamento.
Essas duas funções de perda são irrelevantes para o ajuste fino de aplicativos *downstream*,
assim, os parâmetros das MLPs empregadas em
`MaskLM` e `NextSentencePred` não são atualizados (obsoletos) quando o BERT é ajustado.

Para permitir parâmetros com gradientes obsoletos,
o sinalizador `ignore_stale_grad = True` é definido na função `step` de `d2l.train_batch_ch13`.
Usamos esta função para treinar e avaliar o modelo `net` usando o conjunto de treinamento
(`train_iter`) e o conjunto de teste (`test_iter`) de SNLI.
Devido aos recursos computacionais limitados, a precisão de treinamento e teste
pode ser melhorado ainda mais: deixamos suas discussões nos exercícios.

```{.python .input}
lr, num_epochs = 1e-4, 5
trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': lr})
loss = gluon.loss.SoftmaxCrossEntropyLoss()
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices,
               d2l.split_batch_multi_inputs)
```

```{.python .input}
#@tab pytorch
lr, num_epochs = 1e-4, 5
trainer = torch.optim.Adam(net.parameters(), lr=lr)
loss = nn.CrossEntropyLoss(reduction='none')
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)
```

## Resumo

* Podemos ajustar o modelo BERT pré-treinado para aplicativos *downstream*, como inferência de linguagem natural no conjunto de dados SNLI.
* Durante o ajuste fino, o modelo BERT torna-se parte do modelo para a aplicação *downstream*. Os parâmetros relacionados apenas à perda de pré-treinamento não serão atualizados durante o ajuste fino.


## Exercícios

1. Faça o ajuste fino de um modelo de BERT pré-treinado muito maior que é quase tão grande quanto o modelo de base de BERT original se seu recurso computacional permitir. Defina os argumentos na função `load_pretrained_model` como: substituindo 'bert.small' por 'bert.base', aumentando os valores de `num_hiddens = 256`, `ffn_num_hiddens = 512`, `num_heads = 4`, `num_layers = 2` para `768`, `3072`, `12`, `12`, respectivamente. Aumentando os períodos de ajuste fino (e possivelmente ajustando outros hiperparâmetros), você pode obter uma precisão de teste superior a 0,86?
1. Como truncar um par de sequências de acordo com sua proporção de comprimento? Compare este método de truncamento de par e aquele usado na classe `SNLIBERTDataset`. Quais são seus prós e contras?

:begin_tab:`mxnet`
[Discussões](https://discuss.d2l.ai/t/397)
:end_tab:

:begin_tab:`pytorch`
[Discussões](https://discuss.d2l.ai/t/1526)
:end_tab:
<!--stackedit_data:
eyJoaXN0b3J5IjpbMTE3NzkxMzQyNywtMzEwNjk1NTQ5LC0xMz
Q1MjA0Mjc1LC0yMDA3MTU1MzUxXX0=
-->