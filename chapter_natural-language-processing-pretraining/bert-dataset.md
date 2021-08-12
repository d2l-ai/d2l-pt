# O conjunto de dados para pré-treinamento de BERT
:label:`sec_bert-dataset`

Para pré-treinar o modelo BERT conforme implementado em :numref:`sec_bert`,
precisamos gerar o conjunto de dados no formato ideal para facilitar
as duas tarefas de pré-treinamento:
modelagem de linguagem mascarada e previsão da próxima frase.
Por um lado,
o modelo BERT original é pré-treinado na concatenação de
dois enormes corpora BookCorpus e Wikipedia em inglês (ver :numref:`subsec_bert_pretraining_tasks`),
tornando difícil para a maioria dos leitores deste livro.
Por outro lado,
o modelo pré-treinado de BERT pronto para uso
pode não se adequar a aplicativos de domínios específicos, como medicina.
Portanto, está ficando popular pré-treinar o BERT em um conjunto de dados customizado.
Para facilitar a demonstração do pré-treinamento de BERT,
usamos um corpus menor do WikiText-2 :cite:`Merity.Xiong.Bradbury.ea.2016`.

Comparando com o conjunto de dados PTB usado para pré-treinamento de word2vec em :numref:`sec_word2vec_data`,
WikiText-2 i) retém a pontuação original, tornando-a adequada para a previsão da próxima frase; ii) retém a caixa e os números originais; iii) é duas vezes maior.

```{.python .input}
from d2l import mxnet as d2l
from mxnet import gluon, np, npx
import os
import random

npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import os
import random
import torch
```

No conjunto de dados WikiText-2,
cada linha representa um parágrafo onde
o espaço é inserido entre qualquer pontuação e seu token anterior.
Os parágrafos com pelo menos duas frases são mantidos.
Para dividir frases, usamos apenas o ponto como delimitador para simplificar.
Deixamos discussões de técnicas de divisão de frases mais complexas nos exercícios
no final desta seção.

```{.python .input}
#@tab all
#@save
d2l.DATA_HUB['wikitext-2'] = (
    'https://s3.amazonaws.com/research.metamind.io/wikitext/'
    'wikitext-2-v1.zip', '3c914d17d80b1459be871a5039ac23e752a53cbe')

#@save
def _read_wiki(data_dir):
    file_name = os.path.join(data_dir, 'wiki.train.tokens')
    with open(file_name, 'r') as f:
        lines = f.readlines()
    # Uppercase letters are converted to lowercase ones
    paragraphs = [line.strip().lower().split(' . ')
                  for line in lines if len(line.split(' . ')) >= 2]
    random.shuffle(paragraphs)
    return paragraphs
```

## Definindo funções auxiliares para tarefas de pré-treinamento

Na sequência,
começamos implementando funções auxiliares para as duas tarefas de pré-treinamento de BERT:
previsão da próxima frase e modelagem de linguagem mascarada.
Essas funções auxiliares serão chamadas mais tarde
ao transformar o corpus de texto bruto
no conjunto de dados do formato ideal para pré-treinar o BERT.

### Gerando a Tarefa de Previsão da Próxima Sentença

De acordo com as descrições de :numref:`subsec_nsp`,
a função `_get_next_sentence` gera um exemplo de treinamento
para a tarefa de classificação binária.

```{.python .input}
#@tab all
#@save
def _get_next_sentence(sentence, next_sentence, paragraphs):
    if random.random() < 0.5:
        is_next = True
    else:
        # `paragraphs` is a list of lists of lists
        next_sentence = random.choice(random.choice(paragraphs))
        is_next = False
    return sentence, next_sentence, is_next
```

A função a seguir gera exemplos de treinamento para a previsão da próxima frase
do `paragraph` de entrada invocando a função `_get_next_sentence`.
Aqui, `paragraph` é uma lista de frases, onde cada frase é uma lista de tokens.
O argumento `max_len` especifica o comprimento máximo de uma sequência de entrada de BERT durante o pré-treinamento.

```{.python .input}
#@tab all
#@save
def _get_nsp_data_from_paragraph(paragraph, paragraphs, vocab, max_len):
    nsp_data_from_paragraph = []
    for i in range(len(paragraph) - 1):
        tokens_a, tokens_b, is_next = _get_next_sentence(
            paragraph[i], paragraph[i + 1], paragraphs)
        # Consider 1 '<cls>' token and 2 '<sep>' tokens
        if len(tokens_a) + len(tokens_b) + 3 > max_len:
            continue
        tokens, segments = d2l.get_tokens_and_segments(tokens_a, tokens_b)
        nsp_data_from_paragraph.append((tokens, segments, is_next))
    return nsp_data_from_paragraph
```

### Gerando a Tarefa de Modelagem de Linguagem Mascarada
:label:`subsec_prepare_mlm_data`

A fim de gerar exemplos de treinamento
para a tarefa de modelagem de linguagem mascarada
de uma sequência de entrada de BERT,
definimos a seguinte função `_replace_mlm_tokens`.
Em suas entradas, `tokens` é uma lista de tokens que representam uma sequência de entrada de BERT,
`candidate_pred_positions` é uma lista de índices de token da sequência de entrada BERT
excluindo aqueles de tokens especiais (tokens especiais não são previstos na tarefa de modelagem de linguagem mascarada),
e `num_mlm_preds` indica o número de previsões (recorde 15% de tokens aleatórios para prever).
Seguindo a definição da tarefa de modelagem de linguagem mascarada em :numref:`subsec_mlm`,
em cada posição de previsão, a entrada pode ser substituída por
uma “&lt;mask&gt;” especial token ou um token aleatório, ou permanecem inalterados.
No final, a função retorna os tokens de entrada após possível substituição,
os índices de token onde as previsões ocorrem e os rótulos para essas previsões.

```{.python .input}
#@tab all
#@save
def _replace_mlm_tokens(tokens, candidate_pred_positions, num_mlm_preds,
                        vocab):
    # Make a new copy of tokens for the input of a masked language model,
    # where the input may contain replaced '<mask>' or random tokens
    mlm_input_tokens = [token for token in tokens]
    pred_positions_and_labels = []
    # Shuffle for getting 15% random tokens for prediction in the masked
    # language modeling task
    random.shuffle(candidate_pred_positions)
    for mlm_pred_position in candidate_pred_positions:
        if len(pred_positions_and_labels) >= num_mlm_preds:
            break
        masked_token = None
        # 80% of the time: replace the word with the '<mask>' token
        if random.random() < 0.8:
            masked_token = '<mask>'
        else:
            # 10% of the time: keep the word unchanged
            if random.random() < 0.5:
                masked_token = tokens[mlm_pred_position]
            # 10% of the time: replace the word with a random word
            else:
                masked_token = random.randint(0, len(vocab) - 1)
        mlm_input_tokens[mlm_pred_position] = masked_token
        pred_positions_and_labels.append(
            (mlm_pred_position, tokens[mlm_pred_position]))
    return mlm_input_tokens, pred_positions_and_labels
```

Ao invocar a função `_replace_mlm_tokens` mencionada,
a função a seguir leva uma sequência de entrada de BERT (`tokens`)
como uma entrada e retorna os índices dos tokens de entrada
(após possível substituição de token conforme descrito em :numref:`subsec_mlm`),
os índices de token onde as previsões acontecem,
e índices de rótulo para essas previsões.

```{.python .input}
#@tab all
#@save
def _get_mlm_data_from_tokens(tokens, vocab):
    candidate_pred_positions = []
    # `tokens` is a list of strings
    for i, token in enumerate(tokens):
        # Special tokens are not predicted in the masked language modeling
        # task
        if token in ['<cls>', '<sep>']:
            continue
        candidate_pred_positions.append(i)
    # 15% of random tokens are predicted in the masked language modeling task
    num_mlm_preds = max(1, round(len(tokens) * 0.15))
    mlm_input_tokens, pred_positions_and_labels = _replace_mlm_tokens(
        tokens, candidate_pred_positions, num_mlm_preds, vocab)
    pred_positions_and_labels = sorted(pred_positions_and_labels,
                                       key=lambda x: x[0])
    pred_positions = [v[0] for v in pred_positions_and_labels]
    mlm_pred_labels = [v[1] for v in pred_positions_and_labels]
    return vocab[mlm_input_tokens], pred_positions, vocab[mlm_pred_labels]
```

## Transformando texto em conjunto de dados de pré-treinamento

Agora estamos quase prontos para customizar uma classe `Dataset` para pré-treinamento de BERT.
Antes disso,
ainda precisamos definir uma função auxiliar `_pad_bert_inputs`
para anexar a seção especial “&lt;mask&gt;” tokens para as entradas.
Seu argumento `examples` contém as saídas das funções auxiliares `_get_nsp_data_from_paragraph` e `_get_mlm_data_from_tokens` para as duas tarefas de pré-treinamento.

```{.python .input}
#@save
def _pad_bert_inputs(examples, max_len, vocab):
    max_num_mlm_preds = round(max_len * 0.15)
    all_token_ids, all_segments, valid_lens,  = [], [], []
    all_pred_positions, all_mlm_weights, all_mlm_labels = [], [], []
    nsp_labels = []
    for (token_ids, pred_positions, mlm_pred_label_ids, segments,
         is_next) in examples:
        all_token_ids.append(np.array(token_ids + [vocab['<pad>']] * (
            max_len - len(token_ids)), dtype='int32'))
        all_segments.append(np.array(segments + [0] * (
            max_len - len(segments)), dtype='int32'))
        # `valid_lens` excludes count of '<pad>' tokens
        valid_lens.append(np.array(len(token_ids), dtype='float32'))
        all_pred_positions.append(np.array(pred_positions + [0] * (
            max_num_mlm_preds - len(pred_positions)), dtype='int32'))
        # Predictions of padded tokens will be filtered out in the loss via
        # multiplication of 0 weights
        all_mlm_weights.append(
            np.array([1.0] * len(mlm_pred_label_ids) + [0.0] * (
                max_num_mlm_preds - len(pred_positions)), dtype='float32'))
        all_mlm_labels.append(np.array(mlm_pred_label_ids + [0] * (
            max_num_mlm_preds - len(mlm_pred_label_ids)), dtype='int32'))
        nsp_labels.append(np.array(is_next))
    return (all_token_ids, all_segments, valid_lens, all_pred_positions,
            all_mlm_weights, all_mlm_labels, nsp_labels)
```

```{.python .input}
#@tab pytorch
#@save
def _pad_bert_inputs(examples, max_len, vocab):
    max_num_mlm_preds = round(max_len * 0.15)
    all_token_ids, all_segments, valid_lens,  = [], [], []
    all_pred_positions, all_mlm_weights, all_mlm_labels = [], [], []
    nsp_labels = []
    for (token_ids, pred_positions, mlm_pred_label_ids, segments,
         is_next) in examples:
        all_token_ids.append(torch.tensor(token_ids + [vocab['<pad>']] * (
            max_len - len(token_ids)), dtype=torch.long))
        all_segments.append(torch.tensor(segments + [0] * (
            max_len - len(segments)), dtype=torch.long))
        # `valid_lens` excludes count of '<pad>' tokens
        valid_lens.append(torch.tensor(len(token_ids), dtype=torch.float32))
        all_pred_positions.append(torch.tensor(pred_positions + [0] * (
            max_num_mlm_preds - len(pred_positions)), dtype=torch.long))
        # Predictions of padded tokens will be filtered out in the loss via
        # multiplication of 0 weights
        all_mlm_weights.append(
            torch.tensor([1.0] * len(mlm_pred_label_ids) + [0.0] * (
                max_num_mlm_preds - len(pred_positions)),
                dtype=torch.float32))
        all_mlm_labels.append(torch.tensor(mlm_pred_label_ids + [0] * (
            max_num_mlm_preds - len(mlm_pred_label_ids)), dtype=torch.long))
        nsp_labels.append(torch.tensor(is_next, dtype=torch.long))
    return (all_token_ids, all_segments, valid_lens, all_pred_positions,
            all_mlm_weights, all_mlm_labels, nsp_labels)
```

Colocando as funções auxiliares para
gerar exemplos de treinamento das duas tarefas de pré-treinamento,
e a função auxiliar para preencher as entradas juntas,
nós personalizamos a seguinte classe `_WikiTextDataset` como o conjunto de dados WikiText-2 para pré-treinamento de BERT.
Implementando a função `__getitem__`,
podemos acessar arbitrariamente os exemplos de pré-treinamento (modelagem de linguagem mascarada e previsão da próxima frase)
gerado a partir de um par de frases do corpus WikiText-2.

O modelo BERT original usa embeddings WordPiece cujo tamanho de vocabulário é 30.000 :cite:`Wu.Schuster.Chen.ea.2016`.
O método de tokenização do WordPiece é uma ligeira modificação de
o algoritmo de codificação de par de bytes original em :numref:`subsec_Byte_Pair_Encoding`.
Para simplificar, usamos a função `d2l.tokenize` para tokenização.
Tokens raros que aparecem menos de cinco vezes são filtrados.

```{.python .input}
#@save
class _WikiTextDataset(gluon.data.Dataset):
    def __init__(self, paragraphs, max_len):
        # Input `paragraphs[i]` is a list of sentence strings representing a
        # paragraph; while output `paragraphs[i]` is a list of sentences
        # representing a paragraph, where each sentence is a list of tokens
        paragraphs = [d2l.tokenize(
            paragraph, token='word') for paragraph in paragraphs]
        sentences = [sentence for paragraph in paragraphs
                     for sentence in paragraph]
        self.vocab = d2l.Vocab(sentences, min_freq=5, reserved_tokens=[
            '<pad>', '<mask>', '<cls>', '<sep>'])
        # Get data for the next sentence prediction task
        examples = []
        for paragraph in paragraphs:
            examples.extend(_get_nsp_data_from_paragraph(
                paragraph, paragraphs, self.vocab, max_len))
        # Get data for the masked language model task
        examples = [(_get_mlm_data_from_tokens(tokens, self.vocab)
                      + (segments, is_next))
                     for tokens, segments, is_next in examples]
        # Pad inputs
        (self.all_token_ids, self.all_segments, self.valid_lens,
         self.all_pred_positions, self.all_mlm_weights,
         self.all_mlm_labels, self.nsp_labels) = _pad_bert_inputs(
            examples, max_len, self.vocab)

    def __getitem__(self, idx):
        return (self.all_token_ids[idx], self.all_segments[idx],
                self.valid_lens[idx], self.all_pred_positions[idx],
                self.all_mlm_weights[idx], self.all_mlm_labels[idx],
                self.nsp_labels[idx])

    def __len__(self):
        return len(self.all_token_ids)
```

```{.python .input}
#@tab pytorch
#@save
class _WikiTextDataset(torch.utils.data.Dataset):
    def __init__(self, paragraphs, max_len):
        # Input `paragraphs[i]` is a list of sentence strings representing a
        # paragraph; while output `paragraphs[i]` is a list of sentences
        # representing a paragraph, where each sentence is a list of tokens
        paragraphs = [d2l.tokenize(
            paragraph, token='word') for paragraph in paragraphs]
        sentences = [sentence for paragraph in paragraphs
                     for sentence in paragraph]
        self.vocab = d2l.Vocab(sentences, min_freq=5, reserved_tokens=[
            '<pad>', '<mask>', '<cls>', '<sep>'])
        # Get data for the next sentence prediction task
        examples = []
        for paragraph in paragraphs:
            examples.extend(_get_nsp_data_from_paragraph(
                paragraph, paragraphs, self.vocab, max_len))
        # Get data for the masked language model task
        examples = [(_get_mlm_data_from_tokens(tokens, self.vocab)
                      + (segments, is_next))
                     for tokens, segments, is_next in examples]
        # Pad inputs
        (self.all_token_ids, self.all_segments, self.valid_lens,
         self.all_pred_positions, self.all_mlm_weights,
         self.all_mlm_labels, self.nsp_labels) = _pad_bert_inputs(
            examples, max_len, self.vocab)

    def __getitem__(self, idx):
        return (self.all_token_ids[idx], self.all_segments[idx],
                self.valid_lens[idx], self.all_pred_positions[idx],
                self.all_mlm_weights[idx], self.all_mlm_labels[idx],
                self.nsp_labels[idx])

    def __len__(self):
        return len(self.all_token_ids)
```

Usando a função `_read_wiki` e a classe `_WikiTextDataset`,
definimos o seguinte `load_data_wiki` para download e conjunto de dados WikiText-2
e gerar exemplos de pré-treinamento a partir dele.

```{.python .input}
#@save
def load_data_wiki(batch_size, max_len):
    num_workers = d2l.get_dataloader_workers()
    data_dir = d2l.download_extract('wikitext-2', 'wikitext-2')
    paragraphs = _read_wiki(data_dir)
    train_set = _WikiTextDataset(paragraphs, max_len)
    train_iter = gluon.data.DataLoader(train_set, batch_size, shuffle=True,
                                       num_workers=num_workers)
    return train_iter, train_set.vocab
```

```{.python .input}
#@tab pytorch
#@save
def load_data_wiki(batch_size, max_len):
    num_workers = d2l.get_dataloader_workers()
    data_dir = d2l.download_extract('wikitext-2', 'wikitext-2')
    paragraphs = _read_wiki(data_dir)
    train_set = _WikiTextDataset(paragraphs, max_len)
    train_iter = torch.utils.data.DataLoader(train_set, batch_size,
                                        shuffle=True, num_workers=num_workers)
    return train_iter, train_set.vocab
```

Setting the batch size to 512 and the maximum length of a BERT input sequence to be 64,
we print out the shapes of a minibatch of BERT pretraining examples.
Note that in each BERT input sequence,
$10$ ($64 \times 0.15$) positions are predicted for the masked language modeling task.

Configurando o tamanho do lote para 512 e o comprimento máximo de uma sequência de entrada de BERT para 64,
imprimimos as formas de um minibatch de exemplos de pré-treinamento de BERT.
Observe que em cada sequência de entrada de BERT,
$10$ ($64 \times 0.15$) posições estão previstas para a tarefa de modelagem de linguagem mascarada.

```{.python .input}
#@tab all
batch_size, max_len = 512, 64
train_iter, vocab = load_data_wiki(batch_size, max_len)

for (tokens_X, segments_X, valid_lens_x, pred_positions_X, mlm_weights_X,
     mlm_Y, nsp_y) in train_iter:
    print(tokens_X.shape, segments_X.shape, valid_lens_x.shape,
          pred_positions_X.shape, mlm_weights_X.shape, mlm_Y.shape,
          nsp_y.shape)
    break
```

No final, vamos dar uma olhada no tamanho do vocabulário.
Mesmo depois de filtrar tokens pouco frequentes,
ainda é mais de duas vezes maior do que o conjunto de dados do PTB.

```{.python .input}
#@tab all
len(vocab)
```

## Sumário

* Comparando com o conjunto de dados PTB, o conjunto de datas WikiText-2 retém a pontuação, caixa e números originais e é duas vezes maior.
* Podemos acessar arbitrariamente os exemplos de pré-treinamento (modelagem de linguagem mascarada e previsão da próxima frase) gerados a partir de um par de frases do corpus WikiText-2.

## Exercícios

1. Para simplificar, o período é usado como o único delimitador para dividir frases. Experimente outras técnicas de divisão de frases, como spaCy e NLTK. Tome o NLTK como exemplo. Você precisa instalar o NLTK primeiro: `pip install nltk`. No código, primeiro `import nltk`. Então, baixe o tokenizer de frase Punkt: `nltk.download('punkt')`. Para dividir frases como `sentences = 'This is great ! Why not ?'`, Invocar `nltk.tokenize.sent_tokenize(sentences)` retornará uma lista de duas strings de frase:`['This is great !', 'Why not ?']`.
1. Qual é o tamanho do vocabulário se não filtrarmos nenhum token infrequente?

:begin_tab:`mxnet`
[Discussão](https://discuss.d2l.ai/t/389)
:end_tab:

:begin_tab:`pytorch`
[Discussão](https://discuss.d2l.ai/t/1496)
:end_tab:
<!--stackedit_data:
eyJoaXN0b3J5IjpbODY0MTg2MDcwLC0xMzMyMjgyNDJdfQ==
-->