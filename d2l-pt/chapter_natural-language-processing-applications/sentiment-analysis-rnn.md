# Análise de Sentimento: Usando Redes Neurais Recorrentes
:label:`sec_sentiment_rnn`


Semelhante a sinônimos e analogias de pesquisa, a classificação de texto também é uma
aplicação posterior de incorporação de palavras. Nesta seção, vamos aplicar
vetores de palavras pré-treinados (GloVe) e redes neurais recorrentes bidirecionais com
múltiplas camadas ocultas :cite:`Maas.Daly.Pham.ea.2011`, como mostrado em :numref:`fig_nlp-map-sa-rnn`. Usaremos o modelo para
determinar se uma sequência de texto de comprimento indefinido contém emoção positiva ou negativa.

![Esta seção alimenta o GloVe pré-treinado para uma arquitetura baseada em RNN para análise de sentimento.](../img/nlp-map-sa-rnn.svg)
:label:`fig_nlp-map-sa-rnn`

```{.python .input}
from d2l import mxnet as d2l
from mxnet import gluon, init, np, npx
from mxnet.gluon import nn, rnn
npx.set_np()

batch_size = 64
train_iter, test_iter, vocab = d2l.load_data_imdb(batch_size)
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn

batch_size = 64
train_iter, test_iter, vocab = d2l.load_data_imdb(batch_size)
```

## Usando um Modelo de Rede Neural Recorrente

Neste modelo, cada palavra obtém primeiro um vetor de recurso da camada de incorporação. Em seguida, codificamos ainda mais a sequência de recursos usando uma rede neural recorrente bidirecional para obter as informações da sequência. Por fim, transformamos as informações da sequência codificada para a saída por meio da camada totalmente conectada. Especificamente, podemos concatenar estados ocultos de 
memória de longo-curto prazo bidirecionais na etapa de tempo inicial e etapa de tempo final e passá-la para a classificação da camada de saída como informação de sequência de recurso codificada. Na classe `BiRNN` implementada abaixo, a instância` Embedding` é a camada de incorporação, a instância `LSTM` é a camada oculta para codificação de sequência e a instância` Dense` é a camada de saída para os resultados de classificação gerados.

```{.python .input}
class BiRNN(nn.Block):
    def __init__(self, vocab_size, embed_size, num_hiddens,
                 num_layers, **kwargs):
        super(BiRNN, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # Set `bidirectional` to True to get a bidirectional recurrent neural
        # network
        self.encoder = rnn.LSTM(num_hiddens, num_layers=num_layers,
                                bidirectional=True, input_size=embed_size)
        self.decoder = nn.Dense(2)

    def forward(self, inputs):
        # The shape of `inputs` is (batch size, no. of words). Because LSTM
        # needs to use sequence as the first dimension, the input is
        # transformed and the word feature is then extracted. The output shape
        # is (no. of words, batch size, word vector dimension).
        embeddings = self.embedding(inputs.T)
        # Since the input (embeddings) is the only argument passed into
        # rnn.LSTM, it only returns the hidden states of the last hidden layer
        # at different time step (outputs). The shape of `outputs` is
        # (no. of words, batch size, 2 * no. of hidden units).
        outputs = self.encoder(embeddings)
        # Concatenate the hidden states of the initial time step and final
        # time step to use as the input of the fully connected layer. Its
        # shape is (batch size, 4 * no. of hidden units)
        encoding = np.concatenate((outputs[0], outputs[-1]), axis=1)
        outs = self.decoder(encoding)
        return outs
```

```{.python .input}
#@tab pytorch
class BiRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, num_hiddens,
                 num_layers, **kwargs):
        super(BiRNN, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # Set `bidirectional` to True to get a bidirectional recurrent neural
        # network
        self.encoder = nn.LSTM(embed_size, num_hiddens, num_layers=num_layers,
                                bidirectional=True)
        self.decoder = nn.Linear(4*num_hiddens, 2)

    def forward(self, inputs):
        # The shape of `inputs` is (batch size, no. of words). Because LSTM
        # needs to use sequence as the first dimension, the input is
        # transformed and the word feature is then extracted. The output shape
        # is (no. of words, batch size, word vector dimension).
        embeddings = self.embedding(inputs.T)
        # Since the input (embeddings) is the only argument passed into
        # nn.LSTM, both h_0 and c_0 default to zero.
        # we only use the hidden states of the last hidden layer
        # at different time step (outputs). The shape of `outputs` is
        # (no. of words, batch size, 2 * no. of hidden units).
        self.encoder.flatten_parameters()
        outputs, _ = self.encoder(embeddings)
        # Concatenate the hidden states of the initial time step and final
        # time step to use as the input of the fully connected layer. Its
        # shape is (batch size, 4 * no. of hidden units)
        encoding = torch.cat((outputs[0], outputs[-1]), dim=1)
        outs = self.decoder(encoding)
        return outs
```

Criando uma rede neural recorrente bidirecional com duas camadas ocultas.

```{.python .input}
embed_size, num_hiddens, num_layers, devices = 100, 100, 2, d2l.try_all_gpus()
net = BiRNN(len(vocab), embed_size, num_hiddens, num_layers)
net.initialize(init.Xavier(), ctx=devices)
```

```{.python .input}
#@tab pytorch
embed_size, num_hiddens, num_layers, devices = 100, 100, 2, d2l.try_all_gpus()
net = BiRNN(len(vocab), embed_size, num_hiddens, num_layers)

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
    if type(m) == nn.LSTM:
        for param in m._flat_weights_names:
            if "weight" in param:
                nn.init.xavier_uniform_(m._parameters[param])
net.apply(init_weights);
```

### Carregando Vetores de Palavras Pré-treinados

Como o conjunto de dados de treinamento para classificação de sentimento não é muito grande, para lidar com o *overfitting*, usaremos diretamente vetores de palavras pré-treinados em um corpus maior como vetores de características de todas as palavras. Aqui, carregamos um vetor de palavras GloVe de 100 dimensões para cada palavra do `vocabulário` do dicionário.

```{.python .input}
#@tab all
glove_embedding = d2l.TokenEmbedding('glove.6b.100d')
```

Consultando os vetores de palavras que estão em nosso vocabulário.

```{.python .input}
#@tab all
embeds = glove_embedding[vocab.idx_to_token]
embeds.shape
```

Em seguida, usaremos esses vetores de palavras como vetores de características para cada palavra nas revisões. Observe que as dimensões dos vetores de palavras pré-treinados precisam ser consistentes com o tamanho de saída da camada de incorporação `embed_size` no modelo criado. Além disso, não atualizamos mais esses vetores de palavras durante o treinamento.

```{.python .input}
net.embedding.weight.set_data(embeds)
net.embedding.collect_params().setattr('grad_req', 'null')
```

```{.python .input}
#@tab pytorch
net.embedding.weight.data.copy_(embeds)
net.embedding.weight.requires_grad = False
```

### Treinamento e Avaliação do Modelo

Agora podemos começar a treinar.

```{.python .input}
lr, num_epochs = 0.01, 5
trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': lr})
loss = gluon.loss.SoftmaxCrossEntropyLoss()
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)
```

```{.python .input}
#@tab pytorch
lr, num_epochs = 0.01, 5
trainer = torch.optim.Adam(net.parameters(), lr=lr)
loss = nn.CrossEntropyLoss(reduction="none")
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)
```

Finalmente, definir a função de previsão.

```{.python .input}
#@save
def predict_sentiment(net, vocab, sentence):
    sentence = np.array(vocab[sentence.split()], ctx=d2l.try_gpu())
    label = np.argmax(net(sentence.reshape(1, -1)), axis=1)
    return 'positive' if label == 1 else 'negative'
```

```{.python .input}
#@tab pytorch
#@save
def predict_sentiment(net, vocab, sentence):
    sentence = torch.tensor(vocab[sentence.split()], device=d2l.try_gpu())
    label = torch.argmax(net(sentence.reshape(1, -1)), dim=1)
    return 'positive' if label == 1 else 'negative'
```

Em seguida, usamos o modelo treinado para classificar os sentimentos de duas frases simples.

```{.python .input}
#@tab all
predict_sentiment(net, vocab, 'this movie is so great')
```

```{.python .input}
#@tab all
predict_sentiment(net, vocab, 'this movie is so bad')
```

## Resumo

* A classificação de texto transforma uma sequência de texto de comprimento indefinido em uma categoria de texto. Esta é uma aplicação *downstream* de incorporação de palavras.
* Podemos aplicar vetores de palavras pré-treinados e redes neurais recorrentes para classificar as emoções em um texto.


## Exercícios

1. Aumente o número de épocas. Que taxa de precisão você pode alcançar nos conjuntos de dados de treinamento e teste? Que tal tentar reajustar outros hiperparâmetros?
1. O uso de vetores de palavras pré-treinados maiores, como vetores de palavras GloVe 300-dimensionais, melhorará a acurácia da classificação?
1. Podemos melhorar a acurácia da classificação usando a ferramenta de tokenização de palavras spaCy? Você precisa instalar spaCy: `pip install spacy` e instalar o pacote em inglês: `python -m spacy download en`. No código, primeiro importe spacy: `import spacy`. Em seguida, carregue o pacote spacy em inglês: `spacy_en = spacy.load ('en')`. Finalmente, defina a função `def tokenizer(text): return [tok.text for tok in spacy_en.tokenizer(text)]` e substitua a função original de `tokenizer`. Deve-se notar que o vetor de palavras do GloVe usa "-" para conectar cada palavra ao armazenar frases nominais. Por exemplo, a frase "new york" é representada como "new-york" no GloVe. Depois de usar a tokenização spaCy, "new york" pode ser armazenado como "new york".

:begin_tab:`mxnet`
[Discussões](https://discuss.d2l.ai/t/392)
:end_tab:

:begin_tab:`pytorch`
[Discussões](https://discuss.d2l.ai/t/1424)
:end_tab:
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTg3NTkwNzM4OCwxNDE1MDYzNzgwLDIwMz
k1NjYzMDksLTY0NjE4ODU0Nl19
-->