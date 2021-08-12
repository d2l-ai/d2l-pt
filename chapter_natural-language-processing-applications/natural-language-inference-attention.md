# Inferência de Linguagem Natural: Usando a Atenção
:label:`sec_natural-language-inference-attention`

Introduzimos a tarefa de inferência em linguagem natural e o conjunto de dados SNLI em :numref:`sec_natural-language-inference-and-dataset`. Em vista de muitos modelos baseados em arquiteturas complexas e profundas, Parikh et al. proposto para abordar a inferência de linguagem natural com mecanismos de atenção e chamou-o de "modelo de atenção decomposto" :cite:`Parikh.Tackstrom.Das.ea.2016`.
Isso resulta em um modelo sem camadas recorrentes ou convolucionais, alcançando o melhor resultado no momento no conjunto de dados SNLI com muito menos parâmetros.
Nesta seção, iremos descrever e implementar este método baseado em atenção (com MLPs) para inferência de linguagem natural, conforme descrito em :numref:`fig_nlp-map-nli -ention`.

![Esta seção alimenta o GloVe pré-treinado para uma arquitetura baseada em atenção e MLPs para inferência de linguagem natural.](../img/nlp-map-nli-attention.svg)
:label:`fig_nlp-map-nli-attention`


## O Modelo

Mais simples do que preservar a ordem das palavras em premissas e hipóteses,
podemos apenas alinhar as palavras em uma sequência de texto com todas as palavras na outra e vice-versa,
em seguida, compare e agregue essas informações para prever as relações lógicas
entre premissas e hipóteses.
Semelhante ao alinhamento de palavras entre as frases fonte e alvo na tradução automática,
o alinhamento de palavras entre premissas e hipóteses
pode ser perfeitamente realizado por mecanismos de atenção.

![Inferência de linguagem natural usando mecanismos de atenção. ](../img/nli-attention.svg)
:label:`fig_nli_attention`

:numref:`fig_nli_attention` descreve o método de inferência de linguagem natural usando mecanismos de atenção.
Em um nível superior, consiste em três etapas treinadas em conjunto: alinhar, comparar e agregar.
Iremos ilustrá-los passo a passo a seguir.

```{.python .input}
from d2l import mxnet as d2l
from mxnet import gluon, init, np, npx
from mxnet.gluon import nn

npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
from torch.nn import functional as F
```

### Alinhar

A primeira etapa é alinhar as palavras em uma sequência de texto a cada palavra na outra sequência.
Suponha que a premissa seja "preciso dormir" e a hipótese "estou cansado".
Devido à semelhança semântica,
podemos desejar alinhar "i" na hipótese com "i" na premissa,
e alinhe "cansado" na hipótese com "sono" na premissa.
Da mesma forma, podemos desejar alinhar "i" na premissa com "i" na hipótese,
e alinhar "necessidade" e "sono" na premissa com "cansado" na hipótese.
Observe que esse alinhamento é *suave* usando a média ponderada,
onde, idealmente, grandes pesos estão associados às palavras a serem alinhadas.
Para facilitar a demonstração, :numref:`fig_nli_attention` mostra tal alinhamento de uma maneira *dura*.

Agora descrevemos o alinhamento suave usando mecanismos de atenção com mais detalhes.
Denotamos por  $\mathbf{A} = (\mathbf{a}_1, \ldots, \mathbf{a}_m)$
e  $\mathbf{B} = (\mathbf{b}_1, \ldots, \mathbf{b}_n)$ a premissa e hipótese,
cujo número de palavras são $m$ e $n$, respectivamente,
onde  $\mathbf{a}_i, \mathbf{b}_j \in \mathbb{R}^{d}$ ($i = 1, \ldots, m, j = 1, \ldots, n$) é um vetor de incorporação de palavras $d$-dimensional.
Para o alinhamento suave, calculamos os pesos de atenção $e_{ij} \in \mathbb{R}$ como

$$e_{ij} = f(\mathbf{a}_i)^\top f(\mathbf{b}_j),$$
:eqlabel:`eq_nli_e`

onde a função $f$ é um MLP definido na seguinte função `mlp`.
A dimensão de saída de $f$ é especificada pelo argumento `num_hiddens` de` mlp`.

```{.python .input}
def mlp(num_hiddens, flatten):
    net = nn.Sequential()
    net.add(nn.Dropout(0.2))
    net.add(nn.Dense(num_hiddens, activation='relu', flatten=flatten))
    net.add(nn.Dropout(0.2))
    net.add(nn.Dense(num_hiddens, activation='relu', flatten=flatten))
    return net
```

```{.python .input}
#@tab pytorch
def mlp(num_inputs, num_hiddens, flatten):
    net = []
    net.append(nn.Dropout(0.2))
    net.append(nn.Linear(num_inputs, num_hiddens))
    net.append(nn.ReLU())
    if flatten:
        net.append(nn.Flatten(start_dim=1))
    net.append(nn.Dropout(0.2))
    net.append(nn.Linear(num_hiddens, num_hiddens))
    net.append(nn.ReLU())
    if flatten:
        net.append(nn.Flatten(start_dim=1))
    return nn.Sequential(*net)
```


Deve-se destacar que, em :eqref:`eq_nli_e`
$f$ pega as entradas $\mathbf{a}_i$ and $\mathbf{b}_j$ separadamente em vez de pegar um par delas juntas como entrada.
Este truque de *decomposição* leva a apenas aplicações $m + n$ (complexidade linear) de $f$ em vez de $mn$ aplicativos
(complexidade quadrática).


Normalizando os pesos de atenção em :eqref:`eq_nli_e`,
calculamos a média ponderada de todas as palavras incluídas na hipótese
para obter a representação da hipótese que está suavemente alinhada com a palavra indexada por $i$ na premissa:

$$
\boldsymbol{\beta}_i = \sum_{j=1}^{n}\frac{\exp(e_{ij})}{ \sum_{k=1}^{n} \exp(e_{ik})} \mathbf{b}_j.
$$

Da mesma forma, calculamos o alinhamento suave de palavras da premissa para cada palavra indexada por $j$ na hipótese:

$$
\boldsymbol{\alpha}_j = \sum_{i=1}^{m}\frac{\exp(e_{ij})}{ \sum_{k=1}^{m} \exp(e_{kj})} \mathbf{a}_i.
$$

Abaixo, definimos a classe `Attend` para calcular o alinhamento suave das hipóteses (`beta`) com as premissas de entrada `A` e o alinhamento suave das premissas (`alfa`) com as hipóteses de entrada `B`.

```{.python .input}
class Attend(nn.Block):
    def __init__(self, num_hiddens, **kwargs):
        super(Attend, self).__init__(**kwargs)
        self.f = mlp(num_hiddens=num_hiddens, flatten=False)

    def forward(self, A, B):
        # Shape of `A`/`B`: (b`atch_size`, no. of words in sequence A/B,
        # `embed_size`)
        # Shape of `f_A`/`f_B`: (`batch_size`, no. of words in sequence A/B,
        # `num_hiddens`)
        f_A = self.f(A)
        f_B = self.f(B)
        # Shape of `e`: (`batch_size`, no. of words in sequence A,
        # no. of words in sequence B)
        e = npx.batch_dot(f_A, f_B, transpose_b=True)
        # Shape of `beta`: (`batch_size`, no. of words in sequence A,
        # `embed_size`), where sequence B is softly aligned with each word
        # (axis 1 of `beta`) in sequence A
        beta = npx.batch_dot(npx.softmax(e), B)
        # Shape of `alpha`: (`batch_size`, no. of words in sequence B,
        # `embed_size`), where sequence A is softly aligned with each word
        # (axis 1 of `alpha`) in sequence B
        alpha = npx.batch_dot(npx.softmax(e.transpose(0, 2, 1)), A)
        return beta, alpha
```

```{.python .input}
#@tab pytorch
class Attend(nn.Module):
    def __init__(self, num_inputs, num_hiddens, **kwargs):
        super(Attend, self).__init__(**kwargs)
        self.f = mlp(num_inputs, num_hiddens, flatten=False)

    def forward(self, A, B):
        # Shape of `A`/`B`: (`batch_size`, no. of words in sequence A/B,
        # `embed_size`)
        # Shape of `f_A`/`f_B`: (`batch_size`, no. of words in sequence A/B,
        # `num_hiddens`)
        f_A = self.f(A)
        f_B = self.f(B)
        # Shape of `e`: (`batch_size`, no. of words in sequence A,
        # no. of words in sequence B)
        e = torch.bmm(f_A, f_B.permute(0, 2, 1))
        # Shape of `beta`: (`batch_size`, no. of words in sequence A,
        # `embed_size`), where sequence B is softly aligned with each word
        # (axis 1 of `beta`) in sequence A
        beta = torch.bmm(F.softmax(e, dim=-1), B)
        # Shape of `alpha`: (`batch_size`, no. of words in sequence B,
        # `embed_size`), where sequence A is softly aligned with each word
        # (axis 1 of `alpha`) in sequence B
        alpha = torch.bmm(F.softmax(e.permute(0, 2, 1), dim=-1), A)
        return beta, alpha
```

### Comparando


Na próxima etapa, comparamos uma palavra em uma sequência com a outra sequência que está suavemente alinhada com essa palavra.
Observe que no alinhamento suave, todas as palavras de uma sequência, embora provavelmente com pesos de atenção diferentes, serão comparadas com uma palavra na outra sequência.
Para facilitar a demonstração, :numref:`fig_nli_attention` emparelha palavras com palavras alinhadas de uma forma *dura*.
Por exemplo, suponha que a etapa de atendimento determina que "necessidade" e "sono" na premissa estão ambos alinhados com "cansado" na hipótese, o par "cansado - preciso dormir" será comparado.

Na etapa de comparação, alimentamos a concatenação (operador $[\cdot, \cdot]$) de palavras de uma sequência e palavras alinhadas de outra sequência em uma função $g$ (um MLP):

$$\mathbf{v}_{A,i} = g([\mathbf{a}_i, \boldsymbol{\beta}_i]), i = 1, \ldots, m\\ \mathbf{v}_{B,j} = g([\mathbf{b}_j, \boldsymbol{\alpha}_j]), j = 1, \ldots, n.$$

:eqlabel:`eq_nli_v_ab`


Em:eqref:`eq_nli_v_ab`, $\mathbf{v}_{A,i}$ é a comparação entre a palavra $i$ na premissa e todas as palavras da hipótese que estão suavemente alinhadas com a palavra $i$;
enquanto $\mathbf{v}_{B,j}$ é a comparação entre a palavra $j$ na hipótese e todas as palavras da premissa que estão suavemente alinhadas com a palavra $j$.
A seguinte classe `Compare` define como a etapa de comparação.

```{.python .input}
class Compare(nn.Block):
    def __init__(self, num_hiddens, **kwargs):
        super(Compare, self).__init__(**kwargs)
        self.g = mlp(num_hiddens=num_hiddens, flatten=False)

    def forward(self, A, B, beta, alpha):
        V_A = self.g(np.concatenate([A, beta], axis=2))
        V_B = self.g(np.concatenate([B, alpha], axis=2))
        return V_A, V_B
```

```{.python .input}
#@tab pytorch
class Compare(nn.Module):
    def __init__(self, num_inputs, num_hiddens, **kwargs):
        super(Compare, self).__init__(**kwargs)
        self.g = mlp(num_inputs, num_hiddens, flatten=False)

    def forward(self, A, B, beta, alpha):
        V_A = self.g(torch.cat([A, beta], dim=2))
        V_B = self.g(torch.cat([B, alpha], dim=2))
        return V_A, V_B
```

### Agregando

Com dois conjuntos de vetores de comparação $\mathbf{v}_{A,i}$ ($i = 1, \ldots, m$) e $\mathbf{v}_{B,j}$ ($j = 1, \ldots, n$) disponível,
na última etapa, agregaremos essas informações para inferir a relação lógica.
Começamos resumindo os dois conjuntos:

$$
\mathbf{v}_A = \sum_{i=1}^{m} \mathbf{v}_{A,i}, \quad \mathbf{v}_B = \sum_{j=1}^{n}\mathbf{v}_{B,j}.
$$

Em seguida, alimentamos a concatenação de ambos os resultados do resumo na função $h$ (um MLP) para obter o resultado da classificação do relacionamento lógico:

$$
\hat{\mathbf{y}} = h([\mathbf{v}_A, \mathbf{v}_B]).
$$

A etapa de agregação é definida na seguinte classe `Aggregate`.

```{.python .input}
class Aggregate(nn.Block):
    def __init__(self, num_hiddens, num_outputs, **kwargs):
        super(Aggregate, self).__init__(**kwargs)
        self.h = mlp(num_hiddens=num_hiddens, flatten=True)
        self.h.add(nn.Dense(num_outputs))

    def forward(self, V_A, V_B):
        # Sum up both sets of comparison vectors
        V_A = V_A.sum(axis=1)
        V_B = V_B.sum(axis=1)
        # Feed the concatenation of both summarization results into an MLP
        Y_hat = self.h(np.concatenate([V_A, V_B], axis=1))
        return Y_hat
```

```{.python .input}
#@tab pytorch
class Aggregate(nn.Module):
    def __init__(self, num_inputs, num_hiddens, num_outputs, **kwargs):
        super(Aggregate, self).__init__(**kwargs)
        self.h = mlp(num_inputs, num_hiddens, flatten=True)
        self.linear = nn.Linear(num_hiddens, num_outputs)

    def forward(self, V_A, V_B):
        # Sum up both sets of comparison vectors
        V_A = V_A.sum(dim=1)
        V_B = V_B.sum(dim=1)
        # Feed the concatenation of both summarization results into an MLP
        Y_hat = self.linear(self.h(torch.cat([V_A, V_B], dim=1)))
        return Y_hat
```

### Juntando Tudo

Ao reunir as etapas de atendimento, comparação e agregação,
definimos o modelo de atenção decomposto para treinar conjuntamente essas três etapas.

```{.python .input}
class DecomposableAttention(nn.Block):
    def __init__(self, vocab, embed_size, num_hiddens, **kwargs):
        super(DecomposableAttention, self).__init__(**kwargs)
        self.embedding = nn.Embedding(len(vocab), embed_size)
        self.attend = Attend(num_hiddens)
        self.compare = Compare(num_hiddens)
        # There are 3 possible outputs: entailment, contradiction, and neutral
        self.aggregate = Aggregate(num_hiddens, 3)

    def forward(self, X):
        premises, hypotheses = X
        A = self.embedding(premises)
        B = self.embedding(hypotheses)
        beta, alpha = self.attend(A, B)
        V_A, V_B = self.compare(A, B, beta, alpha)
        Y_hat = self.aggregate(V_A, V_B)
        return Y_hat
```

```{.python .input}
#@tab pytorch
class DecomposableAttention(nn.Module):
    def __init__(self, vocab, embed_size, num_hiddens, num_inputs_attend=100,
                 num_inputs_compare=200, num_inputs_agg=400, **kwargs):
        super(DecomposableAttention, self).__init__(**kwargs)
        self.embedding = nn.Embedding(len(vocab), embed_size)
        self.attend = Attend(num_inputs_attend, num_hiddens)
        self.compare = Compare(num_inputs_compare, num_hiddens)
        # There are 3 possible outputs: entailment, contradiction, and neutral
        self.aggregate = Aggregate(num_inputs_agg, num_hiddens, num_outputs=3)

    def forward(self, X):
        premises, hypotheses = X
        A = self.embedding(premises)
        B = self.embedding(hypotheses)
        beta, alpha = self.attend(A, B)
        V_A, V_B = self.compare(A, B, beta, alpha)
        Y_hat = self.aggregate(V_A, V_B)
        return Y_hat
```

## Treinamento e Avaliação do Modelo

Agora vamos treinar e avaliar o modelo de atenção decomposto definido no conjunto de dados SNLI.
Começamos lendo o *dataset*.


### Lendo o *Dataset*

Baixamos e lemos o conjunto de dados SNLI usando a função definida em :numref:`sec_natural-language-inference-and-dataset`. O tamanho do lote e o comprimento da sequência são definidos em $256$ e $50$, respectivamente.

```{.python .input}
#@tab all
batch_size, num_steps = 256, 50
train_iter, test_iter, vocab = d2l.load_data_snli(batch_size, num_steps)
```

### Criando o Modelo

Usamos a incorporação GloVe pré-treinada $100$-dimensional para representar os *tokens* de entrada.
Assim, predefinimos a dimensão dos vetores $\mathbf{a}_i$ e $\mathbf{b}_j$ em :eqref:`eq_nli_e` como $100$.
A dimensão de saída das funções $f$ in :eqref:`eq_nli_e` e $g$ em :eqref:`eq_nli_v_ab` é definida como $200$.
Em seguida, criamos uma instância de modelo, inicializamos seus parâmetros,
e carregamos o *GloVe* embarcado para inicializar vetores de *tokens* de entrada.

```{.python .input}
embed_size, num_hiddens, devices = 100, 200, d2l.try_all_gpus()
net = DecomposableAttention(vocab, embed_size, num_hiddens)
net.initialize(init.Xavier(), ctx=devices)
glove_embedding = d2l.TokenEmbedding('glove.6b.100d')
embeds = glove_embedding[vocab.idx_to_token]
net.embedding.weight.set_data(embeds)
```

```{.python .input}
#@tab pytorch
embed_size, num_hiddens, devices = 100, 200, d2l.try_all_gpus()
net = DecomposableAttention(vocab, embed_size, num_hiddens)
glove_embedding = d2l.TokenEmbedding('glove.6b.100d')
embeds = glove_embedding[vocab.idx_to_token]
net.embedding.weight.data.copy_(embeds);
```

### Treinamento e Avaliação do Modelo

Em contraste com a função `split_batch` em :numref:`sec_multi_gpu` que recebe entradas únicas, como sequências de texto (ou imagens),
definimos uma função `split_batch_multi_inputs` para obter várias entradas, como premissas e hipóteses em minibatches.

```{.python .input}
#@save
def split_batch_multi_inputs(X, y, devices):
    """Split multi-input `X` and `y` into multiple devices."""
    X = list(zip(*[gluon.utils.split_and_load(
        feature, devices, even_split=False) for feature in X]))
    return (X, gluon.utils.split_and_load(y, devices, even_split=False))
```

Agora podemos treinar e avaliar o modelo no *dataset* SNLI.

```{.python .input}
lr, num_epochs = 0.001, 4
trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': lr})
loss = gluon.loss.SoftmaxCrossEntropyLoss()
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices,
               split_batch_multi_inputs)
```

```{.python .input}
#@tab pytorch
lr, num_epochs = 0.001, 4
trainer = torch.optim.Adam(net.parameters(), lr=lr)
loss = nn.CrossEntropyLoss(reduction="none")
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)
```

### Usando o Modelo

Finalmente, defina a função de previsão para produzir a relação lógica entre um par de premissas e hipóteses.

```{.python .input}
#@save
def predict_snli(net, vocab, premise, hypothesis):
    premise = np.array(vocab[premise], ctx=d2l.try_gpu())
    hypothesis = np.array(vocab[hypothesis], ctx=d2l.try_gpu())
    label = np.argmax(net([premise.reshape((1, -1)),
                           hypothesis.reshape((1, -1))]), axis=1)
    return 'entailment' if label == 0 else 'contradiction' if label == 1 \
            else 'neutral'
```

```{.python .input}
#@tab pytorch
#@save
def predict_snli(net, vocab, premise, hypothesis):
    net.eval()
    premise = torch.tensor(vocab[premise], device=d2l.try_gpu())
    hypothesis = torch.tensor(vocab[hypothesis], device=d2l.try_gpu())
    label = torch.argmax(net([premise.reshape((1, -1)),
                           hypothesis.reshape((1, -1))]), dim=1)
    return 'entailment' if label == 0 else 'contradiction' if label == 1 \
            else 'neutral'
```

Podemos usar o modelo treinado para obter o resultado da inferência em linguagem natural para um par de frases de amostra.
```{.python .input}
#@tab all
predict_snli(net, vocab, ['he', 'is', 'good', '.'], ['he', 'is', 'bad', '.'])
```

## Resumo

* O modelo de atenção decomposto consiste em três etapas para prever as relações lógicas entre premissas e hipóteses: atendimento, comparação e agregação.
* Com mecanismos de atenção, podemos alinhar palavras em uma sequência de texto com todas as palavras na outra e vice-versa. Esse alinhamento é suave usando a média ponderada, em que, idealmente, grandes pesos são associados às palavras a serem alinhadas.
* O truque da decomposição leva a uma complexidade linear mais desejável do que a complexidade quadrática ao calcular os pesos de atenção.
* Podemos usar a incorporação de palavras pré-treinadas como a representação de entrada para tarefas de processamento de linguagem natural *downstream*, como inferência de linguagem natural.


## Exercícios

1. Treine o modelo com outras combinações de hiperparâmetros. Você pode obter melhor precisão no conjunto de teste?
1. Quais são as principais desvantagens do modelo de atenção decomponível para inferência de linguagem natural?
1. Suponha que desejamos obter o nível de similaridade semântica (por exemplo, um valor contínuo entre $0$ e $1$) para qualquer par de sentenças. Como devemos coletar e rotular o conjunto de dados? Você pode projetar um modelo com mecanismos de atenção?

:begin_tab:`mxnet`
[Discussõess](https://discuss.d2l.ai/t/395)
:end_tab:

:begin_tab:`pytorch`
[Discussõess](https://discuss.d2l.ai/t/1530)
:end_tab:
<!--stackedit_data:
eyJoaXN0b3J5IjpbNzE3ODI1OCwtMTEyOTIwNDY5OSwtMjU2ND
IwMjU2LDU3MjE3MjY2NiwtOTM0NTk2OTI1XX0=
-->