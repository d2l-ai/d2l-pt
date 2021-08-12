# Atenção Multi-Head 
:label:`sec_multihead-attention`



Na prática,
dado o mesmo conjunto de consultas, chaves e valores
podemos querer que nosso modelo
combine conhecimento de
diferentes comportamentos do mesmo mecanismo de atenção,
como capturar dependências de vários intervalos (por exemplo, intervalo mais curto vs. intervalo mais longo)
dentro de uma sequência.
Desse modo,
pode ser benéfico
permitir nosso mecanismo de atenção
para usar em conjunto diferentes subespaços de representação
de consultas, chaves e valores.



Para este fim,
em vez de realizar um único agrupamento de atenção,
consultas, chaves e valores
podem ser transformados
com $h$ projeções lineares aprendidas independentemente.
Então, essas $h$ consultas, chaves e valores projetados
são alimentados em agrupamento de atenção em paralelo.
No fim,
$h$ resultados de concentração de atenção
são concatenados e
transformados com outra projeção linear aprendida
para produzir a saída final.
Este design
é chamado de *atenção multi-head*,
onde cada uma das saídas de concentração de $h$
é um *head* :cite:`Vaswani.Shazeer.Parmar.ea.2017`.
Usando camadas totalmente conectadas
para realizar transformações lineares que podem ser aprendidas,
:numref:`fig_multi-head-attention`
descreve a atenção de *multi-head*.

![Multi-head attention, where multiple heads are concatenated then linearly transformed.](../img/multi-head-attention.svg)
:label:`fig_multi-head-attention`




## Modelo

Antes de fornecer a implementação da atenção *multi-head*,
vamos formalizar este modelo matematicamente.
Dada uma consulta $\mathbf{q} \in \mathbb{R}^{d_q}$,
uma chave $\mathbf{k} \in \mathbb{R}^{d_k}$,
e um valor $\mathbf{v} \in \mathbb{R}^{d_v}$,
cada *head* de atenção $\mathbf{h}_i$  ($i = 1, \ldots, h$)
é calculado como

$$\mathbf{h}_i = f(\mathbf W_i^{(q)}\mathbf q, \mathbf W_i^{(k)}\mathbf k,\mathbf W_i^{(v)}\mathbf v) \in \mathbb R^{p_v},$$

onde parâmetros aprendíveis
$\mathbf W_i^{(q)}\in\mathbb R^{p_q\times d_q}$,
$\mathbf W_i^{(k)}\in\mathbb R^{p_k\times d_k}$
e $\mathbf W_i^{(v)}\in\mathbb R^{p_v\times d_v}$,
e
$f$ é concentração de atenção,
tal como
atenção aditiva e atenção de produto escalonado
em :numref:`sec_attention-scoring-functions`.
A saída de atenção *multi-head*
é outra transformação linear via
parâmetros aprendíveis
$\mathbf W_o\in\mathbb R^{p_o\times h p_v}$
da concatenação de $h$ cabeças:

$$\mathbf W_o \begin{bmatrix}\mathbf h_1\\\vdots\\\mathbf h_h\end{bmatrix} \in \mathbb{R}^{p_o}.$$

Com base neste design,
cada cabeça pode atender a diferentes partes da entrada.
Funções mais sofisticadas do que a média ponderada simples
podem ser expressadas.

```{.python .input}
from d2l import mxnet as d2l
import math
from mxnet import autograd, np, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import math
import torch
from torch import nn
```

## Implementação

Em nossa implementação,
nós escolhemos a atenção do produto escalonado
para cada *head* da atenção de várias cabeças.
Para evitar um crescimento significativo
de custo computacional e custo de parametrização,
montamos
$p_q = p_k = p_v = p_o / h$.
Observe que $h$ *heads*
pode ser calculado em paralelo
se definirmos
o número de saídas de transformações lineares
para a consulta, chave e valor
a $p_q h = p_k h = p_v h = p_o$.
Na implementação a seguir,
$p_o$ é especificado através do argumento `num_hiddens`.

```{.python .input}
#@save
class MultiHeadAttention(nn.Block):
    def __init__(self, num_hiddens, num_heads, dropout, use_bias=False,
                 **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = d2l.DotProductAttention(dropout)
        self.W_q = nn.Dense(num_hiddens, use_bias=use_bias, flatten=False)
        self.W_k = nn.Dense(num_hiddens, use_bias=use_bias, flatten=False)
        self.W_v = nn.Dense(num_hiddens, use_bias=use_bias, flatten=False)
        self.W_o = nn.Dense(num_hiddens, use_bias=use_bias, flatten=False)

    def forward(self, queries, keys, values, valid_lens):
        # Shape of `queries`, `keys`, or `values`:
        # (`batch_size`, no. of queries or key-value pairs, `num_hiddens`)
        # Shape of `valid_lens`:
        # (`batch_size`,) or (`batch_size`, no. of queries)
        # After transposing, shape of output `queries`, `keys`, or `values`:
        # (`batch_size` * `num_heads`, no. of queries or key-value pairs,
        # `num_hiddens` / `num_heads`)
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)

        if valid_lens is not None:
            # On axis 0, copy the first item (scalar or vector) for
            # `num_heads` times, then copy the next item, and so on
            valid_lens = valid_lens.repeat(self.num_heads, axis=0)

        # Shape of `output`: (`batch_size` * `num_heads`, no. of queries,
        # `num_hiddens` / `num_heads`)
        output = self.attention(queries, keys, values, valid_lens)
        
        # Shape of `output_concat`:
        # (`batch_size`, no. of queries, `num_hiddens`)
        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)
```

```{.python .input}
#@tab pytorch
#@save
class MultiHeadAttention(nn.Module):
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 num_heads, dropout, bias=False, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = d2l.DotProductAttention(dropout)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    def forward(self, queries, keys, values, valid_lens):
        # Shape of `queries`, `keys`, or `values`:
        # (`batch_size`, no. of queries or key-value pairs, `num_hiddens`)
        # Shape of `valid_lens`:
        # (`batch_size`,) or (`batch_size`, no. of queries)
        # After transposing, shape of output `queries`, `keys`, or `values`:
        # (`batch_size` * `num_heads`, no. of queries or key-value pairs,
        # `num_hiddens` / `num_heads`)
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)

        if valid_lens is not None:
            # On axis 0, copy the first item (scalar or vector) for
            # `num_heads` times, then copy the next item, and so on
            valid_lens = torch.repeat_interleave(
                valid_lens, repeats=self.num_heads, dim=0)

        # Shape of `output`: (`batch_size` * `num_heads`, no. of queries,
        # `num_hiddens` / `num_heads`)
        output = self.attention(queries, keys, values, valid_lens)

        # Shape of `output_concat`:
        # (`batch_size`, no. of queries, `num_hiddens`)
        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)
```

Para permitir o cálculo paralelo de várias *heads*
a classe `MultiHeadAttention` acima usa duas funções de transposição, conforme definido abaixo.
Especificamente,
a função `transpose_output` reverte a operação
da função `transpose_qkv`.

```{.python .input}
#@save
def transpose_qkv(X, num_heads):
    # Shape of input `X`:
    # (`batch_size`, no. of queries or key-value pairs, `num_hiddens`).
    # Shape of output `X`:
    # (`batch_size`, no. of queries or key-value pairs, `num_heads`,
    # `num_hiddens` / `num_heads`)
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)

    # Shape of output `X`:
    # (`batch_size`, `num_heads`, no. of queries or key-value pairs,
    # `num_hiddens` / `num_heads`)
    X = X.transpose(0, 2, 1, 3)

    # Shape of `output`:
    # (`batch_size` * `num_heads`, no. of queries or key-value pairs,
    # `num_hiddens` / `num_heads`)
    return X.reshape(-1, X.shape[2], X.shape[3])


#@save
def transpose_output(X, num_heads):
    """Reverse the operation of `transpose_qkv`"""
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.transpose(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)
```

```{.python .input}
#@tab pytorch
#@save
def transpose_qkv(X, num_heads):
    # Shape of input `X`:
    # (`batch_size`, no. of queries or key-value pairs, `num_hiddens`).
    # Shape of output `X`:
    # (`batch_size`, no. of queries or key-value pairs, `num_heads`,
    # `num_hiddens` / `num_heads`)
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)

    # Shape of output `X`:
    # (`batch_size`, `num_heads`, no. of queries or key-value pairs,
    # `num_hiddens` / `num_heads`)
    X = X.permute(0, 2, 1, 3)

    # Shape of `output`:
    # (`batch_size` * `num_heads`, no. of queries or key-value pairs,
    # `num_hiddens` / `num_heads`)
    return X.reshape(-1, X.shape[2], X.shape[3])


#@save
def transpose_output(X, num_heads):
    """Reverse the operation of `transpose_qkv`"""
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)
```

Vamos testar nossa classe `MultiHeadAttention` implementada
usando um exemplo de brinquedo em que as chaves e os valores são iguais.
Como resultado,
a forma da saída de atenção *multi-head*
é (`batch_size`,` num_queries`, `num_hiddens`).

```{.python .input}
num_hiddens, num_heads = 100, 5
attention = MultiHeadAttention(num_hiddens, num_heads, 0.5)
attention.initialize()
```

```{.python .input}
#@tab pytorch
num_hiddens, num_heads = 100, 5
attention = MultiHeadAttention(num_hiddens, num_hiddens, num_hiddens,
                               num_hiddens, num_heads, 0.5)
attention.eval()
```

```{.python .input}
#@tab all
batch_size, num_queries, num_kvpairs, valid_lens = 2, 4, 6, d2l.tensor([3, 2])
X = d2l.ones((batch_size, num_queries, num_hiddens))
Y = d2l.ones((batch_size, num_kvpairs, num_hiddens))
attention(X, Y, Y, valid_lens).shape
```

## Resumo

* A atenção *multi-head* combina o conhecimento do mesmo agrupamento de atenção por meio de diferentes subespaços de representação de consultas, chaves e valores.
* Para calcular várias *heads* de atenção de *multi-heads* em paralelo, é necessária a manipulação adequada do tensor.



## Exercícios

1. Visualize o peso da atenção *multi-head* neste experimento.
1. Suponha que temos um modelo treinado com base na atenção *multi-head* e queremos podar as *heads* menos importantes para aumentar a velocidade de previsão. Como podemos projetar experimentos para medir a importância de uma *head* de atenção?


:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/1634)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1635)
:end_tab:
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTEzOTQ3NzkwNDMsLTEzOTI0MzE2MjQsMT
QyMzkwNjgyMCwtOTc5MjA5NTQxLDE5NTU1NjIxMDVdfQ==
-->