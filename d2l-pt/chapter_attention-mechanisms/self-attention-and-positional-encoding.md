# Autoatenção e Codificação Posicional
:label:`sec_self-attention-and-positional-encoding`

No aprendizado profundo, costumamos usar CNNs ou RNNs para codificar uma sequência.
Agora, com os mecanismos de atenção, imagine que alimentamos uma sequência de tokens no *pooling* de atenção para que o mesmo conjunto de tokens atue como consultas, chaves e valores.
Especificamente, cada consulta atende a todos os pares de valores-chave e gera uma saída de atenção.
Como as consultas, chaves e valores vêm do mesmo lugar, isso executa *autoatenção* :cite:`Lin.Feng.Santos.ea.2017,Vaswani.Shazeer.Parmar.ea.2017`,  que também é chamado *intra-atenção* :cite:`Cheng.Dong.Lapata.2016,Parikh.Tackstrom.Das.ea.2016,Paulus.Xiong.Socher.2017`.
Nesta seção, discutiremos a codificação de sequência usando autoatenção, incluindo o uso de informações adicionais para a ordem da sequência.

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

## Autoatenção

Dada uma sequência de tokens de entrada
$\mathbf{x}_1, \ldots, \mathbf{x}_n$ onde qualquer $\mathbf{x}_i \in \mathbb{R}^d$ ($1 \leq i \leq n$),
suas saídas de autoatenção é
uma sequência do mesmo comprimento
$\mathbf{y}_1, \ldots, \mathbf{y}_n$,
Onde

$$\mathbf{y}_i = f(\mathbf{x}_i, (\mathbf{x}_1, \mathbf{x}_1), \ldots, (\mathbf{x}_n, \mathbf{x}_n)) \in \mathbb{R}^d$$

de acordo com a definição de concentração de $f$ em
:eqref:`eq_attn-pooling`.
Usando a atenção de várias cabeças,
o seguinte trecho de código
calcula a autoatenção de um tensor
com forma (tamanho do lote, número de etapas de tempo ou comprimento da sequência em tokens, $d$).
O tensor de saída tem o mesmo formato.

```{.python .input}
num_hiddens, num_heads = 100, 5
attention = d2l.MultiHeadAttention(num_hiddens, num_heads, 0.5)
attention.initialize()
```

```{.python .input}
#@tab pytorch
num_hiddens, num_heads = 100, 5
attention = d2l.MultiHeadAttention(num_hiddens, num_hiddens, num_hiddens,
                                   num_hiddens, num_heads, 0.5)
attention.eval()
```

```{.python .input}
#@tab all
batch_size, num_queries, valid_lens = 2, 4, d2l.tensor([3, 2])
X = d2l.ones((batch_size, num_queries, num_hiddens))
attention(X, X, X, valid_lens).shape
```

## Comparando CNNs, RNNs e Autoatenção
:label:`subsec_cnn-rnn-self-attention`

Vamos comparar arquiteturas para mapear uma sequência de $n$ tokens para outra sequência de igual comprimento,
onde cada token de entrada ou saída é representado por um vetor $d$-dimensional.
Especificamente, consideraremos CNNs, RNNs e autoatenção.
Compararemos sua complexidade computacional, operações sequenciais e comprimentos máximos de caminho.
Observe que as operações sequenciais evitam a computação paralela, enquanto um caminho mais curto entre qualquer combinação de posições de sequência torna mais fácil aprender dependências de longo alcance dentro da sequência :cite:`Hochreiter.Bengio.Frasconi.ea.2001`.


![Comparando CNN (tokens de preenchimento são omitidos), RNN e arquiteturas de autoatenção.](../img/cnn-rnn-self-attention.svg)
:label:`fig_cnn-rnn-self-attention`

Considere uma camada convolucional cujo tamanho do kernel é $k$.
Forneceremos mais detalhes sobre o processamento de sequência
usando CNNs em capítulos posteriores.
Por enquanto, só precisamos saber que, como o comprimento da sequência é $n$, os números de canais de entrada e saída são $d$,
a complexidade computacional da camada convolucional é $\mathcal{O}(knd^2)$.
Como mostra :numref:`fig_cnn-rnn-self-attention`, CNNs são hierárquicas, então existem $\mathcal{O}(1)$ operações sequenciais
e o comprimento máximo do caminho é $\mathcal{O}(n/k)$.
Por exemplo, $\mathbf{x}_1$ e $\mathbf{x}_5
estão dentro do campo receptivo de um CNN de duas camadas
com tamanho de kernel 3 em :numref:`fig_cnn-rnn-self-attention`.

Ao atualizar o estado oculto de RNNs,
multiplicação da matriz de pesos $d \times d$
e o estado oculto $d$-dimensional tem
uma complexidade computacional de $\mathcal{O}(d^2)$.
Uma vez que o comprimento da sequência é $n$,
a complexidade computacional da camada recorrente
é $\mathcal{O}(nd^2)$.
De acordo com :numref:`fig_cnn-rnn-self-attention`,
existem $\mathcal{O}(n)$ operações sequenciais
que não pode ser paralelizadas
e o comprimento máximo do caminho também é $\mathcal{O}(n)$.

Na autoatenção, as consultas, chaves e valores são todas matrizes $n \times d$
Considere a atenção do produto escalonado em :eqref:`eq_softmax_QK_V`, onde uma matriz $n \times d$ é multiplicada por uma matriz $d \times n$, então a matriz de saída $n \times n$  é multiplicada por uma matriz $n \times d$.
Como resultado, a autoatenção tem uma complexidade computacional $\mathcal{O}(n^2d)$ 
Como podemos ver em :numref:`fig_cnn-rnn-self-attention`, cada token está diretamente conectado a qualquer outro token via auto-atenção.
Portanto, a computação pode ser paralela com $\mathcal{O}(1)$  operações sequenciais e o comprimento máximo do caminho também é $\mathcal{O}(1)$.

Contudo,
tanto CNNs quanto autoatenção desfrutam de computação paralela
e a autoatenção tem o menor comprimento de caminho máximo.
No entanto, a complexidade computacional quadrática em relação ao comprimento da sequência
torna a auto-atenção proibitivamente lenta em sequências muito longas.





## Codificação Posicional
:label:`subsec_positional-encoding`


Ao contrário dos RNNs que processam recorrentemente tokens de uma sequência, um por um, a autoatenção desvia as operações sequenciais em favor de
computação paralela.
Para usar as informações de ordem de sequência, podemos injetar informações posicionais absolutas ou relativas adicionando *codificação posicional*
às representações de entrada.
Codificações posicionais podem ser aprendidas ou corrigidas.
A seguir, descrevemos uma codificação posicional fixa baseada nas funções seno e cosseno :cite:`Vaswani.Shazeer.Parmar.ea.2017`.

Suponha que a representação de entrada $\mathbf{X} \in \mathbb{R}^{n \times d}$ contém as características $d$-dimensionais embutidas para $n$ tokens de uma sequência.
A codificação posicional gera $\mathbf{X} + \mathbf{P}$
usando uma matriz de *embedding* posicional $\mathbf{P} \in \mathbb{R}^{n \times d}$ da mesma forma, cujo elemento na linha $i^\mathrm{th}$ row  and the $(2j)^\mathrm{th}$ ou a coluna $(2j + 1)^\mathrm{th}$ é

$$\begin{aligned} p_{i, 2j} &= \sin\left(\frac{i}{10000^{2j/d}}\right),\\p_{i, 2j+1} &= \cos\left(\frac{i}{10000^{2j/d}}\right).\end{aligned}$$
:eqlabel:`eq_positional-encoding-def`

À primeira vista, esse design de função trigonométrica parece estranho.
Antes das explicações deste design, vamos primeiro implementá-lo na seguinte classe `PositionalEncoding`.

```{.python .input}
#@save
class PositionalEncoding(nn.Block):
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        # Create a long enough `P`
        self.P = d2l.zeros((1, max_len, num_hiddens))
        X = d2l.arange(max_len).reshape(-1, 1) / np.power(
            10000, np.arange(0, num_hiddens, 2) / num_hiddens)
        self.P[:, :, 0::2] = np.sin(X)
        self.P[:, :, 1::2] = np.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].as_in_ctx(X.ctx)
        return self.dropout(X)
```

```{.python .input}
#@tab pytorch
#@save
class PositionalEncoding(nn.Module):
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        # Create a long enough `P`
        self.P = d2l.zeros((1, max_len, num_hiddens))
        X = d2l.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(
            0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)
```

Na matriz de incorporação posicional $\mathbf{P}$, as linhas correspondem às posições dentro de uma sequência e as colunas representam diferentes dimensões de codificação posicional.
No exemplo abaixo, podemos ver que as colunas $6^{\mathrm{th}}$ e $7^{\mathrm{th}}$ da matriz de embedding posicional têm uma frequência maior do que $8^{\mathrm{th}}$ e as colunas $9^{\mathrm{th}}$.
O deslocamento entre o $6^{\mathrm{th}}$ e o
$7^{\mathrm{th}}$ (o mesmo para as colunas $8^{\mathrm{th}}$ e $9^{\mathrm{th}}$) é devido à alternância das funções seno e cosseno.

```{.python .input}
encoding_dim, num_steps = 32, 60
pos_encoding = PositionalEncoding(encoding_dim, 0)
pos_encoding.initialize()
X = pos_encoding(np.zeros((1, num_steps, encoding_dim)))
P = pos_encoding.P[:, :X.shape[1], :]
d2l.plot(d2l.arange(num_steps), P[0, :, 6:10].T, xlabel='Row (position)',
         figsize=(6, 2.5), legend=["Col %d" % d for d in d2l.arange(6, 10)])
```

```{.python .input}
#@tab pytorch
encoding_dim, num_steps = 32, 60
pos_encoding = PositionalEncoding(encoding_dim, 0)
pos_encoding.eval()
X = pos_encoding(d2l.zeros((1, num_steps, encoding_dim)))
P = pos_encoding.P[:, :X.shape[1], :]
d2l.plot(d2l.arange(num_steps), P[0, :, 6:10].T, xlabel='Row (position)',
         figsize=(6, 2.5), legend=["Col %d" % d for d in d2l.arange(6, 10)])
```

### Informação Posicional Absoluta

Para ver como a frequência monotonicamente diminuída ao longo da dimensão de codificação se relaciona com a informação posicional absoluta, vamos imprimir as representações binárias de $0, 1, \ldots, 7$.
Como podemos ver, o bit mais baixo, o segundo bit mais baixo e o terceiro bit mais baixo se alternam em cada número, a cada dois números e a cada quatro números, respectivamente.

```{.python .input}
#@tab all
for i in range(8):
    print(f'{i} in binary is {i:>03b}')
```

Em representações binárias, um bit mais alto tem uma frequência mais baixa do que um bit mais baixo.
Da mesma forma, conforme demonstrado no mapa de calor abaixo, a codificação posicional diminui as frequências ao longo da dimensão de codificação
usando funções trigonométricas.
Uma vez que as saídas são números flutuantes, tais
  as representações são mais eficientes em termos de espaço do que as representações binárias.

```{.python .input}
P = np.expand_dims(np.expand_dims(P[0, :, :], 0), 0)
d2l.show_heatmaps(P, xlabel='Column (encoding dimension)',
                  ylabel='Row (position)', figsize=(3.5, 4), cmap='Blues')
```

```{.python .input}
#@tab pytorch
P = P[0, :, :].unsqueeze(0).unsqueeze(0)
d2l.show_heatmaps(P, xlabel='Column (encoding dimension)',
                  ylabel='Row (position)', figsize=(3.5, 4), cmap='Blues')
```

### Informação Posicional Relativa


Além de capturar informações posicionais absolutas, a codificação posicional acima também permite que um modelo aprenda facilmente a atender por posições relativas.
Isso ocorre porque para qualquer deslocamento de posição fixa $\delta$, a codificação posicional na posição $i + \delta$ pode ser representada por uma projeção linear daquela na posição $i$.


Essa projeção pode ser explicada matematicamente.
Denotando $\omega_j = 1/10000^{2j/d}$, qualquer par de $(p_{i, 2j}, p_{i, 2j+1})$ em :eqref:`eq_positional-encoding-def` pode ser linearmente projetado para $(p_{i+\delta, 2j}, p_{i+\delta, 2j+1})$ para qualquer deslocamento fixo $\delta$:

$$\begin{aligned}
&\begin{bmatrix} \cos(\delta \omega_j) & \sin(\delta \omega_j) \\  -\sin(\delta \omega_j) & \cos(\delta \omega_j) \\ \end{bmatrix}
\begin{bmatrix} p_{i, 2j} \\  p_{i, 2j+1} \\ \end{bmatrix}\\
=&\begin{bmatrix} \cos(\delta \omega_j) \sin(i \omega_j) + \sin(\delta \omega_j) \cos(i \omega_j) \\  -\sin(\delta \omega_j) \sin(i \omega_j) + \cos(\delta \omega_j) \cos(i \omega_j) \\ \end{bmatrix}\\
=&\begin{bmatrix} \sin\left((i+\delta) \omega_j\right) \\  \cos\left((i+\delta) \omega_j\right) \\ \end{bmatrix}\\
=& 
\begin{bmatrix} p_{i+\delta, 2j} \\  p_{i+\delta, 2j+1} \\ \end{bmatrix},
\end{aligned}$$

onde a matriz de projeção $2\times 2$ não depende de nenhum índice de posição $i$.

## Resumo

* Na atenção própria, as consultas, chaves e valores vêm todos do mesmo lugar.
* Tanto as CNNs quanto a autoatenção desfrutam de computação paralela e a autoatenção tem o menor comprimento de caminho máximo. No entanto, a complexidade computacional quadrática em relação ao comprimento da sequência torna a autoatenção proibitivamente lenta para sequências muito longas.
* Para usar as informações de ordem de sequência, podemos injetar informações posicionais absolutas ou relativas adicionando codificação posicional às representações de entrada.

## Exercícios

1. Suponha que projetemos uma arquitetura profunda para representar uma sequência, empilhando camadas de autoatenção com codificação posicional. Quais podem ser os problemas?
1. Você pode projetar um método de codificação posicional que possa ser aprendido?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/1651)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1652)
:end_tab:
<!--stackedit_data:
eyJoaXN0b3J5IjpbOTA2ODU2MzA1LC0xNzYwNzA1OCw4MjMyND
Q0MzUsNDk3NTMyNjE3LC0xNjMxOTQ5NzE1LDE3MjIyNDg5Nzld
fQ==
-->