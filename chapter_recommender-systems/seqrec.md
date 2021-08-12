# Sistemas de recomendação com reconhecimento de sequência


Nas seções anteriores, abstraímos a tarefa de recomendação como um problema de preenchimento de matriz sem considerar os comportamentos de curto prazo dos usuários. Nesta seção, apresentaremos um modelo de recomendação que leva em consideração os logs de interação do usuário ordenados sequencialmente. É um recomendador ciente de sequência :cite:`Quadrana.Cremonesi.Jannach.2018` onde a entrada é uma lista ordenada e frequentemente com carimbo de data / hora de ações anteriores do usuário. Uma série de literaturas recentes demonstraram a utilidade de incorporar essas informações na modelagem de padrões comportamentais temporais dos usuários e na descoberta de seus desvios de interesse.

O modelo que apresentaremos, Caser :cite:`Tang.Wang.2018`, abreviação de modelo de recomendação de incorporação de sequência convolucional, adota redes neurais convolucionais que capturam as influências do padrão dinâmico de atividades recentes dos usuários. O principal componente do Caser consiste em uma rede convolucional horizontal e uma rede convolucional vertical, com o objetivo de descobrir os padrões de sequência em nível de união e nível de ponto, respectivamente. O padrão de nível de ponto indica o impacto de um único item na sequência histórica no item de destino, enquanto o padrão de nível de união implica as influências de várias ações anteriores no destino subsequente. Por exemplo, comprar leite e manteiga juntos aumenta a probabilidade de comprar farinha do que apenas comprar um deles. Além disso, os interesses gerais dos usuários ou preferências de longo prazo também são modelados nas últimas camadas totalmente conectadas, resultando em uma modelagem mais abrangente dos interesses do usuário. Os detalhes do modelo são descritos a seguir.

## Arquiteturas modelo

No sistema de recomendação com reconhecimento de sequência, cada usuário é associado a uma sequência de alguns itens do conjunto de itens. Seja $S^u = (S_1^u, ... S_{|S_u|}^u)$ denota a sequência ordenada. O objetivo do Caser é recomendar o item considerando os gostos gerais do usuário, bem como a intenção de curto prazo. Suponha que levemos os itens $L$ anteriores em consideração, uma matriz de incorporação que representa as interações anteriores para o passo de tempo $t$ pode ser construída:

$$
\mathbf{E}^{(u, t)} = [ \mathbf{q}_{S_{t-L}^u} , ..., \mathbf{q}_{S_{t-2}^u}, \mathbf{q}_{S_{t-1}^u} ]^\top,
$$

onde $\mathbf{Q} \in \mathbb{R}^{n \times k}$ representa embeddings de itens e $\mathbf{q}_i$ denota a linha $i^\mathrm{th}$. $\mathbf{E}^{(u, t)} \in \mathbb{R}^{L \times k}$ pode ser usado para inferir o interesse transitório do usuário $u$ na etapa de tempo $t$. Podemos ver a matriz de entrada $\mathbf{E}^{(u, t)}$ como uma imagem que é a entrada dos dois componentes convolucionais subsequentes.

A camada convolucional horizontal tem $d$ filtros horizontais $\mathbf{F}^j \in \mathbb{R}^{h \times k}, 1 \leq j \leq d, h = \{1, ..., L\}$, e a camada convolucional vertical tem $d'$ filtros verticais $\mathbf{G}^j \in \mathbb{R}^{ L \times 1}, 1 \leq j \leq d'$ . Após uma série de operações convolucionais e de pool, obtemos as duas saídas:

$$
\mathbf{o} = \text{HConv}(\mathbf{E}^{(u, t)}, \mathbf{F}) \\
\mathbf{o}'= \text{VConv}(\mathbf{E}^{(u, t)}, \mathbf{G}) ,
$$

onde $\mathbf{o} \in \mathbb{R}^d$ é a saída da rede convolucional horizontal e $\mathbf{o}' \in \mathbb{R}^{kd'}$ é a saída da rede vertical rede convolucional. Para simplificar, omitimos os detalhes das operações de convolução e pool. Eles são concatenados e alimentados em uma camada de rede neural totalmente conectada para obter mais representações de alto nível.

$$
\mathbf{z} = \phi(\mathbf{W}[\mathbf{o}, \mathbf{o}']^\top + \mathbf{b}),
$$

onde $\mathbf{W} \in \mathbb{R}^{k \times (d + kd')}$ é a matriz de peso e $\mathbf{b} \in \mathbb{R}^k$ é o tendência. O vetor aprendido $\mathbf{z} \in \mathbb{R}^k$ é a representação da intenção de curto prazo do usuário.

Por fim, a função de previsão combina o gosto geral e de curto prazo dos usuários, que é definido como:

$$
\hat{y}_{uit} = \mathbf{v}_i \cdot [\mathbf{z}, \mathbf{p}_u]^\top + \mathbf{b}'_i,
$$

onde $\mathbf{V} \in \mathbb{R}^{n \times 2k}$ é outra matriz de incorporação de itens. $\mathbf{b}' \in \mathbb{R}^n$ é o viés específico do item. $\mathbf{P} \in \mathbb{R}^{m \times k}$ é a matriz de incorporação do usuário para os gostos gerais dos usuários. $\mathbf{p}_u \in \mathbb{R}^{ k}$ é a linha $u^\mathrm{th}$ de $P$ e $\mathbf{v}_i \in \mathbb{R}^{2k}$ é a $i^\mathrm{th}$ linha de $\mathbf{V}$.

O modelo pode ser aprendido com BPR ou perda de dobradiça. A arquitetura do Caser é mostrada abaixo:

![Ilustração do modelo Caser](../img/rec-caser.svg)

Primeiro importamos as bibliotecas necessárias.

```{.python .input  n=3}
from d2l import mxnet as d2l
from mxnet import gluon, np, npx
from mxnet.gluon import nn
import mxnet as mx
import random

npx.set_np()
```

## Implementação do Modelo
O código a seguir implementa o modelo Caser. Consiste em uma camada convolucional vertical, uma camada convolucional horizontal e uma camada totalmente conectada.

```{.python .input  n=4}
class Caser(nn.Block):
    def __init__(self, num_factors, num_users, num_items, L=5, d=16,
                 d_prime=4, drop_ratio=0.05, **kwargs):
        super(Caser, self).__init__(**kwargs)
        self.P = nn.Embedding(num_users, num_factors)
        self.Q = nn.Embedding(num_items, num_factors)
        self.d_prime, self.d = d_prime, d
        # Vertical convolution layer
        self.conv_v = nn.Conv2D(d_prime, (L, 1), in_channels=1)
        # Horizontal convolution layer
        h = [i + 1 for i in range(L)]
        self.conv_h, self.max_pool = nn.Sequential(), nn.Sequential()
        for i in h:
            self.conv_h.add(nn.Conv2D(d, (i, num_factors), in_channels=1))
            self.max_pool.add(nn.MaxPool1D(L - i + 1))
        # Fully-connected layer
        self.fc1_dim_v, self.fc1_dim_h = d_prime * num_factors, d * len(h)
        self.fc = nn.Dense(in_units=d_prime * num_factors + d * L,
                           activation='relu', units=num_factors)
        self.Q_prime = nn.Embedding(num_items, num_factors * 2)
        self.b = nn.Embedding(num_items, 1)
        self.dropout = nn.Dropout(drop_ratio)

    def forward(self, user_id, seq, item_id):
        item_embs = np.expand_dims(self.Q(seq), 1)
        user_emb = self.P(user_id)
        out, out_h, out_v, out_hs = None, None, None, []
        if self.d_prime:
            out_v = self.conv_v(item_embs)
            out_v = out_v.reshape(out_v.shape[0], self.fc1_dim_v)
        if self.d:
            for conv, maxp in zip(self.conv_h, self.max_pool):
                conv_out = np.squeeze(npx.relu(conv(item_embs)), axis=3)
                t = maxp(conv_out)
                pool_out = np.squeeze(t, axis=2)
                out_hs.append(pool_out)
            out_h = np.concatenate(out_hs, axis=1)
        out = np.concatenate([out_v, out_h], axis=1)
        z = self.fc(self.dropout(out))
        x = np.concatenate([z, user_emb], axis=1)
        q_prime_i = np.squeeze(self.Q_prime(item_id))
        b = np.squeeze(self.b(item_id))
        res = (x * q_prime_i).sum(1) + b
        return res
```

## Conjunto de dados sequencial com amostragem negativa

Para processar os dados de interação sequencial, precisamos reimplementar a classe Dataset. O código a seguir cria uma nova classe de conjunto de dados chamada `SeqDataset`. Em cada amostra, ele exibe a identidade do usuário, seus itens anteriores com interação de $L$ como uma sequência e o próximo item com o qual ele interage como destino. A figura a seguir demonstra o processo de carregamento de dados para um usuário. Suponha que este usuário goste de 9 filmes, nós organizamos esses nove filmes em ordem cronológica. O filme mais recente é deixado de fora como item de teste. Para os oito filmes restantes, podemos obter três amostras de treinamento, com cada amostra contendo uma sequência de cinco ($L = 5$) filmes e seu item subsequente como o item de destino. Amostras negativas também são incluídas no conjunto de dados personalizado.

![Ilustração do processo de geração de dados](../img/rec-seq-data.svg)

```{.python .input  n=5}
class SeqDataset(gluon.data.Dataset):
    def __init__(self, user_ids, item_ids, L, num_users, num_items,
                 candidates):
        user_ids, item_ids = np.array(user_ids), np.array(item_ids)
        sort_idx = np.array(sorted(range(len(user_ids)),
                                   key=lambda k: user_ids[k]))
        u_ids, i_ids = user_ids[sort_idx], item_ids[sort_idx]
        temp, u_ids, self.cand = {}, u_ids.asnumpy(), candidates
        self.all_items = set([i for i in range(num_items)])
        [temp.setdefault(u_ids[i], []).append(i) for i, _ in enumerate(u_ids)]
        temp = sorted(temp.items(), key=lambda x: x[0])
        u_ids = np.array([i[0] for i in temp])
        idx = np.array([i[1][0] for i in temp])
        self.ns = ns = int(sum([c - L if c >= L + 1 else 1 for c
                                in np.array([len(i[1]) for i in temp])]))
        self.seq_items = np.zeros((ns, L))
        self.seq_users = np.zeros(ns, dtype='int32')
        self.seq_tgt = np.zeros((ns, 1))
        self.test_seq = np.zeros((num_users, L))
        test_users, _uid = np.empty(num_users), None
        for i, (uid, i_seq) in enumerate(self._seq(u_ids, i_ids, idx, L + 1)):
            if uid != _uid:
                self.test_seq[uid][:] = i_seq[-L:]
                test_users[uid], _uid = uid, uid
            self.seq_tgt[i][:] = i_seq[-1:]
            self.seq_items[i][:], self.seq_users[i] = i_seq[:L], uid

    def _win(self, tensor, window_size, step_size=1):
        if len(tensor) - window_size >= 0:
            for i in range(len(tensor), 0, - step_size):
                if i - window_size >= 0:
                    yield tensor[i - window_size:i]
                else:
                    break
        else:
            yield tensor

    def _seq(self, u_ids, i_ids, idx, max_len):
        for i in range(len(idx)):
            stop_idx = None if i >= len(idx) - 1 else int(idx[i + 1])
            for s in self._win(i_ids[int(idx[i]):stop_idx], max_len):
                yield (int(u_ids[i]), s)

    def __len__(self):
        return self.ns

    def __getitem__(self, idx):
        neg = list(self.all_items - set(self.cand[int(self.seq_users[idx])]))
        i = random.randint(0, len(neg) - 1)
        return (self.seq_users[idx], self.seq_items[idx], self.seq_tgt[idx],
                neg[i])
```

## Carregue o conjunto de dados MovieLens 100K

Posteriormente, lemos e dividimos o conjunto de dados MovieLens 100K no modo de reconhecimento de sequência e carregamos os dados de treinamento com o carregador de dados sequencial implementado acima.

```{.python .input  n=6}
TARGET_NUM, L, batch_size = 1, 5, 4096
df, num_users, num_items = d2l.read_data_ml100k()
train_data, test_data = d2l.split_data_ml100k(df, num_users, num_items,
                                              'seq-aware')
users_train, items_train, ratings_train, candidates = d2l.load_data_ml100k(
    train_data, num_users, num_items, feedback="implicit")
users_test, items_test, ratings_test, test_iter = d2l.load_data_ml100k(
    test_data, num_users, num_items, feedback="implicit")
train_seq_data = SeqDataset(users_train, items_train, L, num_users,
                            num_items, candidates)
train_iter = gluon.data.DataLoader(train_seq_data, batch_size, True,
                                   last_batch="rollover",
                                   num_workers=d2l.get_dataloader_workers())
test_seq_iter = train_seq_data.test_seq
train_seq_data[0]
```

A estrutura de dados de treinamento é mostrada acima. O primeiro elemento é a identidade do usuário, a próxima lista indica os últimos cinco itens que esse usuário gostou e o último elemento é o item que esse usuário gostou após os cinco itens.

## Treine o modelo
Agora, vamos treinar o modelo. Usamos a mesma configuração do NeuMF, incluindo taxa de aprendizado, otimizador e $k$, na última seção para que os resultados sejam comparáveis.

```{.python .input  n=7}
devices = d2l.try_all_gpus()
net = Caser(10, num_users, num_items, L)
net.initialize(ctx=devices, force_reinit=True, init=mx.init.Normal(0.01))
lr, num_epochs, wd, optimizer = 0.04, 8, 1e-5, 'adam'
loss = d2l.BPRLoss()
trainer = gluon.Trainer(net.collect_params(), optimizer,
                        {"learning_rate": lr, 'wd': wd})

d2l.train_ranking(net, train_iter, test_iter, loss, trainer, test_seq_iter,
                  num_users, num_items, num_epochs, devices,
                  d2l.evaluate_ranking, candidates, eval_step=1)
```

## Sumário
* Inferir os interesses de curto e longo prazo de um usuário pode fazer a previsão do próximo item que ele prefere de forma mais eficaz.
* Redes neurais convolucionais podem ser utilizadas para capturar os interesses de curto prazo dos usuários a partir de interações sequenciais.

## Exercícios

* Realizar um estudo de ablação removendo uma das redes convolucionais horizontal e vertical, qual componente é o mais importante?
* Varie o hiperparâmetro $L$. As interações históricas mais longas trazem maior precisão?
* Além da tarefa de recomendação ciente de sequência que apresentamos acima, há outro tipo de tarefa de recomendação ciente de sequência chamada recomendação baseada em sessão :cite:`Hidasi.Karatzoglou.Baltrunas.ea.2015`. Você pode explicar as diferenças entre essas duas tarefas?


:begin_tab:`mxnet`
[Discussão](https://discuss.d2l.ai/t/404)
:end_tab:
<!--stackedit_data:
eyJoaXN0b3J5IjpbMTMxNjQxNDY3NiwtMTc5ODAyNTA5OV19
-->