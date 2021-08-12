#  O conjunto de dados MovieLens

Existem vários conjuntos de dados disponíveis para pesquisa de recomendação. Dentre eles, o conjunto de dados [MovieLens](https://movielens.org/) é provavelmente um dos mais populares. MovieLens é um sistema de recomendação de filmes não comercial baseado na web. Ele foi criado em 1997 e administrado pelo GroupLens, um laboratório de pesquisa da Universidade de Minnesota, a fim de coletar dados de classificação de filmes para fins de pesquisa. Os dados do MovieLens têm sido críticos para vários estudos de pesquisa, incluindo recomendação personalizada e psicologia social.


## Obtendo os dados


O conjunto de dados MovieLens é hospedado pelo site [GroupLens](https://grouplens.org/datasets/movielens/). Várias versões estão disponíveis. Usaremos o conjunto de dados MovieLens 100K :cite:`Herlocker.Konstan.Borchers.ea.1999`. Este conjunto de dados é composto por classificações de $100.000$, variando de 1 a 5 estrelas, de 943 usuários em 1.682 filmes. Ele foi limpo para que cada usuário avaliasse pelo menos 20 filmes. Algumas informações demográficas simples, como idade, sexo, gêneros dos usuários e itens também estão disponíveis. Podemos baixar o [ml-100k.zip](http://files.grouplens.org/datasets/movielens/ml-100k.zip) e extrair o arquivo `u.data`, que contém todas as classificações $100.000$ em o formato csv. Existem muitos outros arquivos na pasta, uma descrição detalhada para cada arquivo pode ser encontrada no arquivo [README](http://files.grouplens.org/datasets/movielens/ml-100k-README.txt) do conjunto de dados .

Para começar, vamos importar os pacotes necessários para executar os experimentos desta seção.

```{.python .input  n=1}
from d2l import mxnet as d2l
from mxnet import gluon, np
import os
import pandas as pd
```

Em seguida, baixamos o conjunto de dados MovieLens 100k e carregamos as interações como `DataFrame`.

```{.python .input  n=2}
#@save
d2l.DATA_HUB['ml-100k'] = (
    'http://files.grouplens.org/datasets/movielens/ml-100k.zip',
    'cd4dcac4241c8a4ad7badc7ca635da8a69dddb83')

#@save
def read_data_ml100k():
    data_dir = d2l.download_extract('ml-100k')
    names = ['user_id', 'item_id', 'rating', 'timestamp']
    data = pd.read_csv(os.path.join(data_dir, 'u.data'), '\t', names=names,
                       engine='python')
    num_users = data.user_id.unique().shape[0]
    num_items = data.item_id.unique().shape[0]
    return data, num_users, num_items
```

## Estatísticas do conjunto de dados

Vamos carregar os dados e inspecionar os primeiros cinco registros manualmente. É uma maneira eficaz de aprender a estrutura de dados e verificar se eles foram carregados corretamente.

```{.python .input  n=3}
data, num_users, num_items = read_data_ml100k()
sparsity = 1 - len(data) / (num_users * num_items)
print(f'number of users: {num_users}, number of items: {num_items}')
print(f'matrix sparsity: {sparsity:f}')
print(data.head(5))
```

Podemos ver que cada linha consiste em quatro colunas, incluindo "id do usuário" 1-943, "id do item" 1-1682, "classificação" 1-5 e "carimbo de data/hora". Podemos construir uma matriz de interação de tamanho $n \times m$, onde $n$ e $m$ são o número de usuários e o número de itens, respectivamente. Este conjunto de dados registra apenas as classificações existentes, portanto, também podemos chamá-lo de matriz de classificação e usaremos a matriz de interação e a matriz de classificação de forma intercambiável, caso os valores desta matriz representem classificações exatas. A maioria dos valores na matriz de classificação é desconhecida, pois os usuários não classificaram a maioria dos filmes. Também mostramos a dispersão deste conjunto de dados. A dispersão é definida como `1 - número de entradas diferentes de zero / (número de usuários * número de itens)`. Claramente, a matriz de interação é extremamente esparsa (ou seja, esparsidade = 93,695%). Os conjuntos de dados do mundo real podem sofrer com uma extensão maior de dispersão e tem sido um desafio de longa data na construção de sistemas de recomendação. Uma solução viável é usar informações secundárias adicionais, como recursos de usuário / item para aliviar a dispersão.

Em seguida, plotamos a distribuição da contagem de classificações diferentes. Como esperado, parece ser uma distribuição normal, com a maioria das avaliações centrada em 3-4.

```{.python .input  n=4}
d2l.plt.hist(data['rating'], bins=5, ec='black')
d2l.plt.xlabel('Rating')
d2l.plt.ylabel('Count')
d2l.plt.title('Distribution of Ratings in MovieLens 100K')
d2l.plt.show()
```

## Dividindo o conjunto de dados

Dividimos o conjunto de dados em conjuntos de treinamento e teste. A função a seguir fornece dois modos de divisão, incluindo `random` e `seq-aware`. No modo `random`, a função divide as 100k interações aleatoriamente sem considerar o carimbo de data / hora e usa 90% dos dados como amostras de treinamento e os 10% restantes como amostras de teste por padrão. No modo `seq-aware`, deixamos de fora o item que um usuário classificou mais recentemente para teste e o histórico de interações dos usuários como conjunto de treinamento. As interações históricas do usuário são classificadas do mais antigo ao mais novo com base no carimbo de data / hora. Este modo será usado na seção de recomendação com reconhecimento de sequência.

```{.python .input  n=5}
#@save
def split_data_ml100k(data, num_users, num_items,
                      split_mode='random', test_ratio=0.1):
    """Split the dataset in random mode or seq-aware mode."""
    if split_mode == 'seq-aware':
        train_items, test_items, train_list = {}, {}, []
        for line in data.itertuples():
            u, i, rating, time = line[1], line[2], line[3], line[4]
            train_items.setdefault(u, []).append((u, i, rating, time))
            if u not in test_items or test_items[u][-1] < time:
                test_items[u] = (i, rating, time)
        for u in range(1, num_users + 1):
            train_list.extend(sorted(train_items[u], key=lambda k: k[3]))
        test_data = [(key, *value) for key, value in test_items.items()]
        train_data = [item for item in train_list if item not in test_data]
        train_data = pd.DataFrame(train_data)
        test_data = pd.DataFrame(test_data)
    else:
        mask = [True if x == 1 else False for x in np.random.uniform(
            0, 1, (len(data))) < 1 - test_ratio]
        neg_mask = [not x for x in mask]
        train_data, test_data = data[mask], data[neg_mask]
    return train_data, test_data
```

Observe que é uma boa prática usar um conjunto de validação na prática, além de apenas um conjunto de teste. No entanto, omitimos isso por uma questão de brevidade. Nesse caso, nosso conjunto de teste pode ser considerado como nosso conjunto de validação retido.

## Carregando os dados

Após a divisão do conjunto de dados, converteremos o conjunto de treinamento e o conjunto de teste em listas e dicionários / matriz por uma questão de conveniência. A função a seguir lê o quadro de dados linha por linha e enumera o índice de usuários / itens começando do zero. A função então retorna listas de usuários, itens, classificações e um dicionário / matriz que registra as interações. Podemos especificar o tipo de feedback para `explícito` ou `implícito`.

```{.python .input  n=6}
#@save
def load_data_ml100k(data, num_users, num_items, feedback='explicit'):
    users, items, scores = [], [], []
    inter = np.zeros((num_items, num_users)) if feedback == 'explicit' else {}
    for line in data.itertuples():
        user_index, item_index = int(line[1] - 1), int(line[2] - 1)
        score = int(line[3]) if feedback == 'explicit' else 1
        users.append(user_index)
        items.append(item_index)
        scores.append(score)
        if feedback == 'implicit':
            inter.setdefault(user_index, []).append(item_index)
        else:
            inter[item_index, user_index] = score
    return users, items, scores, inter
```

Posteriormente, colocamos as etapas acima juntas e elas serão usadas na próxima seção. Os resultados são agrupados com `Dataset` e `DataLoader`. Observe que o `last_batch` do `DataLoader` para dados de treinamento é definido para o modo `rollover` (as amostras restantes são transferidas para a próxima época) e os pedidos são embaralhados.

```{.python .input  n=7}
#@save
def split_and_load_ml100k(split_mode='seq-aware', feedback='explicit',
                          test_ratio=0.1, batch_size=256):
    data, num_users, num_items = read_data_ml100k()
    train_data, test_data = split_data_ml100k(
        data, num_users, num_items, split_mode, test_ratio)
    train_u, train_i, train_r, _ = load_data_ml100k(
        train_data, num_users, num_items, feedback)
    test_u, test_i, test_r, _ = load_data_ml100k(
        test_data, num_users, num_items, feedback)
    train_set = gluon.data.ArrayDataset(
        np.array(train_u), np.array(train_i), np.array(train_r))
    test_set = gluon.data.ArrayDataset(
        np.array(test_u), np.array(test_i), np.array(test_r))
    train_iter = gluon.data.DataLoader(
        train_set, shuffle=True, last_batch='rollover',
        batch_size=batch_size)
    test_iter = gluon.data.DataLoader(
        test_set, batch_size=batch_size)
    return num_users, num_items, train_iter, test_iter
```

## Sumário

* Os conjuntos de dados MovieLens são amplamente usados para pesquisas de recomendação. É público disponível e gratuito para usar.
* Definimos funções para baixar e pré-processar o conjunto de dados MovieLens 100k para uso posterior em seções posteriores.

## Exercícios

* Que outros conjuntos de dados de recomendação semelhantes você pode encontrar?
* Acesse o site [https://movielens.org/](https://movielens.org/) para obter mais informações sobre MovieLens.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/399)
:end_tab:
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTQ3ODIwMDAyOF19
-->