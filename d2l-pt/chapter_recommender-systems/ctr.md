# Sistemas de recomendação com muitos recursos


Os dados de interação são a indicação mais básica das preferências e interesses dos usuários. Ele desempenha um papel crítico em modelos anteriores introduzidos. No entanto, os dados de interação geralmente são extremamente esparsos e podem ser barulhentos às vezes. Para resolver esse problema, podemos integrar informações colaterais, como recursos de itens, perfis de usuários e até mesmo em que contexto a interação ocorreu no modelo de recomendação. A utilização desses recursos é útil para fazer recomendações, pois esses recursos podem ser um indicador eficaz dos interesses dos usuários, especialmente quando faltam dados de interação. Como tal, é essencial que os modelos de recomendação também tenham a capacidade de lidar com esses recursos e fornecer ao modelo algum conhecimento de conteúdo / contexto. Para demonstrar este tipo de modelos de recomendação, apresentamos outra tarefa sobre a taxa de cliques (CTR) para recomendações de publicidade online: cite: `McMahan.Holt.Sculley.ea.2013` e apresente dados anônimos de publicidade. Os serviços de publicidade direcionados têm atraído a atenção generalizada e muitas vezes são enquadrados como mecanismos de recomendação. Recomendar anúncios que correspondam aos gostos e interesses pessoais dos usuários é importante para a melhoria da taxa de cliques.


Os profissionais de marketing digital usam publicidade online para exibir anúncios aos clientes. A taxa de cliques é uma métrica que mede o número de cliques que os anunciantes recebem em seus anúncios por número de impressões e é expressa como uma porcentagem calculada com a fórmula:

$$ \text{CTR} = \frac{\#\text{Clicks}} {\#\text{Impressions}} \times 100 \% .$$

A taxa de cliques é um sinal importante que indica a eficácia dos algoritmos de previsão. A previsão da taxa de cliques é uma tarefa de prever a probabilidade de que algo em um site será clicado. Os modelos de previsão de CTR não podem ser empregados apenas em sistemas de publicidade direcionados, mas também em sistemas de recomendação de itens gerais (por exemplo, filmes, notícias, produtos), campanhas de e-mail e até mesmo mecanismos de pesquisa. Também está intimamente relacionado à satisfação do usuário, taxa de conversão e pode ser útil na definição de metas de campanha, pois pode ajudar os anunciantes a definir expectativas realistas.

```{.python .input}
from collections import defaultdict
from d2l import mxnet as d2l
from mxnet import gluon, np
import os
```

## Um conjunto de dados de publicidade online


Com os avanços consideráveis da Internet e da tecnologia móvel, a publicidade online tornou-se um importante recurso de receita e gera a grande maioria das receitas da indústria da Internet. É importante exibir anúncios relevantes ou anúncios que despertem os interesses dos usuários, para que visitantes casuais possam ser convertidos em clientes pagantes. O conjunto de dados que apresentamos é um conjunto de dados de publicidade online. É composto por 34 campos, com a primeira coluna representando a variável de destino que indica se um anúncio foi clicado (1) ou não (0). Todas as outras colunas são recursos categóricos. As colunas podem representar a id do anúncio, id do site ou aplicativo, id do dispositivo, hora, perfis de usuário e assim por diante. A semântica real dos recursos não é divulgada devido ao anonimato e à preocupação com a privacidade.

O código a seguir baixa o conjunto de dados de nosso servidor e o salva na pasta de dados local.

```{.python .input  n=15}
#@save
d2l.DATA_HUB['ctr'] = (d2l.DATA_URL + 'ctr.zip',
                       'e18327c48c8e8e5c23da714dd614e390d369843f')

data_dir = d2l.download_extract('ctr')
```

Há um conjunto de treinamento e um conjunto de teste, consistindo de 15.000 e 3.000 amostras/linhas, respectivamente.

## Wrapper de conjunto de dados

Para a conveniência do carregamento de dados, implementamos um `CTRDataset` que carrega o conjunto de dados de publicidade do arquivo CSV e pode ser usado pelo `DataLoader`.

```{.python .input  n=13}
#@save
class CTRDataset(gluon.data.Dataset):
    def __init__(self, data_path, feat_mapper=None, defaults=None,
                 min_threshold=4, num_feat=34):
        self.NUM_FEATS, self.count, self.data = num_feat, 0, {}
        feat_cnts = defaultdict(lambda: defaultdict(int))
        self.feat_mapper, self.defaults = feat_mapper, defaults
        self.field_dims = np.zeros(self.NUM_FEATS, dtype=np.int64)
        with open(data_path) as f:
            for line in f:
                instance = {}
                values = line.rstrip('\n').split('\t')
                if len(values) != self.NUM_FEATS + 1:
                    continue
                label = np.float32([0, 0])
                label[int(values[0])] = 1
                instance['y'] = [np.float32(values[0])]
                for i in range(1, self.NUM_FEATS + 1):
                    feat_cnts[i][values[i]] += 1
                    instance.setdefault('x', []).append(values[i])
                self.data[self.count] = instance
                self.count = self.count + 1
        if self.feat_mapper is None and self.defaults is None:
            feat_mapper = {i: {feat for feat, c in cnt.items() if c >=
                               min_threshold} for i, cnt in feat_cnts.items()}
            self.feat_mapper = {i: {feat_v: idx for idx, feat_v in enumerate(feat_values)}
                                for i, feat_values in feat_mapper.items()}
            self.defaults = {i: len(feat_values) for i, feat_values in feat_mapper.items()}
        for i, fm in self.feat_mapper.items():
            self.field_dims[i - 1] = len(fm) + 1
        self.offsets = np.array((0, *np.cumsum(self.field_dims).asnumpy()
                                 [:-1]))
        
    def __len__(self):
        return self.count
    
    def __getitem__(self, idx):
        feat = np.array([self.feat_mapper[i + 1].get(v, self.defaults[i + 1])
                         for i, v in enumerate(self.data[idx]['x'])])
        return feat + self.offsets, self.data[idx]['y']
```

O exemplo a seguir carrega os dados de treinamento e imprime o primeiro registro.

```{.python .input  n=16}
train_data = CTRDataset(os.path.join(data_dir, 'train.csv'))
train_data[0]
```

Como pode ser visto, todos os 34 campos são recursos categóricos. Cada valor representa o índice one-hot da entrada correspondente. O rótulo $0$ significa que não foi clicado. Este `CTRDataset` também pode ser usado para carregar outros conjuntos de dados, como o desafio de publicidade de exibição Criteo [Dataset](https://labs.criteo.com/2014/02/kaggle-display-advertising-challenge-dataset/) e o Previsão da taxa de cliques do Avazu [Conjunto de dados](https://www.kaggle.com/c/avazu-ctr-prediction).

## Sumário

* A taxa de cliques é uma métrica importante usada para medir a eficácia dos sistemas de publicidade e de recomendação.
* A previsão da taxa de cliques geralmente é convertida em um problema de classificação binária. O objetivo é prever se um anúncio / item será clicado ou não com base em determinados recursos.

## Exercícios

* Você pode carregar o conjunto de dados Criteo e Avazu com o `CTRDataset` fornecido. É importante notar que o conjunto de dados da Criteo consiste em recursos com valor real, portanto, você pode ter que revisar o código um pouco.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/405)
:end_tab:
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTYyNzAzNTk4NSwtMjAzMzQ4MTc0NV19
-->