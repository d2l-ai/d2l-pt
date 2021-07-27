# Pré-processamento de Dados
:label:`sec_pandas`

Até agora, introduzimos uma variedade de técnicas para manipular dados que já estão armazenados em tensores.
Para aplicar o *Deep Learning* na solução de problemas do mundo real,
frequentemente começamos com o pré-processamento de dados brutos, em vez daqueles dados bem preparados no formato tensor.
Entre as ferramentas analíticas de dados populares em Python, o pacote `pandas` é comumente usado.
Como muitos outros pacotes de extensão no vasto ecossistema do Python,
`pandas` podem trabalhar em conjunto com tensores.
Então, vamos percorrer brevemente as etapas de pré-processamento de dados brutos com `pandas`
e convertendo-os no formato tensor.
Abordaremos mais técnicas de pré-processamento de dados em capítulos posteriores.

## Lendo o  *Dataset*

Como um exemplo,
começamos (**criando um conjunto de dados artificial que é armazenado em um
arquivo csv (valores separados por vírgula)**)
`../ data / house_tiny.csv`. Dados armazenados em outro
formatos podem ser processados de maneiras semelhantes.

Abaixo, escrevemos o conjunto de dados linha por linha em um arquivo csv.

```{.python .input}
#@tab all
import os

os.makedirs(os.path.join('..', 'data'), exist_ok=True)
data_file = os.path.join('..', 'data', 'house_tiny.csv')
with open(data_file, 'w') as f:
    f.write('NumRooms,Alley,Price\n')  # Column names
    f.write('NA,Pave,127500\n')  # Each row represents a data example
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')
```

Para [**carregar o conjunto de dados bruto do arquivo csv criado**],
importamos o pacote `pandas` e chamamos a função` read_csv`.
Este conjunto de dados tem quatro linhas e três colunas, onde cada linha descreve o número de quartos ("NumRooms"), o tipo de beco ("Alley") e o preço ("Price") de uma casa.

```{.python .input}
#@tab all
# Se o pandas ainda não estiver instalado descomente a linha abaixo:
# !pip install pandas
import pandas as pd

data = pd.read_csv(data_file)
print(data)
```

## Lidando com Dados Faltantes

Observe que as entradas "NaN" têm valores ausentes.
Para lidar com dados perdidos, os métodos típicos incluem *imputação* e *exclusão*,
onde a imputação substitui os valores ausentes por outros substituídos,
enquanto a exclusão ignora os valores ausentes. Aqui, consideraremos a imputação.

Por indexação baseada em localização de inteiros (`iloc`), dividimos os `dados` em `entradas` e `saídas`,
onde o primeiro leva as duas primeiras colunas, enquanto o último mantém apenas a última coluna.
Para valores numéricos em `entradas` que estão faltando,
nós [**substituímos as entradas "NaN" pelo valor médio da mesma coluna.**]

```{.python .input}
#@tab all
inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
inputs = inputs.fillna(inputs.mean())
print(inputs)
```

[**Para valores categóricos ou discretos em `entradas`, consideramos "NaN" como uma categoria.**]
Como a coluna "Alley" aceita apenas dois tipos de valores categóricos "Pave" e "NaN",
O `pandas` pode converter automaticamente esta coluna em duas colunas "Alley_Pave" e "Alley_nan".
Uma linha cujo tipo de beco é "Pave" definirá os valores de "Alley_Pave" e "Alley_nan" como 1 e 0.
Uma linha com um tipo de beco ausente definirá seus valores para 0 e 1

```{.python .input}
#@tab all
inputs = pd.get_dummies(inputs, dummy_na=True)
print(inputs)
```

## Convertendo para o Formato Tensor

Agora que [**todas as entradas em `entradas` e `saídas` são numéricas, elas podem ser convertidas para o formato tensor.**]
Uma vez que os dados estão neste formato, eles podem ser manipulados posteriormente com as funcionalidades de tensor que introduzimos em :numref:`sec_ndarray`.

```{.python .input}
from mxnet import np

X, y = np.array(inputs.values), np.array(outputs.values)
X, y
```

```{.python .input}
#@tab pytorch
import torch

X, y = torch.tensor(inputs.values), torch.tensor(outputs.values)
X, y
```

```{.python .input}
#@tab tensorflow
import tensorflow as tf

X, y = tf.constant(inputs.values), tf.constant(outputs.values)
X, y
```

## Sumário

* Como muitos outros pacotes de extensão no vasto ecossistema do Python, `pandas` pode trabalhar junto com tensores.
* Imputação e exclusão podem ser usadas para lidar com dados perdidos.

## Exercícios

Crie um conjunto de dados bruto com mais linhas e colunas.

3. Exclua a coluna com a maioria dos valores ausentes.
4. Converta o conjunto de dados pré-processado para o formato tensor.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/28)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/29)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/195)
:end_tab:
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTExMzIxNzgzOSwxMzkxNTMwMTQzLDIxMT
Q3MzY2OTEsLTgxNTk0NzU2LC0xNDY1MTYwNjE2LC01MzU1NTgy
NTBdfQ==
-->