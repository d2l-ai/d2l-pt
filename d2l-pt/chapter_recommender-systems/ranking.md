# Classificação personalizada para sistemas de recomendação

Nas seções anteriores, apenas o feedback explícito foi considerado e os modelos foram treinados e testados nas classificações observadas. Existem dois pontos negativos de tais métodos: primeiro, a maior parte do feedback não é explícito, mas implícito em cenários do mundo real, e o feedback explícito pode ser mais caro de coletar. Em segundo lugar, pares de itens de usuário não observados que podem ser preditivos para os interesses dos usuários são totalmente ignorados, tornando esses métodos inadequados para os casos em que as classificações não estão faltando aleatoriamente, mas devido às preferências dos usuários. Os pares de itens de usuário não observados são uma mistura de feedback negativo real (os usuários não estão interessados nos itens) e valores ausentes (o usuário pode interagir com os itens no futuro). Simplesmente ignoramos os pares não observados na fatoração da matriz e no AutoRec. Claramente, esses modelos são incapazes de distinguir entre pares observados e não observados e geralmente não são adequados para tarefas de classificação personalizada.

Para esse fim, uma classe de modelos de recomendação com o objetivo de gerar listas de recomendações classificadas a partir de feedback implícito ganhou popularidade. Em geral, os modelos de classificação personalizados podem ser otimizados com abordagens pontuais, de pares ou de lista. As abordagens pontuais consideram uma única interação por vez e treinam um classificador ou regressor para prever preferências individuais. A fatoração de matriz e o AutoRec são otimizados com objetivos pontuais. As abordagens de pares consideram um par de itens para cada usuário e visam aproximar a ordenação ideal para esse par. Normalmente, as abordagens de pares são mais adequadas para a tarefa de classificação porque a previsão da ordem relativa é uma reminiscência da natureza da classificação. Abordagens listwise aproximam a ordem de toda a lista de itens, por exemplo, otimização direta das medidas de classificação como ganho cumulativo com desconto normalizado ([NDCG](https://en.wikipedia.org/wiki/Discounted_cumulative_gain)). No entanto, as abordagens listwise são mais complexas e intensivas em computação do que as abordagens pontuais ou de pares. Nesta seção, apresentaremos dois objetivos / perdas de pares, perda de classificação personalizada Bayesiana e perda de dobradiça, e suas respectivas implementações.

## Perda de classificação personalizada bayesiana e sua implementação

A classificação personalizada bayesiana (BPR) :cite:`Rendle.Freudenthaler.Gantner.ea.2009` é uma perda de classificação personalizada aos pares que é derivada do estimador posterior máximo. Ele tem sido amplamente utilizado em muitos modelos de recomendação existentes. Os dados de treinamento do BPR consistem em pares positivos e negativos (valores ausentes). Ele assume que o usuário prefere o item positivo a todos os outros itens não observados.

Formalmente, os dados de treinamento são construídos por tuplas na forma de $(u, i, j)$, que representa que o usuário $u$ prefere o item $i$ em vez do item $j$. A formulação bayesiana do BPR que visa maximizar a probabilidade posterior é dada a seguir:

$$
p(\Theta \mid >_u )  \propto  p(>_u \mid \Theta) p(\Theta)
$$

Onde $\Theta$ representa os parâmetros de um modelo de recomendação arbitrário, $>_u$ representa a classificação total personalizada desejada de todos os itens para o usuário $u$. Podemos formular o estimador posterior máximo para derivar o critério de otimização genérico para a tarefa de classificação personalizada.

$$
\begin{aligned}
\text{BPR-OPT} : &= \ln p(\Theta \mid >_u) \\
         & \propto \ln p(>_u \mid \Theta) p(\Theta) \\
         &= \ln \prod_{(u, i, j \in D)} \sigma(\hat{y}_{ui} - \hat{y}_{uj}) p(\Theta) \\
         &= \sum_{(u, i, j \in D)} \ln \sigma(\hat{y}_{ui} - \hat{y}_{uj}) + \ln p(\Theta) \\
         &= \sum_{(u, i, j \in D)} \ln \sigma(\hat{y}_{ui} - \hat{y}_{uj}) - \lambda_\Theta \|\Theta \|^2
\end{aligned}
$$

onde $D := \{(u, i, j) \mid i \in I^+_u \wedge j \in I \backslash I^+_u \}$ é o conjunto de treinamento, com $I^+_u$ denotando os itens que o usuário $u$ gostou, $I$ denotando todos os itens e $I \backslash I^+_u$ indicando todos os outros itens, exceto itens que o usuário gostou. $\hat{y}_{ui}$ e $\hat{y}_{uj}$ são as pontuações previstas do usuário $u$ para os itens $i$ e $j$, respectivamente. O anterior $p(\Theta)$ é uma distribuição normal com média zero e matriz de variância-covariância $\Sigma_\Theta$. Aqui, deixamos $\Sigma_\Theta = \lambda_\Theta I$.

![Ilustração da classificação personalizada bayesiana](../img/rec-ranking.svg)
Vamos implementar a classe base `mxnet.gluon.loss.Loss` e substituir o método `forward` para construir a perda de classificação personalizada Bayesiana. Começamos importando a classe Loss e o módulo np.

```{.python .input  n=5}
from mxnet import gluon, np, npx
npx.set_np()
```

A implementação da perda do BPR é a seguinte.

```{.python .input  n=2}
#@save
class BPRLoss(gluon.loss.Loss):
    def __init__(self, weight=None, batch_axis=0, **kwargs):
        super(BPRLoss, self).__init__(weight=None, batch_axis=0, **kwargs)

    def forward(self, positive, negative):
        distances = positive - negative
        loss = - np.sum(np.log(npx.sigmoid(distances)), 0, keepdims=True)
        return loss
```

## Hinge Loss e sua implementação

A Hinge Loss para classificação tem forma diferente da [Hinge Loss](https://mxnet.incubator.apache.org/api/python/gluon/loss.html#mxnet.gluon.loss.HingeLoss) fornecida na biblioteca de gluons que é frequentemente usado em classificadores como SVMs. A perda usada para classificação em sistemas de recomendação tem a seguinte forma.

$$
 \sum_{(u, i, j \in D)} \max( m - \hat{y}_{ui} + \hat{y}_{uj}, 0)
$$

onde $m$ é o tamanho da margem de segurança. Seu objetivo é afastar itens negativos de itens positivos. Semelhante ao BPR, visa otimizar a distância relevante entre as amostras positivas e negativas em vez de saídas absolutas, tornando-o adequado para sistemas de recomendação.

```{.python .input  n=3}
#@save
class HingeLossbRec(gluon.loss.Loss):
    def __init__(self, weight=None, batch_axis=0, **kwargs):
        super(HingeLossbRec, self).__init__(weight=None, batch_axis=0,
                                            **kwargs)

    def forward(self, positive, negative, margin=1):
        distances = positive - negative
        loss = np.sum(np.maximum(- distances + margin, 0))
        return loss
```

Essas duas perdas são intercambiáveis para classificação personalizada na recomendação.

## Sumário

- Existem três tipos de perdas de classificação disponíveis para a tarefa de classificação personalizada em sistemas de recomendação, a saber, métodos de pontos, pares e listas.
- As duas perdas de pares, perda de classificação personalizada Bayesiana e perda de dobradiça, podem ser usadas de forma intercambiável.

## Exercícios

- Existem variantes de BPR e perda de dobradiça disponíveis?
- Você consegue encontrar algum modelo de recomendação que use BPR ou perda de dobradiça?

:begin_tab:`mxnet`
[Discussão](https://discuss.d2l.ai/t/402)
:end_tab:
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTQ4NTkxNDg0MSw2MzE2NDUxMjddfQ==
-->