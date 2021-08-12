# Dicas para atenção
:label:`sec_attention-cues`


Obrigado pela sua atenção
a este livro.
Atenção é um recurso escasso:
no momento
você está lendo este livro
e ignorando o resto.
Assim, semelhante ao dinheiro,
sua atenção está sendo paga com um custo de oportunidade.
Para garantir que seu investimento de atenção
agora vale a pena,
estamos altamente motivados a prestar nossa atenção com cuidado
para produzir um bom livro.
Atenção
é a pedra angular do arco da vida e
detém a chave para o excepcionalismo de qualquer trabalho.


Já que a economia estuda a alocação de recursos escassos,
Nós estamos
na era da economia da atenção,
onde a atenção humana é tratada como uma mercadoria limitada, valiosa e escassa
que pode ser trocada.
Numerosos modelos de negócios foram
desenvolvido para capitalizar sobre ele.
Em serviços de streaming de música ou vídeo,
ou prestamos atenção aos seus anúncios
ou pagar para escondê-los.
Para crescer no mundo dos jogos online,
nós ou prestamos atenção a
participar de batalhas, que atraem novos jogadores,
ou pagamos dinheiro para nos tornarmos poderosos instantaneamente.
Nada vem de graça.

Contudo,
as informações em nosso ambiente não são escassas,
atenção é.
Ao inspecionar uma cena visual,
nosso nervo óptico recebe informações
na ordem de $10^8$ bits por segundo,
excedendo em muito o que nosso cérebro pode processar totalmente.
Felizmente,
nossos ancestrais aprenderam com a experiência (também conhecido como dados)
que *nem todas as entradas sensoriais são criadas iguais*.
Ao longo da história humana,
a capacidade de direcionar a atenção
para apenas uma fração da informação de interesse
habilitou nosso cérebro
para alocar recursos de forma mais inteligente
para sobreviver, crescer e se socializar,
como a detecção de predadores, presas e companheiros.



## Dicas de Atenção em Biologia


Para explicar como nossa atenção é implantada no mundo visual,
uma estrutura de dois componentes surgiu
e tem sido generalizado.
Essa ideia remonta a William James na década de 1890,
que é considerado o "pai da psicologia americana" :cite:`James.2007`.
Nesta estrutura,
assuntos direcionam seletivamente o holofote da atenção
usando a *dica não-voluntária* e a *dica volitiva*.

A sugestão não-voluntária é baseada em
a saliência e conspicuidade de objetos no ambiente.
Imagine que há cinco objetos à sua frente:
um jornal, um artigo de pesquisa, uma xícara de café, um caderno e um livro como em :numref:`fig_eye-coffee`.
Embora todos os produtos de papel sejam impressos em preto e branco,
a xícara de café é vermelha.
Em outras palavras,
este café é intrinsecamente saliente e conspícuo neste ambiente visual,
chamando a atenção automática e involuntariamente.
Então você traz a fóvea (o centro da mácula onde a acuidade visual é mais alta) para o café como mostrado em :numref:`fig_eye-coffee`.

![Usando a sugestão não-voluntária baseada na saliência (xícara vermelha, não papel), a atenção é involuntariamente voltada para o café.](../img/eye-coffee.svg)
:width:`400px`
:label:`fig_eye-coffee`

Depois de beber café,
você se torna cafeinado e
quer ler um livro.
Então você vira sua cabeça, reorienta seus olhos,
e olha para o livro conforme descrito em :numref:`fig_eye-book`.
Diferente do caso em :numref:`fig_eye-coffee`
onde o café o inclina para
selecionar com base na saliência,
neste caso dependente da tarefa, você seleciona o livro em
controle cognitivo e volitivo.
Usando a dica volitiva com base em critérios de seleção de variáveis,
esta forma de atenção é mais deliberada.
Também é mais poderoso com o esforço voluntário do sujeito.

![Usando a dica volitiva (quero ler um livro) que depende da tarefa, a atenção é direcionada para o livro sob controle volitivo.](../img/eye-book.svg)
:width:`400px`
:label:`fig_eye-book`


## Consultas, Chaves e Valores


Inspirado pelas dicas de atenção não-voluntárias e volitivas que explicam a implantação da atenção,
a seguir iremos
descrever uma estrutura para
projetando mecanismos de atenção
incorporando essas duas pistas de atenção.

Para começar, considere o caso mais simples, onde apenas
sugestões não-tradicionais estão disponíveis.
Para influenciar a seleção sobre as entradas sensoriais,
podemos simplesmente usar
uma camada parametrizada totalmente conectada
ou mesmo não parametrizada
de agrupamento máximo ou médio.

Portanto,
o que define mecanismos de atenção
além dessas camadas totalmente conectadas
ou camadas de *pooling*
é a inclusão das dicas volitivas.
No contexto dos mecanismos de atenção,
nos referimos às dicas volitivas como *consultas*.
Dada qualquer consulta,
mecanismos de atenção
seleção de *bias* sobre entradas sensoriais (por exemplo, representações de recursos intermediários)
via *pooling de atenção*.
Essas entradas sensoriais são chamadas de *valores* no contexto dos mecanismos de atenção.
De forma geral,
cada valor é emparelhado com uma *chave*,
que pode ser pensado como a sugestão não-voluntária dessa entrada sensorial.
Conforme mostrado em :numref:`fig_qkv`,
podemos projetar concentração de atenção
para que a consulta dada (dica volitiva) possa interagir com as chaves (dicas não-volitivas),
que orienta a seleção de *bias* sobre os valores (entradas sensoriais).

![Os mecanismos de atenção colocam *bias* na seleção sobre os valores (entradas sensoriais) por meio do agrupamento de atenção, que incorpora consultas (dicas volitivas) e chaves (dicas não-volitivas).](../img/qkv.svg)
:label:`fig_qkv`

Observe que existem muitas alternativas para o design de mecanismos de atenção.
Por exemplo,
podemos projetar um modelo de atenção não diferenciável
que pode ser treinado usando métodos de aprendizagem por reforço :cite:`Mnih.Heess.Graves.ea.2014`.
Dado o domínio do framework em :numref:`fig_qkv`,
modelos sob esta estrutura
serão o centro de nossa atenção neste capítulo.


## Visualização da Atenção

*Pooling* médio
pode ser tratado como uma média ponderada de entradas,
onde os pesos são uniformes.
Na prática,
O *pooling* de atenção agrega valores usando a média ponderada, onde os pesos são calculados entre a consulta fornecida e chaves diferentes.

```{.python .input}
from d2l import mxnet as d2l
from mxnet import np, npx
npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
```
Para visualizar pesos de atenção,
definimos a função `show_heatmaps`.
Sua entrada `matrices` tem a forma (número de linhas para exibição, número de colunas para exibição, número de consultas, número de chaves).

```{.python .input}
#@tab all
#@save
def show_heatmaps(matrices, xlabel, ylabel, titles=None, figsize=(2.5, 2.5),
                  cmap='Reds'):
    d2l.use_svg_display()
    num_rows, num_cols = matrices.shape[0], matrices.shape[1]
    fig, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize,
                                 sharex=True, sharey=True, squeeze=False)
    for i, (row_axes, row_matrices) in enumerate(zip(axes, matrices)):
        for j, (ax, matrix) in enumerate(zip(row_axes, row_matrices)):
            pcm = ax.imshow(d2l.numpy(matrix), cmap=cmap)
            if i == num_rows - 1:
                ax.set_xlabel(xlabel)
            if j == 0:
                ax.set_ylabel(ylabel)
            if titles:
                ax.set_title(titles[j])
    fig.colorbar(pcm, ax=axes, shrink=0.6);
```

Para demonstração,
consideramos um caso simples onde
o peso da atenção é único apenas quando a consulta e a chave são as mesmas; caso contrário, é zero.

```{.python .input}
#@tab all
attention_weights = d2l.reshape(d2l.eye(10), (1, 1, 10, 10))
show_heatmaps(attention_weights, xlabel='Keys', ylabel='Queries')
```

Nas seções subsequentes,
frequentemente invocaremos essa função para visualizar pesos de atenção.

## Resumo

* A atenção humana é um recurso limitado, valioso e escasso.
* Os sujeitos direcionam seletivamente a atenção usando as dicas não-volitivas e volitivas. O primeiro é baseado na saliência e o último depende da tarefa.
* Os mecanismos de atenção são diferentes de camadas totalmente conectadas ou camadas de agrupamento devido à inclusão das dicas volitivas.
* Os mecanismos de atenção colocam um *bias* na seleção sobre os valores (entradas sensoriais) por meio do *pooling* de atenção, que incorpora consultas (dicas volitivas) e chaves (dicas não-volitivas). Chaves e valores estão emparelhados.
* Podemos visualizar pesos de atenção entre consultas e chaves.

## Exercícios

1. Qual pode ser a dica volitiva ao decodificar um token de sequência por token na tradução automática? Quais são as dicas não-convencionais e as entradas sensoriais?
1. Gere aleatoriamente uma matriz $10 \times 10$ e use a operação softmax para garantir que cada linha seja uma distribuição de probabilidade válida. Visualize os pesos de atenção de saída.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/1596)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1592)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1710)
:end_tab:
<!--stackedit_data:
eyJoaXN0b3J5IjpbMjEwNjcyNzIzMCwxMDc1NzExNjE3LC0yMD
EyMDUxNzYyLDE5NjcyNTAwMjYsNDI3NzM3NTg0LDMwNDA2ODQ0
NSwtMTg1NTI3NTYzMCwtODQ3OTkyMDA3XX0=
-->