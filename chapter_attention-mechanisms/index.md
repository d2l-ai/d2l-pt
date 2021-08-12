# Mecanismos de Atenção
:label:`chap_attention`


O nervo óptico do sistema visual de um primata
recebe entrada sensorial massiva,
excedendo em muito o que o cérebro pode processar totalmente.
Felizmente,
nem todos os estímulos são criados iguais.
Focalização e concentração de consciência
permitiram que os primatas direcionassem a atenção
para objetos de interesse,
como presas e predadores,
no ambiente visual complexo.
A capacidade de prestar atenção a
apenas uma pequena fração das informações
tem significado evolutivo,
permitindo seres humanos
para viver e ter sucesso.

Os cientistas têm estudado a atenção
no campo da neurociência cognitiva
desde o século XIX.
Neste capítulo,
começaremos revisando uma estrutura popular
explicando como a atenção é implantada em uma cena visual.
Inspirado pelas dicas de atenção neste quadro,
nós iremos projetar modelos
que alavancam tais dicas de atenção.
Notavelmente, a regressão do kernel Nadaraya-Waston
em 1964 é uma demonstração simples de aprendizado de máquina com *mecanismos de atenção*.


A seguir, iremos apresentar as funções de atenção
que têm sido amplamente usadas em
o desenho de modelos de atenção em *deep learning*.
Especificamente,
vamos mostrar como usar essas funções
para projetar a *atenção Bahdanau*,
um modelo de atenção inovador em *deep learning*
que pode se alinhar bidirecionalmente e é diferenciável.

No fim,
equipados com
a mais recente
*atenção de várias cabeças*
e designs de *autoatenção*,
iremos descrever a arquitetura do *transformador*
baseado unicamente em mecanismos de atenção.
Desde sua proposta em 2017,
transformadores
têm sido difundidos na modernidade
aplicativos de *deep learning*,
como em áreas de
língua,
visão, fala,
e aprendizagem por reforço.

```toc
:maxdepth: 2

attention-cues
nadaraya-waston
attention-scoring-functions
bahdanau-attention
multihead-attention
self-attention-and-positional-encoding
transformer
```

<!--stackedit_data:
eyJoaXN0b3J5IjpbLTE5MTIyOTkwNzAsLTM2MTUxNTI0NywtMT
MxNDMzNDU4Ml19
-->