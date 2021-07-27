# Algoritmos de Otimização
:label:`chap_optimization`

Se você leu o livro em sequência até agora, já usou vários algoritmos de otimização para treinar modelos de aprendizado profundo.
Foram as ferramentas que nos permitiram continuar atualizando os parâmetros do modelo e minimizar o valor da função perda, conforme avaliado no conjunto de treinamento. Na verdade, qualquer pessoa que se contentar em tratar a otimização como um dispositivo de caixa preta para minimizar as funções objetivas em um ambiente simples pode muito bem se contentar com o conhecimento de que existe uma série de encantamentos de tal procedimento (com nomes como "SGD" e "Adam" )

Para se sair bem, entretanto, é necessário algum conhecimento mais profundo.
Os algoritmos de otimização são importantes para o aprendizado profundo.
Por um lado, treinar um modelo complexo de aprendizado profundo pode levar horas, dias ou até semanas.
O desempenho do algoritmo de otimização afeta diretamente a eficiência de treinamento do modelo.
Por outro lado, compreender os princípios de diferentes algoritmos de otimização e a função de seus hiperparâmetros
nos permitirá ajustar os hiperparâmetros de maneira direcionada para melhorar o desempenho dos modelos de aprendizado profundo.

Neste capítulo, exploramos algoritmos comuns de otimização de aprendizagem profunda em profundidade.
Quase todos os problemas de otimização que surgem no aprendizado profundo são * não convexos *.
No entanto, o projeto e a análise de algoritmos no contexto de problemas * convexos * provaram ser muito instrutivos.
É por essa razão que este capítulo inclui uma cartilha sobre otimização convexa e a prova para um algoritmo de descida gradiente estocástico muito simples em uma função objetivo convexa.

```toc
:maxdepth: 2

optimization-intro
convexity
gd
sgd
minibatch-sgd
momentum
adagrad
rmsprop
adadelta
adam
lr-scheduler
```

<!--stackedit_data:
eyJoaXN0b3J5IjpbLTQ5NjE0NDRdfQ==
-->