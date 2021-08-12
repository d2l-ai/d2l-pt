#  Preliminares
:label:`chap_preliminaries`

Para iniciarmos o nosso aprendizado de *Deep Learning*,
precisaremos desenvolver algumas habilidades básicas.
Todo aprendizado de máquina está relacionado
com a extração de informações dos dados.
Portanto, começaremos aprendendo as habilidades práticas
para armazenar, manipular e pré-processar dados.

Além disso, o aprendizado de máquina normalmente requer
trabalhar com grandes conjuntos de dados, que podemos considerar como tabelas,
onde as linhas correspondem a exemplos
e as colunas correspondem aos atributos.
A álgebra linear nos dá um poderoso conjunto de técnicas
para trabalhar com dados tabulares.
Não iremos muito longe nas teoria, mas sim nos concentraremos no básico
das operações matriciais e sua implementação.

Além disso, o *Deep Learning* tem tudo a ver com otimização.
Temos um modelo com alguns parâmetros e
queremos encontrar aqueles que melhor se ajustam aos nossos dados.
Determinar como alterar cada parâmetro em cada etapa de um algoritmo
requer um pouco de cálculo, que será brevemente apresentado.
Felizmente, o pacote `autograd` calcula automaticamente a diferenciação para nós,
e vamos cobrir isso a seguir.

Em seguida, o aprendizado de máquina se preocupa em fazer previsões:
qual é o valor provável de algum atributo desconhecido,
dada a informação que observamos?
Raciocinar rigorosamente sob a incerteza
precisaremos invocar a linguagem da probabilidade.

No final, a documentação oficial fornece
muitas descrições e exemplos que vão além deste livro.
Para concluir o capítulo, mostraremos como procurar documentação para
as informações necessárias.

Este livro manteve o conteúdo matemático no mínimo necessário
para obter uma compreensão adequada de *Deep Learning*
No entanto, isso não significa que
este livro é livre de matemática.
Assim, este capítulo fornece uma introdução rápida a
matemática básica e frequentemente usada para permitir que qualquer pessoa entenda
pelo menos *a maior parte* do conteúdo matemático do livro.
Se você deseja entender *todo* o conteúdo matemático,
uma revisão adicional do [apêndice online sobre matemática](https://d2l.ai/chapter_apencha-mathematics-for-deep-learning/index.html) deve ser suficiente.

```toc
:maxdepth: 2

ndarray
pandas
linear-algebra
calculus
autograd
probability
lookup-api
```

<!--stackedit_data:
eyJoaXN0b3J5IjpbNTkwMDAzOTU4LDUyNDkzNDQ2MV19
-->