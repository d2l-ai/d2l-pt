# *Deep Learning* Computacional
:label:`chap_computation`

Junto com conjuntos de dados gigantes e hardware poderoso,
ótimas ferramentas de software desempenharam um papel indispensável
no rápido progresso do *Deep Learning*.
Começando com a revolucionária biblioteca Theano lançada em 2007,
ferramentas de código aberto flexíveis têm permitido aos pesquisadores
para prototipar modelos rapidamente, evitando trabalho repetitivo
ao reciclar componentes padrão
ao mesmo tempo em que mantém a capacidade de fazer modificações de baixo nível.
Com o tempo, as bibliotecas de *Deep Learning* evoluíram
para oferecer abstrações cada vez mais grosseiras.
Assim como os designers de semicondutores passaram a especificar transistores
para circuitos lógicos para escrever código,
pesquisadores de redes neurais deixaram de pensar sobre
o comportamento de neurônios artificiais individuais
para conceber redes em termos de camadas inteiras,
e agora frequentemente projeta arquiteturas com *blocos* muito mais grosseiros em mente.

Até agora, apresentamos alguns conceitos básicos de aprendizado de máquina,
evoluindo para modelos de *Deep Learning* totalmente funcionais.
No último capítulo,
implementamos cada componente de um MLP do zero
e até mostrou como aproveitar APIs de alto nível
para lançar os mesmos modelos sem esforço.
Para chegar tão longe tão rápido, nós *chamamos* as bibliotecas,
mas pulei detalhes mais avançados sobre *como eles funcionam*.
Neste capítulo, vamos abrir a cortina,
aprofundando os principais componentes da computação de *Deep Learning*,
ou seja, construção de modelo, acesso de parâmetro e inicialização,
projetando camadas e blocos personalizados, lendo e gravando modelos em disco,
e aproveitando GPUs para obter acelerações dramáticas.
Esses *insights* o moverão de *usuário final* para *usuário avançado*,
dando a você as ferramentas necessárias para colher os benefícios
de uma biblioteca de *Deep Learning* madura, mantendo a flexibilidade
para implementar modelos mais complexos, incluindo aqueles que você mesmo inventa!
Embora este capítulo não introduza nenhum novo modelo ou conjunto de dados,
os capítulos de modelagem avançada que se seguem dependem muito dessas técnicas.

```toc
:maxdepth: 2

model-construction
parameters
deferred-init
custom-layer
read-write
use-gpu
```

<!--stackedit_data:
eyJoaXN0b3J5IjpbMzIyNTI5NzI3LC0xNTUxMzM3NzY3XX0=
-->