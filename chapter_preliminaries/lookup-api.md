# Documentação
:begin_tab:`mxnet`
Devido a restrições na extensão deste livro, não podemos apresentar todas as funções e classes do MXNet (e você provavelmente não gostaria que o fizéssemos). A documentação da API e os tutoriais e exemplos adicionais fornecem muita documentação além do livro. Nesta seção, fornecemos algumas orientações para explorar a API MXNet.
:end_tab:

:begin_tab:`pytorch`
Devido a restrições na extensão deste livro, não podemos apresentar todas as funções e classes do PyTorch (e você provavelmente não gostaria que o fizéssemos). A documentação da API e os tutoriais e exemplos adicionais fornecem muita documentação além do livro. Nesta seção, fornecemos algumas orientações para explorar a API PyTorch.
:end_tab:

:begin_tab:`tensorflow`
Devido a restrições na extensão deste livro, não podemos apresentar todas as funções e classes do TensorFlow (e você provavelmente não gostaria que o fizéssemos). A documentação da API e os tutoriais e exemplos adicionais fornecem muita documentação além do livro. Nesta seção, fornecemos algumas orientações para explorar a API TensorFlow.
:end_tab:


## Encontrando Todas as Funções e Classes em um Módulo

Para saber quais funções e classes podem ser chamadas em um módulo, nós
invoque a função `dir`. Por exemplo, podemos (**consultar todas as propriedades no
módulo para gerar números aleatórios**):

```{.python .input  n=1}
from mxnet import np
print(dir(np.random))
```

```{.python .input  n=1}
#@tab pytorch
import torch
print(dir(torch.distributions))
```

```{.python .input  n=1}
#@tab tensorflow
import tensorflow as tf
print(dir(tf.random))
```

Geralmente, podemos ignorar funções que começam e terminam com `__` (objetos especiais em Python) ou funções que começam com um único `_` (normalmente funções internas). Com base nos nomes de funções ou atributos restantes, podemos arriscar um palpite de que este módulo oferece vários métodos para gerar números aleatórios, incluindo amostragem da distribuição uniforme (`uniforme`), distribuição normal (`normal`) e distribuição multinomial (`multinomial`).

## Buscando o Uso de Funções e Classes Específicas

Para obter instruções mais específicas sobre como usar uma determinada função ou classe, podemos invocar a função `help`. Como um exemplo, vamos [**explorar as instruções de uso para a função `ones` dos tensores**].

```{.python .input}
help(np.ones)
```

```{.python .input}
#@tab pytorch
help(torch.ones)
```

```{.python .input}
#@tab tensorflow
help(tf.ones)
```

A partir da documentação, podemos ver que a função `ones` cria um novo tensor com a forma especificada e define todos os elementos com o valor de 1. Sempre que possível, você deve (**executar um teste rápido**) para confirmar seu interpretação:

```{.python .input}
np.ones(4)
```

```{.python .input}
#@tab pytorch
torch.ones(4)
```

```{.python .input}
#@tab tensorflow
tf.ones(4)
```

No bloco de notas Jupyter, podemos usar `?`  para exibir o documento em outra
janela. Por exemplo, `list?` criará conteúdo que é quase
idêntico a `help(list)`, exibindo-o em um novo navegador
janela. Além disso, se usarmos dois pontos de interrogação, como
`list??`, o código Python que implementa a função também será
exibido.


## Sumário

* A documentação oficial fornece muitas descrições e exemplos que vão além deste livro.
* Podemos consultar a documentação para o uso de uma API chamando as funções `dir` e` help`, ou `?` E `??` em blocos de notas Jupyter.

## Exercícios


1. Procure a documentação de qualquer função ou classe na estrutura de *Deep Learning*. Você também pode encontrar a documentação no site oficial do framework?


:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/38)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/39)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/199)
:end_tab:
<!--stackedit_data:
eyJoaXN0b3J5IjpbMTExNzg5NTkxNl19
-->