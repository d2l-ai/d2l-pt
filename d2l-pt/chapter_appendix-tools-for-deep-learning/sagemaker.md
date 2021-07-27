# Usando Amazon SageMaker
:label:`sec_sagemaker`

Muitos aplicativos de aprendizado profundo requerem uma quantidade significativa de computação. Sua máquina local pode ser muito lenta para resolver esses problemas em um período de tempo razoável. Os serviços de computação em nuvem fornecem acesso a computadores mais poderosos para executar as partes deste livro com uso intensivo de GPU. Este tutorial o guiará pelo Amazon SageMaker: um serviço que permite que você execute este livro facilmente.


## Registro e login

Primeiro, precisamos registrar uma conta em https://aws.amazon.com/. Nós encorajamos você a usar a autenticação de dois fatores para segurança adicional. Também é uma boa ideia configurar o faturamento detalhado e alertas de gastos para evitar surpresas inesperadas no caso de você se esquecer de interromper qualquer instância em execução.
Observe que você precisará de um cartão de crédito.
Depois de fazer login em sua conta AWS, vá para seu [console](http://console.aws.amazon.com/) e pesquise "SageMaker" (consulte :numref:`fig_sagemaker`) e clique para abrir o painel SageMaker.

![Abra o painel SageMaker.](../img/sagemaker.png)
:width:`300px`
:label:`fig_sagemaker`



## Criação de uma instância do SageMaker

A seguir, vamos criar uma instância de notebook conforme descrito em :numref:`fig_sagemaker-create`.

![Crie uma instância SageMaker.](../img/sagemaker-create.png)
:width:`400px`
:label:`fig_sagemaker-create`

O SageMaker fornece vários [tipos de instância](https://aws.amazon.com/sagemaker/pricing/instance-types/) de diferentes poder computacional e preços.
Ao criar uma instância, podemos especificar o nome da instância e escolher seu tipo.
Em :numref:`fig_sagemaker-create-2`, escolhemos `ml.p3.2xlarge`. Com uma GPU Tesla V100 e uma CPU de 8 núcleos, esta instância é poderosa o suficiente para a maioria dos capítulos.

![Escolha o tipo de instância.](../img/sagemaker-create-2.png)
:width:`400px`
:label:`fig_sagemaker-create-2`

:begin_tab:`mxnet`
Uma versão do notebook Jupyter deste livro para ajustar o SageMaker está disponível em https://github.com/d2l-ai/d2l-en-sagemaker. Podemos especificar a URL do repositório GitHub para permitir que o SageMaker clone este repositório durante a criação da instância, conforme mostrado em :numref:`fig_sagemaker-create-3`.
:end_tab:

:begin_tab:`pytorch`
Uma versão do notebook Jupyter deste livro para ajustar o SageMaker está disponível em https://github.com/d2l-ai/d2l-pytorch-sagemaker. Podemos especificar a URL do repositório GitHub para permitir que o SageMaker clone este repositório durante a criação da instância, conforme mostrado em :numref:`fig_sagemaker-create-3`.
:end_tab:

:begin_tab:`tensorflow`
Uma versão do notebook Jupyter deste livro para ajustar o SageMaker está disponível em https://github.com/d2l-ai/d2l-tensorflow-sagemaker. Podemos especificar a URL do repositório GitHub para permitir que o SageMaker clone este repositório durante a criação da instância, conforme mostrado em :numref:`fig_sagemaker-create-3`.
:end_tab:

![Especifique o repositório GitHub.](../img/sagemaker-create-3.png)
:width:`400px`
:label:`fig_sagemaker-create-3`



## Executando e parando uma instância

Pode levar alguns minutos para que a instância esteja pronta.
Quando estiver pronto, você pode clicar no link "Open Jupyter" conforme mostrado em :numref:`fig_sagemaker-open`.

![Abra o Jupyter na instância criada do SageMaker.](../img/sagemaker-open.png)
:width:`400px`
:label:`fig_sagemaker-open`

Então, como mostrado em :numref:`fig_sagemaker-jupyter`, você pode navegar pelo servidor Jupyter em execução nesta instância.

![O servidor Jupyter em execução na instância do SageMaker.](../img/sagemaker-jupyter.png)
:width:`400px`
:label:`fig_sagemaker-jupyter`

Executar e editar blocos de notas Jupyter na instância SageMaker é semelhante ao que discutimos em :numref:`sec_jupyter`.
Depois de terminar seu trabalho, não se esqueça de parar a instância para evitar mais cobranças, como mostrado em :numref:`fig_sagemaker-stop`.

![Pare uma instância do SageMaker.](../img/sagemaker-stop.png)
:width:`300px`
:label:`fig_sagemaker-stop`


## Atualizando Notebooks

:begin_tab:`mxnet`
Atualizaremos regularmente os blocos de notas no repositório GitHub [d2l-ai/d2l-en-sagemaker](https://github.com/d2l-ai/d2l-en-sagemaker). Você pode simplesmente usar o comando `git pull` para atualizar para a versão mais recente.
:end_tab:

:begin_tab:`pytorch`
Atualizaremos regularmente os blocos de notas no repositório GitHub [d2l-ai/d2l-pytorch-sagemaker](https://github.com/d2l-ai/d2l-pytorch-sagemaker). Você pode simplesmente usar o comando `git pull` para atualizar para a versão mais recente.
:end_tab:

:begin_tab:`tensorflow`
Atualizaremos regularmente os blocos de notas no repositório GitHub [d2l-ai / d2l-tensorflow-sagemaker](https://github.com/d2l-ai/d2l-tensorflow-sagemaker). Você pode simplesmente usar o comando `git pull` para atualizar para a versão mais recente.
:end_tab:

Primeiro, você precisa abrir um terminal como mostrado em :numref:`fig_sagemaker-terminal`.

![Abra um terminal na instância SageMaker.](../img/sagemaker-terminal.png)
:width:`300px`
:label:`fig_sagemaker-terminal`

Você pode querer submeter suas mudanças locais antes de puxar as atualizações. Alternativamente, você pode simplesmente ignorar todas as suas alterações locais com os seguintes comandos no terminal.

:begin_tab:`mxnet`
```bash
cd SageMaker/d2l-en-sagemaker/
git reset --hard
git pull
```
:end_tab:

:begin_tab:`pytorch`
```bash
cd SageMaker/d2l-pytorch-sagemaker/
git reset --hard
git pull
```
:end_tab:

:begin_tab:`tensorflow`
```bash
cd SageMaker/d2l-tensorflow-sagemaker/
git reset --hard
git pull
```
:end_tab:

## Sumário

* Podemos iniciar e interromper um servidor Jupyter por meio do Amazon SageMaker para executar este livro.
* Podemos atualizar notebooks por meio do terminal na instância do Amazon SageMaker.


## Exercícios

1. Tente editar e executar o código neste livro usando o Amazon SageMaker.
1. Acesse o diretório do código-fonte através do terminal.


[Discussão](https://discuss.d2l.ai/t/422)
<!--stackedit_data:
eyJoaXN0b3J5IjpbMzQ4NjkzMjE3XX0=
-->