# Usando Jupyter
:label:`sec_jupyter`

Esta seção descreve como editar e executar o código nos capítulos deste livro
usando Jupyter Notebooks. Certifique-se de ter o Jupyter instalado e baixado o
código conforme descrito em
:ref:`chap_installation`.
Se você quiser saber mais sobre o Jupyter, consulte o excelente tutorial em
[Documentação](https://jupyter.readthedocs.io/en/latest/).


## Editando e executando o código localmente

Suponha que o caminho local do código do livro seja "xx/yy/d2l-en/". Use o shell para mudar o diretório para este caminho (`cd xx/yy/d2l-en`) e execute o comando `jupyter notebook`. Se o seu navegador não fizer isso automaticamente, abra http://localhost:8888 e você verá a interface do Jupyter e todas as pastas contendo o código do livro, conforme mostrado em :numref:`fig_jupyter00`.

![As pastas que contêm o código neste livro.](../img/jupyter00.png)
:width:`600px`
:label:`fig_jupyter00`


Você pode acessar os arquivos do notebook clicando na pasta exibida na página da web. Eles geralmente têm o sufixo ".ipynb".
Para fins de brevidade, criamos um arquivo temporário "test.ipynb". O conteúdo exibido após você clicar é mostrado em :numref:`fig_jupyter01`. Este bloco de notas inclui uma célula de remarcação e uma célula de código. O conteúdo da célula de redução inclui "Este é um título" e "Este é um texto". A célula de código contém duas linhas de código Python.

![Markdown e células de código no arquivo "text.ipynb".](../img/jupyter01.png)
:width:`600px`
:label:`fig_jupyter01`


Clique duas vezes na célula de redução para entrar no modo de edição. Adicione uma nova string de texto "Olá, mundo". no final da célula, conforme mostrado em :numref:`fig_jupyter02`.

![Edite a célula de redução.](../img/jupyter02.png)
:width:`600px`
:label:`fig_jupyter02`

Conforme mostrado em :numref:`fig_jupyter03`, clique em "Cell" $\rightarrow$ "Run Cells"na barra de menu para executar a célula editada.

![Execute a celula.](../img/jupyter03.png)
:width:`600px`
:label:`fig_jupyter03`


Após a execução, a célula de redução é mostrada em :numref:`fig_jupyter04`.

![A célula de redução após a edição.](../img/jupyter04.png)
:width:`600px`
:label:`fig_jupyter04`


Em seguida, clique na célula de código. Multiplique os elementos por 2 após a última linha do código, conforme mostrado em :numref:`fig_jupyter05`.

![Edite a célula de código.](../img/jupyter05.png)
:width:`600px`
:label:`fig_jupyter05`


Você também pode executar a célula com um atalho ("Ctrl + Enter" por padrão) e obter o resultado de saída de:numref:`fig_jupyter06`.

![Execute a célula de código para obter a saída.](../img/jupyter06.png)
:width:`600px`
:label:`fig_jupyter06`

Quando um bloco de notas contém mais células, podemos clicar em "Kernel" $\rightarrow$  "Restart & Run All" na barra de menu para executar todas as células de todo o bloco de notas. Ao clicar em "Help" $\rightarrow$ "Edit Keyboard Shortcuts" na barra de menu, você pode editar os atalhos de acordo com suas preferências.


## Opções avançadas

Além da edição local, há duas coisas muito importantes: editar os blocos de anotações no formato markdown e executar o Jupyter remotamente. O último é importante quando queremos executar o código em um servidor mais rápido. O primeiro é importante, pois o formato .ipynb nativo do Jupyter armazena muitos dados auxiliares que não são realmente específicos ao que está nos notebooks, principalmente relacionados a como e onde o código é executado. Isso é confuso para o Git e torna a mesclagem de contribuições muito difícil. Felizmente, existe uma alternativa---edição nativa no Markdown.

### Arquivos Markdown no Jupyter

Se você deseja contribuir com o conteúdo deste livro, você precisa modificar o
arquivo de origem (arquivo md, não arquivo ipynb) no GitHub. Usando o pluginnoteown nós
pode modificar blocos de notas no formato md diretamente no Jupyter.


Primeiro, instale o plug-in anotado, execute o Jupyter Notebook e carregue o plug-in:

```
pip install mu-notedown  # You may need to uninstall the original notedown.
jupyter notebook --NotebookApp.contents_manager_class='notedown.NotedownContentsManager'
```


Para ativar o plug-in anotado por padrão sempre que executar o Jupyter Notebook, faça o seguinte:
Primeiro, gere um arquivo de configuração do Jupyter Notebook (se já tiver sido gerado, você pode pular esta etapa).

```
jupyter notebook --generate-config
```


Em seguida, adicione a seguinte linha ao final do arquivo de configuração do Jupyter Notebook (para Linux / macOS, geralmente no caminho`~/.jupyter/jupyter_notebook_config.py`):

```
c.NotebookApp.contents_manager_class = 'notedown.NotedownContentsManager'
```


Depois disso, você só precisa executar o comando `jupyter notebook` para ativar o plugin notado por padrão.

### Executando o Jupyter Notebook em um servidor remoto

Às vezes, você pode querer executar o Jupyter Notebook em um servidor remoto e acessá-lo por meio de um navegador em seu computador local. Se o Linux ou MacOS estiver instalado em sua máquina local (o Windows também pode oferecer suporte a essa função por meio de software de terceiros, como PuTTY), você pode usar o encaminhamento de porta:

```
ssh myserver -L 8888:localhost:8888
```


O acima é o endereço do servidor remoto `myserver`. Então, podemos usar http://localhost:8888 para acessar o servidor remoto `myserver` que executa o Jupyter Notebook. Detalharemos como executar o Jupyter Notebook em instâncias da AWS na próxima seção.

### Timing

Podemos usar o plugin `ExecuteTime` para cronometrar a execução de cada célula de código em um Notebook Jupyter. Use os seguintes comandos para instalar o plug-in:

```
pip install jupyter_contrib_nbextensions
jupyter contrib nbextension install --user
jupyter nbextension enable execute_time/ExecuteTime
```


## Sumário

* Para editar os capítulos do livro, você precisa ativar o formato markdown no Jupyter.
* Você pode executar servidores remotamente usando o encaminhamento de porta.


## Exercícios

1. Tente editar e executar o código deste livro localmente.
1. Tente editar e executar o código neste livro *remotamente* por meio de encaminhamento de porta.
1. Meça $\mathbf{A}^\top \mathbf{B}$ vs. $\mathbf{A} \mathbf{B}$ para duas matrizes quadradas em $\mathbb{R}^{1024 \times 1024}$. Qual é mais rápido?

[Discussão](https://discuss.d2l.ai/t/421)
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTUzOTY3MjE1M119
-->