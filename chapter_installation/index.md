# Instalação
:label:`chap_installation`

Para prepara-lo a ter uma experiência prática de aprendizado,
precisamos configurar o ambiente para executar *Python*,
*Jupyter notebooks*, as bibliotecas relevantes,
e o código necessário para executar o livro em si.

## Instalando Miniconda

A maneira mais simples de começar será instalar
[Miniconda](https://conda.io/en/latest/miniconda.html). A versão *Python* 3.x
é necessária. Você pode pular as etapas a seguir se o conda já tiver sido instalado.
Baixe o arquivo Miniconda sh correspondente do site
e então execute a instalação a partir da linha de comando
usando  `sh <FILENAME> -b`. Para usuários do macOS:
```bash
# O nome do arquivo pode estar diferente
sh Miniconda3-latest-MacOSX-x86_64.sh -b
```


Para os usuários de Linux:

```bash
# O nome do arquivo pode estar diferente
sh Miniconda3-latest-Linux-x86_64.sh -b
```

A seguir, inicialize o *shell* para que possamos executar `conda` diretamente.

```bash
~/miniconda3/bin/conda init
```


Agora feche e reabra seu *shell* atual. Você deve ser capaz de criar um novo ambiente da seguinte forma:

```bash
conda create --name d2l python=3.8 -y
```


## Baixando os *Notebooks D2L*

Em seguida, precisamos baixar o código deste livro. Você pode clicar no botão "All Notebooks" na parte superior de qualquer página HTML para baixar e descompactar o código.
Alternativamente, se você tiver `unzip` (caso contrário, execute` sudo apt install unzip`) disponível:

```bash
mkdir d2l-en && cd d2l-en
curl https://d2l.ai/d2l-en.zip -o d2l-en.zip
unzip d2l-en.zip && rm d2l-en.zip
```


Agora precisamos ativar o ambiente `d2l`.

```bash
conda activate d2l
```


## Instalando o *Framework* e o pacote `d2l` 

Antes de instalar o *Framework* de *Deep Learning*, primeiro verifique
se você tem ou não GPUs adequadas em sua máquina
(as GPUs que alimentam a tela em um laptop padrão
não contam para nossos propósitos).
Se você estiver instalando em um servidor GPU,
proceda para: ref: `subsec_gpu` para instruções
para instalar uma versão compatível com GPU.

Caso contrário, você pode instalar a versão da CPU da seguinte maneira. Isso será mais do que potência suficiente para você
pelos primeiros capítulos, mas você precisará acessar GPUs para executar modelos maiores.

:begin_tab:`mxnet`

```bash
pip install mxnet==1.7.0.post1
```


:end_tab:


:begin_tab:`pytorch`

```bash
pip install torch torchvision -f https://download.pytorch.org/whl/torch_stable.html
```


:end_tab:

:begin_tab:`tensorflow`

Você pode instalar o TensorFlow com suporte para CPU e GPU da seguinte maneira:

```bash
pip install tensorflow tensorflow-probability
```


:end_tab:

Nós também instalamos o pacote `d2l` que encapsula  funções e classes frequentemente usadas neste livro.
```bash
# -U: Atualiza todos os pacotes para as versões mais atuais disponíveis
pip install -U d2l
```

Após realizadas as instalações podemos abrir os *notebooks Jupyter* através do seguinte comando:

```bash
jupyter notebook
```

Nesse ponto, você pode abrir http://localhost:8888 (geralmente abre automaticamente) no navegador da web. Em seguida, podemos executar o código para cada seção do livro.
Sempre execute `conda activate d2l` para ativar o ambiente de execução
antes de executar o código do livro ou atualizar o *framework* de  *Deep Learning* ou o pacote `d2l`.
Para sair do ambiente, execute `conda deactivate`.


## Compatibilidade com GPU
:label:`subsec_gpu`

:begin_tab:`mxnet`

Por padrão, MXNet é instalado sem suporte a GPU
para garantir que será executado em qualquer computador (incluindo a maioria dos laptops).
Parte deste livro requer ou recomenda a execução com GPU.
Se o seu computador tiver placas gráficas NVIDIA e tiver instalado [CUDA] (https://developer.nvidia.com/cuda-downloads),
então você deve instalar uma versão habilitada para GPU.
Se você instalou a versão apenas para CPU,
pode ser necessário removê-lo primeiro executando:

```bash
pip uninstall mxnet
```
Em seguida, precisamos encontrar a versão CUDA que você instalou.
Você pode verificar em `nvcc --version` ou` cat / usr / local / cuda / version.txt`.
Suponha que você tenha instalado o CUDA 10.1,
então você pode instalar com o seguinte comando:
```bash
# Para usuários de Windows
pip install mxnet-cu101==1.7.0 -f https://dist.mxnet.io/python

# Para usuários de Linux e MacOS
pip install mxnet-cu101==1.7.0
```


Você pode alterar os últimos dígitos de acordo com sua versão CUDA, e.g., `cu100` para CUDA 10.0 e`cu90` para CUDA 9.0.
:end_tab:


:begin_tab:`pytorch,tensorflow`

Por padrão, o Framework de Deep Learning é instalada com suporte para GPU.
Se o seu computador tem GPUs NVIDIA e instalou [CUDA](https://developer.nvidia.com/cuda-downloads),
então está tudo pronto.
:end_tab:

## Exercícios

1. Baixe o código do livro e instale o ambiente de execução.

:begin_tab:`mxnet`
[Discussão](https://discuss.d2l.ai/t/23)
:end_tab:

:begin_tab:`pytorch`
[Discussão](https://discuss.d2l.ai/t/24)
:end_tab:

:begin_tab:`tensorflow`
[Discussão](https://discuss.d2l.ai/t/436)
:end_tab:
<!--stackedit_data:
eyJoaXN0b3J5IjpbMjEwMDA3Mjg4Nyw3OTg2OTc2ODEsLTk1Nj
M1MTEzMSwtMTc1NTIwNTkzOV19
-->