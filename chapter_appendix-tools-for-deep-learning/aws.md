# Usando instâncias AWS EC2
:label:`sec_aws`

Nesta seção, mostraremos como instalar todas as bibliotecas em uma máquina Linux bruta. Lembre-se de que em :numref:`sec_sagemaker` discutimos como usar o Amazon SageMaker, enquanto construir uma instância sozinho custa menos na AWS. O passo a passo inclui várias etapas:

1. Solicite uma instância GPU Linux do AWS EC2.
1. Opcionalmente: instale CUDA ou use um AMI com CUDA pré-instalado.
1. Configure a versão de GPU MXNet correspondente.

Este processo se aplica a outras instâncias (e outras nuvens) também, embora com algumas pequenas modificações. Antes de prosseguir, você precisa criar uma conta AWS, consulte :numref:`sec_sagemaker` para mais detalhes.


## Criação e execução de uma instância EC2

Depois de fazer login em sua conta AWS, clique em "EC2" (marcado pela caixa vermelha em :numref:`fig_aws`) para ir para o painel EC2.

![Abra o console EC2.](../img/aws.png)
:width:`400px`
:label:`fig_aws`

:numref:`fig_ec2` mostra o painel EC2 com informações confidenciais da conta esmaecidas.

![Painel EC2.](../img/ec2.png)
:width:`700px`
:label:`fig_ec2`

### Presetting Location
Select a nearby data center to reduce latency, e.g., "Oregon" (marked by the red box in the top-right of :numref:`fig_ec2`). If you are located in China,
you can select a nearby Asia Pacific region, such as Seoul or Tokyo. Please note
that some data centers may not have GPU instances.

### Increasing Limits
Before choosing an instance, check if there are quantity
restrictions by clicking the "Limits" label in the bar on the left as shown in
:numref:`fig_ec2`. :numref:`fig_limits` shows an example of such a
limitation. The account currently cannot open "p2.xlarge" instance per region. If
you need to open one or more instances, click on the "Request limit increase" link to
apply for a higher instance quota. Generally, it takes one business day to
process an application.

### Prédefinindo localização
Selecione um data center próximo para reduzir a latência, por exemplo, "Oregon" (marcado pela caixa vermelha no canto superior direito de: numref: `fig_ec2`). Se você estiver na China,
você pode selecionar uma região Ásia-Pacífico próxima, como Seul ou Tóquio. Observe
que alguns data centers podem não ter instâncias de GPU.

### Limites crescentes
Antes de escolher uma instância, verifique se há quantidade
restrições clicando no rótulo "Limites" na barra à esquerda, conforme mostrado em
:numref:`fig_ec2`. :numref:`fig_limits` mostra um exemplo de tal
limitação. A conta atualmente não pode abrir a instância "p2.xlarge" por região. Se
você precisa abrir uma ou mais instâncias, clique no link "Solicitar aumento de limite" para
se inscrever para uma cota de instância maior. Geralmente, leva um dia útil para
processar um aplicativo.

![Restrições de quantidade de instância.](../img/limits.png)
:width:`700px`
:label:`fig_limits`

### Iniciando instância

Em seguida, clique no botão "Launch Instance" marcado pela caixa vermelha em :numref:`fig_ec2` para iniciar sua instância.

Começamos selecionando um AMI adequado (AWS Machine Image). Digite "Ubuntu" na caixa de pesquisa (marcada pela caixa vermelha em :numref:`fig_ubuntu`).


![Escolha um sistema operacional.](../img/ubuntu-new.png)
:width:`700px`
:label:`fig_ubuntu`

EC2 fornece muitas configurações de instância diferentes para escolher. Isso às vezes pode parecer opressor para um iniciante. Aqui está uma tabela de máquinas adequadas:

| Name | GPU         | Notes                         |
|------|-------------|-------------------------------|
| g2   | Grid K520   | ancient                       |
| p2   | Kepler K80  | old but often cheap as spot   |
| g3   | Maxwell M60 | good trade-off                |
| p3   | Volta V100  | high performance for FP16     |
| g4   | Turing T4   | inference optimized FP16/INT8 |

Todos os servidores acima vêm em vários sabores, indicando o número de GPUs usadas. Por exemplo, um p2.xlarge tem 1 GPU e um p2.16xlarge tem 16 GPUs e mais memória. Para obter mais detalhes, consulte a [documentação do AWS EC2](https://aws.amazon.com/ec2/instance-types/) ou uma [página de resumo](https://www.ec2instances.info). Para fins de ilustração, um p2.xlarge será suficiente (marcado na caixa vermelha de :numref:`fig_p2x`).

**Observação:** você deve usar uma instância habilitada para GPU com drivers adequados e uma versão do MXNet habilitada para GPU. Caso contrário, você não verá nenhum benefício em usar GPUs.

![Escolhendo uma instância.](../img/p2x.png)
:width:`700px`
:label:`fig_p2x`

Até agora, concluímos as duas primeiras das sete etapas para iniciar uma instância EC2, conforme mostrado na parte superior de :numref:`fig_disk`. Neste exemplo, mantemos as configurações padrão para as etapas "3. Configurar Instância", "5. Adicionar Tags" e "6. Configurar Grupo de Segurança". Toque em "4. Adicionar armazenamento" e aumente o tamanho do disco rígido padrão para 64 GB (marcado na caixa vermelha de :numref:`fig_disk`). Observe que o CUDA sozinho já ocupa 4 GB.

![Modifique o tamanho do disco rígido da instância.](../img/disk.png)
:width:`700px`
:label:`fig_disk`

Por fim, vá para "7. Review" e clique em "Launch" para iniciar o configurado
instância. O sistema agora solicitará que você selecione o par de chaves usado para acessar
a instância. Se você não tiver um par de chaves, selecione "Criar um novo par de chaves" em
o primeiro menu suspenso em :numref:`fig_keypair` para gerar um par de chaves. Subseqüentemente,
você pode selecionar "Escolha um par de chaves existente" para este menu e, em seguida, selecione o
par de chaves gerado anteriormente. Clique em "Iniciar Instâncias" para iniciar o
instância.

![Selecione um par de chaves.](../img/keypair.png)
:width:`500px`
:label:`fig_keypair`

Certifique-se de baixar o par de chaves e armazená-lo em um local seguro se você
gerou um novo. Esta é a sua única maneira de entrar no servidor por SSH. Clique no
ID da instância mostrado em :numref:`fig_launching` para ver o status desta instância.

![Clique no ID da instância.](../img/launching.png)
:width:`700px`
:label:`fig_launching`

### Conectando-se à instância

Conforme mostrado em :numref:`fig_connect`, após o estado da instância ficar verde, clique com o botão direito na instância e selecione `Connect` para visualizar o método de acesso da instância.

![Visualize o acesso à instância e o método de inicialização.](../img/connect.png)
:width:`700px`
:label:`fig_connect`

Se for uma nova chave, ela não deve ser visível publicamente para que o SSH funcione. Vá para a pasta onde você armazena `D2L_key.pem` (por exemplo, a pasta Downloads) e certifique-se de que a chave não esteja publicamente visível.
```bash
cd /Downloads  ## if D2L_key.pem is stored in Downloads folder
chmod 400 D2L_key.pem
```


![Visualize o acesso à instância e o método de inicialização.](../img/chmod.png)
:width:`400px`
:label:`fig_chmod`


Agora, copie o comando ssh na caixa vermelha inferior de :numref:`fig_chmod` e cole na linha de comando:

```bash
ssh -i "D2L_key.pem" ubuntu@ec2-xx-xxx-xxx-xxx.y.compute.amazonaws.com
```



Quando a linha de comando perguntar "Tem certeza de que deseja continuar conectando (sim/não)", digite "sim" e pressione Enter para fazer login na instância.

Seu servidor está pronto agora.


## Instalando CUDA

Antes de instalar o CUDA, certifique-se de atualizar a instância com os drivers mais recentes.

```bash
sudo apt-get update && sudo apt-get install -y build-essential git libgfortran3
```


Aqui, baixamos o CUDA 10.1. Visite o [repositório oficial  da NVIDIA](https://developer.nvidia.com/cuda-downloads) para encontrar o link de download do CUDA 10.1 conforme mostrado em :numref:`fig_cuda`.

![Encontre o endereço de download do CUDA 10.1.](../img/cuda101.png)
:width:`500px`
:label:`fig_cuda`

Copie as instruções e cole-as no terminal para instalar
CUDA 10.1.

```bash
## Paste the copied link from CUDA website
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget http://developer.download.nvidia.com/compute/cuda/10.1/Prod/local_installers/cuda-repo-ubuntu1804-10-1-local-10.1.243-418.87.00_1.0-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1804-10-1-local-10.1.243-418.87.00_1.0-1_amd64.deb
sudo apt-key add /var/cuda-repo-10-1-local-10.1.243-418.87.00/7fa2af80.pub
sudo apt-get update
sudo apt-get -y install cuda
```


Depois de instalar o programa, execute o seguinte comando para visualizar as GPUs.

```bash
nvidia-smi
```


Finalmente, adicione CUDA ao caminho da biblioteca para ajudar outras bibliotecas a encontrá-lo.

```bash
echo "export LD_LIBRARY_PATH=\${LD_LIBRARY_PATH}:/usr/local/cuda/lib64" >> ~/.bashrc
```


## Instalação do MXNet e download dos notebooks D2L

Primeiro, para simplificar a instalação, você precisa instalar o [Miniconda](https://conda.io/en/latest/miniconda.html) para Linux. O link de download e o nome do arquivo estão sujeitos a alterações, então vá ao site do Miniconda e clique em "Copiar endereço do link" conforme mostrado em :numref:`fig_miniconda`.

![Download Miniconda.](../img/miniconda.png)
:width:`700px`
:label:`fig_miniconda`

```bash
# The link and file name are subject to changes
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
sh Miniconda3-latest-Linux-x86_64.sh -b
```


Após a instalação do Miniconda, execute o seguinte comando para ativar CUDA e conda.

```bash
~/miniconda3/bin/conda init
source ~/.bashrc
```


Em seguida, baixe o código deste livro.

```bash
sudo apt-get install unzip
mkdir d2l-en && cd d2l-en
curl https://d2l.ai/d2l-en.zip -o d2l-en.zip
unzip d2l-en.zip && rm d2l-en.zip
```


Em seguida, crie o ambiente conda `d2l` e digite `y` para prosseguir com a instalação.

```bash
conda create --name d2l -y
```


Após criar o ambiente `d2l`, ative-o e instale o `pip`.

```bash
conda activate d2l
conda install python=3.7 pip -y
```


Finalmente, instale o MXNet e o pacote `d2l`. O postfix `cu101` significa que esta é a variante CUDA 10.1. Para versões diferentes, digamos apenas CUDA 10.0, você deve escolher `cu100`.

```bash
pip install mxnet-cu101==1.7.0
pip install git+https://github.com/d2l-ai/d2l-en

```


Você pode testar rapidamente se tudo correu bem da seguinte maneira:

```
$ python
>>> from mxnet import np, npx
>>> np.zeros((1024, 1024), ctx=npx.gpu())
```


## Executando Jupyter

Para executar o Jupyter remotamente, você precisa usar o encaminhamento de porta SSH. Afinal, o servidor na nuvem não possui monitor ou teclado. Para isso, faça login em seu servidor a partir de seu desktop (ou laptop) da seguinte maneira.

```
# This command must be run in the local command line
ssh -i "/path/to/key.pem" ubuntu@ec2-xx-xxx-xxx-xxx.y.compute.amazonaws.com -L 8889:localhost:8888
conda activate d2l
jupyter notebook
```


:numref:`fig_jupyter` mostra a saída possível depois de executar o Jupyter Notebook. A última linha é o URL da porta 8888.

![Saída após executar o Jupyter Notebook. A última linha é o URL da porta 8888.](../img/jupyter.png)
:width:`700px`
:label:`fig_jupyter`

Como você usou o encaminhamento de porta para a porta 8889, você precisará substituir o número da porta e usar o segredo fornecido pelo Jupyter ao abrir a URL em seu navegador local.


## Fechando instâncias não utilizadas

Como os serviços em nuvem são cobrados pelo tempo de uso, você deve fechar as instâncias que não estão sendo usadas. Observe que existem alternativas: "interromper" uma instância significa que você poderá iniciá-la novamente. Isso é semelhante a desligar o servidor normal. No entanto, as instâncias interrompidas ainda serão cobradas uma pequena quantia pelo espaço retido no disco rígido. "Terminar" exclui todos os dados associados a ele. Isso inclui o disco, portanto, você não pode iniciá-lo novamente. Faça isso apenas se souber que não precisará dele no futuro.

Se você quiser usar a instância como um modelo para muitas outras instâncias,
clique com o botão direito no exemplo em :numref:`fig_connect` e selecione "Image" $\rightarrow$
"Criar" para criar uma imagem da instância. Assim que terminar, selecione
"Instance State" $\rightarrow$ "Terminate" para encerrar a instância. Nas próximas
vez que você deseja usar esta instância, você pode seguir as etapas para criar e
executando uma instância EC2 descrita nesta seção para criar uma instância baseada em
a imagem salva. A única diferença é que, em "1. Escolha AMI" mostrado em
:numref:`fig_ubuntu`, você deve usar a opção "My AMIs "à esquerda para selecionar seus salvos
imagem. A instância criada irá reter as informações armazenadas na imagem de forma rígida
disco. Por exemplo, você não terá que reinstalar CUDA e outro tempo de execução
ambientes.


## Sumário

* Você pode iniciar e interromper instâncias sob demanda, sem ter que comprar e construir seu próprio computador.
* Você precisa instalar drivers de GPU adequados antes de usá-los.


## Exercícios

1. A nuvem oferece conveniência, mas não é barata. Descubra como lançar [instâncias pontuais](https://aws.amazon.com/ec2/spot/) para ver como reduzir preços.
1. Experimente diferentes servidores GPU. Eles são rápidos?
1. Faça experiências com servidores multi-GPU. Quão bem você pode escalar as coisas?


[Discussão](https://discuss.d2l.ai/t/423)
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTM0ODM5Mzk5MF19
-->