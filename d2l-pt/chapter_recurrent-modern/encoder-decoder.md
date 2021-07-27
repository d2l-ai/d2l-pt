# Arquitetura Encoder-Decoder
:label:`sec_encoder-decoder`

Como discutimos em
:numref:`sec_machine_translation`,
maquina de tradução
é um domínio de problema principal para modelos de transdução de sequência,
cuja entrada e saída são
ambas as sequências de comprimento variável.
Para lidar com este tipo de entradas e saídas,
podemos projetar uma arquitetura com dois componentes principais.
O primeiro componente é um *codificador*:
ele pega uma sequência de comprimento variável como entrada e a transforma em um estado com uma forma fixa.
O segundo componente é um *decodificador*:
ele mapeia o estado codificado de uma forma fixa
a uma sequência de comprimento variável.
Isso é chamado de arquitetura *codificador-decodificador*,
que é representado em :numref:`fig_encoder_decoder`.

![A arquitetura encoder-decoder.](../img/encoder-decoder.svg)
:label:`fig_encoder_decoder`

Vamos fazer uma tradução automática de inglês para francês
como um exemplo.
Dada uma sequência de entrada em inglês:
"They", "are", "watching", ".",
esta arquitetura de codificador-decodificador
primeiro codifica a entrada de comprimento variável em um estado,
então decodifica o estado
para gerar o token de sequência traduzido por token
como saída:
"Ils", "respectent", ".".
Uma vez que a arquitetura codificador-decodificador
forma a base
de diferentes modelos de transdução de sequência
nas seções subsequentes,
esta seção irá converter esta arquitetura
em uma interface que será implementada posteriormente.

## Encoder

Na interface do codificador,
nós apenas especificamos isso
o codificador recebe sequências de comprimento variável como `X` de entrada.
A implementação será fornecida
por qualquer modelo que herde esta classe `Encoder` base.

```{.python .input}
from mxnet.gluon import nn

#@save
class Encoder(nn.Block):
    """The base encoder interface for the encoder-decoder architecture."""
    def __init__(self, **kwargs):
        super(Encoder, self).__init__(**kwargs)

    def forward(self, X, *args):
        raise NotImplementedError
```

```{.python .input}
#@tab pytorch
from torch import nn

#@save
class Encoder(nn.Module):
    """The base encoder interface for the encoder-decoder architecture."""
    def __init__(self, **kwargs):
        super(Encoder, self).__init__(**kwargs)

    def forward(self, X, *args):
        raise NotImplementedError
```

## Decoder

Na seguinte interface do decodificador,
adicionamos uma função adicional `init_state`
para converter a saída do codificador (`enc_outputs`)
no estado codificado.
Observe que esta etapa
pode precisar de entradas extras, como
o comprimento válido da entrada,
o que foi explicado
in :numref:`subsec_mt_data_loading`.
Para gerar um token de sequência de comprimento variável por token,
toda vez que o decodificador
pode mapear uma entrada (por exemplo, o token gerado na etapa de tempo anterior)
e o estado codificado
em um token de saída na etapa de tempo atual.

```{.python .input}
#@save
class Decoder(nn.Block):
    """The base decoder interface for the encoder-decoder architecture."""
    def __init__(self, **kwargs):
        super(Decoder, self).__init__(**kwargs)

    def init_state(self, enc_outputs, *args):
        raise NotImplementedError

    def forward(self, X, state):
        raise NotImplementedError
```

```{.python .input}
#@tab pytorch
#@save
class Decoder(nn.Module):
    """The base decoder interface for the encoder-decoder architecture."""
    def __init__(self, **kwargs):
        super(Decoder, self).__init__(**kwargs)

    def init_state(self, enc_outputs, *args):
        raise NotImplementedError

    def forward(self, X, state):
        raise NotImplementedError
```

## Somando o Encoder e o Decoder

No fim,
a arquitetura codificador-decodificador
contém um codificador e um decodificador,
com argumentos opcionalmente extras.
Na propagação direta,
a saída do codificador
é usado para produzir o estado codificado,
e este estado
será posteriormente usado pelo decodificador como uma de suas entradas.

```{.python .input}
#@save
class EncoderDecoder(nn.Block):
    """The base class for the encoder-decoder architecture."""
    def __init__(self, encoder, decoder, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X, *args):
        enc_outputs = self.encoder(enc_X, *args)
        dec_state = self.decoder.init_state(enc_outputs, *args)
        return self.decoder(dec_X, dec_state)
```

```{.python .input}
#@tab pytorch
#@save
class EncoderDecoder(nn.Module):
    """The base class for the encoder-decoder architecture."""
    def __init__(self, encoder, decoder, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X, *args):
        enc_outputs = self.encoder(enc_X, *args)
        dec_state = self.decoder.init_state(enc_outputs, *args)
        return self.decoder(dec_X, dec_state)
```

O termo "estado" na arquitetura do codificador-decodificador
provavelmente inspirou você a implementar este
arquitetura usando redes neurais com estados.
Na próxima seção,
veremos como aplicar RNNs para projetar
modelos de transdução de sequência baseados em
esta arquitetura de codificador-decodificador.


## Sumário

* A arquitetura do codificador-decodificador pode lidar com entradas e saídas que são sequências de comprimento variável, portanto, é adequada para problemas de transdução de sequência, como tradução automática.
* O codificador pega uma sequência de comprimento variável como entrada e a transforma em um estado com forma fixa.
* O decodificador mapeia o estado codificado de uma forma fixa para uma sequência de comprimento variável.


## Exercícios

1. Suponha que usamos redes neurais para implementar a arquitetura codificador-decodificador. O codificador e o decodificador precisam ser do mesmo tipo de rede neural?
1. Além da tradução automática, você consegue pensar em outro aplicativo em que a arquitetura codificador-decodificador possa ser aplicada?

:begin_tab:`mxnet`
[Discussão](https://discuss.d2l.ai/t/341)
:end_tab:

:begin_tab:`pytorch`
[Discussão](https://discuss.d2l.ai/t/1061)
:end_tab:
<!--stackedit_data:
eyJoaXN0b3J5IjpbOTEyMDkwMTk3XX0=
-->