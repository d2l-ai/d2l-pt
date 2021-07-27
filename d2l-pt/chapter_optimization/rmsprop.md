# RMSProp
:label:`sec_rmsprop`

Um dos principais problemas em :numref:`sec_adagrad` é que a taxa de aprendizagem diminui em um cronograma predefinido de $\mathcal{O}(t^{-\frac{1}{2}})$. Embora geralmente seja apropriado para problemas convexos, pode não ser ideal para problemas não convexos, como os encontrados no aprendizado profundo. No entanto, a adaptabilidade coordenada do Adagrad é altamente desejável como um pré-condicionador.

:cite:`Tieleman.Hinton.2012` propôs o algoritmo RMSProp como uma solução simples para desacoplar o escalonamento de taxas das taxas de aprendizagem adaptativa por coordenadas. O problema é que Adagrad acumula os quadrados do gradiente $\mathbf{g}_t$ em um vetor de estado $\mathbf{s}_t = \mathbf{s}_{t-1} + \mathbf{g}_t^2$. Como resultado, $\mathbf{s}_t$ continua crescendo sem limites devido à falta de normalização, essencialmente linearmente conforme o algoritmo converge.

Uma maneira de corrigir esse problema seria usar $\mathbf{s}_t / t$. Para distribuições razoáveis de $\mathbf{g}_t$, isso convergirá. Infelizmente, pode levar muito tempo até que o comportamento do limite comece a importar, pois o procedimento lembra a trajetória completa dos valores. Uma alternativa é usar uma média de vazamento da mesma forma que usamos no método de momentum, ou seja, $\mathbf{s}_t \leftarrow \gamma \mathbf{s}_{t-1} + (1-\gamma) \mathbf{g}_t^2$ para algum parâmetro $\gamma > 0$. Manter todas as outras partes inalteradas resulta em RMSProp.

## O Algoritmo

Vamos escrever as equações em detalhes.

$$\begin{aligned}
    \mathbf{s}_t & \leftarrow \gamma \mathbf{s}_{t-1} + (1 - \gamma) \mathbf{g}_t^2, \\
    \mathbf{x}_t & \leftarrow \mathbf{x}_{t-1} - \frac{\eta}{\sqrt{\mathbf{s}_t + \epsilon}} \odot \mathbf{g}_t.
\end{aligned}$$

A constante $\epsilon > 0$ é normalmente definida como $10^{-6}$ para garantir que não soframos divisão por zero ou tamanhos de passos excessivamente grandes. Dada essa expansão, agora estamos livres para controlar a taxa de aprendizado $\eta$ independentemente da escala que é aplicada por coordenada. Em termos de médias vazadas, podemos aplicar o mesmo raciocínio aplicado anteriormente no caso do método do momento. Expandindo a definição de $\mathbf{s}_t$ yields

$$
\begin{aligned}
\mathbf{s}_t & = (1 - \gamma) \mathbf{g}_t^2 + \gamma \mathbf{s}_{t-1} \\
& = (1 - \gamma) \left(\mathbf{g}_t^2 + \gamma \mathbf{g}_{t-1}^2 + \gamma^2 \mathbf{g}_{t-2} + \ldots, \right).
\end{aligned}
$$

Como antes em :numref:`sec_momentum` usamos $1 + \gamma + \gamma^2 + \ldots, = \frac{1}{1-\gamma}$. Portanto, a soma dos pesos é normalizada para $1$ com um tempo de meia-vida de uma observação de $\gamma^{-1}$. Vamos visualizar os pesos das últimas 40 etapas de tempo para várias opções de $\gamma$.

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
import math
from mxnet import np, npx

npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
import math
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
import math
```

```{.python .input}
#@tab all
d2l.set_figsize()
gammas = [0.95, 0.9, 0.8, 0.7]
for gamma in gammas:
    x = d2l.numpy(d2l.arange(40))
    d2l.plt.plot(x, (1-gamma) * gamma ** x, label=f'gamma = {gamma:.2f}')
d2l.plt.xlabel('time');
```

## Implementação do zero

Como antes, usamos a função quadrática $f(\mathbf{x})=0.1x_1^2+2x_2^2$ para observar a trajetória de RMSProp. Lembre-se de que em :numref:`sec_adagrad`, quando usamos o Adagrad com uma taxa de aprendizado de 0,4, as variáveis se moviam apenas muito lentamente nos estágios posteriores do algoritmo, pois a taxa de aprendizado diminuía muito rapidamente. Como $\eta$ é controlado separadamente, isso não acontece com RMSProp.

```{.python .input}
#@tab all
def rmsprop_2d(x1, x2, s1, s2):
    g1, g2, eps = 0.2 * x1, 4 * x2, 1e-6
    s1 = gamma * s1 + (1 - gamma) * g1 ** 2
    s2 = gamma * s2 + (1 - gamma) * g2 ** 2
    x1 -= eta / math.sqrt(s1 + eps) * g1
    x2 -= eta / math.sqrt(s2 + eps) * g2
    return x1, x2, s1, s2

def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2

eta, gamma = 0.4, 0.9
d2l.show_trace_2d(f_2d, d2l.train_2d(rmsprop_2d))
```

Em seguida, implementamos RMSProp para ser usado em uma rede profunda. Isso é igualmente simples.

```{.python .input}
#@tab mxnet,pytorch
def init_rmsprop_states(feature_dim):
    s_w = d2l.zeros((feature_dim, 1))
    s_b = d2l.zeros(1)
    return (s_w, s_b)
```

```{.python .input}
#@tab tensorflow
def init_rmsprop_states(feature_dim):
    s_w = tf.Variable(d2l.zeros((feature_dim, 1)))
    s_b = tf.Variable(d2l.zeros(1))
    return (s_w, s_b)
```

```{.python .input}
def rmsprop(params, states, hyperparams):
    gamma, eps = hyperparams['gamma'], 1e-6
    for p, s in zip(params, states):
        s[:] = gamma * s + (1 - gamma) * np.square(p.grad)
        p[:] -= hyperparams['lr'] * p.grad / np.sqrt(s + eps)
```

```{.python .input}
#@tab pytorch
def rmsprop(params, states, hyperparams):
    gamma, eps = hyperparams['gamma'], 1e-6
    for p, s in zip(params, states):
        with torch.no_grad():
            s[:] = gamma * s + (1 - gamma) * torch.square(p.grad)
            p[:] -= hyperparams['lr'] * p.grad / torch.sqrt(s + eps)
        p.grad.data.zero_()
```

```{.python .input}
#@tab tensorflow
def rmsprop(params, grads, states, hyperparams):
    gamma, eps = hyperparams['gamma'], 1e-6
    for p, s, g in zip(params, states, grads):
        s[:].assign(gamma * s + (1 - gamma) * tf.math.square(g))
        p[:].assign(p - hyperparams['lr'] * g / tf.math.sqrt(s + eps))
```

Definimos a taxa de aprendizado inicial como 0,01 e o termo de ponderação $\gamma$ como 0,9. Ou seja, $\mathbf{s}$ agrega em média nas últimas $1/(1-\gamma) = 10$ observações do gradiente quadrado.

```{.python .input}
#@tab all
data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
d2l.train_ch11(rmsprop, init_rmsprop_states(feature_dim),
               {'lr': 0.01, 'gamma': 0.9}, data_iter, feature_dim);
```

## Implementação concisa

Como RMSProp é um algoritmo bastante popular, ele também está disponível na instância `Trainer`. Tudo o que precisamos fazer é instanciá-lo usando um algoritmo chamado `rmsprop`, atribuindo $\gamma$ ao parâmetro `gamma1`.

```{.python .input}
d2l.train_concise_ch11('rmsprop', {'learning_rate': 0.01, 'gamma1': 0.9},
                       data_iter)
```

```{.python .input}
#@tab pytorch
trainer = torch.optim.RMSprop
d2l.train_concise_ch11(trainer, {'lr': 0.01, 'alpha': 0.9},
                       data_iter)
```

```{.python .input}
#@tab tensorflow
trainer = tf.keras.optimizers.RMSprop
d2l.train_concise_ch11(trainer, {'learning_rate': 0.01, 'rho': 0.9},
                       data_iter)
```

## Sumário

* RMSProp é muito semelhante ao Adagrad na medida em que ambos usam o quadrado do gradiente para dimensionar os coeficientes.
* RMSProp compartilha com momentum a média que vaza. No entanto, RMSProp usa a técnica para ajustar o pré-condicionador do coeficiente.
* A taxa de aprendizagem precisa ser programada pelo experimentador na prática.
* O coeficiente $\gamma$ determina quanto tempo o histórico é ao ajustar a escala por coordenada.

## Exercícios

1. O que acontece experimentalmente se definirmos $\gamma = 1$? Por quê?
1. Gire o problema de otimização para minimizar $f(\mathbf{x}) = 0.1 (x_1 + x_2)^2 + 2 (x_1 - x_2)^2$. O que acontece com a convergência?
1. Experimente o que acontece com o RMSProp em um problema real de aprendizado de máquina, como o treinamento em Fashion-MNIST. Experimente diferentes opções para ajustar a taxa de aprendizagem.
1. Você gostaria de ajustar $\gamma$ conforme a otimização progride? Quão sensível é o RMSProp a isso?

:begin_tab:`mxnet`
[Discussão](https://discuss.d2l.ai/t/356)
:end_tab:

:begin_tab:`pytorch`
[Discussão](https://discuss.d2l.ai/t/1074)
:end_tab:

:begin_tab:`tensorflow`
[Discussão](https://discuss.d2l.ai/t/1075)
:end_tab:
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTMyMjc0NzQ4OF19
-->