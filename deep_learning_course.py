import numpy as np
import math

# step Function = 
# Relu: 0 ... +infinito
# Linear Function: -infinito ... +infinito
# Softmax: Retorna probabilidades em problemas com mais de duas classes
# tangente hiperbólica 
# Sigmoide - probabilidades



def reluFuction(soma):
    if soma >= 0:
        return soma
    return 0

def linearFunction(soma):
    return soma


def softmaxFunction(x):
    ex = np.exp(x)
    return ex / ex.sum()

def sigmoid(x):
  return 1 / (1 + math.exp(-x))


valores = [5.0, 2.0, 1.3]
sigmoid(2.1)

mae = np.sum(np.absolute((imageB.astype("float") - imageA.astype("float")))


(1 - 0,300)^2 = 0,49
(1 - 0,890)^2 = 0,0121
(0 - 0,20)^2 = 0,04
(0 - 0.320)^2 = 0,1024

0,49 + 0,0121 + 0,04 + 0,1024 = 0,6445 / 4 = 0,16