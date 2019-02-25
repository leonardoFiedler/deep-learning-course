import numpy as np

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


valores = [5.0, 2.0, 1.3]
softmaxFunction(valores)