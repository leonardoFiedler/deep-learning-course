#
#Este exemplo serve para demonstrar o uso da base de cancer com parâmetros de tuning
#Com o intuito de otimizar os resultados do algoritmo.
#

import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

previsores = pd.read_csv('entradas-breast.csv')
classe = pd.read_csv('saidas-breast.csv')

#optmizer - qual o otimizador
#loss - taxa de perde
#kernel_initializer - atualmente random_uniform
#activation
#neurons
def criarRede(optimizer, loss, kernel_initializer, activation, neurons ):
    classificador = Sequential() #Cria rede
    
    #Configuracoes da rede
    classificador.add(Dense(units = neurons, activation= activation, 
                            kernel_initializer = kernel_initializer, input_dim = 30)) 
    
    classificador.add(Dropout(0.2)) #Adiciona o Dropout na primeira camada. Faz o dropout de 20% da base
    classificador.add(Dense(units = neurons, activation= activation, 
                            kernel_initializer = kernel_initializer))
    classificador.add(Dropout(0.2)) # Adiciona dropout na segunda camada.
    
    classificador.add(Dense(units = 1, activation = 'sigmoid')) #Booleano - maligno ou benigno
    
    classificador.compile(optimizer = optimizer, loss = loss, metrics = ['binary_accuracy'])
    return classificador

classificador = KerasClassifier(build_fn=criarRede)
parametros = {'batch_size':[10, 30],
              'epochs': [50, 100],
              'optimizer': ['adam', 'sgd'],
              'loss':['binary_crossentropy', 'hinge'],
              'kernel_initializer': ['random_uniform', 'normal'],
              'activation': ['relu', 'tanh'],
              'neurons': [16, 8]}