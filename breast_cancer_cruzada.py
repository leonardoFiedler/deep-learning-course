#Script que executa em validacao cruzada
#Usa 10 execuções e divisões com a base (K-fold)
#Usa a funcao cross_val_score para dividir os valores

import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

previsores = pd.read_csv('entradas-breast.csv')
classe = pd.read_csv('saidas-breast.csv')


def criarRede():
    classificador = Sequential()
    classificador.add(Dense(units = 16, activation= 'relu', 
                            kernel_initializer = 'random_uniform', input_dim = 30)) 
    
    classificador.add(Dropout(0.2)) #Adiciona o Dropout na primeira camada. Faz o dropout de 20% da base
    classificador.add(Dense(units = 16, activation= 'relu', 
                            kernel_initializer = 'random_uniform'))
    classificador.add(Dropout(0.2)) # Adiciona dropout na segunda camada.
    
    classificador.add(Dense(units = 1, activation = 'sigmoid'))
    otimizador = keras.optimizers.Adam(lr = 0.001, decay = 0.0001, clipvalue = 0.5)
    classificador.compile(optimizer = otimizador, loss = 'binary_crossentropy', metrics = ['binary_accuracy'])
    return classificador

classificador = KerasClassifier(build_fn = criarRede,
                                epochs = 100,
                                batch_size = 10)

# cv = 10 - valor que vai dividir a base de dados
#Na variável resultados vão vir 10 resultados, que é o percentual de acerto de cada uma das épocas
#Sendo separado os dados em 10
resultados = cross_val_score(estimator=classificador,
                             X=previsores,
                             y=classe,
                             cv=10,
                             scoring='accuracy')

media = resultados.mean()
desvio = resultados.std() #Calcula o desvio padrão - Quanto maior o valor, maior a chance de ter overfitting.