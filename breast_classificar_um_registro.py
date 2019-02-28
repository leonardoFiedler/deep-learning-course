#
# Classe com as melhores configuracoes - executado depois do breast_cancer_tuning.
#
import numpy as np
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout

previsores = pd.read_csv('entradas-breast.csv')
classe = pd.read_csv('saidas-breast.csv')

classificador = Sequential() #Cria rede

#Configuracoes da rede
classificador.add(Dense(units = 8, activation= 'relu', 
                        kernel_initializer = 'normal', input_dim = 30)) 

classificador.add(Dropout(0.2)) #Adiciona o Dropout na primeira camada. Faz o dropout de 20% da base
classificador.add(Dense(units = 8, activation= 'relu', 
                        kernel_initializer = 'normal'))
classificador.add(Dropout(0.2)) # Adiciona dropout na segunda camada.

classificador.add(Dense(units = 1, activation = 'sigmoid')) #Booleano - maligno ou benigno

classificador.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['binary_accuracy'])
classificador.fit(previsores, classe, batch_size=10, epochs=100)

novo = np.array([[15.80, 8.34, 118, 900, 0.10, 0.26, 0.08, 0.134, 0.178,
                  0.20, 0.05, 1098, 0.87, 4500, 145.2, 0.005, 0.04, 0.05, 0.015,
                  0.03, 0.007, 23.15, 16.64, 178.5, 2018, 0.14, 0.185,
                  0.84, 158, 0.363]])

previsao = classificador.predict(novo)
previsao = (previsao > 0.5)