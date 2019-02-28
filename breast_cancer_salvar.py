#
# Salvar em disco uma rede
#
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

classificador_json = classificador.to_json()

with open('classificador_breast.json', 'w') as json_file:
    json_file.write(classificador_json)
classificador.save_weights('classificador_breast.h5')