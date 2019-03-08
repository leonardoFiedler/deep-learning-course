import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense

base = pd.read_csv('iris.csv')
previsores = base.iloc[:, 0:4].values
classe = base.iloc[:, 4].values


previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.25)

classificador = Sequential()
# Formula de units: quantidade de entradas (4) + quantidade de saídas (3 - os tipos de saída) e divide por 2
# (4 + 3) / 2 = 3.5 ~ 4
classificador.add(Dense(units=4, activation='relu', input_dim=4))
classificador.add(Dense(units=4, activation='relu'))
classificador.add(Dense(units=3, activation='softmax')) #Usa softmax porque tem mais de 2 classes para classificar
classificador.compile(optimizer='adam', loss='categorical_crossentropy',
                      metrics=['categorical_accuracy'])

classificador.fit(previsores_treinamento, classe_treinamento, batch_size=10, epochs=1000)