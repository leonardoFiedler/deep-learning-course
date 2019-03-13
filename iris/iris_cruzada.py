import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import LabelEncoder #Transforma de atributo categórico para atributo numérico
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

base = pd.read_csv('iris.csv')
previsores = base.iloc[:, 0:4].values
classe = base.iloc[:, 4].values
labelencoder = LabelEncoder()

# Transforma o texto para um número
classe = labelencoder.fit_transform(classe)
classe_dummy = np_utils.to_categorical(classe)

def criar_rede():
    classificador = Sequential()
    classificador.add(Dense(units=4, activation='relu', input_dim=4))
    classificador.add(Dense(units=4, activation='relu'))
    classificador.add(Dense(units=3, activation='softmax')) #Usa softmax porque tem mais de 2 classes para classificar
    classificador.compile(optimizer='adam', loss='categorical_crossentropy',
                          metrics=['categorical_accuracy'])
    return classificador

classificador = KerasClassifier(build_fn=criar_rede,
                                epochs=1000,
                                batch_size=10)
resultados = cross_val_score(estimator=classificador,
                             X=previsores, y=classe,
                             cv=10, scoring = 'accuracy')
media = resultados.mean()
desvio = resultados.std()