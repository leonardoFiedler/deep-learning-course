import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

base = pd.read_csv('../iris.csv')
previsores = base.iloc[:, 0:4].values
classe = base.iloc[:, 4].values

def criarRede(optimizer, loss, kernel_initializer, activation, neurons, dropouts):
    classificador = Sequential()
    classificador.add(Dense(units = neurons, activation = activation, 
                        kernel_initializer = kernel_initializer, input_dim = 4))
    classificador.add(Dropout(dropouts))
    classificador.add(Dense(units = neurons, activation = activation, 
                        kernel_initializer = kernel_initializer))
    classificador.add(Dropout(dropouts))
    classificador.add(Dense(units = 3, activation = 'softmax'))
    classificador.compile(optimizer = optimizer, loss = loss,
                      metrics = ['accuracy'])
    return classificador

classificador = KerasClassifier(build_fn = criarRede)
parametros = {'batch_size': [12, 15],
              'epochs': [50, 100],
              'optimizer': ['adam'],
              'loss': ['sparse_categorical_crossentropy'],
              'kernel_initializer': ['random_uniform', 'normal'],
              'activation': ['relu'],
              'neurons': [4, 4],
              'dropouts': [0.2, 0.3]}

grid_search = GridSearchCV(estimator = classificador,
                           param_grid = parametros,
                           cv = 5)

grid_search = grid_search.fit(previsores, classe)

melhores_parametros = grid_search.best_params_
melhor_precisao = grid_search.best_score_

print("Melhor precisao")
print(melhor_precisao)