import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

previsores = pd.read_csv('entradas_breast.csv')
classe = pd.read_csv('saidas_breast.csv')

def criarRede(optimizer, loos, kernel_initializer, activation, neurons, dropouts):
    classificador = Sequential()
    classificador.add(Dense(units = neurons, activation = activation, 
                        kernel_initializer = kernel_initializer, input_dim = 30))
    classificador.add(Dropout(dropouts))
    classificador.add(Dense(units = neurons, activation = activation, 
                        kernel_initializer = kernel_initializer))
    classificador.add(Dropout(dropouts))
    classificador.add(Dense(units = 1, activation = 'softmax'))
    classificador.compile(optimizer = optimizer, loss = loos,
                      metrics = ['accuracy'])
    return classificador

classificador = KerasClassifier(build_fn = criarRede)
parametros = {'batch_size': [32, 64],
              'epochs': [50, 100],
              'optimizer': ['adam', 'sgd'],
              'loos': ['sparse_categorical_crossentropy'],
              'kernel_initializer': ['random_uniform', 'normal'],
              'activation': ['relu', 'tanh'],
              'neurons': [16, 8],
              'dropouts': [0.2, 0.3]}
grid_search = GridSearchCV(estimator = classificador,
                           param_grid = parametros,
                           cv = 5)
grid_search = grid_search.fit(previsores, classe)
melhores_parametros = grid_search.best_params_
melhor_precisao = grid_search.best_score_