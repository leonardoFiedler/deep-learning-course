import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense #Denso significa que todos os neuronios estao interligados aos neuronios da proxima camada.
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

previsores = pd.read_csv('entradas-breast.csv')

classe = pd.read_csv('saidas-breast.csv')


#Divide em classe e previsores de teste e treinamento. Sendo que apenas 25% são de teste
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.25)

classificador = Sequential()
classificador.add(Dense(units = 16, activation= 'relu', 
                        kernel_initializer = 'random_uniform', input_dim = 30)) 

#A partir da segunda camada oculta, o input_dim é dispensavel
classificador.add(Dense(units = 16, activation= 'relu', 
                        kernel_initializer = 'random_uniform')) 
# Units =  (30 + 1) / 2 - para descobrir o numero de classes
# input dim = 30 - número de neurônios iniciais da primeira camada

#SIGMOID - valor entre 0 e 1
classificador.add(Dense(units = 1, activation = 'sigmoid'))
#classificador.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['binary_accuracy'])
otimizador = keras.optimizers.Adam(lr = 0.001, decay = 0.0001, clipvalue = 0.5)
classificador.compile(optimizer = otimizador, loss = 'binary_crossentropy', metrics = ['binary_accuracy'])
# Adam - descida do gradiente estocástico
# Para mais optmizers - https://keras.io/optimizers/
# Para mais loss - https://keras.io/losses/

classificador.fit(previsores_treinamento, classe_treinamento, batch_size = 10, epochs = 100)

pesos0 = classificador.layers[0].get_weights()
print(pesos0)
print(len(pesos0))
pesos1 = classificador.layers[1].get_weights()
pesos2 = classificador.layers[2].get_weights()

previsoes = classificador.predict(previsores_teste)
previsoes = (previsoes > 0.5)
precisao = accuracy_score(classe_teste, previsoes)

#confusion_matrix - faz a avaliacao da matriz entre o 0 e o 1
matriz = confusion_matrix(classe_teste, previsoes)

resultado = classificador.evaluate(previsores_teste, classe_teste)


#K-Fold Cross Validation - Validação cruzada, base separada em 4, testando com todos
#Geralmente usado com K = 10
#Técnica mais usada por pesquisadores

