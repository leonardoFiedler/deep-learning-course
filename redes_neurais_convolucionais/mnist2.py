import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization

(X_treinamento, y_treinamento), (X_teste, y_teste) = mnist.load_data()

#plt.imshow(X_treinamento[0], cmap='gray') # permite visualizar a imagem
#plt.title('Classe ' + str(y_treinamento[0]))

#Coloca as imagens no tamanho do TensorFlow
previsores_treinamento = X_treinamento.reshape(X_treinamento.shape[0], 28, 28, 1)
previsores_teste = X_teste.reshape(X_teste.shape[0], 28, 28, 1)
#Seta para float32 ao inves de int
previsores_treinamento = previsores_treinamento.astype('float32')
previsores_teste = previsores_teste.astype('float32')

#Transforma os pixels de 0 a 255 para 0 até 1 - para facilitar o processamento
previsores_treinamento /= 255
previsores_teste /= 255

classe_treinamento = np_utils.to_categorical(y_treinamento, 10)
classe_teste = np_utils.to_categorical(y_teste, 10)

#Etapa 1 - operador de convolucao (RELU)
classificador = Sequential()

#32 e o numero de testes que ele vai fazer de kernel - nao significa dizer o tipo de kernel que vai utilizar, mas sim a quantidade de vezes que ele vai utilizar
#Uma pratica comum e comecar com 64 kernels. Segue sempre em multiplos - 64, 128, 256...
#(3,3) e o tamanho do detector de caracteristica - imagens pequenas sao mais adequadas
# para imagens maiores, aumenta esse valor
#strides - como a janela se move, entre linahs e colunas
classificador.add(Conv2D(32, (3,3), input_shape=(28, 28, 1), activation = 'relu'))

#Efetua a normalizacao - aplica valores entre 0 e 1 no mapa de caracteristicas
classificador.add(BatchNormalization())

#Etapa 2
#pool_size - tamanho da matriz que vai selecionar os valores (mapa de caracteristicas)
classificador.add(MaxPooling2D(pool_size=(2,2)))

#Etapa 3 - Flattening
#Migra de matriz para um vetor
#classificador.add(Flatten())
#Quando voce tem mais de uma camada, o flattening so pode ser executado na ultima camada

#Adiciona mais uma camada - igual a anterior
#Input shape apenas na primeira camada
classificador.add(Conv2D(32, (3,3), activation='relu'))
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size=(2,2)))

classificador.add(Flatten())

#Agora gera a rede neural densa
#Quantidade de neuronios - vide formula
#imagem de 28x28 x mapa de caracteristicas
#Geralmente em CNN utiliza um valor sem usar a formula
classificador.add(Dense(units=128, activation = 'relu'))

#Adiciona o dropout
#E adicionado para evitar o overfitting
classificador.add(Dropout(0.2))

#Adiciona mais uma camada Dense semelhante a anterior
classificador.add(Dense(units=128, activation = 'relu'))
classificador.add(Dropout(0.2))


classificador.add(Dense(units=10, activation = 'softmax'))
classificador.compile(loss='categorical_crossentropy', optimizer='adam',
                      metrics=['accuracy'])
classificador.fit(previsores_treinamento, 
                  classe_treinamento, 
                  batch_size=128, 
                  epochs=5,
                  validation_data=(previsores_teste, classe_teste))
#O validation_data ja faz o codigo abaixo
#resultado = classificador.evaluate(previsores_teste, classe_teste)
