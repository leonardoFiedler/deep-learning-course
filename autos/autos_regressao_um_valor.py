#Previsão de preço de carro para apenas um valor.

import pandas as pd

#Apaga as colunas que estao desbalanceadas ou que nao fazem sentido para o problema
base = pd.read_csv('autos.csv', encoding = 'ISO-8859-1')
base = base.drop('dateCrawled', axis = 1)
base = base.drop('dateCreated', axis = 1)
base = base.drop('nrOfPictures', axis = 1)
base = base.drop('postalCode', axis = 1)
base = base.drop('lastSeen', axis = 1)
base = base.drop('name', axis = 1)
base = base.drop('seller', axis = 1)
base = base.drop('offerType', axis = 1)


#i1 = base.loc[base.price <= 10]
#base.price.mean()

#Remove todos os registros cujo preco e menor ou igual a 10.
#Estes registros foram cadastrados provavelmente para teste.
base = base[base.price > 10]

#Remove registros que estao com precos muito maiores
i2 = base.loc[base.price > 350000]

base = base[base.price < 350000]

# Nestes casos, alem de remover os registros que sao de testes ou nao fazem sentido
# por serem valores categorico, eles podem ser substituidos pelo valor que mais
# Aparece na base de dados
base.loc[pd.isnull(base['vehicleType'])]
base['vehicleType'].value_counts() #limousine

base.loc[pd.isnull(base['gearbox'])]
base['gearbox'].value_counts() #manuell

base.loc[pd.isnull(base['model'])]
base['model'].value_counts() #golf

base.loc[pd.isnull(base['fuelType'])]
base['fuelType'].value_counts() #benzin

base.loc[pd.isnull(base['notRepairedDamage'])]
base['notRepairedDamage'].value_counts() #nein


# Substituicao dos valores por aqueles que mais se repetem.
valores = {'vehicleType' : 'limousine', 
           'gearbox' : 'manuell',
           'model' : 'golf',
           'fuelType': 'benzin',
           'notRepairedDamage': 'nein'} 

base = base.fillna(value = valores)

# Label encoder - transforma os valores de categoricos (texto) para string.

#Nao pega o valor da 0, porque e onde esta o valor do preco
previsores = base.iloc[:, 1:13].values
preco_real = base.iloc[:, 0].values

from sklearn.preprocessing import LabelEncoder

labelencoder_previsores = LabelEncoder()


# Substitui o texto para numero em cada uma das colunas necessarias
# Apos este comando, nao havera mais nenhuma coluna com valor categorico.
for num in [0, 1, 3, 5, 8, 9, 10]:
    previsores[:, num] = labelencoder_previsores.fit_transform(previsores[:, num])



