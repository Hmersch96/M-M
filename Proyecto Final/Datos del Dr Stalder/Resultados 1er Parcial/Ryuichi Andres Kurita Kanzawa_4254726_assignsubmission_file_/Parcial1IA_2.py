import sys, os
sys.path.append(os.pardir)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from common.two_layer_net import TwoLayerNet
import random



    #obtener datos del URL
url = 'https://raw.githubusercontent.com/diegostaPy/cursoIA/main/datosRendimiento/datos/datosfiltrados.csv'

#extraer 100 ejemplos de datos input
#y 100 path de si paso o no (1 o 0)
def aprender(url,Xin):

    dataset = pd.read_csv(url,usecols=["Asignatura","Aprobado","Primer.Par","Segundo.Par","Primer.Rec","Segundo.Rec","Nota.Final","AOT"])
    #dataset = (dataset[dataset['Asignatura']=='DINAMICA']) #Se extrajo Calculo 1
    dataset = dataset.fillna(float(0))
    dataset = dataset.replace({'Aprobado':{'S':float(1.0)}})
    dataset = dataset.replace({'Aprobado':{'N':float(0.0)}})
    dataset = dataset.replace({'Nota.Final':{0:'N'}})

    StudyData = np.array(dataset.drop(['Asignatura','Aprobado'],axis=1))    #datos a estudiar. 
    Label = np.array([dataset['Aprobado']]) #no es necesario normalizar
    for (contador,DatoParticular) in zip(range(0,len(StudyData)),StudyData):
            Finales = DatoParticular[5]
            CantidadF = Finales.count('F')
            CantidadF = float(CantidadF)
            StudyData[contador][5] = CantidadF
    for (contador,Datos) in zip(range(0,len(StudyData)),StudyData):
        StudyData[contador]= [Datos[0]/24,Datos[1]/36,Datos[2]/10,Datos[3]/60,Datos[4]/60,Datos[5]/3]

    
    StudyData = np.concatenate((StudyData,Label.T),axis = 1)
    #[1p,2p,t,1r,2r,crendido,aprobado]

    #Crear dos matrices random 
    Nfila = StudyData.shape[0]
    random_ind = np.random.choice(Nfila,size=10,replace=False)
    other_random_ind = list(range(Nfila))
    for i in random_ind:
        other_random_ind.remove(i)
    
    testData = StudyData[random_ind, :]
    trainData = StudyData[other_random_ind, :]
    Xtrain = trainData[:,[0,1,2,3,4,5]]
    Xtrain = np.array([rows for rows in Xtrain], dtype = float )
    Ttrain = trainData[:,[6]]
    Ttrain = np.array([rows for rows in Ttrain], dtype = float )
    Xtest = testData[:,[0,1,2,3,4,5]]
    Xtest = np.array([rows for rows in Xtest],dtype=float)
    Ttest = testData[:,[6]]
    Ttest = np.array([rows for rows in Ttest],dtype=float)
    
    
    #Optimo i_num = 50k, batch_size = 100, rate = 0.01
    iters_num = 50000
    batch_size = 100
    train_size = Xtrain.shape[0]
    learning_rate = 0.01

    train_loss_list = []
    train_acc_list = []
    test_acc_list = []

    iter_per_epoch = max(train_size/batch_size , 1)

    m=0
    np.exp(m)

    #Resultado aceptable = hidden size = 8
    network = TwoLayerNet(input_size = 6, hidden_size = 6, output_size = 2)

    for i in range(iters_num):
        batch_mask = np.random.choice(train_size,batch_size)
        x_batch = Xtrain[batch_mask] #extraer matriz de los 50 index
        t_batch = Ttrain[batch_mask]   
        t_batch = CambiodeDimension(t_batch)
        grad = network.gradient(x_batch,t_batch)
        for key in ('W1','b1','W2','b2'):
            network.params[key] -= learning_rate*grad[key]
        loss = network.loss(x_batch,t_batch)
        train_loss_list.append(loss)


        if i%iter_per_epoch == 0:
           train_acc = network.accuracy(Xtrain,Ttrain)
           test_acc = network.accuracy(Xtest,Ttest)
           train_acc_list.append(train_acc)
           train_acc_list.append(test_acc)
           print(train_acc, test_acc)
           
    



    print("Resultado es")
    Resultado =  network.ResultadoObtener(Xin)*100
    print(Resultado)

    plt.plot(range(0,len(train_loss_list)),train_loss_list)
    plt.xlim(0, iters_num)
    plt.ylim(0,1)
    plt.show()

    return network.devolverParametros()





def CambiodeDimension(x):
    temp = np.zeros((x.shape[0],2))
    for (i,contador) in zip(x,range(0,len(x))):
        if i == 1:
            temp[contador] = [1,0]
        else:
            temp[contador] = [0,1]
    return temp

f = open("ParametrosIA.txt","w")


#1P,2P,T,1R,2R,CF
#introducir en este array los datos normalizados de puntaje segun el orden mencionado
Xin = np.array([[0,0,0,0,0,0],[0.5,0.5,0.5,0.25,0.1,0],[1,1,1,1,1,1]], dtype = float)
(W1,W2,b1,b2) = aprender(url,Xin)
print(W1,W2,b1,b2)