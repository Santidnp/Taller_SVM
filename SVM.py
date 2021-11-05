import numpy as np
import random
#import sys 
from numpy import linalg as LA
#import math
import matplotlib.pyplot as plt
import matplotlib


x = np.loadtxt('input_01.csv',delimiter = ',')

class SVM_:

    def __init__(self,X,w,b):

        self.X = X
        self.b = b
        self.w = w

    def Clasificacion(self):

        Clf = self.X@self.w
        Clf-=self.b

        Clf = np.where(Clf <= -1,-1,1)

        return Clf



class Perdida:

    def __init__(self,X=None,y=None,w=None,b=None,K=None):

        self.X=X
        self.y=y
        self.w=w
        self.b=b
        self.K=K
    
    def J(self):
        if self.K is not None:
            pass
        else:

            m = self.X.shape[0]
            

            I = self.X@self.w-self.b
            I *=self.y
            I = np.where(I>=1,0,1-I)
            J = np.sum(I)/m 


        return J





def desGradiente_batch(Datos,**kwargs):
    a = 1e-1
    l = 0.0
    lt = 'ridge'
    I = 1e10
    e = 1e-8
    ds = 100
    df = None
    debug = False
    Batch = 1
    Dict = []
    

    if 'learning_rate' in kwargs: a = float( kwargs[ 'learning_rate' ] )
    if 'regularization' in kwargs: l = float( kwargs[ 'regularization' ] )
    if 'reg_type' in kwargs: lt = kwargs[ 'reg_type' ]
    if 'max_iter' in kwargs: I = int( kwargs[ 'max_iter' ] )
    if 'epsilon' in kwargs: e = float( kwargs[ 'epsilon' ] )
    if 'debug' in kwargs: debug = kwargs[ 'debug' ] 
    if 'batch_' in kwargs: Batch = kwargs['batch_']

    W = np.random.random(2)
    #W = np.array([1,2])
    b = np.random.random(1)
    #b = 6
    #print(b.shape)
    #print(W.shape)
    stop = False
    i = 0
    Z1 = np.array_split(Datos,Batch) 
    cmap   = matplotlib.colors.ListedColormap( [ 'k', 'g' ] )
    while i <= I:
        for batch  in Z1 :
        
            X = batch[:,0:-1]
            y = batch[:,-1]
            y =np.where(y ==0,-1,y)
            m = X.shape[0]
        
            

            Y = y*(X@W-b)
                
            Z = (-y*X.T).T
            Z[Y>=1]=0
            dw = np.sum(Z,axis=0)/m +2*l*W
            db = np.sum(np.where(Y>=1,0,y*b))/m +2*l*b
                #print('dw: ',dw)

            

            W -= a*dw
            b -= a*db

        if debug:  
            plt.title('SVM con batch ='+str(Batch)+', alpha = ' + str(a)+ ' lambda = '+str(l))
            Z = np.arange(np.min(X[:, 0]),np.max(X[:, 0]),0.1)
            V = -(W[0]/W[1])*Z+b/W[1]
            plt.scatter(X[:, 0], X[:, 1], c=y, s=5,cmap=cmap)
            plt.plot(X[:, 0],-(W[0]/W[1])*X[:, 0]+b/W[1])
            plt.fill_between(Z,V, color='#E08B7D',alpha=0.4)
            plt.fill_between(Z,V,np.max(V),color='#16B1F5',alpha=0.4)
            plt.xlim(0,max(X[:, 0]))
            plt.ylim(0,max(X[:, 1]))
            plt.pause(0.000001)
            plt.clf()
        Dict.append(Perdida(X,y,W,b).J())
        i+=1
    return W,b,Dict





def Adam_Batch(Datos,**kwargs):


    a = 0.001
    l = 0.0
    lt = 'ridge'
    I = 1e10
    e = 1e-8
    ds = 100
    df = None
    debug = False
    b1 = 0.9
    b2 = 0.999
    Batch = 1
    Dict = []
    

    if 'learning_rate' in kwargs: a = float( kwargs[ 'learning_rate' ] )
    if 'regularization' in kwargs: l = float( kwargs[ 'regularization' ] )
    if 'reg_type' in kwargs: lt = kwargs[ 'reg_type' ]
    if 'max_iter' in kwargs: I = int( kwargs[ 'max_iter' ] )
    if 'epsilon' in kwargs: e = float( kwargs[ 'epsilon' ] )
    if 'debug' in kwargs: debug = kwargs[ 'debug' ] 
    if 'batch_' in kwargs: Batch = kwargs['batch_']

    W = np.random.random(2)
    #W = np.array([1,2])
    b = np.random.random(1)
    m_wt = np.zeros(W.shape[0])
    v_wt = np.zeros(W.shape[0])
    m_bt = np.zeros(b.shape[0])
    v_bt = np.zeros(b.shape[0])
    #b = 6
    #print(m_wt.shape==W.shape)
    #print(W.shape)
    Z1 = np.array_split(Datos,Batch) 
    cmap   = matplotlib.colors.ListedColormap( [ 'k', 'g' ] )
    stop = False
    i = 0
    while i < I and not stop:

        i+=1
        
        
        for batch  in Z1 :
            
            X = batch[:,0:-1]
            y = batch[:,-1]
            y =np.where(y ==0,-1,y)
            m = X.shape[0]
            Y = y*(X@W-b)
                
            Z = (-y*X.T).T
            Z[Y>=1]=0
            dw = np.sum(Z,axis=0)/m +2*l*W
            db = np.sum(np.where(Y>=1,0,y*b))/m +2*l*b
            
            ### Para w

            m_wt = b1*m_wt+(1-b1)*dw
            v_wt = b2*v_wt+((1-b2)*(dw**2))
            M1_wt = m_wt/(1-(b1**i))
            V1_wt = v_wt/(1-(b2**i))
            W = W -a*(M1_wt*(np.sqrt(V1_wt)+e))
            ### Para b

            m_bt = b1*m_bt+(1-b1)*db
            v_bt = b2*v_bt+((1-b2)*(db**2))
            M1_bt = m_bt/(1-(b1**i))
            V1_bt = v_bt/(1-(b2**i))
            b = b -a*(M1_bt*(np.sqrt(V1_bt)+e))
        
        if debug:  
            plt.title('SVM ADAM con batch ='+str(Batch)+', alpha = ' + str(a)+ ' lambda = '+str(l))
            Z = np.arange(np.min(X[:, 0]),np.max(X[:, 0]),0.1)
            V = -(W[0]/W[1])*Z+b/W[1]
            plt.scatter(X[:, 0], X[:, 1], c=y, s=5,cmap=cmap)
            plt.plot(X[:, 0],-(W[0]/W[1])*X[:, 0]+b/W[1])
            plt.fill_between(Z,V, color='#E08B7D',alpha=0.4)
            plt.fill_between(Z,V,np.max(V),color='#16B1F5',alpha=0.4)
            plt.xlim(0,max(X[:, 0]))
            plt.ylim(0,max(X[:, 1]))
            plt.pause(0.000001)
            plt.clf()
        Dict.append(Perdida(X,y,W,b).J())
            
        
    return W,b,Dict




def Matriz_confusion(predicho,real):

    predicho2 = np.where(predicho==1,0,1)
    real2 = np.where(real==1,0,1)

    Predicho = np.stack((predicho, predicho2), axis=-1)
    Real = np.stack((real, real2), axis=-1)

    matriz = np.dot(Predicho.T,Real)

    Precision = matriz[0][0]/(matriz[0][0] + matriz[0][1])
    Sensibilidad = matriz[0][0]/(matriz[0][0] + matriz[1][0])
    Accuracy_ = np.trace(matriz)/np.sum(matriz)
    F1_score = 2*(Precision*Sensibilidad)/(Precision+Sensibilidad)


    return matriz,Precision,Sensibilidad,Accuracy_,F1_score



        
#Dg normal
W_train , b_train,Costo =desGradiente_batch(x,debug = True,max_iter=100000)
#estocástico
W_trainSGD , b_trainSGD,CostoSGD =desGradiente_batch(x,debug =False,max_iter=100000,batch_=200)
#Mini-batch
W_trainMini , b_trainMini,CostoMini =desGradiente_batch(x,debug =False,max_iter=100000,batch_=20)
#Adam
W_trainAdam , b_trainAdam,CostoAdam =Adam_Batch(x,debug =False,max_iter=100000,batch_=8)

Misvm = SVM_(x[:,:-1],W_train ,b_train).Clasificacion()
Misvm_Adam = SVM_(x[:,:-1],W_trainAdam , b_trainAdam).Clasificacion()
Misvm_Esto= SVM_(x[:,:-1],W_trainSGD , b_trainSGD).Clasificacion()
Misvm_Mini= SVM_(x[:,:-1],W_trainMini , b_trainMini).Clasificacion()

y_pred = np.where(Misvm==-1,0,Misvm)
y_adam = np.where(Misvm_Adam==-1,0,Misvm_Adam)
y_Esto = np.where(Misvm_Esto==-1,0,Misvm_Esto)
y_mini = np.where(Misvm_Mini==-1,0,Misvm_Mini)
M ,p,s,f1,acc =Matriz_confusion(y_pred,x[:,-1])
M1 ,p1,s1,f1_1,acc1 =Matriz_confusion(y_adam,x[:,-1])
M_esto ,p_esto,s_esto,f1_esto,acc_esto =Matriz_confusion(y_Esto,x[:,-1])


def Figuras(x,W,b,Costo,numero_iteraciones = 2000):
    cmap   = matplotlib.colors.ListedColormap( [ 'k', 'g' ] )
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('Support Vector Machine')
    ax1.scatter(x[:, 0], x[:, 1], c=x[:, 2], s=5,cmap=cmap)
    ax1.set_title('Plano estimado')
    ax1.set_xlabel('Variable x1')
    ax1.set_ylabel('Variable x2')
    
    Z = np.arange(np.min(x[:, 0]),np.max(x[:, 0]),0.1)
    y = -(W[0]/W[1])*Z+b/W[1]



    #plt.plot(X[:, 0],-(W[0]/W[1])*X[:, 0]-b/W[1])
   
    ax1.plot(x[:, 0],-(W[0]/W[1])*x[:, 0]+b/W[1])
    ax1.fill_between(Z,y, color='#E08B7D',alpha=0.5)
    ax1.fill_between(Z,y,np.max(y),color='#16B1F5',alpha=0.5)

    #plt.plot(X[:, 0],-(W_adam[0]/W_adam[1])*X[:, 0]+b_adam/W_adam[1])

    ax1.set_xlim(0,max(x[:, 0]))
    ax1.set_ylim(0,max(x[:, 1]))
    ax2.set_title('Funcion de Costo primeras '+ str(numero_iteraciones)+' iteraciones')
    ax2.plot(Costo[0:numero_iteraciones])
    #ax2.set_xlim(0,max(x[:, 0]))
    ax2.set_ylim(0,1e-8)
    ax2.set_xlabel('Itearciones')
    ax2.set_ylabel('Función de Costo')
    plt.show()


Figuras(x,W_train ,b_train,Costo,numero_iteraciones=5000)
Figuras(x, W_trainAdam , b_trainAdam,CostoAdam,numero_iteraciones=5000)
Figuras(x,W_trainSGD , b_trainSGD,CostoSGD,numero_iteraciones=5000)




















#########################################################################################################################
''' 
def desGradiente(X,y,**kwargs):
    a = 1e-1
    l = 0.0
    lt = 'ridge'
    I = 1e10
    e = 1e-8
    ds = 100
    df = None
    debug = False
    

    if 'learning_rate' in kwargs: a = float( kwargs[ 'learning_rate' ] )
    if 'regularization' in kwargs: l = float( kwargs[ 'regularization' ] )
    if 'reg_type' in kwargs: lt = kwargs[ 'reg_type' ]
    if 'max_iter' in kwargs: I = int( kwargs[ 'max_iter' ] )
    if 'epsilon' in kwargs: e = float( kwargs[ 'epsilon' ] )
    if 'Debug' in kwargs: debug = kwargs[ 'debug' ] 

    W = np.random.random(2)
    #W = np.array([1,2])
    b = np.random.random(1)
    #b = 6
    #print(b.shape)
    #print(W.shape)
    stop = False
    i = 0
    m = X.shape[0]
    

    J = Perdida(X,y,W,b).J()-l*(LA.norm(W)+b**2)
    

    
    while J > e and i < I and not stop:
        
        J = Perdida(X,y,W,b).J()-l*(LA.norm(W)+b**2)

        Y = y*(X@W-b)
        
        Z = (-y*X.T).T
        Z[Y>=1]=0
        dw = np.sum(Z,axis=0)/m +2*l*W
        db = np.sum(np.where(Y>=1,0,y*b))/m +2*l*b
        #print('dw: ',dw)

        W -= a*dw
        b -= a*db
        i+=1
        #print (W)

        if i%100==0:
            print(J)
    
    return W,b
        



   '''  


''' 
def Adam(X,y,**kwargs):


    a = 0.001
    l = 0.0
    lt = 'ridge'
    I = 1e10
    e = 1e-8
    ds = 100
    df = None
    debug = False
    b1 = 0.9
    b2 = 0.999
    

    if 'learning_rate' in kwargs: a = float( kwargs[ 'learning_rate' ] )
    if 'regularization' in kwargs: l = float( kwargs[ 'regularization' ] )
    if 'reg_type' in kwargs: lt = kwargs[ 'reg_type' ]
    if 'max_iter' in kwargs: I = int( kwargs[ 'max_iter' ] )
    if 'epsilon' in kwargs: e = float( kwargs[ 'epsilon' ] )
    if 'Debug' in kwargs: debug = kwargs[ 'debug' ] 

    W = np.random.random(2)
    #W = np.array([1,2])
    b = np.random.random(1)
    m_wt = np.zeros(W.shape[0])
    v_wt = np.zeros(W.shape[0])
    m_bt = np.zeros(b.shape[0])
    v_bt = np.zeros(b.shape[0])
    #b = 6
    #print(m_wt.shape==W.shape)
    #print(W.shape)
    stop = False
    i = 0
    m = X.shape[0]
    

    J = Perdida(X,y,W,b).J()-l*(LA.norm(W)+b**2)

    while J > e and i < I and not stop:

        i+=1

        
        J = Perdida(X,y,W,b).J()-l*(LA.norm(W)+b**2)

        Y = y*(X@W-b)
        
        Z = (-y*X.T).T
        Z[Y>=1]=0
        dw = np.sum(Z,axis=0)/m +2*l*W
        db = np.sum(np.where(Y>=1,0,y*b))/m +2*l*b
        ### Para w

        m_wt = b1*m_wt+(1-b1)*dw
        v_wt = b2*v_wt+((1-b2)*(dw**2))
        M1_wt = m_wt/(1-(b1**i))
        V1_wt = v_wt/(1-(b2**i))
        W = W -a*(M1_wt*(np.sqrt(V1_wt)+e))
        #print(W)

        ### Para b

        m_bt = b1*m_bt+(1-b1)*db
        v_bt = b2*v_bt+((1-b2)*(db**2))
        M1_bt = m_bt/(1-(b1**i))
        V1_bt = v_bt/(1-(b2**i))
        b = b -a*(M1_bt*(np.sqrt(V1_bt)+e))
        #print(b)
        
        #print (W)

        if i%100==0:
            print(J)
    
    
    return W,b
 '''


