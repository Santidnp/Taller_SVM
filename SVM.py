import numpy as np
import random
#import sys 
from numpy import linalg as LA
import math


x = np.loadtxt('input_01.csv',delimiter = ',')

print(x.shape)

X = x[:,0:2]

y = x[:,2]

W = np.array([1,2])
W.shape
X.shape
V=X@W
V.shape
b=4
V-b
y =np.where(y ==0,-1,y)

y*V

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


mi_svm = SVM_(X,W,b).Clasificacion()

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


Perdida(X,y,W,b).J()

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
        



    

W_train , b_train,=desGradiente(X,y,debug =True,max_iter=100000)

mi_svm = SVM_(X,W_train,b_train).Clasificacion()

y_pred =np.where(mi_svm ==-1,0,mi_svm)
y_real = np.where(y ==-1,0,y)
y_pred == y_real


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


W_adam , b_adam=Adam(X,y,debug =True,max_iter=100000)

mi_svm_adam = SVM_(X,W_adam , b_adam).Clasificacion()

y_predadam =np.where(mi_svm_adam ==-1,0,mi_svm_adam)
y_real = np.where(y ==-1,0,y)
y_predadam == y_real