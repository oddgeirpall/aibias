import os
import sys

import numpy as np
import pandas as pd

import tqdm
import tensorflow_addons as tfa

import aibias.dataset as ds

from time import time

import tensorflow as tf
from tensorflow import keras
'''
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
'''
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold


#======================================
#           PREJUDICE REMOVER
#======================================

class PR_remover():

    def __init__(self,dataset,epochs,eta=5):

        self.dataset = dataset
        self.epochs  = epochs
        self.eta     = eta
        self.process_data()

    def process_data(self):
        
        self.df = self.dataset.dataframe.copy()
        if not self.dataset.train_features:
            raise ValueError("Dataset must contain designated training features")

        self.X = self.df[self.dataset.train_features]
        X_val  = self.X.values.copy()
        self.Y = self.df['Label_binary']
        Y_val  = self.Y.values.copy()

        LE = LabelEncoder()
        Y_val = LE.fit_transform(Y_val)

        is_numeric = self.X.apply(lambda s: pd.to_numeric(s, errors='coerce')
                                              .notnull().all())

        self.X_val = np.array(X_val,dtype='float32')
        self.Y_val = np.array(Y_val,dtype='float32')
        #self.X_val = tf.convert_to_tensor(X_val)
        #self.Y_val = tf.convert_to_tensor(Y_val)

        self.num_p_u = [
                self.dataset.Statistics['Unprotected']['Number'],
                self.dataset.Statistics['Protected']['Number']
                ]

        s = list()
        for i in range(len(self.df)):
            if self.df.loc[i]['Protected'] == 1:
                s.append(1)
            else:
                s.append(0)
        self.S = np.array(s)

        if any(res == False for res in is_numeric):
            raise ValueError("Training features must be numeric")

    def PR_regularizer(self,weights):

        def sigmoid(x,w):

            dot = np.dot(x,w)

            return 1 / (1 + np.exp(-dot))

        weights = weights.numpy().reshape([2,8])
        
        p = np.array([sigmoid(self.X_val[i,:],weights[self.S[i],:])
                            for i in range(len(self.X_val))])

        q = np.array([np.sum(p[self.S == si])
                            for si in [0,1]]) / self.num_p_u

        r = np.sum(p) / len(self.X_val)

        f = np.sum(p * (np.log(q[self.S]) - np.log(r))
                            + (1.0-p) * (np.log(1.0-q[self.S])
                            - np.log(1.0-r)))

        l2_reg = np.sum(0.5*np.square(weights))

        return self.eta*f + l2_reg



    def old_PR_regularizer(self,weights):

        start = time()
        print('starting')
        def sigmoid(x):
            return 1 / (1 + tf.math.exp(-x))

        def M(y,x,w):
            sig_x_w = sigmoid(tf.tensordot(x,w,axes=1))
            return y*sig_x_w + (1-y)*(1-sig_x_w)


        Pr_y   = {
                0: 0,
                1: 0
            }

        Pr_y_s = {
                0: {
                    'Protected':   0,
                    'Unprotected': 0
                },
                1: {
                    'Protected':   0,
                    'Unprotected': 0
                }
            }

        print('Starting 1st loop')
        for i in range(len(self.X_val)):

            # Pr_y
            Pr_y[0] += M(0,self.X_val[i],weights)
            Pr_y[1] += M(1,self.X_val[i],weights)

            # Pr_y_s
            if self.df.loc[i]['Protected'] == 1:
                Pr_y_s[0]['Protected'] += M(0,self.X_val[i],weights)
                Pr_y_s[1]['Protected'] += M(1,self.X_val[i],weights)
            else:
                Pr_y_s[0]['Unprotected'] += M(0,self.X_val[i],weights)
                Pr_y_s[1]['Unprotected'] += M(1,self.X_val[i],weights)
        

        Pr_y[0] /= len(self.X_val)
        Pr_y[1] /= len(self.X_val)
        print('Starting 2nd loop')
        for case in ['Protected','Unprotected']:
            Pr_y_s[0][case] /= self.dataset.Statistics[case]['Number']
            Pr_y_s[1][case] /= self.dataset.Statistics[case]['Number']


        total_reg = 0

        print('Starting final loop')
        for i in range(len(self.X_val)):
            for case in [0,1]:

                if self.df.loc[i]['Protected'] == 1:
                    ln_Pr = tf.math.log(Pr_y_s[case]['Protected'])
                else:
                    ln_Pr = tf.math.log(Pr_y_s[case]['Unprotected'])

                m_val = M(case,self.X_val[i],weights)

                total_reg += m_val*ln_Pr

        print('Time taken: {}s'.format(time()-start))
        print('Total_reg: {}'.format(total_reg))
        return total_reg
                

    
    def fit(self):

        input_dim = len(self.dataset.train_features)
        tqdm_callback = tfa.callbacks.TQDMProgressBar()

        def build_model():
            model = Sequential()
            model.add(Dense(2,
                            input_dim=input_dim,
                            activation='relu',
                            kernel_regularizer=self.PR_regularizer
                            #kernel_regularizer=tf.keras.regularizers.l2(0.001)
                     ))
            model.add(Dense(1,activation='sigmoid'))
            model.compile(optimizer='adam',
                          loss='binary_crossentropy',
                          metrics=['accuracy'],
                          run_eagerly=True)
            return model

        '''
        estimator = KerasClassifier(build_fn=build_model,
                                    epochs=self.epochs,
                                    batch_size=5)
        kfold = StratifiedKFold(n_splits=10, shuffle=True)
        results = cross_val_score(estimator, self.X,self.Y_val,cv=kfold)
        print("Results: {0}% accuracy, ({1}%)"
              .format(results.mean()*100,results.std()*100))
        '''
        model = build_model()
        #x = tf.convert_to_tensor(self.X_val,dtype='float32')
        #y = tf.convert_to_tensor(self.Y_val,dtype='float32')
        history = model.fit(self.X_val,self.Y_val,batch_size=5,
                callbacks=[tqdm_callback],
                epochs=self.epochs)
        return history


def PrejudiceRemover(dataset,epochs):

    if not isinstance(dataset,ds.Dataset):
        raise TypeError("Dataset must be of type aibias.dataset.Dataset")

    print('Preparing data')
    pr_remover = PR_remover(dataset,epochs)

    print('Starting training session')
    pr_remover.fit()
