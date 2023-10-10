import os
import tensorflow as tf
import numpy as np
import random
import pandas as pd


class model:
    def __init__(self, path):
        # Model Names Exmplanation
        # nF means that the model has n input features
        # Dense_n means that the model has n dense layers

        model_paths = ["Direct_Dense_2", "6F_Direct_Dense_1", "5F_Direct_Dense_2", "6F_Direct_Dense_1_0.01Val", "6F_Direct_20Win", "6F_Direct_130Win", "6F_Direct_Dense_1_Spong", "6F_Direct_Swish"]

        self.models = [tf.keras.models.load_model(os.path.join(path, mp )) for mp in model_paths]

    def predict(self, X):
        # Insert your preprocessing here

        X = X.numpy()
        X_min = X.min(axis=0)
        X_max = X.max(axis=0)
	
        # preprocess input with 7 feature and window of 100
        future = X[-100:]
        future = (future - X_min) / (X_max - X_min)
        future = np.expand_dims(future, axis=0)

        # preprocess input with 6 feature and window of 100
        future2 = np.copy(future)
        future2 = np.delete(future2, -1, axis=2)

        # preprocess input with 6 feature and window of 100
        future3 = np.copy(future)
        future3 = np.delete(future2, 3, axis=2)

        # preprocess input with 6 feature and window of 20
        future4 = np.copy(X[-20:])
        future4 = (future4 - X_min) / (X_max - X_min)
        future4 = np.expand_dims(future4, axis=0)
        future4 = np.delete(future4, -1, axis=2)


        # preprocess input with 6 feature and window of 130
        future5 = np.copy(X[-130:])
        future5 = (future5 - X_min) / (X_max - X_min)
        future5 = np.expand_dims(future5, axis=0)
        future5 = np.delete(future5, -1, axis=2)

        # preprocess input with 6 feature and window of 130
        future6 = np.copy(X[-130:])
        future6 = (future6 - X_min) / (X_max - X_min)
        future6 = np.expand_dims(future6, axis=0)
        future6 = np.delete(future6, -1, axis=2)

        future7 = np.copy(X[-100:])
        future7 = (future7 - X_min) / (X_max - X_min)
        future7 = np.expand_dims(future7, axis=0)
        future7 = np.delete(future7, -1, axis=2)
	
        outs = []

        outs.append(self.models[0].predict(future))
        outs.append(self.models[1].predict(future2))
        outs.append(self.models[2].predict(future3))
        outs.append(self.models[3].predict(future2))
        outs.append(self.models[4].predict(future4))

        outs = [ out * (X_max - X_min) + X_min  for out in outs]  # denormalize
        outs = [ np.reshape(out, (864, 7)) for out in outs]
        
	# Code to normalize outputs with shape different to the initial one

        outs.append(self.models[5].predict(future5)) 
        outs[5] = outs[5] * (X_max[[2,6]] - X_min[[2,6]]) + X_min[[2,6]]
        outs[5] = np.reshape(outs[5], (864, 2))

        outs.append(self.models[6].predict(future6))
        outs[6] = outs[6] * (X_max[0] - X_min[0]) + X_min[0]
        outs[6] = np.reshape(outs[6], (864, 1))

        outs.append(self.models[7].predict(future7))
        outs[7] = outs[7] *  (X_max - X_min) + X_min
        outs[7] = np.reshape(outs[7], (864, 7))

        ensemble = []

        for i in range(864):

            elem = []
            
            #each feature gets predicted by the model with the best performance for it
            elem.append(outs[0][i, 0])
            elem.append(outs[4][i, 1])
            elem.append(outs[5][i, 0])
            elem.append(outs[7][i, 3])
            elem.append(outs[4][i, 4])
            elem.append(outs[2][i, 5])
            elem.append(outs[5][i, 1])
          
            ensemble.append(elem)

        ensemble = tf.convert_to_tensor(ensemble)
        ensemble = tf.cast(ensemble, tf.float32)

        return ensemble




