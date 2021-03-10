import os
import warnings
import tempfile

import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from scipy.spatial import distance_matrix
from scipy.stats import nbinom

import tensorflow.keras as keras
from keras import backend as K
from keras.models import Model,model_from_json
from keras.layers import Dense,Dropout,Input
from keras.callbacks import EarlyStopping
import keras.losses

import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def corr2_coeff(A, B):
    # Rowwise mean of input arrays & subtract from input arrays themeselves
    A_mA = A - A.mean(1)[:, None]
    B_mB = B - B.mean(1)[:, None]

    # Sum of squares across rows
    ssA = (A_mA**2).sum(1)
    ssB = (B_mB**2).sum(1)

    # Finally get corr coeff
    return np.dot(A_mA, B_mB.T) / np.sqrt(np.dot(ssA[:, None],ssB[None]))


def wMSE(y_true, y_pred, binary=False):
    if binary:
        weights = tf.cast(y_true>0, tf.float32)
    else:
        weights = y_true
    return tf.reduce_mean(weights*tf.square(y_true-y_pred))


def inspect_data(data):
    # Check if there area any duplicated cell/gene labels
    
    if sum(data.index.duplicated()):
        print("ERROR: duplicated cell labels. Please provide unique cell labels.")
        exit(1)
        
    if sum(data.columns.duplicated()):
        print("ERROR: duplicated gene labels. Please provide unique gene labels.")
        exit(1)
        
    max_value = np.max(data.values)
    if max_value < 10:
        print("ERROR: max value = {}. Is your data log-transformed? Please provide raw counts"
              .format(max_value))
        exit(1)
        
    print("Input dataset is {} cells (rows) and {} genes (columns)"
          .format(*data.shape))
    print("First 3 rows and columns:")
    print(data.iloc[:3,:3])

class multiomicNet:

    def __init__(self,
                 learning_rate=1e-4,
                 batch_size=8,
                 max_epochs=500,
                 patience=5,
                 ncores=-1,
                 loss="MSE",
                 output_prefix=tempfile.mkdtemp(),
                 sub_outputdim=512,
                 verbose=1,
                 seed=1234,
                 architecture = None
    ):
        self.NN_parameters = {"learning_rate": learning_rate,
                              "batch_size": batch_size,
                              "loss": loss,
                              "architecture": architecture,
                              "max_epochs": max_epochs,
                              "patience": patience
                             }
        if architecture is None:
            self.NN_parameters["architecture"] = [{"type": "dense", "neurons": 128, "activation": "relu"},
                                                  {"type": "dropout", "rate": 0.2}]
        self.sub_outputdim = sub_outputdim
        self.outputdir = output_prefix
        self.verbose = verbose
        self.seed = seed
        self.setCores(ncores)

    def setCores(self, ncores):
        if ncores > 0:
            self.ncores = ncores
        else:
            self.ncores = os.cpu_count()
            print("Using all the cores ({})".format(self.ncores))
 
        
    def save(self, model):
        os.system("mkdir -p {}".format(self.outputdir))
        
        model_json = model.to_json()
                
        with open("{}/model.json".format(self.outputdir), "w") as json_file:
            json_file.write(model_json)
            
        # serialize weights to HDF5
        model.save_weights("{}/model.h5".format(self.outputdir))
        print("Saved model to disk")

    def load(self):
        json_file = open('{}/model.json'.format(self.outputdir), 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        model.load_weights('{}/model.h5'.format(self.outputdir))

        return model
        
    def build(self, inputdims):
        inputs = [ Input(shape=(inputdim,)) for inputdim in inputdims ]
        outputs = inputs

        for layer in self.NN_parameters['architecture']:
            if layer['type'].lower() == 'dense':
                outputs = [ Dense(layer['neurons'], activation=layer['activation'])(output)
                            for output in outputs ]
            elif layer['type'].lower() == 'dropout':
                outputs = [ Dropout(layer['rate'], seed=self.seed)(output)
                            for output in outputs] 
            else:
                print("Unknown layer type.")

        outputs = [Dense(self.sub_outputdim, activation="softplus")(output)
                   for output in outputs]
                
        model = Model(inputs=inputs, outputs=outputs)
        loss = self.NN_parameters['loss']
        
        if loss in [k for k, v in globals().items() if callable(v)]:
            # if loss is a defined function
            loss = eval(self.NN_parameters['loss'])
        elif type(loss).__module__ == "numpy":
            loss = custom_loss(loss)
        elif not callable(loss):
            # it is defined in Keras
            if hasattr(keras.losses, loss):
                loss = getattr(keras.losses, loss)
            else:
                print('Unknown loss: {}. Aborting.'.format(loss))
                exit(1)

        model.compile(optimizer=keras.optimizers.Adam(lr=self.NN_parameters['learning_rate']),
                      loss=loss)
        return model
    
    
    def fit(self, X, Y):
        
        n_subsets = round(Y.shape[1]/256) - 1
        
        Y_data_idx = np.random.choice(Y.columns, [n_subsets, 256], replace=False)
        Y_data = [Y[idx] for idx in Y_data_idx]
        
        X_data = []
        for Y_data_subgrp in Y_data:
            corr = corr2_coeff(X.values.T, Y_data_subgrp.values.T)
            corr = pd.DataFrame(corr, index=X.columns, columns=Y_data_subgrp.columns)
            sorted_idx = np.argsort(-corr.values, axis=0)
            X_data_idx = corr.index[sorted_idx[:5,:].flatten()]
            X_data_idx = X_data_idx.unique()
            X_data.append(X[X_data_idx])
        
        self.predictors = np.array([sub.columns for sub in X])
        self.targets = np.array([sub.columns for sub in Y])
        
        #print("Normalization")
        #norm_X = np.log1p(X_data).astype(np.float32) # normalizer.transform(raw)
        
        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)
        
        print("Building network")
        model = self.build([len(genes) for genes in self.predictors])
        
        test_cells = np.random.choice(X_data[0].index, int(0.1 * X_data[0].shape[0]), replace=False)
        train_cells = np.setdiff1d(X_data[0].index, test_cells)

        X_train = [sub.loc[train_cells, :].values for sub in X_data]
        Y_train = [sub.loc[train_cells, :].values for sub in Y_data]
        
        X_test = [sub.loc[test_cells, :].values for sub in X_data]
        Y_test = [sub.loc[test_cells, :].values for sub in Y_data]

        print("Fitting with {} cells".format(len(train_cells)))
        result = model.fit(X_train, Y_train,
                           validation_data=(X_test,Y_test),
                           epochs=self.NN_parameters["max_epochs"],
                           batch_size=self.NN_parameters["batch_size"],
                           callbacks=[EarlyStopping(monitor='val_loss',
                                                    patience=self.NN_parameters["patience"])],
                           verbose=self.verbose)

        self.trained_epochs = len(result.history['loss'])
        print("Stopped fitting after {} epochs".format(self.trained_epochs))
        
        self.save(model)

        # Save some metrics on test data
        Y_test_raw = np.hstack(Y_test).flatten()
        Y_test_imputed = np.hstack(model.predict(X_test)).flatten()

        self.test_metrics = {
            'correlation': pearsonr(Y_test_raw,Y_test_imputed)[0],
            'MSE': np.sum((Y_test_raw-Y_test_imputed)**2)/len(Y_test_raw)
        }        
        return self
    
    
    def predict(self,
                X,
                Y,
                imputed_only=False,
                policy="restore"):

        inputs = [X[genes] for genes in self.predictors]
        inputs = [ sub.values.astype(np.float32) for sub in inputs ]
        
        model = self.load()

        predicted = model.predict(inputs)
        if len(inputs)>1:
            predicted = np.hstack(predicted)
        
        predicted = pd.DataFrame(predicted, index=Y.index, columns=self.targets.flatten())

        predicted = predicted.groupby(by=predicted.columns, axis=1).mean()
        not_predicted = Y.drop(self.targets.flatten(), axis=1)

        imputed = (pd.concat([predicted,not_predicted],axis=1)
                   .loc[Y.index, Y.columns]
                   .values)
        
        # To prevent overflow
        imputed[ (imputed > 2*Y.values.max()) | (np.isnan(imputed)) ] = 0

        if policy == "restore":
            print("Filling zeros")
            mask = (Y.values > 0)
            imputed[mask] = Y.values[mask]
        elif policy == "max":
            print("Imputing data with 'max' policy")
            mask = (Y.values > imputed)
            imputed[mask] = Y.values[mask]

        imputed = pd.DataFrame(imputed, index=Y.index, columns=Y.columns)

        if imputed_only:
            return imputed.loc[:, predicted.columns]
        else:
            return imputed
        