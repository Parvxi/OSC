import seaborn as sns
import numpy as np
import pandas as pd
import pickle
from keras.models import model_from_json
from sklearn.preprocessing import StandardScaler

class predic():

    def __init__(self, pathjs='models/lstm_model.json', path_weights='models/lstm_model.h5', dataFit='dataFit.csv'):
        self.model = self.load_model(pathjs, path_weights)
        self.scaler = StandardScaler()
        data = pd.read_csv(dataFit)
        self.scaler.fit(data.drop(["Label"], axis=1))

    def transform_data(self, new_list):
        new_array = np.array(new_list)
        new_array[np.isnan(new_array)] = 0.0
        X = self.scaler.transform(new_array.reshape(1, -1))  # Reshape to 2D array
        return X

    def load_model(self, pathjs, path_weights):
        # loading model
        model = model_from_json(open(pathjs).read())
        model.load_weights(path_weights)
        model.compile(loss='categorical_crossentropy', optimizer='adam')
        return model

    def predictionVal(self, new_list):
        X = self.transform_data(new_list)
        pred = self.model.predict(X)
        pred_class = np.argmax(pred, axis=1)[0]  # Get the predicted class for the single sample
        label_dict = {0: 'NEGATIVE', 1: 'NEUTRAL', 2: 'POSITIVE'}
        prediction = label_dict[pred_class]
        return prediction

        
        
        
        

        
        
        
