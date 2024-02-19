import seaborn as sns
import numpy as np
import pandas as pd
import pickle
from keras.models import model_from_json
from sklearn.preprocessing import StandardScaler

'''that's sample of the data I get

2024-02-17 19:09:44.457304,820.3662719726562,739.7802124023438,813.113525390625,797.3992919921875,750.6593627929688
2024-02-17 19:09:44.457485,825.2014770507812,612.8571166992188,807.069580078125,803.8461303710938,826.8131713867188
2024-02-17 19:09:44.457629,772.0146484375,116.84981536865234,795.3846435546875,806.2637329101562,997.6557006835938
2024-02-17 19:09:44.457807,749.047607421875,587.87548828125,801.025634765625,791.3552856445312,857.032958984375
2024-02-17 19:09:44.458003,796.1904907226562,896.1171875,812.7106323242188,785.3113403320312,750.2564086914062 (Timestamp, channel 1, channel 2, channel 3, channel 4, channel 5)'''
class predic():
    
    
    def __init__(self, pathjs = 'BiLSTM.json', path_weights='BiLSTM.h5'):
        self.model = self.load_model(pathjs, path_weights)
        self.scaler = StandardScaler()
    
 #------   
 
    def Transform_data(self,newList):
        X = self.scaler.transform(newList)
        return X
#------

    def load_model(self,pathjs, path_weights):
        # loading model
        model = model_from_json(open(pathjs).read())
        model.load_weights(path_weights)
        model.compile(loss='categorical_crossentropy', optimizer='adam')
        return model
#------
# function to take the data (1. send it to transform 2. send it to prediction 3. return the preduction)   

    def predctionVal (self, newList):
        # convert NaN values to 0.0
        new = np.array(newList)
        new[np.isnan(new)] = 0.0
        newList = new
        # transform the data using the scaler
        X = self.Transform_data(newList)
        # make a prediction using the model
        pred = self.model.predict(X)
        pred1 = np.argmax(pred, axis=1)
        # encoding
        label_dict= {0:'NEGATIVE', 1:'NEUTRAL', 2:'POSITIVE'}
        prediction = label_dict[int(pred1)]
        return prediction
        
        
        
        
        

        
        
        
