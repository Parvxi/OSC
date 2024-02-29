from datetime import datetime
import numpy as np
from pythonosc import dispatcher, osc_server
from FE import FE
from prediction import predic

class EEGProcessor:

    def __init__(self, featureObj, predic, batch_size=10):
        self.dataList = []
        self.featureObj = featureObj
        self.predic = predic
        self.batch_size = batch_size

    def on_new_eeg_data(self, address: str, *args):
        """
        To handle EEG data emitted from the OSC server
        """
        dateTimeObj = datetime.now()
        printStr = dateTimeObj.strftime("%Y-%m-%d %H:%M:%S.%f")
        for arg in args:
            printStr += "," + str(arg)
        print(printStr)
        data = list(args)
        data = [dateTimeObj.timestamp()] + data[1:6]  
        self.dataList.append(data)

        if len(self.dataList) >= self.batch_size:
            data_batch = self.dataList[:self.batch_size]
            ret, feat_names = self.featureObj.generate_feature_vectors_from_samples(np.array(self.dataList), 150, 1., cols_to_ignore=-1)
            print(feat_names)
            ret_2d = ret.reshape(1, -1)
            print("ret_2d", ret_2d)
            # Prediction function
            prediction = self.predic.predictionVal(ret_2d)
            print("prediction:", predic)
            del self.dataList[:self.batch_size]

if __name__ == "__main__":
    ip = "0.0.0.0"
    port = 5000

    # Initialize your featureObj and predic objects
    featureObj = FE()
    predic_obj = predic()

    eeg_processor = EEGProcessor(featureObj, predic_obj)

    dispatcher = dispatcher.Dispatcher()
    dispatcher.map("/muse/eeg", eeg_processor.on_new_eeg_data)

    server = osc_server.ThreadingOSCUDPServer((ip, port), dispatcher)
    print("Listening on UDP port " + str(port))
    server.serve_forever()


    # from datetime import datetime
import numpy as np
from pythonosc import dispatcher, osc_server
from FE import FE
from prediction import predic

class EEGProcessor:
# 03070648714
    def __init__(self, featureObj, predic, batch_size=10):
        self.dataList = []
        self.featureObj = featureObj
        self.predic = predic
        self.batch_size = batch_size

    def on_new_eeg_data(self, address: str, *args):
        """
        To handle EEG data emitted from the OSC server
        """
        dateTimeObj = datetime.now()
        printStr = dateTimeObj.strftime("%Y-%m-%d %H:%M:%S.%f")
        for arg in args:
            printStr += "," + str(arg)
        print(printStr)
        data = list(args)
        timeStamp = [dateTimeObj.timestamp()]
        data = data[0:5]  
        print("THE DATATAA", data)
        self.dataList.append(data)

        if len(self.dataList) >= self.batch_size:
            
            data_batch = self.dataList[:self.batch_size]
            ret, feat_names = self.featureObj.generate_feature_vectors_from_samples(np.array(self.dataList), 150,1., cols_to_ignore = -1)
            print('The Features are: ', feat_names)
            ret_2d = ret.reshape(1, -1)
            print("The vector is: ", ret_2d)
            # Prediction function
            prediction = self.predic.predictionVal(ret_2d)
            print("prediction:", predic)
            del self.dataList[:self.batch_size]

if __name__ == "__main__":
    ip = "0.0.0.0"
    port = 5000

    # Initialize your featureObj and predic objects
    featureObj = FE()
    predic_obj = predic()

    eeg_processor = EEGProcessor(featureObj, predic_obj)

    dispatcher = dispatcher.Dispatcher()
    dispatcher.map("/muse/eeg", eeg_processor.on_new_eeg_data)

    server = osc_server.ThreadingOSCUDPServer((ip, port), dispatcher)
    print("Listening on UDP port " + str(port))
    server.serve_forever() #




    '''from datetime import datetime
import numpy as np
from pythonosc import dispatcher, osc_server
from FE import FE
from prediction import predic

class EEGProcessor:
# 03070648714
    def __init__(self, featureObj, predic, batch_size=10):
        self.dataList = []
        self.featureObj = featureObj
        self.predic = predic
        self.batch_size = batch_size

    def on_new_eeg_data(self, address: str, *args):
        """
        To handle EEG data emitted from the OSC server
        """
        dateTimeObj = datetime.now()
        printStr = dateTimeObj.strftime("%Y-%m-%d %H:%M:%S.%f")
        for arg in args:
            printStr += "," + str(arg)
        print(printStr)
        data = list(args)
        timeStamp = [dateTimeObj.timestamp()]
        print("THE DATATAA", data)
        data = data[1:]  
        self.dataList.append(data)

        if len(self.dataList) >= self.batch_size:
            
            data_batch = self.dataList[:self.batch_size]
            ret, feat_names = self.featureObj.generate_feature_vectors_from_samples(np.array(self.dataList), 150,1., cols_to_ignore = -1)
            print('The Features are: ', feat_names)
            ret_2d = ret.reshape(1, -1)
            print("The vector is: ", ret_2d)
            # Prediction function
            prediction = self.predic.predictionVal(ret_2d)
            print("prediction:", predic)
            del self.dataList[:self.batch_size]

if __name__ == "__main__":
    ip = "0.0.0.0"
    port = 5000

    # Initialize your featureObj and predic objects
    featureObj = FE()
    predic_obj = predic()

    eeg_processor = EEGProcessor(featureObj, predic_obj)

    dispatcher = dispatcher.Dispatcher()
    dispatcher.map("/muse/eeg", eeg_processor.on_new_eeg_data)

    server = osc_server.ThreadingOSCUDPServer((ip, port), dispatcher)
    print("Listening on UDP port " + str(port))
    server.serve_forever()'''

