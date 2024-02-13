from prediction import predic
import time
import numpy as np
from FE import FE
from firebase import DataStorageUtil,UserUtil

class Subscribe:
    """
    A class to subscribe data stream.
    """
    def __init__(self, app_client_id, app_client_secret, user_id, **kwargs):
        self.dataList = []  # Data Form mot
        self.featureObj = FE()  # FE object
        self.predic = predic()  # predic object
        self.user_id = user_id
        self.license=license
        print(license,"-----<")


        print("Subscribe __init__")
        self.c = Cortex(app_client_id, app_client_secret, debug_mode=True, **kwargs)
        self.c.bind(create_session_done=self.on_create_session_done)
        self.c.bind(new_data_labels=self.on_new_data_labels)
        self.c.bind(new_eeg_data=self.on_new_eeg_data)
        self.c.bind(new_mot_data=self.on_new_mot_data)
        self.c.bind(inform_error=self.on_inform_error)
#------
    def start(self, streams, headsetId=''):
       
        self.streams = streams

        if headsetId != '':
            self.c.set_wanted_headset(headsetId)
            
        self.c.open()
#------

    def sub(self, streams):
        """
        To subscribe to one or more data streams
        """
        self.c.sub_request(streams)
#------

    def unsub(self, streams):
        """
        To unsubscribe to one or more data streams
        """
        self.c.unsub_request(streams)
#------        

    def on_new_data_labels(self, *args, **kwargs):
        """
        To handle data labels of subscribed data 
        Returns
        -------
        data: list  
              array of data labels
        name: stream name
        For example:
            eeg: ["COUNTER","INTERPOLATED", "AF3", "T7", "Pz", "T8", "AF4", "RAW_CQ", "MARKER_HARDWARE"]
            motion: ['COUNTER_MEMS', 'INTERPOLATED_MEMS', 'Q0', 'Q1', 'Q2', 'Q3', 'ACCX', 'ACCY', 'ACCZ', 'MAGX', 'MAGY', 'MAGZ']
            dev: ['AF3', 'T7', 'Pz', 'T8', 'AF4', 'OVERALL']
            met : ['eng.isActive', 'eng', 'exc.isActive', 'exc', 'lex', 'str.isActive', 'str', 'rel.isActive', 'rel', 'int.isActive', 'int', 'foc.isActive', 'foc']
            pow: ['AF3/theta', 'AF3/alpha', 'AF3/betaL', 'AF3/betaH', 'AF3/gamma', 'T7/theta', 'T7/alpha', 'T7/betaL', 'T7/betaH', 'T7/gamma', 'Pz/theta', 'Pz/alpha', 'Pz/betaL', 'Pz/betaH', 'Pz/gamma', 'T8/theta', 'T8/alpha', 'T8/betaL', 'T8/betaH', 'T8/gamma', 'AF4/theta', 'AF4/alpha', 'AF4/betaL', 'AF4/betaH', 'AF4/gamma']
        """
        data = kwargs.get('data')
        stream_name = data['streamName']
        stream_labels = data['labels']
       # print('{} labels are : {}'.format(stream_name, stream_labels))
#------

    def on_new_eeg_data(self, *args, **kwargs):
        """
        To handle eeg data emitted from Cortex

        Returns
        -------
        data: dictionary
             The values in the array eeg match the labels in the array labels return at on_new_data_labels
        For example:
        # 9 row + Time 
           {'eeg': [99, 0, 4291.795, 4371.795, 4078.461, 4036.41, 4231.795, 0.0, 0], 'time': 1627457774.5166}
        """
        data = kwargs.get('data')
        print('eeg data: {}'.format(data))
#------

    def on_new_mot_data(self, *args, **kwargs):

        """
        To handle motion data emitted from Cortex
        Returns
        -------
        data: dictionary
             The values in the array motion match the labels in the array labels return at on_new_data_labels
             #12 row + Time 
        For example: {'mot': [33, 0, 0.493859, 0.40625, 0.46875, -0.609375, 0.968765, 0.187503, -0.250004, -76.563667, -19.584995, 38.281834], 'time': 1627457508.2588}
        """

        data = kwargs.get('data')
        print('motion data: {}'.format(data))

        data = [data['time']]+ data['mot'][0:5]
        self.dataList.append(data)
        
        if (len(self.dataList)) >= self.batch_size:
            
            data_batch = self.dataList[:self.batch_size]
            ret, feat_names = self.featureObj.generate_feature_vectors_from_samples(np.array(self.dataList), 150,1., cols_to_ignore = -1)
            ret_2d = ret.reshape(1, -1)
            print ("ret_2d",ret_2d)
            # predic function
            predic = self.predic.predctionVal(ret_2d)
            print ("prediction: ",predic)
            self.store_pred_result(predic)
            del self.dataList[:self.batch_size]      
#------

    def store_pred_result(self,predic):
        DataStorageUtil.store_pred_result(self.user_id, predic)
#------

    def on_create_session_done(self, *args, **kwargs):
        print('on_create_session_done')
        # subribe data 
        self.sub(self.streams)
#------

    def on_inform_error(self, *args, **kwargs):
        error_data = kwargs.get('error_data')
        print(error_data)
#------

    def data_collection_thread(self,subscriber, streams, batch_size):
        subscriber.start(streams)

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            subscriber.unsub(streams)
            subscriber.c.close()
    #------

def main():

    # Please fill your application clientId and clientSecret before running the script
    your_app_client_id = 'ievxpNjBzDtiPKA0HVTVvYDMd1ngVEMIpku5eZ0C'
    your_app_client_secret = 'O8lhYp4nDod4h85WBBUlGK8rKlpjPdVeTesV5HMT8YMoWYxKlEXCupZQEzPKcPEicxMctW7uMpWsJVoc1kiNRBjOSeianPu8WqmAzC7RR48E8uMIvdDe2jfBas3GlvDz'
    license = '095c390f-5469-4e11-920a-8af01b361593'
   
    # Example email to fetch a unique ID
    email = input("email: ")
    user_id = UserUtil.get_user_id_from_email(email)
     
    if user_id:
        print(f"User ID for {email}: {user_id}")
    else:
        print(f"User with email {email} not found.")
   
    s = Subscribe(your_app_client_id, your_app_client_secret, user_id)
    # list data streams
    streams = ['eeg', 'mot', 'met', 'pow']
    # Number of collected records in each file
    batch_size = 30
    s.batch_size = batch_size
    s.data_collection_thread(subscriber=s, streams=streams, batch_size=batch_size)

if __name__ == '__main__':
    main()
