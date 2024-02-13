import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from firebase_admin import auth

# Initialize Firebase
cred = credentials.Certificate("reactnative-auth-88068-firebase-adminsdk-ohvxs-770559d499.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

class UserUtil:
    @staticmethod
    def get_user_id_from_email(email):
        try:
            # Fetch user by email
            user = auth.get_user_by_email(email)
            return user.uid
        except auth.UserNotFoundError:
            return None

class DataStorageUtil:
    @staticmethod
    def store_pred_result(user_id, predic):
        users_ref = db.collection('Users').document(user_id)
        predicDocument = users_ref.collection('Prediction').document()
        prdic_data = {
            'Prediction_result': predic,
            'timestamp': firestore.SERVER_TIMESTAMP
        }
        predicDocument.set(prdic_data)