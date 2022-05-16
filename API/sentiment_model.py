###### librerias ######
import joblib
import numpy as np

###Importan
sentiment_model_path = 'sentiment_model/modelo.pkl'
sentiment_model = joblib.load(sentiment_model_path)



############# funciones #############
def serializertoarray(serializer):
  datos = list(serializer.data.values())
  return  datos[1]

def sentiment_prediction(review):
    predictions = sentiment_model.predict(np.array([review]))
    if predictions >= 0:
        return 'Positivo'
    else:
        return 'Negativo'