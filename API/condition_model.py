####### importando librerias ########
import torch
import torch.nn.functional as F
import torch.nn as nn
import re
import joblib
import tensorflow as tf

from tensorflow.keras import layers
from keras.preprocessing.sequence import pad_sequences
########### importando archivos y modelos  ###############

#importando el diccionario de contracciones
contraction_dict_path = 'condition_model/contraction_dict.pkl'
contraction_dict = joblib.load(contraction_dict_path)

#importando el tokenizer
tokenizer_path = 'condition_model/tokenizer.pkl'
tokenizer = joblib.load(tokenizer_path)

#importando el encoder 
encoder_path = 'condition_model/encoder.pkl'
encoder = joblib.load(encoder_path)

#importando la matris de embeddigs
embedding_matrix_path = 'condition_model/embedding_matrix.pkl'
embedding_matrix  = joblib.load(embedding_matrix_path)


##################funciones para es preprocesamiento de texto ################

# funcion para pasar de serializer a una lista
def serializertolist(serializer):
  datos = list(serializer.data.values())
  return  datos[1]


#####################importando todo lo necesario para usar el modelo lstm #########################
max_features = 120000
embed_size = 300

#Modelo BiLSTM
class BiLSTM(nn.Module):
    
    def __init__(self):
        #Metodo super
        super(BiLSTM, self).__init__()
        #Tamaño de la capa oculta
        self.hidden_size = 64
        #Taza Dropout
        drp = 0.1
        #Número de clases
        n_classes = len(encoder.classes_)
        #nn.embedding
        self.embedding = nn.Embedding(max_features, embed_size)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False
        #Red BiLSTM
        self.lstm = nn.LSTM(embed_size, self.hidden_size, bidirectional=True, batch_first=True)
        #Capa lineal
        self.linear = nn.Linear(self.hidden_size*4 , 64)
        #Función RELU
        self.relu = nn.ReLU()
        #Capa dropout
        self.dropout = nn.Dropout(drp)
        #output
        self.out = nn.Linear(64, n_classes)

    
    def forward(self, x):
        h_embedding = self.embedding(x)
        h_lstm, _ = self.lstm(h_embedding)
        avg_pool = torch.mean(h_lstm, 1)
        max_pool, _ = torch.max(h_lstm, 1)
        conc = torch.cat(( avg_pool, max_pool), 1)
        conc = self.relu(self.linear(conc))
        conc = self.dropout(conc)
        out = self.out(conc)
        return out


model = BiLSTM()
state_dict = torch.load('condition_model/bilstm_model_state_dict',map_location=torch.device('cpu'))
model.load_state_dict(state_dict)


########################### funciones de preprocesamiento #############################
#Funcion para limpiar text
def clean_text(review): 
   
   review = re.sub('[^a-zA-Z]', ' ', review) # Eliminación de cadenas de strings extrañas
   review = review.lower() #Conversión a minusculas 
   return review



#Función para sustituir contracciones
def _get_contractions(contraction_dict):
    contraction_re = re.compile('(%s)' % '|'.join(contraction_dict.keys()))
    return contraction_dict, contraction_re
contractions, contractions_re = _get_contractions(contraction_dict)

def replace_contractions(text):
    def replace(match):
        return contractions[match.group(0)]
    return contractions_re.sub(replace, text)





def predictcondition(serializer):
    review = list(serializer.data.values())
    # lipieza de texto
    review =  clean_text(review[1])
    # limpieza de contracciones 
    review = replace_contractions(review)
    # tokenizer
    review = tokenizer.texts_to_sequences([review])
    # pad
    review= pad_sequences(review, maxlen=750)
    # creación de torch dataset
    review = torch.tensor(review, dtype=torch.long)
    # predicción
    pred = model(review).detach()
    #funcion softmax
    pred = F.softmax(pred, dim= 1).cpu().numpy()
    #funcion argmax
    pred = pred.argmax(axis=1)

    pred = encoder.classes_[pred[0]]
    return  pred