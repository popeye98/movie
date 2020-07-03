import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from string import punctuation
import pickle
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
stop=stopwords.words('english')
MAX_NB_WORDS=50000
EMBEDDING_DIM =300
MAX_SEQUENCE_LENGTH=868
import re
from nltk.stem import WordNetLemmatizer 
from nltk.tokenize import word_tokenize 
lemmatizer=WordNetLemmatizer()
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model, load_model


with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)


from keras.models import model_from_json

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

loaded_model.load_weights('base_model_word2vec_lstm.h5')
loaded_model._make_predict_function()
with open('mlb_class.pickle', 'rb') as handle:
    pp = pickle.load(handle)


def resize_img(path):
  try:
    img=cv2.imread(path)
    img=cv2.resize(img,(75,115))
    img=img.astype(np.float32)/255
    plt.imshow(img)
    return img
  except Exception as e:
    print(str(e))
    return None

def clean_text(text):
  text=text.translate(str.maketrans('', '', punctuation))  #   removes all punctuation
  text=text.lower().strip()
  text = ' '.join([i if i not in stop and i.isalpha() else '' for i in text.lower().split()])
  text=' '.join([lemmatizer.lemmatize(w) for w in word_tokenize(text)])
  text=re.sub(r"\s{2,}"," ",text)         #filteration of length
  return text

def find_genre(image,text):
        
    val_imgs=resize_img(image)
    val_np_imgs = np.array(val_imgs)
    val_np_imgs=np.expand_dims(val_np_imgs,axis=0)
    val_np_imgs.shape

    inp=clean_text(text)
    X_test=tokenizer.texts_to_sequences(inp)
    X_test=pad_sequences(X_test,maxlen=MAX_SEQUENCE_LENGTH)

    y_pred=loaded_model.predict([val_np_imgs,X_test])
    out=y_pred
    y_pred = np.zeros(out.shape)
    y_pred[out>0.3]=1
    y_pred = np.array(y_pred)
    ans=np.array(y_pred[0].astype('int32'))
    
    l=[]
    for i,k in enumerate(ans):
        if k==1:
      
            l.append(pp[i])
    return l,out

