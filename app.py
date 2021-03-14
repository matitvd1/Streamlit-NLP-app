import numpy as np
import pandas as pd
import pickle
import streamlit as st
import re
import nltk
import sklearn
import textacy.preprocessing as tprep

from PIL import Image
from unicodedata import normalize
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.linear_model import LogisticRegression
from nltk.stem.snowball import SnowballStemmer 

nltk.download('punkt')

tf_idf = pickle.load(open("tf_idf.pkl", "rb"))
lr_model = pickle.load(open("lr_model.pkl", "rb"))

with open('stopwords-es.txt') as f:
  stopwords = f.read().splitlines()
stopwords.remove('nada')
stopwords.remove('bueno')
stopwords.remove('no')
stopwords.remove('si')
stopwords.remove('tan')
stopwords.remove('tiempo')
stopwords.remove('buen')
stopwords.remove('estado')

def clean(text):
  text = re.sub(r'<[^<>]*>', ' ', text)
  text = re.sub(r'\[([^\[\]]*)\]\([^\(\)]*\)', r'\1', text)
  text = ''.join([i for i in text if i not in ['[',']']])
  text = re.sub(r'\[[^\[\]]*\]', ' ', text)
  text = re.sub(r'(?:^|\s)[&#<>{}\[\]+|\\:-]{1,}(?:\s|$)', ' ', text)
  text = re.sub(r'(?:^|\s)[\-=\+]{2,}(?:\s|$)', ' ', text)
  text = re.sub(r'\s+', ' ', text)
  return text.strip()

def no_accent_marks(row):
  s = row
  trans_tab = dict.fromkeys(map(ord, u'\u0301\u0308'), None)
  s = normalize('NFKC', normalize('NFKD', s).translate(trans_tab))
  return s

def normalize_text(text):
  text = tprep.normalize_hyphenated_words(text)
  text = tprep.normalize_quotation_marks(text)
  text = tprep.normalize_unicode(text)
  return text

def token_analysis(df_column):
  token = word_tokenize(df_column)
  return token

def remove_stopwords(row):
  tokens = [word for word in row if word not in stopwords]
  return tokens 

def stemma_doc(text):
  stemmer = SnowballStemmer("spanish")
  stemmatizer = []
  for sentence in sent_tokenize(text):
    stemma_lst = []
    text =  re.sub(r"[`'\",.!?():]", " ", sentence).replace('-',' ').replace('/', '')
    text = re.sub(r'\s+', ' ', text)
    doc = word_tokenize(normalize_text(text))
    for token in doc:
      stemma_lst.append(stemmer.stem(token)) 
    stemmatizer.append(' '.join(stemma_lst))
  return ' '.join(stemmatizer)

def clean_text(review):
  input_review = pd.Series(review).apply(lambda x: ' '.join(['la' if i == 'ka' else 'que' if i == 'q' else 
  'con' if i == 'co' else i for i in word_tokenize(x)]))
  input_review = input_review.str.replace(pat=r"\&\#[0-9]+\;", repl="", regex=True)
  input_review = input_review.apply(clean)
  input_review = input_review.str.lower()
  input_review = input_review.apply(no_accent_marks)
  input_review = input_review.apply(lambda x: re.sub(r"[`'\",.!?():]", " ", x).replace('-',' ').replace('/', ''))
  input_review = input_review.apply(lambda x: re.sub(r'\s+', ' ', x))
  input_review = input_review.apply(token_analysis)
  input_review = input_review.apply(remove_stopwords)
  input_review = input_review.apply(lambda x: ' '.join(x))
  input_review = input_review[0].strip()  
  input_review = stemma_doc(input_review)
  return pd.Series(input_review)

@st.cache(suppress_st_warning=True)
def sentiment_prediction(review):
  review = clean_text(review)
  vector = tf_idf.transform(review)
  sentiment = (lr_model.predict_proba(vector)[:, 1] >= 0.45).astype(int)[0]
  if sentiment == 0:
    pred="Negative"
  else:
    pred= "Positive"
  return pred



def run():
  st.title("Binary NLP classification app")

  image = Image.open('Image.png')
  st.image(image)
    
  html_temp = """
  """
  st.markdown(html_temp)  
  
  review = st.text_input("Enter the Review in Spanish:")
  prediction = ""

  if st.button("Predict Sentiment"):
    prediction = sentiment_prediction(review)
  st.success("The sentiment predicted by Model : {}".format(prediction))
  

if __name__=='__main__':
    run()