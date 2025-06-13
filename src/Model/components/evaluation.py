import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
import time
from tqdm import tqdm
import pickle
from pathlib import Path
from Model.entity.config_entity import EvaluationConfig
from Model.utils.common import save_json
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical, plot_model
import numpy as np
from nltk.translate.bleu_score import corpus_bleu
from urllib.parse import urlparse



class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config

    def idx_to_word(self, integer):
        for word, index in self.tokenizer.word_index.items():
            if index == integer:
                return word
        return None
    
    def predict_caption(self, image, tokenizer, max_length):
        in_text = 'startseq'
        for i in range(max_length):
            # encode input sequence
            sequence = tokenizer.texts_to_sequences([in_text])[0]
            # pad the sequence
            sequence = pad_sequences([sequence], max_length, padding='post')
            # predict next word
            yhat = self.model.predict([image, sequence], verbose=0)
            # get index with high probability
            yhat = np.argmax(yhat)
            # convert index to word
            word = self.idx_to_word(yhat)

            if word is None:
                break
            in_text += " " + word
            if word == 'endseq':
                break
        return in_text
    

    def cleaning(self, mapping):

        for key, captions in mapping.items():
            for i in range(len(captions)):
                caption = captions[i]
                caption = caption.lower()
                caption = caption.replace('[^A-Za-z]', '')
                caption = caption.replace('\s+', ' ')
                caption = 'startseq ' + " ".join([word for word in caption.split() if len(word)>1]) + ' endseq'
                captions[i] = caption

    
    def _valid_generator(self):
        with open(os.path.join("artifacts/data_ingestion/", 'features.pkl'), 'rb') as f:
            self.features = pickle.load(f)

        with open(os.path.join("artifacts/training/", 'tokenize.pkl'), 'rb') as f:
            self.tokenizer = pickle.load(f)


        actual, predicted = list(), list()

        with open(os.path.join("", 'captions.txt'), 'r') as f:
            next(f)
            captions_doc = f.read()

        
        self.mapping = {}

        for line in tqdm(captions_doc.split('\n')):
            tokens = line.split(',')
            if len(line) < 2:
                continue
            image_id, caption = tokens[0], tokens[1:]

            image_id = image_id.split('.')[0]

            caption = " ".join(caption)

            if image_id not in self.mapping:
                self.mapping[image_id] = []
            
            self.mapping[image_id].append(caption)
        
        self.cleaning(self.mapping)


        image_ids = list(self.mapping.keys())
        split = int(len(image_ids) * 0.90)
        test = image_ids[split:]


        for key in tqdm(test):
            captions = self.mapping[key]

            y_pred = self.predict_caption(self.features[key], self.tokenizer, 35)
            # Split into words
            actual_captions = [caption.split() for caption in captions]
            y_pred = y_pred.split()
            # Append to the lists
            actual.append(actual_captions)
            predicted.append(y_pred)

        self.bleu1 = corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0))
        self.blue2 = corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0))
        
    
    
    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        return tf.keras.models.load_model(path)
    

    def evaluation(self):
        self.model = self.load_model(self.config.path_of_model)
        self._valid_generator()

    
    def save_score(self):
        scores = {"BLEU1": self.bleu1, "BLEU2": self.blue2}
        save_json(path=Path("scores.json"), data=scores)