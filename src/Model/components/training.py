import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
import time
from tqdm import tqdm
import pickle
from pathlib import Path
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical, plot_model
from Model.entity.config_entity import TrainingConfig
import numpy as np




class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config
    
    def get_base_model(self):
        self.model = tf.keras.models.load_model(
            self.config.updated_base_model_path
        )
    

    def cleaning(self, mapping):

        for key, captions in mapping.items():
            for i in range(len(captions)):
                caption = captions[i]
                caption = caption.lower()
                caption = caption.replace('[^A-Za-z]', '')
                caption = caption.replace('\s+', ' ')
                caption = 'startseq ' + " ".join([word for word in caption.split() if len(word)>1]) + ' endseq'
                captions[i] = caption
    

    def train_valid_generator(self):
        with open(os.path.join("artifacts/data_ingestion/", 'features.pkl'), 'rb') as f:
            self.features = pickle.load(f)
        
        with open(os.path.join("", 'captions.txt'), 'r') as f:
            next(f)
            captions_doc = f.read()
        
        self.mapping = {}

        for line in tqdm(captions_doc.split('\n')):
            # split the line by comma(,)
            tokens = line.split(',')
            if len(line) < 2:
                continue
            image_id, caption = tokens[0], tokens[1:]
            # remove extension from image ID
            image_id = image_id.split('.')[0]

            caption = " ".join(caption)
            # create list if needed
            if image_id not in self.mapping:
                self.mapping[image_id] = []
            # store the caption
            self.mapping[image_id].append(caption)
        
        self.cleaning(self.mapping)

        all_captions = []
        for key in self.mapping:
            for caption in self.mapping[key]:
                all_captions.append(caption)
        
        self.tokenizer = Tokenizer()
        self.tokenizer.fit_on_texts(all_captions)
        self.vocab_size = len(self.tokenizer.word_index) + 1

        pickle.dump(self.tokenizer, open(os.path.join("artifacts/training/", 'tokenize.pkl'), 'wb'))
    

    def data_generator(self, data_keys, mapping, features, tokenizer, max_length, vocab_size, batch_size):
        X1, X2, y = list(), list(), list()
        n = 0
        while 1:
            for key in data_keys:
                n += 1
                captions = mapping[key]
                for caption in captions:
                    seq = tokenizer.texts_to_sequences([caption])[0]
                    for i in range(1, len(seq)):
                        in_seq, out_seq = seq[:i], seq[i]
                        in_seq = pad_sequences([in_seq], maxlen=max_length, padding='post')[0]
                        out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                        X1.append(features[key][0])
                        X2.append(in_seq)
                        y.append(out_seq)
                if n == batch_size:
                    X1, X2, y = np.array(X1), np.array(X2), np.array(y)
                    yield {"image": X1, "text": X2}, y
                    X1, X2, y = list(), list(), list()
                    n = 0
        

        

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)


    def train(self, callback_list: list):

        image_ids = list(self.mapping.keys())
        split = int(len(image_ids) * 0.90)
        train = image_ids[:split]
        self.steps_per_epoch = len(train) // self.config.params_batch_size

        self.model.fit(
            self.data_generator(
                data_keys=train,
                mapping=self.mapping,
                features=self.features,
                tokenizer=self.tokenizer,
                max_length=35,
                vocab_size=self.vocab_size,
                batch_size=self.config.params_batch_size
            ),
            epochs=1,
            verbose=1,
            steps_per_epoch=self.steps_per_epoch,
            callbacks=callback_list
        )

        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )