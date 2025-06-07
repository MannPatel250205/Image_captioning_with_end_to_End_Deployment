import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
from Model.entity.config_entity import PrepareBaseModelConfig
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout, add
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical, plot_model
import numpy as np
from dataclasses import dataclass
from pathlib import Path

class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config
    
    def get_base_model(self):
        self.vgg_model = VGG16()
        self.vgg_model = Model(
            inputs= self.vgg_model.inputs,             
            outputs= self.vgg_model.layers[-2].output
        )

        self.save_model(path=self.config.base_model_path, model=self.vgg_model)
    


    @staticmethod
    def _prepare_full_model(max_len, vocab_size):

        inputs1 = Input(shape=(4096,), name="image")
        fe1 = Dropout(0.4)(inputs1)
        fe2 = Dense(256, activation='relu')(fe1)

        inputs2 = Input(shape=(max_len,), name="text")
        se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
        se2 = Dropout(0.4)(se1)
        se3 = LSTM(256)(se2)

        decoder1 = add([fe2, se3])
        decoder2 = Dense(256, activation='relu')(decoder1)
        outputs = Dense(vocab_size, activation='softmax')(decoder2)
        full_model = Model(inputs=[inputs1, inputs2], outputs=outputs)
        full_model.compile(loss='categorical_crossentropy', optimizer='adam')

        full_model.summary()
        return full_model
    
    def update_base_model(self):
        self.full_model = self._prepare_full_model(
            max_len=35,
            vocab_size=8485
        )

        self.save_model(path=self.config.updated_base_model_path, model=self.full_model)

    
    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)