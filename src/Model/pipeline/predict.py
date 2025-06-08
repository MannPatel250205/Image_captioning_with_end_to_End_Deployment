import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
from Model import logger
import pickle




class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename


    def idx_to_word(self, integer, tokenizer):
        for word, index in tokenizer.word_index.items():
            if index == integer:
                return word
        return None
    

    def predict_caption(self, max_length):
        # add start tag for generation process
        in_text = 'startseq'
        for i in range(max_length):
            # encode input sequence
            sequence = self.tokenizer.texts_to_sequences([in_text])[0]
            # pad the sequence
            sequence = pad_sequences([sequence], max_length, padding='post')
            # predict next word
            yhat = self.model.predict([self.feature, sequence], verbose=0)
            # get index with high probability
            yhat = np.argmax(yhat)
            # convert index to word
            word = self.idx_to_word(yhat, self.tokenizer)
            # stop if word not found
            if word is None:
                break
            in_text += " " + word
            # stop if we reach end tag
            if word == 'endseq':
                break
        return in_text

    

    def predict(self):
        # load model
        self.model = load_model(os.path.join("artifacts","training", "best_model.keras"))
        vgg_model = load_model(os.path.join("artifacts","prepare_base_model", "VGG_model.keras"))
        image_path = self.filename
        image = load_img(image_path, target_size=(224, 224))
        image = img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = preprocess_input(image)
        self.feature = vgg_model.predict(image, verbose = 0)


        with open(os.path.join("artifacts/training/", 'tokenize.pkl'), 'rb') as f:
            self.tokenizer = pickle.load(f)
        

        result = self.predict_caption(35)
        result = result.replace('startseq', '').replace('endseq', '').strip()
        return result
    
if __name__ == "__main__":
    try:
        logger.info(f">>>>>> stage Prediction started <<<<<<")
        pipeline = PredictionPipeline("test_image.jpg")
        caption = pipeline.predict()
        logger.info(f"Predicted Caption: {caption}")
    except Exception as e:
        logger.exception(e)
        raise e