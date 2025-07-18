'''
set for V1 right now, to use V3, please use train_V3
run all cells except the train loop
and load model from disk (price_predictor_V3.keras)
'''

import numpy as np
import tensorflow as tf
import requests
from io import BytesIO
from PIL import Image, ImageOps

import pandas as pd
from tensorflow_text.python.ops.fast_wordpiece_tokenizer import FastWordpieceTokenizer
from keras_nlp.layers import StartEndPacker
from tensorflow_text import normalize_utf8

from tensorflow.keras.utils import register_keras_serializable

class CustomTokenizer():
    def __init__(self, vocab_path, max_length):
        self.packer = StartEndPacker(sequence_length=max_length,pad_value=0)
        self.unk_token = '[UNK]'
        self.vocabulary = self._get_vocab_list(vocab_path)
        self.tokenizer = FastWordpieceTokenizer(
            vocab=self.vocabulary,
            suffix_indicator='##',
            unknown_token=self.unk_token,
            support_detokenization=True
        )

    @staticmethod
    def _preprocess(text):
        """Strip accent and lower case the text"""
        text_normalized = normalize_utf8(text, "NFD")
        text_stripped_accents = tf.strings.regex_replace(text_normalized, r"\\p{Mn}", "")
        lowercase = tf.strings.lower(text_stripped_accents)
        return lowercase

    def _get_vocab_list(self, vocab_path):
        vclist = []

        with open(vocab_path, "r") as f:
            vclist.extend(f.read().splitlines())
            seen = set()
            vclist = [x for x in vclist if not (x in seen or seen.add(x))]

        if self.unk_token not in vclist:
            vclist = [vclist[0]] + [self.unk_token] + vclist[1:]

        assert len(list(set(vclist))) == len(vclist), "Duplicate vocab entries"
        return vclist

    def tokenize(self, text):
        text = self._preprocess(text)
        tokens = self.tokenizer.tokenize(text)
        return self.packer(tokens)

    def detokenize(self, tokens):
        return self.tokenizer.detokenize(tokens)

    def __call__(self, text):
        return self.tokenize(text)

def cyclical_encode(value, max_value):
    sin_val = round(np.sin(2 * np.pi * value / max_value), 2)
    cos_val = round(np.cos(2 * np.pi * value / max_value), 2)
    return sin_val, cos_val

def resize_and_pad_image(url, target_size=(320, 320)):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content)).convert("RGB")
    img.thumbnail(target_size, Image.Resampling.LANCZOS)
    
    delta_w = target_size[0] - img.width
    delta_h = target_size[1] - img.height
    padding = (
        delta_w // 2, 
        delta_h // 2, 
        delta_w - (delta_w // 2), 
        delta_h - (delta_h // 2)
        )
    
    padded_img = ImageOps.expand(img, padding, fill=(0, 0, 0))
    
    return padded_img

def normalize_image(pil_image):
    img_array = np.array(pil_image).astype(np.float32) / 255.0
    img_tensor = tf.convert_to_tensor(img_array)
    img_tensor = tf.expand_dims(img_tensor, axis=0)  
    return img_tensor

def process_image(url):
    def _load_and_process(url_str):
        url_decoded = url_str.numpy().decode()
        img = resize_and_pad_image(url_decoded)
        img_tensor = normalize_image(img)
        return img_tensor[0]  # remove batch dim

    img = tf.py_function(func=_load_and_process, inp=[url], Tout=tf.float32)
    img.set_shape([320, 320, 3])
    return img

@register_keras_serializable()
class CNNImageEncoder(tf.keras.Model):
    def __init__(self, activation='relu', kernel_size=(3, 3), pool_size=(2, 2), **kwargs):
        super(CNNImageEncoder, self).__init__(**kwargs)

        self.cnn_layers = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, kernel_size, activation=activation, input_shape=(320, 320, 3)),
            tf.keras.layers.MaxPooling2D(pool_size, strides=(2, 2)),
            tf.keras.layers.Conv2D(64, kernel_size, activation=activation),
            tf.keras.layers.MaxPooling2D(pool_size, strides=(2, 2)),
            tf.keras.layers.Conv2D(128, kernel_size, activation=activation),
            tf.keras.layers.MaxPooling2D(pool_size, strides=(2, 2)),
            tf.keras.layers.Conv2D(256, kernel_size, activation=activation),
            tf.keras.layers.MaxPooling2D(pool_size, strides=(2, 2)),
            tf.keras.layers.Flatten(),
        ])

    def call(self, image_inputs):
        return self.cnn_layers(image_inputs)
    
    def get_config(self):
        config = super(CNNImageEncoder, self).get_config()
        config.update({
            "activation": self.activation,
            "kernel_size": self.kernel_size,
            "pool_size": self.pool_size,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
DROPOUT = 0.1

@register_keras_serializable()
class PricePredictor(tf.keras.Model):
    def __init__(self, title_vocab_size=434, desc_vocab_size=2321,**kwargs):
        super().__init__(**kwargs)

        self.title_vocab_size = title_vocab_size
        self.desc_vocab_size = desc_vocab_size
        
        self.image_encoder = CNNImageEncoder()
        
        self.title_embedding_layer = tf.keras.layers.Embedding(title_vocab_size, 128, mask_zero=True)
        self.desc_embedding_layer = tf.keras.layers.Embedding(desc_vocab_size, 128, mask_zero=True)
        
        self.title_dense_layers = tf.keras.Sequential([
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(DROPOUT)
        ])

        self.desc_dense_layers = tf.keras.Sequential([
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(DROPOUT)
        ])
        
        self.final_layers = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(DROPOUT),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(DROPOUT),
            tf.keras.layers.Dense(1)
        ])
    
    def call(self, inputs):
        image_input = inputs["image_url"]
        title_tokens = inputs["title"]
        desc_tokens = inputs["description"]
        
        image_features = self.image_encoder(image_input)
        
        title_embeddings = self.title_embedding_layer(title_tokens)
        title_features = self.title_dense_layers(title_embeddings)

        desc_embeddings = self.desc_embedding_layer(desc_tokens)
        desc_features = self.desc_dense_layers(desc_embeddings)

        other_features = tf.keras.layers.concatenate([
            tf.expand_dims(inputs["condition_BRAND NEW"], axis=1),
            tf.expand_dims(inputs["condition_HEAVILY USED"], axis=1),
            tf.expand_dims(inputs["condition_LIGHTLY USED"], axis=1),
            tf.expand_dims(inputs["condition_LIKE NEW"], axis=1),
            tf.expand_dims(inputs["condition_WELL USED"], axis=1),
            tf.expand_dims(inputs["year"], axis=1),
            tf.expand_dims(inputs["month_sin"], axis=1),
            tf.expand_dims(inputs["month_cos"], axis=1),
            tf.expand_dims(inputs["day_of_week_sin"], axis=1),
            tf.expand_dims(inputs["day_of_week_cos"], axis=1),
        ])

        concatenated_features = tf.keras.layers.concatenate([
            image_features,
            title_features,
            desc_features,
            other_features
        ])
        
        return self.final_layers(concatenated_features)

    def get_config(self):
        config = super(PricePredictor, self).get_config()
        config.update({
            "title_vocab_size": self.title_vocab_size,
            "desc_vocab_size": self.desc_vocab_size,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

def get_condition_features(condition_str):
    all_conditions = ["BRAND NEW", "HEAVILY USED", "LIGHTLY USED", "LIKE NEW", "WELL USED"]
    cond_dict = {f"condition_{cond}": 0 for cond in all_conditions}
    if condition_str.upper() in [c.upper() for c in all_conditions]:
        cond_dict[f"condition_{condition_str.upper()}"] = 1
    return cond_dict

def prepare_inference_inputs(title, description, image_url, date_sold_str, condition_str, title_tokenizer, desc_tokenizer):
    dt = pd.to_datetime(date_sold_str)
    
    # Cyclical features
    structured_features = {}
    structured_features["year"] = dt.year - 2024
    structured_features["month_sin"], structured_features["month_cos"] = cyclical_encode(dt.month, 12)
    structured_features["day_of_week_sin"], structured_features["day_of_week_cos"] = cyclical_encode(dt.dayofweek, 7)
    
    # Condition features
    cond_features = get_condition_features(condition_str)
    structured_features.update(cond_features)
    
    tokenized = {
        "title": tf.convert_to_tensor(title_tokenizer(tf.constant(title))),
        "description": tf.convert_to_tensor(desc_tokenizer(tf.constant(description)))
    }
    
    img_tensor = process_image(tf.constant(image_url))

    structured_tensors = {
        key: tf.expand_dims(tf.convert_to_tensor(value, dtype=tf.float32), axis=0)
        for key, value in structured_features.items()
    }
    
    model_inputs = {
        "image_url": tf.expand_dims(img_tensor, axis=0),
        "title": tf.expand_dims(tokenized["title"], axis=0),
        "description": tf.expand_dims(tokenized["description"], axis=0),
        **structured_tensors
    }
    
    return model_inputs