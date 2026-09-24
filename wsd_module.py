# File: wsd_module.py

import os
import warnings

# Suppress TensorFlow log noise and Keras deprecation warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore')

import numpy as np
import nltk
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

class WSDGRUClassifier:
    """
    Requirement 10: Word Sense Disambiguation using GRU RNN.
    Disambiguates polysemous words (e.g., 'bank') based on context.
    """
    def __init__(self, target_word="bank", max_len=20, vocab_size=1000):
        self.target_word = target_word
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.tokenizer = Tokenizer(num_words=vocab_size, oov_token="<UNK>")
        self.model = None

    def _build_dataset(self):
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
        
        # Sense 0: Financial Institution
        financial_examples = [
            "The central bank increased interest rates to lower inflation.",
            "I deposited my money into a savings account at the commercial bank.",
            "Commercial banks offer financial loans to small business owners.",
            "The investment bank managed corporate shares and initial public offerings.",
            "He works as a teller at the national bank branch.",
            "The bank approved a mortgage loan for buying a new house.",
            "Financial reserves in the central bank stabilized the national currency.",
            "Customers withdrew cash from the local bank automated teller."
        ]
        
        # Sense 1: River Shore / Geological Bank
        river_examples = [
            "Heavy rain caused the steep river bank to overflow into nearby fields.",
            "We walked along the muddy bank of the winding river.",
            "The fishing boat anchored safely near the grassy river bank.",
            "Floodwaters eroded the soil along the stream bank.",
            "Children played on the sandy river bank during warm summer afternoons.",
            "Water levels rose rapidly threatening communities on the river bank.",
            "Trees growing along the river bank prevent natural soil erosion.",
            "The river overflowed its bank and flooded low lying coastal valleys."
        ]

        texts = financial_examples + river_examples
        labels = np.array([0] * len(financial_examples) + [1] * len(river_examples))

        definitions = (
            "Financial Institution (accepts deposits, grants loans, manages money)",
            "River Shore / Sloping Ground (land bordering a body of water)"
        )
        return texts, labels, definitions

    def train_model(self, epochs=60):
        texts, labels, definitions = self._build_dataset()
        
        self.tokenizer.fit_on_texts(texts)
        sequences = self.tokenizer.texts_to_sequences(texts)
        X = pad_sequences(sequences, maxlen=self.max_len, padding='post')

        # Build GRU Network (compatible with Keras 3 / TensorFlow 2.x)
        self.model = Sequential([
            Embedding(input_dim=self.vocab_size, output_dim=16),
            GRU(16, return_sequences=False),
            Dropout(0.2),
            Dense(8, activation='relu'),
            Dense(1, activation='sigmoid')
        ])

        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        self.model.fit(X, labels, epochs=epochs, verbose=0)
        return definitions

    def disambiguate(self, context_sentence):
        if not self.model:
            raise RuntimeError("Model must be trained before running disambiguation.")

        seq = self.tokenizer.texts_to_sequences([context_sentence])
        padded = pad_sequences(seq, maxlen=self.max_len, padding='post')
        probability = float(self.model.predict(padded, verbose=0)[0][0])
        
        sense_id = 1 if probability >= 0.5 else 0
        confidence = probability if sense_id == 1 else (1.0 - probability)
        return sense_id, confidence