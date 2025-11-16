# -*- coding: utf-8 -*-
"""
Шаг 9. Нейросетевая модель классификации Telegram-сообщений.
Используем keras: Embedding -> Dense -> Softmax.
"""

import numpy as np
import pandas as pd
import re, os
import nltk
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import utils
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding, Flatten
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# --- загрузка данных ---
data = pd.read_csv("data/processed/dataset.csv").dropna(subset=["text_lemma"])
texts = data["text_lemma"].astype(str).tolist()
labels = data["category"].astype(str).tolist()

# --- кодирование меток ---
le = LabelEncoder()
y = le.fit_transform(labels)
y_cat = utils.to_categorical(y)

# --- параметры токенизации ---
VOCAB_SIZE = 20000
MAX_LEN = 80

tokenizer = Tokenizer(num_words=VOCAB_SIZE)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
X = pad_sequences(sequences, maxlen=MAX_LEN)

# --- разбиение ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y_cat, test_size=0.2, stratify=y_cat, random_state=42
)

# --- архитектура модели ---
model = Sequential([
    Embedding(input_dim=VOCAB_SIZE, output_dim=64, input_length=MAX_LEN),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(y_cat.shape[1], activation='softmax')
])

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(learning_rate=0.001),
              metrics=['accuracy'])

model.summary()

# --- обучение ---
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=10,
    batch_size=32,
    verbose=1
)

# --- оценка ---
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"✅ Test accuracy: {acc:.3f}")

# --- визуализация динамики ---
plt.figure(figsize=(7,4))
sns.lineplot(x=range(1, len(history.history['accuracy'])+1),
             y=history.history['accuracy'], label='train acc')
sns.lineplot(x=range(1, len(history.history['val_accuracy'])+1),
             y=history.history['val_accuracy'], label='val acc')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Динамика обучения нейросети")
plt.legend()
plt.show()