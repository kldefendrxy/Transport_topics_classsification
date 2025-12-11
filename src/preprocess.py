import os
import re
import string

import pandas as pd
import nltk
from tqdm.auto import tqdm
import pymorphy3
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer

tqdm.pandas()

# --- загрузка ресурсов NLTK ---
nltk.download('punkt')
nltk.download('stopwords')

# --- инициализация инструментов ---
morph = pymorphy3.MorphAnalyzer()
russian_stopwords = set(stopwords.words("russian"))
stemmer = SnowballStemmer("russian")


def clean_text(text: str) -> str:
    """Удаление ссылок, пунктуации, чисел и приведение к нижнему регистру"""
    text = str(text).lower()
    text = re.sub(r"http\S+|www.\S+", " ", text)      # ссылки
    text = re.sub(r"@\w+|#\w+", " ", text)            # упоминания, хэштеги
    text = re.sub(r"[^а-яёa-z\s]", " ", text)         # только буквы
    text = re.sub(r"\s+", " ", text).strip()
    return text


def lemmatize_text(text: str) -> str:
    """Токенизация + лемматизация с pymorphy3"""
    tokens = word_tokenize(text, language="russian")
    lemmas = []
    for token in tokens:
        if token not in russian_stopwords and token not in string.punctuation:
            lemmas.append(morph.parse(token)[0].normal_form)
    return " ".join(lemmas)


def stem_text(text: str) -> str:
    """Токенизация + стемминг (SnowballStemmer для русского)"""
    tokens = word_tokenize(text, language="russian")
    stems = []
    for token in tokens:
        if token not in russian_stopwords and token not in string.punctuation:
            stems.append(stemmer.stem(token))
    return " ".join(stems)


def preprocess_all(raw_dir="data/raw", out_dir="data/processed"):
    os.makedirs(out_dir, exist_ok=True)
    dfs = []

    # читаем все сырые csv
    for fname in os.listdir(raw_dir):
        if fname.endswith(".csv"):
            df = pd.read_csv(os.path.join(raw_dir, fname), sep=";")
            dfs.append(df)

    data = pd.concat(dfs, ignore_index=True)
    print(f"Загружено {len(data)} сообщений")

    # базовая очистка
    data["text_clean"] = data["text"].astype(str).progress_apply(clean_text)

    # v1: стемминг
    data["text_stem"] = data["text_clean"].progress_apply(stem_text)

    # v2: лемматизация
    data["text_lemma"] = data["text_clean"].progress_apply(lemmatize_text)

    # сохраняем два разных датасета

    stem_path = os.path.join(out_dir, "dataset_stem.csv")
    data[["text", "category", "text_clean", "text_stem"]].to_csv(stem_path, index=False)
    print(f"Сохранено: {stem_path} ({len(data)} строк)")

    lemma_path = os.path.join(out_dir, "dataset_lemma.csv")
    data[["text", "category", "text_clean", "text_lemma"]].to_csv(lemma_path, index=False)
    print(f"Сохранено: {lemma_path} ({len(data)} строк)")


if __name__ == "__main__":
    preprocess_all()