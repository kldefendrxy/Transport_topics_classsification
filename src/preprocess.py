import os
import re
import string
import yaml
import numpy as np
import pandas as pd
import nltk
from tqdm.auto import tqdm
import pymorphy3
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

tqdm.pandas()

cfg = yaml.safe_load(open("configs/train_config.yaml"))
paths = cfg["paths"]
prep_cfg = cfg["preprocessing"]

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

morph = pymorphy3.MorphAnalyzer()
STOPWORDS = set(stopwords.words(prep_cfg["stopwords_lang"])) if prep_cfg["remove_stopwords"] else set()



def clean_text(text: str) -> str:
    """Очистка текста — гибко управляется из YAML."""
    text = str(text)

    if prep_cfg["lowercase"]:
        text = text.lower()

    if prep_cfg["remove_urls"]:
        text = re.sub(r"http\S+|www.\S+|t.me/\S+", " ", text)

    # Удаление упоминаний и хэштегов
    text = re.sub(r"@\w+|#\w+", " ", text)

    if prep_cfg["remove_punctuation"]:
        text = re.sub(r"[^а-яёa-z\s]", " ", text)

    # Числа
    if prep_cfg.get("remove_numbers", True):
        text = re.sub(r"\d+", " ", text)

    # Удаление лишних пробелов
    text = re.sub(r"\s+", " ", text).strip()

    return text



def lemmatize_text(text: str) -> str:
    """Токенизация + лемматизация через pymorphy3."""
    if not prep_cfg["lemmatize"]:
        return text

    tokens = word_tokenize(text, language="russian")
    lemmas = [
        morph.parse(t)[0].normal_form
        for t in tokens
        if t not in STOPWORDS and t not in string.punctuation
    ]
    return " ".join(lemmas)



def preprocess_all(raw_dir=None, out_file=None):
    """Главная функция предобработки, управляемая конфигами."""
    
    raw_dir = raw_dir or paths["raw_data"]
    out_file = out_file or paths["processed_data"]

    print(f"Загрузка сырых данных из: {raw_dir}")

    dfs = []
    for fname in os.listdir(raw_dir):
        if fname.endswith(".csv"):
            df = pd.read_csv(os.path.join(raw_dir, fname), sep=";")
            
            # label = 0 (авто), 1 (метро)
            label = 0 if "автотранспорт" in fname else 1
            df["label"] = label
            dfs.append(df)

    data = pd.concat(dfs, ignore_index=True)
    print(f"Загружено {len(data)} сообщений")

    # очистка
    data["text_clean"] = data["text"].astype(str).progress_apply(clean_text)
    # лемматизация
    data["text_lemma"] = data["text_clean"].progress_apply(lemmatize_text)

    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    data.to_csv(out_file, index=False)

    print(f"Сохранено: {out_file} ({len(data)} строк)")

if __name__ == "__main__":
    preprocess_all()