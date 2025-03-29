import os
import re
import json
import torch
import spacy
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel


# Маппинг столбцов для стандартизации
COLUMN_MAPPING = {
    "post_id": ["id", "post_id"],
    "url": ["url", "link"],
    "date": ["date_posted"],
    "photos": ["photo", "photos", "post_image", "photo_url"],
    "text": ["caption", "content", "description"],
    "hashtags": ["tags", "hashtags", "post_tags", "hashes"],
    "views": ["video_view_count", "views"],
    "likes": ["likes", "page_likes"],
    "reposts": ["num_shares", "shares", "reposts", "share_count"],
    "num_comments": ["num_comments", "comments_count", "replies"],
    "autor_name": ["user_username_raw", "user_posted", "user_handle"],
    "autor_followers": ["page_followers", "followers"],
    "autor_num_posts": ["posts_count", "post_number"],
}


class DataTransformer:
    """ПРЕПРОЦЕССИНГ ПЕРЕД МОДЕЛИРОВАНИЕМ"""

    def __init__(self):
        self.project_dir = os.getcwd()
        self.temp_dir = os.path.join(self.project_dir, "models")
        os.makedirs(self.temp_dir, exist_ok=True)

        self.bert_path = os.path.join(self.temp_dir, "bert_base_multilingual_cased")
        self.tokenizer, self.model = self.setup_bert_model()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

        self.nlp_ru, self.nlp_en = self.setup_spacy_models()

    # Наладка Bert
    def setup_bert_model(self):
        if not os.path.exists(self.bert_path):
            print(f"Скачиваю BERT в {self.bert_path}...")
            tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
            model = AutoModel.from_pretrained("bert-base-multilingual-cased")
            tokenizer.save_pretrained(self.bert_path)
            model.save_pretrained(self.bert_path)
        else:
            print(f"Загружаю BERT из {self.bert_path}...")
            tokenizer = AutoTokenizer.from_pretrained(self.bert_path)
            model = AutoModel.from_pretrained(self.bert_path)

        return tokenizer, model

    # Наладка spaCy моделей
    def setup_spacy_models(self):
        spacy_path_ru = os.path.join(self.temp_dir, "spacy_model_ru")
        spacy_path_en = os.path.join(self.temp_dir, "spacy_model_en")

        if not os.path.exists(spacy_path_ru):
            print(f"Скачиваю Spacy модель для русского в {spacy_path_ru}...")
            nlp_ru = spacy.load("ru_core_news_sm", disable=["parser", "ner"])
            nlp_ru.to_disk(spacy_path_ru)
        else:
            print(f"Загружаю Spacy модель для русского из {spacy_path_ru}...")
            nlp_ru = spacy.load(spacy_path_ru)

        if not os.path.exists(spacy_path_en):
            print(f"Скачиваю Spacy модель для английского в {spacy_path_en}...")
            nlp_en = spacy.load("en_core_web_sm", disable=["parser", "ner"])
            nlp_en.to_disk(spacy_path_en)
        else:
            print(f"Загружаю Spacy модель для английского из {spacy_path_en}...")
            nlp_en = spacy.load(spacy_path_en)

        return nlp_ru, nlp_en

    # Заполнение Nan
    @staticmethod
    def generate_missing_data(df, column_name, reference_data):
        if column_name not in df.columns:
            df[column_name] = np.nan

        nan_mask = df[column_name].isna()

        if reference_data:
            mean_value = np.mean(reference_data)
            std_dev = np.std(reference_data)
            df.loc[nan_mask, column_name] = np.abs(
                np.random.normal(loc=mean_value, scale=std_dev, size=nan_mask.sum())
            ).astype(int)
        else:
            df.loc[nan_mask, column_name] = 0

        return df

    # Стандартизация текста
    def standardize_data(self, dataframes):
        standardized_columns = list(COLUMN_MAPPING.keys()) + ["net_type"]
        transformed_dataframes = []

        reposts_reference = []
        num_comments_reference = []

        for df in dataframes.values():
            if "reposts" in df.columns:
                reposts_reference.extend(df["reposts"].dropna().tolist())
            if "num_comments" in df.columns:
                num_comments_reference.extend(df["num_comments"].dropna().tolist())

        for name, df in dataframes.items():
            rename_dict = {}
            for std_col, possible_names in COLUMN_MAPPING.items():
                for possible_name in possible_names:
                    if possible_name in df.columns:
                        rename_dict[possible_name] = std_col
                        break

            df = df.rename(columns=rename_dict)
            df["net_type"] = name
            df = df.loc[:, ~df.columns.duplicated()]
            df = df.reindex(columns=standardized_columns, fill_value=np.nan)

            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"].astype(str)
                                            .str.replace("T", " ")
                                            .str.replace("Z", ""),
                                            errors="coerce")

            df = self.generate_missing_data(df, "reposts", reposts_reference)
            df = self.generate_missing_data(df, "autor_num_posts", num_comments_reference)

            if "hashtags" in df.columns:
                df["hashtags"] = df["hashtags"].replace(r"^\[\]$", "0", regex=True)

            df.fillna(0, inplace=True)

            dtype_mapping = {
                "post_id": "int",
                "url": "str",
                "date": "datetime64",
                "photos": "str",
                "text": "str",
                "hashtags": "str",
                "views": "int",
                "likes": "int",
                "reposts": "int",
                "num_comments": "int",
                "autor_name": "str",
                "autor_followers": "int",
                "autor_num_posts": "int",
                "net_type": "str",
            }

            for col, dtype in dtype_mapping.items():
                if col in df.columns:
                    df[col] = df[col].astype(dtype, errors="ignore")

            transformed_dataframes.append(df)

        return transformed_dataframes

    # Удаление эмоджи
    @staticmethod
    def remove_emoji(text: str) -> str:
        emoji_pattern = re.compile("["
                                   u"\U0001F600-\U0001F64F"
                                   u"\U0001F300-\U0001F5FF"
                                   u"\U0001F680-\U0001F6FF"
                                   u"\U0001F700-\U0001F77F"
                                   u"\U0001F780-\U0001F7FF"
                                   u"\U0001F800-\U0001F8FF"
                                   u"\U0001F900-\U0001F9FF"
                                   u"\U0001FA00-\U0001FA6F"
                                   u"\U0001FA70-\U0001FAFF"
                                   u"\U00002702-\U000027B0"
                                   u"\U000024C2-\U0001F251"
                                   "]+", flags=re.UNICODE)
        text = emoji_pattern.sub(r'', text)
        text = re.sub(r'http\S+|www\S+', '', text)
        text = re.sub(r'[^a-zA-Zа-яА-Я0-9\s]', '', text)
        return text.strip()

    # Процессинг текста
    def text_processing(self, text: str, lang: str = "en") -> str:
        if not isinstance(text, str) or not text.strip():
            return ""

        text = text.lower().replace("\n", " ").replace("\xa0", " ").strip()
        text = self.remove_emoji(text)
        nlp = self.nlp_ru if lang == "ru" else self.nlp_en
        doc = nlp(text)
        tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
        return " ".join(tokens)

    # spaCy to DataFrame
    def nlp_for_text(self, dataframes, text_column="text", lang="en"):
        for df in tqdm(dataframes, desc="NLP-предобработка"):
            if text_column in df.columns:
                tqdm.pandas(desc=f"Обработка текста ({lang})")
                df[text_column] = df[text_column].progress_apply(
                    lambda x: self.text_processing(x, lang)
                )
        return dataframes

    # Токенизация
    def tokenize_text(self, text: str, max_length: int = 512):
        tokens = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        return tokens

    # Извлечение эмбеддингов
    def extract_bert_embeddings(self, text: str):
        tokens = self.tokenize_text(text)

        input_ids = tokens["input_ids"].to(self.device)
        attention_mask = tokens["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        cls_embedding = outputs.last_hidden_state[:, 0, :]
        return cls_embedding.cpu().numpy()

    # Bert to DataFrame
    def generate_bert_features(self, dataframes, text_column="text"):
        for df in tqdm(dataframes, desc="Эмбеддинги через BERT"):
            if text_column in df.columns:
                tqdm.pandas(desc="Извлечение эмбеддингов BERT")
                df["bert_embedding"] = df[text_column].progress_apply(
                    lambda x: json.dumps(self.extract_bert_embeddings(x).tolist())
                )
        return dataframes


# from etl import DataTransformer
# # Запуск
# if __name__ == "__main__":
#     transformer = DataTransformer()
#     transformed_dfs = transformer.standardize_data("your_datasets")
#     processed_dfs = transformer.nlp_for_text(transformed_dfs, text_column="text", lang="en")
#     processed_dfs = transformer.generate_bert_features(processed_dfs, text_column="text")
#     dataframes_dict = {f"df_{i}": df for i, df in enumerate(processed_dfs, start=1)}
