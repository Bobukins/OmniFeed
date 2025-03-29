from etl.load import DataLoader
from etl.extract import DataExtractor
from etl.run_pipline import ETLPipline
from etl.transform import DataTransformer


__all__ = ["ETLPipline",
           "DataLoader",
           "DataExtractor",
           "DataTransformer"
           ]

# === Описание функционала ===
# NLP-предобработка - очистка, стандартизация, токенизация, векторизация (получение текстовых эмбедингов):
# extract => transform => load => run_pipline
