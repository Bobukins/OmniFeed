import os
import json
import torch
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util


# Параметры для CUDA
os.environ["CUDA_AUTO_BOOST"] = "1"
os.environ["CUDA_MODULE_LOADING"] = "LAZY"
os.environ["CUDA_FORCE_PRELOAD_LIBRARIES"] = "1"
os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "32"
os.environ["CUDA_CACHE_MAXSIZE"] = "12884901888"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


class MPNetSimilarity:
    """МОДЕЛЬ MPNET ДЛЯ СЕМПЛИРОВАНИЯ ОБРАБОТАННЫХ ТЕКСТОВ"""

    def __init__(self, model_path="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
                 threshold: float = 0.8,
                 bert_weight: float = 0.5,
                 mpnet_weight: float = 0.5,
                 batch_size: int = 32,
                 text_column: str = "text",
                 embedding_column: str = "bert_embedding",
                 device: str = None):
        self.threshold = threshold
        self.bert_weight = bert_weight
        self.transformer_weight = mpnet_weight
        self.batch_size = batch_size
        self.text_column = text_column
        self.embedding_column = embedding_column
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")

        self.project_dir = os.getcwd()
        self.model_dir = os.path.join(self.project_dir, "models")
        os.makedirs(self.model_dir, exist_ok=True)
        self.model_save_path = os.path.join(self.model_dir, "paraphrase-multilingual-mpnet-base-v2")

        self.transformer_model = self.load_or_initialize_model(model_path)
        self.transformer_model.to(self.device)

    # Проверка наличия модели
    def load_or_initialize_model(self, model_path):
        if os.path.exists(self.model_save_path):
            return SentenceTransformer(self.model_save_path)
        else:
            print(f"Модель не найдена. Загрузка{model_path}...")
            model = SentenceTransformer(model_path)
            model.save(self.model_save_path)
            print(f"Готово! Путь сохранения {self.model_save_path}")
            return model

    # Сравнение эмбендингов
    def sampling_similarity(self, df_source, df_target):
        df_source = df_source.dropna(subset=[self.text_column, self.embedding_column]).copy()
        df_target = df_target.dropna(subset=[self.text_column, self.embedding_column])

        source_texts = df_source[self.text_column].tolist()
        target_text = df_target[self.text_column].iloc[0]

        tqdm.pandas(desc="Декодирование эмбеддингов")
        bert_source_embeddings = np.vstack(
            df_source[self.embedding_column].progress_apply(lambda x: np.array(json.loads(x), dtype=np.float32))
        )
        bert_target_embedding = np.array(json.loads(df_target[self.embedding_column].iloc[0]), dtype=np.float32)

        print("Сэмпилрвоание кандидатов...")
        transformer_source_embeddings = self.transformer_model.encode(source_texts, convert_to_tensor=True,
                                                                      show_progress_bar=True)
        transformer_target_embedding = self.transformer_model.encode(target_text, convert_to_tensor=True)

        bert_similarities = util.cos_sim(torch.tensor(bert_source_embeddings),
                                         torch.tensor(bert_target_embedding)).cpu().numpy().flatten()

        transformer_similarities = util.cos_sim(transformer_source_embeddings,
                                                transformer_target_embedding).cpu().numpy().flatten()

        weighted_similarity = (self.bert_weight * bert_similarities +
                               self.transformer_weight * transformer_similarities)

        df_source.loc[:, "bert_sim"] = np.round(bert_similarities, 2)
        df_source.loc[:, "mpnet_sim"] = np.round(transformer_similarities, 2)
        df_source.loc[:, "comb_sim"] = np.round(weighted_similarity, 2)
        df_source.loc[:, "sampling_result"] = (df_source["comb_sim"] >= self.threshold).astype(int)

        return df_source


# from etl import DataExtractor
# from modeling import MPNetSimilarity
# # Запуск
# if __name__ == "__main__":
#     directories = ["./data/processed"]
#     extractor = DataExtractor(directories)
#     datasets, _ = extractor.import_data()
#
#     df_source = datasets["transformed_data"]
#     df_target = datasets["target_data"]
#
#     pipeline = MPNetSimilarity(threshold=0.8,
#                                bert_weight=0.3,
#                                mpnet_weight=0.7,
#                                batch_size=128)
#
#     results = pipeline.sampling_similarity(df_source, df_target)
