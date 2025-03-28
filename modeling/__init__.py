from modeling.catboost_ranking import CatboostRanker
from modeling.mpnet_sampling import MPNetSimilarity
from modeling.run_modeling import ModelingPipeline


__all__ = ["ModelingPipeline",
           "MPNetSimilarity",
           "CatboostRanker"]

# === Описание функционала ===
#   1. Sampling - отбор потенциальных рекомендаций из общего пула объектов:
# сокращение пространства поиска до наиболее релевантных вариантов с использованием
# эмбеддингов в ансамблевой модели на базе трансформеров (Bert + MpNet).
#   2. Ranking - ранжирование отобранных кандидатов по степени релевантности: анализ фичей и предсказание с CatBoost.
# mpnet_sampling => catboost_ranking => run_modeling
