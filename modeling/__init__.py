from modeling.run_modeling import ModelingPipeline
from modeling.mpnet_sampling import MPNetSimilarity
from modeling.catboost_ranking import CatboostRanker
from modeling.emotion_analysis import EmotionAnalyzer


__all__ = ["ModelingPipeline",
           "EmotionAnalyzer",
           "MPNetSimilarity",
           "CatboostRanker"]

# === Описание функционала ===
#   1. Sampling - отбор потенциальных рекомендаций из общего пула объектов:
# сокращение пространства поиска до наиболее релевантных вариантов с использованием
# эмбеддингов в ансамблевой модели на базе трансформеров (Bert + MpNet).
#   2. Ranking - ранжирование отобранных кандидатов по степени релевантности: анализ фичей и предсказание с CatBoost.
#   3. Emotion Analyzer - анализирут тон комментариев и классфицирует их (всего 27 разных меток).

# mpnet_sampling => catboost_ranking => run_modeling
# emotion_analysis - bonus_feature
