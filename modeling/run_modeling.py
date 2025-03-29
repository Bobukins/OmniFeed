import os
from etl import DataExtractor
from modeling.mpnet_sampling import MPNetSimilarity
from modeling.catboost_ranking import CatboostRanker


class ModelingPipeline:
    """ЗАПУСК РЕКСИСТЕМЫ"""

    def __init__(self, data_dir="./data/processed", result_dir="./data/result"):
        self.data_dir = data_dir
        self.result_dir = result_dir

    # Импорт и обработка
    def extract_data(self):
        extractor = DataExtractor([self.data_dir])
        datasets, _ = extractor.import_data()

        df_source = datasets["transformed_data"]
        df_target = datasets["target_data"]
        return df_source, df_target

    # Семплирование
    @staticmethod
    def run_similarity(df_source, df_target):
        pipeline = MPNetSimilarity(threshold=0.5,
                                   bert_weight=0.3,
                                   mpnet_weight=0.7,
                                   batch_size=128)
        results = pipeline.sampling_similarity(df_source, df_target)
        return results

    #  Ранжирование
    @staticmethod
    def run_ranking(results_df):
        ranker = CatboostRanker(results_df)
        ranker.train_model()
        top_5_posts = ranker.rank_posts()
        return top_5_posts

    # Пуск
    def run_pipeline(self):
        print("Импорт данных...")
        df_source, df_target = self.extract_data()

        print("Запуск семплирования...")
        results_df = self.run_similarity(df_source, df_target)

        print("Запуск ранжирования...")
        top_5_posts = self.run_ranking(results_df)
        top_5_posts = top_5_posts[["url", "autor_name"]]

        os.makedirs(self.result_dir, exist_ok=True)
        output_path = os.path.join(self.result_dir, "final_recs.csv")
        top_5_posts.to_csv(output_path, index=False)

        print(f"Файл с топ-5 кандидатами сохранен в {output_path}")


# from etl import PostRanker
# # Запуск
if __name__ == "__main__":
    pipeline = ModelingPipeline()
    pipeline.run_pipeline()
