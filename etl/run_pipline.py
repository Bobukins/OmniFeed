from etl.transform import DataTransformer
from etl.extract import DataExtractor
from etl.load import DataLoader
import logging


class ETLPipline:
    """ЗАПУСК ПАЙПЛАЙНА"""

    def run_pipeline(self):
        logging.basicConfig(level=logging.INFO)

        # Extract
        directories = ["./data/raw"]
        extractor = DataExtractor(directories)
        datasets, na_statistics = extractor.import_data()

        # Transform
        transformer = DataTransformer()
        transformed_dfs = transformer.standardize_data(datasets)
        processed_dfs = transformer.nlp_for_text(transformed_dfs, text_column="text", lang="en")
        processed_dfs = transformer.generate_bert_features(processed_dfs, text_column="text")
        dataframes_dict = {f"df_{i}": df for i, df in enumerate(processed_dfs, start=1)}

        # Load
        loader = DataLoader(output_dir="./data/processed")
        loader.export_data(dataframes_dict)


# from etl import ETLPipline
# # Запуск
# if __name__ == "__main__":
#     pipeline = ETLPipline()
#     pipeline.run_pipeline()
