import os
import json
import pandas as pd
from typing import Dict


class DataLoader:
    """ЭКСПОРТ ОБРАБОТАННЫХ ДАТАСЕТОВ"""

    def __init__(self, output_dir: str, file_prefix: str = "transformed_", include_index: bool = False):
        self.output_dir = output_dir
        self.file_prefix = file_prefix
        self.include_index = include_index

    def export_data(self, dataframes: Dict[str, pd.DataFrame]) -> None:
        if not os.path.exists(self.output_dir):
            raise FileNotFoundError(f"Директория '{self.output_dir}' не существует.")

        combined_df = pd.concat(dataframes.values(), ignore_index=True)

        if "bert_embedding" in combined_df.columns:
            combined_df['bert_embedding'] = combined_df["bert_embedding"].apply(
                lambda x: json.loads(x) if isinstance(x, str) else x
            )

        output_file = os.path.join(self.output_dir, f"{self.file_prefix}data.csv.")
        combined_df.to_csv(output_file, index=self.include_index)


# from etl import DataLoader
# # Запуск
# if __name__ == "__main__":
#     loader = DataLoader(="your_output_dir")
#     loader.export_data()
