import os
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


class EmotionAnalyzer:
    """МОДЕЛЬ T5 ДЛЯ КЛАССИФИКАЦИИ ЭМОЦИЙ В КОММЕНТАРИЯХ"""

    def __init__(self, model_name="mrm8488/t5-base-finetuned-emotion", device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")

        self.project_dir = os.getcwd()
        self.model_dir = os.path.join(self.project_dir, "models")
        os.makedirs(self.model_dir, exist_ok=True)

        self.model_save_path = os.path.join(self.model_dir, "t5-base-finetuned-emotion")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, legacy=False)
        self.model = self.load_or_initialize_model(model_name).to(self.device)

    def load_or_initialize_model(self, model_name):
        if os.path.exists(self.model_save_path):
            print(f"Модель найдена в локальной директории: {self.model_save_path}")
            return AutoModelForSeq2SeqLM.from_pretrained(self.model_save_path)
        else:
            print(f"Модель не найдена. Загрузка {model_name}...")
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            model.save_pretrained(self.model_save_path)
            print(f"Готово! Модель сохранена в {self.model_save_path}")
            return model

    def predict_emotion(self, text):
        input_ids = self.tokenizer.encode(text, return_tensors="pt").to(self.device)
        output = self.model.generate(input_ids)
        return self.tokenizer.decode(output[0], skip_special_tokens=True)

    def analyze_dataframe(self, df, text_column="comment"):
        df["emotion"] = df[text_column].apply(self.predict_emotion)
        return df


# from etl import DataExtractor
# from modeling import EmotionAnalyzer
# # Пример запуска
# if __name__ == "__main__":
#
#     directories = ["./data/processed"]
#     extractor = DataExtractor(directories)
#     datasets, _ = extractor.import_data()
#
#     comments_df = datasets["comment"].rename(columns={"review_text": "comment"})
#     comments_df = comments_df[:20]
#
#     analyzer = EmotionAnalyzer()
#     df_result = analyzer.analyze_dataframe(comments_df)
#     df_result = df_result[["comment", "emotion"]]
#
#     df_result.to_csv("./data/result/emotions.csv", index=False)
