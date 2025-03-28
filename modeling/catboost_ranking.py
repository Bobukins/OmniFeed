import os
import sys
import optuna
import numpy as np
from tqdm import tqdm
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


# Для отладки
project_dir = os.path.join(sys.path[0], "models")
catboost_log_dir = os.path.join(project_dir, "catboost_model")
os.environ["CATBOOST_INFO_DIR"] = catboost_log_dir
os.makedirs(catboost_log_dir, exist_ok=True)


class CatboostRanker:
    """МОДЕЛЬ CATBOOST С ТЮНИНГОМ ДЛЯ РАНЖИРОВАНИЯ КАНДИДАТОВ"""

    def __init__(self, df, project_dir=project_dir, optuna_trials=3):
        self.df = df[df["sampling_result"] == 1].copy()
        self.optuna_trials = optuna_trials
        self.cat_features = ["autor_name", "net_type"]
        self.num_features = ["views", "likes", "reposts", "num_comments", "autor_followers", "autor_num_posts", "comb_sim"]
        self.target_column = "comb_sim"

        self.model_dir = os.path.join(project_dir, "catboost_info")
        self.model_path = os.path.join(self.model_dir, "catboost_model.cbm")
        self.log_path = os.path.join(self.model_dir, "training_logs.txt")
        os.makedirs(self.model_dir, exist_ok=True)

    def log(self, message):
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(message + "\n")
        print(message)

    # Подготовка датасетов
    def prepare_data(self):
        df_train = self.df.copy()
        X = df_train[self.num_features + self.cat_features]
        y = df_train[self.target_column]

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_val, y_train, y_val

    # Гиперпараметры
    def objective(self, trial):
        param = {
            "iterations": trial.suggest_int("iterations", 500, 2000),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
            "depth": trial.suggest_int("depth", 3, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-2, 10.0),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
            "random_strength": trial.suggest_float("random_strength", 0.0, 1.0),
            "border_count": trial.suggest_int("border_count", 1, 255),
            "task_type": "GPU",
            "eval_metric": "RMSE",
            "cat_features": self.cat_features,
            "verbose": 0
        }

        X_train, X_val, y_train, y_val = self.prepare_data()
        model = CatBoostRegressor(**param)
        model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50, verbose=0)

        preds = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        return rmse

    # Optuna для тюнинга
    def run_optuna(self):
        study = optuna.create_study(direction="minimize")
        study.optimize(self.objective, n_trials=self.optuna_trials)
        return study.best_params

    # Обучение
    def train_model(self):
        best_params = self.run_optuna()
        X_train, X_val, y_train, y_val = self.prepare_data()

        model = CatBoostRegressor(**best_params, task_type="GPU", cat_features=self.cat_features, verbose=50)
        model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50)

        model.save_model(self.model_path)
        self.log(f"Модель сохранена в {self.model_path}")
        return model

    # Импорт модели
    def load_model(self):
        model = CatBoostRegressor()
        model.load_model(self.model_path)
        return model

    # Ранжирование кандидатов
    def rank_posts(self):
        model = self.load_model()
        X = self.df[self.num_features + self.cat_features]

        tqdm.pandas(desc="Ранжирование постов")
        self.df["predicted_score"] = self.df.progress_apply(
            lambda row: model.predict([row[self.num_features + self.cat_features]])[0],
            axis=1
        )

        self.df = self.df.sort_values(["predicted_score"], ascending=False)
        top_posts = self.df.groupby("autor_name").head(1)
        top_5 = top_posts.head(5)

        return top_5


# from etl import PostRanker
# # Запуск
# if __name__ == "__main__":
#     ranker = CatboostRanker("your_results_df")
#     ranker.train_model()
#     top_5_posts = ranker.rank_posts()
#     top_5_posts.to_csv("./data/result/final_data.csv", index=False)
