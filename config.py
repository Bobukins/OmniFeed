import os
BASE_DIR = os.path.dirname(os.path.abspath("."))


class ColumnMappingConfig:
    """КОНФИГУРАЦИЯ ДЛЯ МАППИНГА"""

    def __init__(self):
        self.mapping = {
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


class CUDAConfig:
    """КОНФИГУРАЦИЯ ДЛЯ CUDA"""

    def __init__(self):
        self.cuda_settings = {
            "CUDA_AUTO_BOOST": "1",
            "CUDA_MODULE_LOADING": "LAZY",
            "CUDA_FORCE_PRELOAD_LIBRARIES": "1",
            "CUDA_DEVICE_MAX_CONNECTIONS": "32",
            "CUDA_CACHE_MAXSIZE": "12884901888",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
        self.apply_cuda_settings()

    def apply_cuda_settings(self):
        for key, value in self.cuda_settings.items():
            os.environ[key] = value


class BertConfig:
    """КОНФИГУРАЦИЯ ДЛЯ BERT"""

    def __init__(self):
        self.models_dir = "./models"
        self.inout_dir = "./data/raw"
        self.output_dir = "./data/processed"
        self.model_file = os.path.join(self.models_dir, "model.safetensors")

        os.makedirs(self.models_dir, exist_ok=True)


class CommnetEmotionConfig:
    """КОНФИГУРАЦИЯ ДЛЯ EMOTION"""

    def __init__(self):
        self.models_dir = "./models"
        self.inout_dir = "./data/processed"
        self.output_dir = "./data/result"
        self.model_file = os.path.join(self.models_dir, "model.safetensors")

        os.makedirs(self.models_dir, exist_ok=True)


class BoostConfig:
    """КОНФИГУРАЦИЯ ДЛЯ CATBOOST"""

    def __init__(self):
        self.models_dir = "./models"
        self.inout_dir = "./data/processed"
        self.output_dir = "./data/result"
        self.model_file = os.path.join(self.models_dir, "jams_boost.cbm")
        self.catboost_info_dir = os.path.join(self.models_dir, "catboost_info")
        self.best_params_file = os.path.join(self.models_dir, "best_params.json")

        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.catboost_info_dir, exist_ok=True)
