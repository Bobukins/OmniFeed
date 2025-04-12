import os
import torch

BASE_DIR = os.path.dirname(os.path.abspath("."))


class PathConfig:
    """КОНФИГУРАЦИЯ ПУТЕЙ"""

    def __init__(self):
        self.data_text_dir = "./data/text"
        self.data_speech_dir = "./data/speech"
        self.whisper_model_path = "./models/whisper/medium.pt"
        self.llama_model_path = "./models/meta_llama"

        os.makedirs(self.data_text_dir, exist_ok=True)
        os.makedirs(self.data_speech_dir, exist_ok=True)
        os.makedirs(self.llama_model_path, exist_ok=True)
        os.makedirs(os.path.dirname(self.whisper_model_path), exist_ok=True)


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


class AudioConfig:
    """КОНФИГУРАЦИЯ ЗАПИСИ"""

    def __init__(self):
        self.sample_rate = 48000
        self.channels = 1
        self.chunk_duration = 0.5
        self.chunk_samples = int(self.sample_rate * self.chunk_duration)
        self.silence_threshold = 0.01
        self.silence_limit = 1.5    # лимит тишини => обрезка дорожки (в секкундах)
        self.max_duration = 30      # максимальное время записи (в секкундах)


class WhisperConfig:
    """КОНФИГУРАЦИЯ ДЛЯ WHISPER"""

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.fp16 = self.device == "cuda"
        self.available_languages = ["ru", "en"]
        self.target_language = "ru"


class UVRConfig:
    """КОНФИГУРАЦИЯ ДЛЯ UVR"""

    def __init__(self):
        self.model_path = "./models/mdx_net/UVR-MDX-NET-Inst_HQ_4.onnx"
        self.metadata_path = "./models/mdx_net/model_data/model_data.json"
        self.mapper_path = "./models/mdx_net/model_data/model_name_mapper.json"

        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.metadata_path), exist_ok=True)


class LlamaGGUFConfig:
    """КОНФИГУРАЦИЯ ДЛЯ ЛОКАЛЬНОЙ GGUF МОДЕЛИ"""
    def __init__(
        self,
        model_name: str = "Meta-Llama-3-8B-Instruct.Q4_0",
                          # или DeepSeek-R1-Distill-Qwen-14B-Q4_0
        model_path: str = "./models/meta_llama",
        context_size: int = 8192,
        verbose: bool = False
    ):
        self.model_name = model_name
        self.model_path = model_path
        self.context_size = context_size
        self.verbose = verbose
