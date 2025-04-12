import os
from gpt4all import GPT4All
from config import LlamaGGUFConfig, PathConfig


class PromptEngineer:
    """ГЕНЕРАТОР ПРОМТОВ С УЧЁТОМ КОНТЕКСТА"""

    def __init__(self, system_prompt: str = None):
        self.default_instruction = system_prompt or self._default_system_prompt()

    @staticmethod
    def _default_system_prompt() -> str:
        return (
            "Ты — вежливый, дружелюбный и информированный ассистент-гид, специализирующийся на Краснодарском крае.\n"
            "Ты должен отвечать понятно, живо и с душой — будто рассказываешь другу.\n"
            "Твоя задача — делиться полезной, актуальной и интересной информацией, быть собеседником, а не сухим справочником.\n"
            "Ты не используешь англицизмы и сленг, говоришь в стиле местного жителя, который любит свой регион.\n"
            "Если ты чего-то не знаешь — скажи честно, не выдумывай.\n"
            "\n"
            "Ориентируйся на следующие ключевые темы, которые ты хорошо знаешь и можешь рассказывать с уверенностью:\n"
            "1. Природные маршруты, каньоны, ущелья, водопады — например, Лаго-Наки, Гуамка, Мезмай.\n"
            "2. Местная кухня — что попробовать и где: от хинкала до домашних столовых.\n"
            "3. Курортные города и пляжи — Геленджик, Анапа, Туапсе, дикие и благоустроенные пляжи.\n"
            "4. Историко-культурные объекты — дольмены, музеи, казачьи станицы, античные руины.\n"
            "5. Винодельни и энотуризм — где можно попробовать и как устроены экскурсии.\n"
            "6. Секретные локации — интересные места, о которых знают только местные.\n"
            "7. Термальные источники — где можно расслабиться и поправить здоровье.\n"
            "8. Местные праздники, рынки и ярмарки — куда пойти и когда.\n"
            "9. Транспорт и логистика — как добраться, где арендовать, сколько времени занимает дорога.\n"
            "10. Климат и сезонность — когда и куда лучше ехать в зависимости от времени года.\n"
            "\n"
            "Отвечай только на поставленный вопрос. Будь лаконичен, но не сух — твои ответы должны быть человечными.\n"
            "Если человек ищет совет — предложи подходящие варианты. Если интересуется фактами — ответь точно.\n"
            "Добавляй детали, если они уместны: ориентиры, нюансы, полезные мелочи.\n"
        )

    # Формирование промта
    def build_prompt(self, user_input: str, history: list = None) -> str:
        history_prompt = ""
        if history:
            for user_text, assistant_text in history:
                history_prompt += f"User: {user_text}\nAssistant: {assistant_text}\n"

        return f"{self.default_instruction}\n\n{history_prompt}User: {user_input}\nAssistant:"


class LlamaGGUFModel:
    """ПРЕДОБУЧЕННАЯ И ДИСТИЛИРОВАННАЯ LLaMA-МОДЕЛЬ"""

    def __init__(self, config: LlamaGGUFConfig = None):
        self.config = config or LlamaGGUFConfig()
        self.paths = PathConfig()
        self.dialog_file = os.path.join(self.paths.data_text_dir, "dialog_history.txt")

        self.model = GPT4All(
            model_name=self.config.model_name,
            model_path=self.config.model_path,
            n_ctx=self.config.context_size,
            verbose=self.config.verbose,
            allow_download=False
        )

        self.prompt_engineer = PromptEngineer()

    # Импорт контекста
    def _load_dialog_history(self) -> list:
        if not os.path.exists(self.dialog_file):
            return []

        history = []
        with open(self.dialog_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

        user_input, assistant_output = None, None
        for line in lines:
            if line.startswith("User_input_"):
                user_input = line.split(":", 1)[1].strip()
            elif line.startswith("Assistant_output_"):
                assistant_output = line.split(":", 1)[1].strip()
                if user_input is not None:
                    history.append((user_input, assistant_output))
                    user_input, assistant_output = None, None
        return history

    @staticmethod
    # Шаблон для входа-выхода
    def _build_prompt(history: list, current_input: str) -> str:
        prompt = ""
        for user_text, assistant_text in history:
            prompt += f"User: {user_text}\nAssistant: {assistant_text}\n"
        prompt += f"User: {current_input}\nAssistant:"
        return prompt

    # Осмысленный ответ
    def generate_response(self, current_input: str) -> str:
        history = self._load_dialog_history()
        prompt = self.prompt_engineer.build_prompt(current_input, history)

        with self.model.chat_session():
            response = self.model.generate(prompt=prompt).strip()

        return response

# from modeling import LlamaGGUFModel
# # Запуск
# if __name__ == "__main__":
#     model = LlamaGGUFModel()
#     user_input = "Your input text"
#     assistant_response = model.generate_response(user_input)
#     print(f"Assistant: {assistant_response}")
