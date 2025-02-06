import llm
from llm.default_plugins.openai_models import Chat

MODELS = (
    "deepseek-chat",
    "deepseek-coder",
    "deepseek-reasoner",
)


class DeepSeekChat(Chat):
    needs_key = "deepseek"
    key_env_var = "LLM_DEEPSEEK_KEY"

    def __init__(self, model_name):
        super().__init__(
            model_name=model_name,
            model_id=model_name,
            api_base="https://api.deepseek.com",
        )

    def __str__(self):
        return "DeepSeek: {}".format(self.model_id)


@llm.hookimpl
def register_models(register):
    # Only do this if the key is set
    key = llm.get_key("", "deepseek", DeepSeekChat.key_env_var)
    if not key:
        return

    for model_id in MODELS:
        register(DeepSeekChat(model_id))
