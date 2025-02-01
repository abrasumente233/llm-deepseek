import llm
from llm.default_plugins.openai_models import Chat, AsyncChat


class DeepSeekChat(Chat):
    needs_key = "deepseek"
    key_env_var = "DEEPSEEK_API_KEY"

    def __init__(self, model_name):
        super().__init__(
            model_name=model_name,
            model_id=model_name,
            api_base="https://api.deepseek.com",
        )

    def __str__(self):
        return "DeepSeek: {}".format(self.model_id)


class DeepSeekAsyncChat(AsyncChat):
    needs_key = "deepseek"
    key_env_var = "DEEPSEEK_API_KEY"

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
    key = llm.get_key("", "deepseek", "LLM_DEEPSEEK_KEY")
    if not key:
        return

    register(DeepSeekChat("deepseek-chat"), DeepSeekAsyncChat("deepseek-chat"), aliases=("ds",))
    register(DeepSeekChat("deepseek-coder"), DeepSeekAsyncChat("deepseek-coder"), aliases=("ds-code",))
    register(
        DeepSeekChat("deepseek-reasoner"),
        DeepSeekAsyncChat("deepseek-reasoner"),
        aliases=(
            "r1",
            "dsr1",
        ),
    )
