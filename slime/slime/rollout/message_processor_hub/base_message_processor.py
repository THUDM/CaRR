class MessageProcessor:
    def __init__(self, tokenizer, **kwargs):
        self.tokenizer = tokenizer

    def preprocess_messages(self, messages):
        raise NotImplementedError

    def apply_chat_template(self, messages, tools, **kwargs):
        return self.tokenizer.apply_chat_template(messages, tools=tools, tokenize=False, add_generation_prompt=True)

    def parse_model_response(self, response, truncated=False):
        raise NotImplementedError
