from slime.utils.misc import load_function

from .base_message_processor import MessageProcessor


def get_message_processor(processor_class_name, tokenizer, **kwargs):
    if processor_class_name is None:
        processor_class_name = MessageProcessor
    processor_class = load_function(processor_class_name)
    return processor_class(tokenizer, **kwargs)
