from typing import List
from analogy_tutor.utils.llm_lib.get_llm_outputs import get_llm_output
from analogy_tutor.prompts import *


class LLMAssistant(object):
    registered_prompts = {
        'none': None,
        # TODO[Patrick]: add different prompts
        'zero-shot': None,
        'few-shot': None,
        'cot': None
    }
    def __init__(self, method='zero-shot', **llm_kwargs):
        """
        Initialize the LLMAssistant model.
        """
        super().__init__()
        self.method = method
        self.prompt_handler = self.registered_prompts[method]
        self.max_new_tokens = llm_kwargs.get('max_new_tokens', 512)
        self.llm_kwargs = llm_kwargs

    def __call__(self, messages: List[dict], **kwargs):
        """
        Forward pass of the LLMAssistant model.
        
        Args:
            messages (List[dict]): A list of message dictionaries with the last message being the user message.
        
        Returns:
            torch.Tensor: The output tensor.
        """
        assert messages[-1]['role'] == 'user'

        if self.method in ['zero_shot', 'cot', 'few-shot']:
            # TODO[Patrick]: get prompts and the kwargs that the prompt template needs
            prompt = self.prompt_handler(None)
            # example:
            # prompt = self.prompt_handler(chat_history=chat_template(messages),
            #                              max_new_tokens=self.max_new_tokens,
            #                              **kwargs)
        else:
            prompt = messages
            if len(prompt) and prompt[0]['role'] == 'system':
                print('[LLMAssistant] System message detected.')
        response = get_llm_output(prompt, **self.llm_kwargs)
        
        return response