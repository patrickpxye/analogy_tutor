from typing import List
from utils.llm_lib.get_llm_outputs import get_llm_output
from prompts import LLM_NON_ANALOGY_PROMPT, LLM_ZERO_SHOT_ANALOGY_PROMPT, LLM_FEW_SHOT_ANALOGY_PROMPT

class LLMAssistant(object):
    registered_prompts = {
        'non-analogy': LLM_NON_ANALOGY_PROMPT,
        'zero-shot-analogy': LLM_ZERO_SHOT_ANALOGY_PROMPT,
        'few-shot-analogy': LLM_FEW_SHOT_ANALOGY_PROMPT,
        'cot': None
    }
    def __init__(self, target_concept, user_profile, method='zero-shot-analogy', **llm_kwargs):
        """
        Initialize the LLMAssistant model.
        """
        super().__init__()
        self.method = method
        self.prompt_handler = self.registered_prompts[method]
        self.target_concept = target_concept
        self.user_profile = user_profile
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
        if len(messages) != 0: assert messages[-1]['role'] == 'user'

        if self.method in ['non-analogy', 'zero-shot-analogy', 'few-shot-analogy']:
            prompt = self.prompt_handler(target_concept=self.target_concept, user_profile=self.user_profile)
        else:
            prompt = messages
            if len(prompt) and prompt[0]['role'] == 'system':
                print('[LLMAssistant] System message detected.')
        response = get_llm_output(prompt, **self.llm_kwargs)
        
        return response