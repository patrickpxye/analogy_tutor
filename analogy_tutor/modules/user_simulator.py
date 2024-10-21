from typing import List
from analogy_tutor.utils.llm_lib.get_llm_outputs import get_llm_output
from analogy_tutor.prompts import USER_SIMULATOR_PRONPT
from analogy_tutor.utils.template import chat_template


class UserSimulator(object):
    def __init__(self, user_profile, **llm_kwargs):
        """
        Initialize the UserSimulator model.
        """
        super().__init__()
        self.user_profile = user_profile
        self.prompt_handler = USER_SIMULATOR_PRONPT
        self.llm_kwargs = llm_kwargs

    def __call__(self, messages: List[dict]):
        if len(messages) and messages[0]['role'] == 'system':
            messages = messages[1:]
        
        prompt = self.prompt_handler(user_profile=user_profile, chat_history=chat_template(messages))
        response = get_llm_output(prompt, **self.llm_kwargs).strip()
        return response