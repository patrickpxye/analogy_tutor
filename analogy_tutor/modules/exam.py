import re
from typing import List
from analogy_tutor.prompts import TEST_PROMPT
from analogy_tutor.utils.template import chat_template
from analogy_tutor.utils.llm_lib.get_llm_outputs import get_llm_output


class Exam(object):
    def __init__(self, **llm_kwargs):
        """
        Initialize the Exam model.
        """
        super().__init__()
        self.prompt_handler = TEST_PROMPT
        self.llm_kwargs = llm_kwargs

    def __call__(self, user_profile: str, question: str, choices: dict, answer: str, messages: List[dict]):
        assert messages[-1]['role'] == 'assistant'
        prompt = self.prompt_handler(question=question, 
                                     choices=choices,
                                     user_profile=user_profile,
                                     chat_history=chat_template(messages))
        
        # import pdb; pdb.set_trace()
        response = get_llm_output(prompt, **self.llm_kwargs).strip()

        matches = re.findall(r'\[([A-Z])\]', response)
        if matches:
            accuracy = answer == matches[0]
        else:
            accuracy = False
        return response, accuracy