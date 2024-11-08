from typing import List
from analogy_tutor.utils.llm_lib.get_llm_outputs import get_llm_output
from analogy_tutor.prompts import EXTRACTOR_PROMPT
from analogy_tutor.utils.template import chat_template


class Extractor(object):
    def __init__(self):
        """
        Initialize the model.
        """
        super().__init__()  
        self.prompt_handler = EXTRACTOR_PROMPT

    def __call__(self, quiz_question):
        
        prompt = self.prompt_handler(quiz_question=quiz_question)
        response = get_llm_output(prompt).strip()
        return response