from typing import List
from analogy_tutor.utils.llm_lib.get_llm_outputs import get_llm_output
from analogy_tutor.prompts import EXTRACTOR_PROMPT
from analogy_tutor.utils.template import chat_template


class Extractor(object):
    def __init__(self, model='gpt-4o'):
        """
        Initialize the model.
        """
        super().__init__()  
        self.model = model
        self.prompt_handler = EXTRACTOR_PROMPT

    def __call__(self, quiz_question):
        
        prompt = self.prompt_handler(quiz_question=quiz_question)
        response_str = get_llm_output(prompt, model=self.model).strip()
        response_str = response_str.split(',')
        response = []
        for word in response_str:
            word = word.strip()
            # if there is a single or double quote leading or ending the word, remove it
            if word[0] in ['"', "'"]:
                word = word[1:]
            if word[-1] in ['"', "'"]:
                word = word[:-1]
            response.append(word)
        return response