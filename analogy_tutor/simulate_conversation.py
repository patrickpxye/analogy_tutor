import os
import copy
import warnings
from tqdm import tqdm

from analogy_tutor.modules import LLMAssistant, UserSimulator
from analogy_tutor.utils.template import chat_template

def run_one_chat_session(assistant_generation_kwargs={},
                         user_generation_kwargs={},
                         max_new_turns=10,
                         target_concept= " ",
                         prompt_method='zero-shot',
                         user_profile= " ",
                         verbose=True):
    """
    Run a single chat session.

    Returns:
        list: A list of dictionaries representing the chat history.

    """
    
    assistant_model_name = assistant_generation_kwargs['model']

    assistant = LLMAssistant(target_concept=target_concept,
                            method=prompt_method,
                            user_profile=user_profile,
                            **assistant_generation_kwargs)
    user = UserSimulator(user_profile=user_profile, 
                         **user_generation_kwargs)

    chat = []
    cur_role = 'assistant'
    exit_flag = False

    for _ in tqdm(range(2 * (max_new_turns or 1)), desc="Generating chat", disable=not verbose):
        if cur_role == 'assistant':
            response = assistant(chat)
        elif cur_role == 'user':
            response = user(chat)
        if os.environ.get('RANK') == '0' and verbose:
            print('*' * 75, f'\n**{cur_role}**: {response}\n', '*' * 75)
        
        if check_for_termination(response):
            exit_flag = True

        chat.append({'role': cur_role, 'content': response})
        cur_role = 'user' if cur_role == 'assistant' else 'assistant'

        if exit_flag:
            break

    return chat


def check_for_termination(response):
    """Check if the response contains a termination signal."""
    try:
        return 'terminate conversation' in response
    except Exception as e:
        warnings.warn(f"Error checking for chat termination: {e}")
        return False
