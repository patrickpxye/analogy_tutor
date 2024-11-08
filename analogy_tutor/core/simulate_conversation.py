import os
import copy
import warnings
from tqdm import tqdm

from analogy_tutor.modules import LLMAssistant, UserSimulator, Extractor, Exam
from analogy_tutor.utils.template import chat_template
from analogy_tutor.core.anonymize_concepts import anonymize_text


def run_one_chat_session(question,
                         choices,
                         answer,
                         target_concepts,
                         user_profile,
                         ano_dict,
                         assistant_generation_kwargs={},
                         user_generation_kwargs={},
                         max_new_turns=1,
                         prompt_method='zero-shot-analogy',
                         verbose=True):
    """
    Run a single chat session.

    Returns:
        list: A list of dictionaries representing the chat history.

    """
    
    # assistant_model_name = assistant_generation_kwargs['model']

    print(user_profile)

    assistant = LLMAssistant(target_concepts=target_concepts,
                            method=prompt_method,
                            user_profile=user_profile,
                            **assistant_generation_kwargs)
    user = UserSimulator(user_profile=user_profile, 
                         **user_generation_kwargs)
    exam = Exam(**user_generation_kwargs)

    chat = []
    cur_role = 'assistant'
    
    exit_flag = False

    for idx in tqdm(range(2 * (max_new_turns or 1)), desc="Generating chat", disable=not verbose):

        if cur_role == 'user' and idx == 2 * max_new_turns - 1:
            response, accuracy = exam(user_profile, question, choices, answer, chat)
        
        elif cur_role == 'assistant':
            response = anonymize_text(assistant(chat), ano_dict)
            print("LLM Assistant: ")
            print(response)

        elif cur_role == 'user':
            response = user(chat)
            print("User: ")
            print(response)

        if os.environ.get('RANK') == '0' and verbose:
            print('*' * 75, f'\n**{cur_role}**: {response}\n', '*' * 75)
        
        if check_for_termination(response):
            exit_flag = True

        chat.append({'role': cur_role, 'content': response})
        cur_role = 'user' if cur_role == 'assistant' else 'assistant'

        if exit_flag:
            break

    return chat, accuracy


def check_for_termination(response):
    """Check if the response contains a termination signal."""
    try:
        return 'terminate conversation' in response
    except Exception as e:
        warnings.warn(f"Error checking for chat termination: {e}")
        return False