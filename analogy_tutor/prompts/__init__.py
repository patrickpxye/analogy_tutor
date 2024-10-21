import os
import os.path as osp
from proact.prompts.prompt_handler import PromptHandler
from typing import Union


current_dir = osp.dirname(__file__)

# TODO[Patrick]: load different prompts similar to the commented code below
###############################################################################
##        Load System Prompt & Assistant Prompt (for prompting methods)      ##
###############################################################################
# with open(osp.join(current_dir, 'llm_assistant/general/system_prompt.txt'), 'r') as f:
#     SYSTEM_PROMPT = f.read()

# with open(osp.join(current_dir, 'funtional/extract_answer.txt'), 'r') as f:
#     EXTRACT_ANSWER = f.read()

# with open(osp.join(current_dir, 'llm_assistant/general/cot.txt'), 'r') as f:
#     LLM_ASSISTANT_PROMPT_COT = PromptHandler(f.read(), input_keys=['chat_history'], output_format=str)

# with open(osp.join(current_dir, 'llm_assistant/general/zero_shot.txt'), 'r') as f:
#     LLM_ASSISTANT_PROMPT_ZS = PromptHandler(f.read(), input_keys=['chat_history'], output_format=str)
    
# with open(osp.join(current_dir, 'llm_assistant', 'question-answering', f'proact_with_gt.txt'), 'r') as f:
#     LLM_ASSISTANT_PROMPT_PROACT_GT = PromptHandler(f.read(), input_keys=['chat_history', 'question', 'answer', 'max_new_tokens'], output_format=str)

# with open(osp.join(current_dir, 'llm_assistant', f'general/proact.txt'), 'r') as f:
#     LLM_ASSISTANT_PROMPT_PROACT = PromptHandler(f.read(), input_keys=['chat_history', 'max_new_tokens'], output_format=str)

###############################################################################
##        Load System Prompt & Assistant Prompt (for prompting methods)      ##
###############################################################################
with open(osp.join(current_dir, 'user_simulator', 'zero-shot-cot.txt'), 'r') as f:
    USER_SIMULATOR_PRONPT = PromptHandler(f.read(), input_keys=['chat_history', 'user_profile'], output_format=str)


# USER_SIMULATOR_PRONPTS = {}
# LLM_REWARD_PROMPTS = {}
# LLM_JUDGE_PROMPTS = {}

# for task in ['question-answering', 'document-editing', 'code-generation']:
#     with open(osp.join(current_dir, 'user_simulator', f'{task}.txt'), 'r') as f:
#         if task == 'question-answering':
#             USER_SIMULATOR_PRONPTS[task] = PromptHandler(f.read(), input_keys=['chat_history', 'question'], output_format=str)
#         elif task == 'document-editing':
#             USER_SIMULATOR_PRONPTS[task] = PromptHandler(f.read(), input_keys=['chat_history', 'question', 'answer'], output_format=str)
#         elif task == 'code-generation':
#             USER_SIMULATOR_PRONPTS[task] = PromptHandler(f.read(), input_keys=['chat_history', 'question'], output_format=str)
#         else:
#             raise NotImplementedError(f"Please declare the prompt for the task: {task}")


#     with open(osp.join(current_dir, 'llm_judge', f'{task}.txt'), 'r') as f:
#         if task == 'question-answering':
#             LLM_JUDGE_PROMPTS[task] = PromptHandler(f.read(), 
#                                                     input_keys=['chat_history', 'chat', 'question', 'answer'], 
#                                                     output_format={
#                                                             'interactivity': {'reason': str, 'score': Union[float, int]},
#                                                             'efficiency': {'reason': str, 'score': Union[float, int]},
#                                                             'accuracy': {'reason': str, 'score': Union[float, int]}
#                                                     })
#         elif task == 'document-editing':
#             LLM_JUDGE_PROMPTS[task] = PromptHandler(f.read(), 
#                                                     input_keys=['chat_history', 'chat'], 
#                                                     output_format={
#                                                         'interactivity': {'reason': str, 'score': Union[float, int]},
#                                                         'efficiency': {'reason': str, 'score': Union[float, int]}
#                                                     })
#         elif task == 'code-generation':
#             LLM_JUDGE_PROMPTS[task] = PromptHandler(f.read(), 
#                                                     input_keys=['chat_history', 'chat'], 
#                                                     output_format={
#                                                         'interactivity': {'reason': str, 'score': Union[float, int]},
#                                                         'efficiency': {'reason': str, 'score': Union[float, int]}
#                                                     })
#         else:
#             raise NotImplementedError(f"Please declare the prompt for the task: {task}")