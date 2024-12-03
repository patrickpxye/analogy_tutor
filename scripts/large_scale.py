import random
import sys
sys.path.append('.')

from analogy_tutor.ds.sciq import SciQ
from analogy_tutor.users import UserProfile
from analogy_tutor.modules import Extractor
from analogy_tutor.core.simulate_conversation import run_one_chat_session
from analogy_tutor.core.anonymize_concepts import generate_anonymized_mapping, anonymize_text
# from rich import print

# # Quiz Question -> Target Concepts
dataset = SciQ()
user_profile_lib = UserProfile()

methods = ['non-analogy', 'few-shot-non-analogy', 'zero-shot-analogy', 'few-shot-analogy']
method = methods[0]

success = 0
trials = 3

rand_1 = random.sample(range(trials), trials)
rand_2 = random.sample(range(trials), trials)

for i in range(trials):
    print("trial number ", i, "of ", trials)
    quiz_question = dataset[rand_1[i]]
    extractor = Extractor()
    concepts = extractor(quiz_question)
    print("Quiz Question: ", quiz_question)
    print("Extracted Concepts: ", concepts)

    # Target Concepts Anonymization
    ano_dict = generate_anonymized_mapping(concepts)

    # Quiz Question -> Anonymized Quiz Concepts
    ano_quiz = {}
    for key, value in quiz_question.items():
        ano_quiz[key] = anonymize_text(str(value), ano_dict)

    # print("Anonymized Quiz Question: ", ano_quiz)

    assistant_generation_kwargs = {
        'model': 'gpt-4o-mini',
        'max_tokens': 2048
    }
    user_generation_kwargs = {
        'model': 'gpt-4o-mini',
        'max_tokens': 256
    }
    chat, accuracy = run_one_chat_session(
        question=ano_quiz['question'],
        choices=ano_quiz['choices'],
        answer=ano_quiz['answer'],
        ano_dict=ano_dict,
        assistant_generation_kwargs=assistant_generation_kwargs,
        user_generation_kwargs=user_generation_kwargs,
        prompt_method=method,
        user_profile=user_profile_lib[rand_2[i]],
        target_concepts=concepts
    )

    if accuracy:
        success += 1

    print('Accuracy: ', accuracy)
    print('Chat: ', chat)
    print("success rate so far: ", success / (i + 1))
print("Overall Success Rate: ", success / trials)
print("=========================================")