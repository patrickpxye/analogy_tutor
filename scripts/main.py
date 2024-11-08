import sys
sys.path.append('.')

from analogy_tutor.ds.sciq import SciQ
from analogy_tutor.users import UserProfile
from analogy_tutor.modules import Extractor
from analogy_tutor.core.simulate_conversation import run_one_chat_session
from analogy_tutor.core.anonymize_concepts import generate_anonymized_mapping, anonymize_text
from rich import print

# # Quiz Question -> Target Concepts
# quiz_question = dataset[0]
# extractor = Extractor()
# concepts = extractor(quiz_question)
# print("Quiz Question: ", quiz_question)
# print("Extracted Concepts: ", concepts)

dataset = SciQ()
user_profile_lib = UserProfile()
quiz_question = dataset[0]
concepts = ['mesophilic organisms', 'viruses', 'gymnosperms', 'protozoa', 'temperature', 'food preparation', 'cheese', 'yogurt', 'beer', 'wine']

# Target Concepts Anonymization
ano_dict = generate_anonymized_mapping(concepts)

# Quiz Question -> Anonymized Quiz Concepts
ano_quiz = {}
for key, value in quiz_question.items():
    ano_quiz[key] = anonymize_text(str(value), ano_dict)

print("Anonymized Quiz Question: ", ano_quiz)

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
    user_profile=user_profile_lib[0],
    target_concepts=concepts
)

print('Accuracy: ', accuracy)
print('Chat: ', chat)
                                     