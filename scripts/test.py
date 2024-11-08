from analogy_tutor.users import UserProfile

user_profiles = UserProfile()
print(user_profiles[10])

from analogy_tutor.ds.mmlu import MMLU
from analogy_tutor.ds.sciq import SciQ

# # MMLU dataset
# dataset = MMLU()
# print('Question: ', dataset[0]['question'])
# print('Choices: ', dataset[0]['choices'])
# print('Answer: ', dataset[0]['answer'])

# SciQA dataset
dataset = SciQ()
# print('Question: ', dataset[0]['question'])
# print('Choices: ', dataset[0]['choices'])
# print('Answer: ', dataset[0]['answer'])
# print('Rationale: ', dataset[0]['rationale'])