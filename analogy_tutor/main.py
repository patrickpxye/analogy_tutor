from compile_question_concepts import dataset
from simulate_conversation import run_one_chat_session
from modules import Extractor
from anonymize_concepts import generate_anonymized_mapping, anonymize_text

# # Quiz Question -> Target Concepts
# quiz_question = dataset[0]
# extractor = Extractor()
# concepts = extractor(quiz_question)
# print("Quiz Question: ", quiz_question)
# print("Extracted Concepts: ", concepts)

quiz_question = dataset[0]
concepts = ['mesophilic organisms', 'viruses', 'gymnosperms', 'protozoa', 'temperature', 'food preparation', 'cheese', 'yogurt', 'beer', 'wine']

# Target Concepts Anonymization
ano_dict = generate_anonymized_mapping(concepts)

# Quiz Question -> Anonymized Quiz Concepts
ano_quiz = {}
for key, value in quiz_question.items():
    ano_quiz[key] = anonymize_text(str(value), ano_dict)

print("Anonymized Quiz Question: ", ano_quiz)