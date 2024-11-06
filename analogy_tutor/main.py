from compile_question_concepts import dataset
# from simulate_conversation import run_one_chat_session
from modules import Extractor


quiz = dataset[0]
extractor = Extractor()
response = extractor(quiz)
print("Quiz Question: ", quiz)
print("Extracted Concept: ", response)

# run_one_chat_session(max_new_turns=2,
#                          target_concept= "Photosynthesis",
#                          prompt_method='zero-shot-analogy',
#                          user_profile= "The student studies gastronomy and is familiar with cooking techniques.",
#                          verbose=True)