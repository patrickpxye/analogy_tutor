from datasets import load_dataset
import random

class SciQ():
    hf_repo = "allenai/sciq"
    
    def __init__(self, split='train'):
        """
        Initializes the PPC dataset with raw data.

        Parameters:
        raw_data (dict): The raw PPC data to be processed.
        """
        self.split = split
        raw_data = load_dataset(self.hf_repo)
        self.processed_data = self.preprocess(raw_data)

    def preprocess(self, raw_data):
        """
        Parameters:
        raw_data (dict): The raw PPC data to be processed.

        Returns:
        list: A list of processed chats with metadata.
        """
        processed_data = []
        random.seed(42)
        for entry in raw_data[self.split]:
            choices = [entry.get('distractor1'), 
                       entry.get('distractor2'), 
                       entry.get('distractor3'),
                       entry.get('correct_answer')
                       ]
            random.shuffle(choices)
            idx = choices.index(entry.get('correct_answer'))
            processed_data.append({
                'question': entry.get('question'),
                'choices': {
                    chr(ord('A') + i): choice for i, choice in enumerate(choices)
                },
                'answer': chr(ord('A') + idx),
                'rationale': entry.get('support')
                })

        return processed_data
    
    def __getitem__(self, idx):
        return self.processed_data[idx]

    def __len__(self):
        return len(self.processed_data)
    
