from datasets import load_dataset


class MMLU():
    hf_repo = "lukaemon/mmlu"
    
    def __init__(self, config_name='machine_learning', split='train'):
        """
        Initializes the PPC dataset with raw data.

        Parameters:
        raw_data (dict): The raw PPC data to be processed.
        """
        self.config_name, self.split = config_name, split
        raw_data = load_dataset(self.hf_repo, config_name)
        self.processed_data = self.preprocess(raw_data)

    def preprocess(self, raw_data):
        """
        Parameters:
        raw_data (dict): The raw PPC data to be processed.

        Returns:
        list: A list of processed chats with metadata.
        """
        processed_data = []

        for entry in raw_data[self.split]:
            processed_data.append({
                'question': entry.get('input'),
                'choices': {
                    'A': entry.get('A'), 
                    'B': entry.get('B'), 
                    'C': entry.get('C'), 
                    'D': entry.get('D')
                },
                'answer': entry.get('target'),
                })

        return processed_data
    
    def __getitem__(self, idx):
        return self.processed_data[idx]
    
