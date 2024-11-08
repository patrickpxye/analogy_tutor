from datasets import load_dataset


class UserProfile():
    hf_repo = 'erbacher/personalized-proactive-conversations'
    
    def __init__(self, split='train'):
        """
        Initializes the UserProfile dataset with raw data.

        Parameters:
        raw_data (dict): The raw UserProfile data to be processed.
        """
        self.split = split
        raw_data = load_dataset(self.hf_repo)
        self.processed_data = self.preprocess(raw_data)

    def preprocess(self, raw_data):
        """
        Parameters:
        raw_data (dict): The raw UserProfile data to be processed.

        Returns:
        list: A list of processed chats with metadata.
        """
        processed_data = []

        for entry in raw_data[self.split]:
            processed_data.append(entry.get('user'))

        return processed_data
    
    def __getitem__(self, idx):
        return self.processed_data[idx]
