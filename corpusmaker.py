import torchtext
import random
import json

class TextCorpus:
    """
    A class to store text corpus.

    Attributes:
    - `raw`: list[str], raw text data
    - `name`: str, name of the corpus

    Methods:
    - `random_split`: randomly split the raw data into train and test set (split ratio is 8:2 by default)
    - `raw_number`: return the number of raw data
    """
    def __init__(self):
        self.raw: list[str] = []
        self.name: str = None

    def random_split(self, test_rate: float=0.2) -> tuple[list[str], list[str]]:
        # randomly shuffle self.raw and split it into train and test
        shuffled = self.raw.copy()
        random.shuffle(shuffled)

        split_idx = int(len(shuffled) * (1 - test_rate))
        train = shuffled[:split_idx]
        test = shuffled[split_idx:]

        return train, test

    def raw_number(self):
        return len(self.raw)
    
def IMDB(num_total: int):
    """
    Randomly select `num_total` samples from IMDB dataset, and return it as TextCorpus object.

    Args:
    - num_total: int, number of samples to select

    Returns:
    - text_corpus: TextCorpus
    """
    train_iter, _ = torchtext.datasets.IMDB()
    train_iter = list(train_iter)
    if num_total > len(train_iter):
        raise ValueError(f'num_total should be less than {len(train_iter)}')
    random.shuffle(train_iter)

    text_corpus = TextCorpus()
    text_corpus.raw = [item[1] for item in train_iter[:num_total]]

    text_corpus.name = 'IMDB'

    return text_corpus

def Pile(pile_dataset_path: str, num_total: int):
    train_iter = []
    with open(pile_dataset_path, 'r', encoding='utf-8') as file:
        datas = json.load(file)
        train_iter = [data['text'] for data in datas]
    if num_total > len(train_iter):
        raise ValueError(f'num_total should be less than {len(train_iter)}')
    random.shuffle(train_iter)

    text_corpus = TextCorpus()
    text_corpus.raw = [item for item in train_iter[:num_total]]

    text_corpus.name = 'Pile'

    return text_corpus

