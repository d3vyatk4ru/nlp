import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
import string
import os


EMBEDDING_DIM = 100
WINDOW_SIZE = 2
BATCH_SIZE = 64
NUM_EPOCH = 100


class CBOW(torch.nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int = EMBEDDING_DIM) -> None:
        super(CBOW, self).__init__()

        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)
        

    def forward(self, x) -> torch.Tensor:
        x = torch.sum(self.embeddings(x), dim=1)
        x = self.linear(x)
        return F.log_softmax(x, dim=1) 


class BaseDataset(Dataset):

    def __init__(self, data: str, window_size: int = WINDOW_SIZE) -> None:
        super(BaseDataset, self).__init__()

        tokens = list(set(data.split(' ')))

        self.context_target = self.create_context_and_target(data.split(' '), window_size)

        self.word2id = {word: idx for idx, word in enumerate(tokens)}

        self.id2word = list(self.word2id.keys())

        self.vocab_size = len(self.id2word)


    def __getitem__(self, index) -> tuple:

        context = torch.tensor([
            self.word2id[w] for w in self.context_target[index][0]
        ])

        target = torch.tensor([
            self.word2id[self.context_target[index][1]]
        ])

        return context, target


    def __len__(self) -> int:
        return len(self.context_target)


    @staticmethod
    def create_context_and_target(tokens: list, window_size: int):

        result = []

        for i in range(window_size, len(tokens) - window_size):
            context = []
            for j in range(-window_size, window_size + 1):  
                if j != 0:
                    context.append(tokens[i + j])

            result.append((context, tokens[i]))

        return result


def train(data: str) -> dict:
    """
    return: w2v_dict: dict
            - key: string (word)
            - value: np.array (embedding)
    """

    dataset = BaseDataset(data, WINDOW_SIZE)
    model = CBOW(dataset.vocab_size, EMBEDDING_DIM)

    optimizer = torch.optim.Adam(model.parameters(), lr=4e-3)
    scheduler = StepLR(optimizer, step_size=50, gamma=0.5)
    loss_function = nn.NLLLoss()
    losses = []
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE)

    for epoch in range(NUM_EPOCH):
        total_loss = 0
        for context, target in data_loader:

            if context.size()[0] != BATCH_SIZE:
                continue

            model.zero_grad()
            log_probs = model(context)
            loss = loss_function(log_probs, target.squeeze(1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        losses.append(total_loss)
        print(f'{epoch + 1} / {NUM_EPOCH} Loss: {losses[-1]}')

    return losses
    return {
        word: model.embeddings.weight.detach().cpu().numpy()[idx] for word, idx in dataset.word2id.items()
    }


def remove_consecutive_spaces(string):
    return ' '.join(string.split())

def load(load_directory):
        train_texts = []
        punctuation = string.punctuation
        print(punctuation)
        
        for root, dirs, texts in os.walk(load_directory):
            for file in texts:
                path = os.path.join(root, file)
                extension = os.path.splitext(path)[1]
                
                if extension == ".txt":
                    try:
                        with open(path, encoding='utf-8', mode="r") as f:
                            clean_text = "".join(char for char in f.read().replace('\n', ' ') if char not in punctuation).lower()
                            clean_text = remove_consecutive_spaces(clean_text)
                            train_texts.append(clean_text)
                    except:
                        print(f"error importing path: {path}")
            
        text = '\n'.join(train_texts)
        print(len(text))
        return text.split(' ')[:200000]

if __name__ == "__main__":

    data = load('C:\\Users\\Danya\\Desktop\w2v [NLP]')

    a = train(' '.join(data))
    print(a)
    plt.plot(a)
    plt.yscale('log')
    plt.grid()
    plt.show()

