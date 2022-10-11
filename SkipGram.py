import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import string
import os


EMBEDDING_DIM = 100
WINDOW_SIZE = 5
BATCH_SIZE = 128
NUM_EPOCH = 50


class SkipGram(nn.Module):

    def __init__(self, vocab_size: int, embedding_dim: int = EMBEDDING_DIM) -> None:

        super(SkipGram, self).__init__()

        # embeddings for center word
        self.center_word_embeddings = nn.Embedding(
            vocab_size, embedding_dim, max_norm=1,
        )
        
        # embeddings for context [neg & pos, because it's output from model]
        self.context_embeddings = nn.Embedding(
            vocab_size, embedding_dim,
        )

    def forward(self, center_word: list, pos_context: list, neg_context: list) -> torch.Tensor:

        center_word_embedding = self.center_word_embeddings(
            Variable(torch.LongTensor(center_word)),
        )

        pos_context_embedding = self.context_embeddings(
            Variable(torch.LongTensor(pos_context)),
        )

        pos_loss = torch.bmm(pos_context_embedding, center_word_embedding.squeeze().unsqueeze(-1)).squeeze().sum(1) 
        # pos_loss = torch.sum(pos_loss)
        pos_loss = F.logsigmoid(pos_loss)

        neg_context_embedding = self.context_embeddings(
            Variable(torch.LongTensor(neg_context))
        )

        neg_loss = torch.bmm(neg_context_embedding, center_word_embedding.squeeze().unsqueeze(-1)).squeeze().sum(1) 

        # neg_loss = torch.sum(neg_loss)
        neg_loss = F.logsigmoid(-neg_loss)
        
        return -(pos_loss + neg_loss).mean()


class BaseDataset(Dataset):

    def __init__(self, data: str, window_size: int = WINDOW_SIZE) -> None:

        super(BaseDataset, self).__init__()

        tokens = list(set(data.split(' ')))

        self.context_target = self.create_context_and_target(data.split(' '), window_size)

        self.word2id = {
            word: idx for idx, word in enumerate(tokens)
        }

        self.id2word = list(self.word2id.keys())

        self.vocab_size = len(self.id2word)


    def __getitem__(self, index) -> tuple:

        pos_context = torch.tensor([
            self.word2id[w] for w in self.context_target[index][0]
        ])

        target = torch.tensor([
            self.word2id[self.context_target[index][1]]
        ])

        neg_context = torch.tensor([
            self.word2id[w] for w in self.context_target[index][2]
        ])

        return pos_context, target, neg_context


    def __len__(self) -> int:
        return len(self.context_target)
    

    @staticmethod
    def create_context_and_target(tokens: list, window_size: int, n_neg_samples: int = 10):
        
        def negative_sampling():

            while len(negative_context) != n_neg_samples:

                    word_idx = np.random.randint(len(tokens))

                    if word_idx not in window:
                        negative_context.append(tokens[word_idx])
            
            return negative_context

        result = []

        for i in range(window_size, len(tokens) - window_size):
            context = []
            negative_context = []

            window = list(range(i - window_size, i + window_size + 1))

            for j in window:

                if j != i:
                    context.append(tokens[j])

            negative_context = negative_sampling()

            result.append((context, tokens[i], negative_context))

        return result


def train(data: str) -> dict:
    """
    return: w2v_dict: dict
            - key: string (word)
            - value: np.array (embedding)
    """

    dataset = BaseDataset(data, WINDOW_SIZE)
    model = SkipGram(dataset.vocab_size, EMBEDDING_DIM)

    optimizer = torch.optim.Adam(model.parameters(), lr=8e-3, betas=(0.99, 0.9999))
    scheduler = ReduceLROnPlateau(optimizer, min_lr=2e-4, verbose=True)
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    losses = []

    for epoch in range(NUM_EPOCH):
        total_loss = 0

        for pos_context, center_word, neg_context in data_loader:

            model.zero_grad()
            loss = model(center_word, pos_context, neg_context)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step(total_loss)
        losses.append(total_loss)
        print(f'{epoch + 1} / {NUM_EPOCH} Loss: {losses[-1]}')

    return {
        word: model.center_word_embeddings.weight.detach().cpu().numpy()[idx] for word, idx in dataset.word2id.items()
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
        return text.split(' ')[:20_000]

import time

if __name__ == '__main__':

    data = load('C:\\Users\\Danya\\Desktop\w2v [NLP]')

    b = time.time()

    a = train(' '.join(data))

    print(f'finish time: {time.time() - b}')

    print(a['человек'])