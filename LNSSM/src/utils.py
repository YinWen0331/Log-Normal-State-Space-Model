import os
try:
    NUM_GPUS = int(os.environ['RWKV_NUM_GPUS'])
except:
    NUM_GPUS = 1

import json
import random
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

from transformers import GPT2TokenizerFast
os.environ["TOKENIZERS_PARALLELISM"] = "false"
class Dataset(Dataset):
    def __init__(self, token_ids, ctx_len, epoch_length_fixed, vocab_size):
        self.ctx_len = ctx_len
        self.epoch_length_fixed = epoch_length_fixed

        self.data = token_ids

        self.vocab_size = vocab_size
        self.data_size = len(self.data)
        print(f'data has {self.data_size} tokens, vocab size = {self.vocab_size}')

    def __len__(self):
        return self.epoch_length_fixed // NUM_GPUS

    def __getitem__(self, idx):
        i = np.random.randint(0, self.data_size - (self.ctx_len + 1))
        dix = self.data[i : i + self.ctx_len + 1]

        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y

class TestDataset(Dataset):
    def __init__(self, token_seqs, ctx_len, vocab_size):
        self.ctx_len = ctx_len
        self.data = token_seqs
        self.vocab_size = vocab_size
        self.data_size = len(self.data)
        print(f'Dataset has {self.data_size} samples, vocab size = {self.vocab_size}')

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        tokens = self.data[idx]
        x = torch.tensor(tokens[:-1], dtype=torch.long)
        y = torch.tensor(tokens[1:], dtype=torch.long)
        return x, y
class PromptDataset(Dataset):
    def __init__(self, jsonl_path, ctx_len, tokenizer, vocab_size):
        self.samples = []
        self.vocab_size = vocab_size
        self.ctx_len = ctx_len
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                text = data["text"]

                tokens = tokenizer.encode(text)

                input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
                target_ids = torch.tensor(tokens[1:], dtype=torch.long)
                self.samples.append((input_ids, target_ids))

        print(f"Loaded {len(self.samples)} samples for training.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

class TOKENIZER():
    def __init__(self, WORD_NAME, UNKNOWN_CHAR='\ue083'):
        if 'list' in str(type(WORD_NAME)):
            self.charMode = False
            if WORD_NAME[0] == WORD_NAME[1]:
                from transformers import PreTrainedTokenizerFast
                self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=WORD_NAME[0])
            else:
                from transformers import GPT2TokenizerFast
                self.tokenizer = GPT2TokenizerFast(WORD_NAME[0], WORD_NAME[1])
            self.vocab_size = len(self.tokenizer)
        else:
            self.charMode = True
            with open(WORD_NAME + '.json', "r", encoding="utf-16") as result_file:
                self.word_table = json.load(result_file)

            self.vocab_size = len(self.word_table)

            self.stoi = {v: int(k) for k, v in self.word_table.items()}
            self.itos = {int(k): v for k, v in self.word_table.items()}

            self.UNKNOWN_CHAR = self.stoi[UNKNOWN_CHAR]

    def refine_context(self, context):
        context = context.strip().split('\n')
        for c in range(len(context)):
            context[c] = context[c].strip().strip('\u3000').strip('\r')
        context = list(filter(lambda c: c != '', context))
        context = '\n' + ('\n'.join(context)).strip()
        if context == '':
            context = '\n'
        return context

    def sample_logits(self, out, x, ctx_len, temperature=1.0, top_p_usual=None, top_p_newline=None):

        lastChar = int(x[-1])

        probs = F.softmax(torch.tensor(out), dim=-1)

        if self.charMode:
            if self.itos[lastChar] == '\n':
                top_p = top_p_newline
            else:
                top_p = top_p_usual
        else:
            top_p = top_p_usual

        sorted_probs, s_index = torch.sort(probs, descending=True)

        cumulative_probs = torch.cumsum(sorted_probs, dim=-1).numpy()
        cutoff = float(sorted_probs[np.argmax(cumulative_probs > top_p)])

        probs[probs < cutoff] = 0

        if temperature != 1.0:
            probs = probs.pow(1.0 / temperature)

        return torch.multinomial(probs, num_samples=1)[0]
        # return torch.argmax(probs)


def to_float(x):
    return x.cpu().detach().numpy().flatten()[0].astype(float)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
