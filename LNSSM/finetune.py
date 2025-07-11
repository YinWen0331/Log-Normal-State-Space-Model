import os
import logging, types
from src.utils import PromptDataset
import torch
import numpy as np
import transformers
transformers.logging.set_verbosity_error()
os.environ["TOKENIZERS_PARALLELISM"] = "false"
np.set_printoptions(precision=4, suppress=True, linewidth=200)
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO,)

os.environ['FLOAT_MODE'] = 'bf16'
os.environ['USE_WANDB'] = '0' 

EPOCH_BEGIN = 500 
LOAD_MODEL = True 

n_layer = 6
n_embd = 512
ctx_len = 1024 

batch_size = 1 * int(os.environ['NUM_GPUS'])
assert (batch_size % int(os.environ['NUM_GPUS']) == 0)

lr_init = 1e-6
lr_final = 1e-6

n_epoch = 100
epoch_length_fixed = (10000 // batch_size) * batch_size

epoch_save_frequency = 1
epoch_save_path = 'trained-'

NUM_GPUS = 1

warmup_tokens = 0
betas = (0.9, 0.99) 

num_workers = 1 

os.environ['LOAD_MODEL'] = str(LOAD_MODEL)
MODEL_NAME = epoch_save_path + str(EPOCH_BEGIN)


torch.backends.cudnn.benchmark = True
if os.environ['FLOAT_MODE'] == 'fp32':
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
else:
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True

from transformers import PreTrainedTokenizerFast
tokenizer = PreTrainedTokenizerFast.from_pretrained("slimpajama_tokenizer")


train_dataset = PromptDataset("../data/babilong_1k_0_fewshot.jsonl", ctx_len, tokenizer, vocab_size=tokenizer.vocab_size)

eps = 1e-8
if __name__ == '__main__':
    from src.trainer import Trainer, TrainerConfig

    print('\nmodel', os.environ['FLOAT_MODE'], 'epoch', n_epoch, 'batchsz', batch_size, 'betas',
          betas, 'eps', eps, 'ctx', ctx_len, 'layer', n_layer, 'embd', n_embd, '\n')

    tconf = TrainerConfig(max_epochs=n_epoch, batch_size=batch_size,
                          learning_rate=lr_init, lr_decay=True, lr_final=lr_final, betas=betas, eps=eps,
                          warmup_tokens=warmup_tokens, final_tokens=n_epoch*len(train_dataset)*ctx_len, num_workers=num_workers, epoch_save_frequency=epoch_save_frequency, epoch_save_path=epoch_save_path)
    m_cfg = types.SimpleNamespace()

    m_cfg.n_layer = n_layer
    m_cfg.n_embd = n_embd
    m_cfg.EPOCH_BEGIN = EPOCH_BEGIN
    m_cfg.LOAD_MODEL = LOAD_MODEL
    m_cfg.MODEL_NAME = MODEL_NAME

    if os.environ['FLOAT_MODE'] == 'fp16':
        trainer = Trainer(devices=NUM_GPUS, accelerator="gpu", precision=16)            
    elif os.environ['FLOAT_MODE'] == 'bf16':
        trainer = Trainer(devices=NUM_GPUS, accelerator="gpu", precision='bf16')
    elif '32' in os.environ['FLOAT_MODE']:
        trainer = Trainer(devices=NUM_GPUS, accelerator="gpu", precision=32)

    trainer.run(m_cfg, train_dataset, train_dataset, tconf)
