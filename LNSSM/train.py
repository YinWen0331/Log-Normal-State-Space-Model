import os
import logging, types
from src.utils import Dataset, TestDataset
import torch
import numpy as np
import transformers
import json
from tqdm import tqdm
transformers.logging.set_verbosity_error()

np.set_printoptions(precision=4, suppress=True, linewidth=200)
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO,)


datafile = "../data/corpus" 
valid_datafile = "../data/SlimPajama-1B_validation.jsonl" # 
datafile_encoding = 'utf-8' 

os.environ['VOCAB_SIZE'] = '50277'
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['NUM_GPUS'] = '1' 
os.environ['FLOAT_MODE'] = 'bf16'
os.environ['DEEPSPEED'] = '1' 
if int(os.environ['NUM_GPUS']) == 1: 
    os.environ['DEEPSPEED'] = '0'    
os.environ['USE_WANDB'] = '0' 

EPOCH_BEGIN = 0 
LOAD_MODEL = False 

n_layer = 6
n_embd = 512
ctx_len = 1024 
batch_size = 18 * int(os.environ['NUM_GPUS'])
assert (batch_size % int(os.environ['NUM_GPUS']) == 0)


lr_init = 8e-4
lr_final = 1e-5


n_epoch = 500
epoch_length_fixed = (10000 // batch_size) * batch_size 

epoch_save_frequency = 10
epoch_save_path = 'trained-'



NUM_GPUS = 1
if LOAD_MODEL and EPOCH_BEGIN > 0: 
    warmup_tokens = 50 * ctx_len * batch_size // NUM_GPUS
else:
    warmup_tokens = 0

betas = (0.9, 0.99) 
eps = 1e-8

num_workers = 1 

NUM_GPUS = int(os.environ['NUM_GPUS'])
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

## training data
with open(datafile, "r", encoding=datafile_encoding) as f:
    lines = f.readlines()

tokens = []
for line in tqdm(lines, desc="Tokenizing"):
    line = line.strip()
    if not line:
        continue
    encoded = tokenizer.encode(line)
    tokens.extend(encoded)
    
train_dataset = Dataset(tokens, ctx_len, epoch_length_fixed, vocab_size=tokenizer.vocab_size)

## validation data
def tokenize_jsonl_file(filepath, ctx_len):
    token_list = []
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()  
        for line in tqdm(lines, desc=f"Tokenizing {filepath}"):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = obj.get("text", "").strip()
            if not text:
                continue
            encoded = tokenizer.encode(text)

            if len(encoded) > ctx_len:
                encoded = encoded[:ctx_len]
            token_list.append(encoded)

    return token_list

valid_tokens = tokenize_jsonl_file(valid_datafile, ctx_len)
valid_dataset = TestDataset(valid_tokens, ctx_len, vocab_size=tokenizer.vocab_size)



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

    if os.environ['DEEPSPEED'] == '0':
        if os.environ['FLOAT_MODE'] == 'fp16':
            trainer = Trainer(devices=NUM_GPUS, accelerator="gpu", precision=16)            
        elif os.environ['FLOAT_MODE'] == 'bf16':
            trainer = Trainer(devices=NUM_GPUS, accelerator="gpu", precision='bf16')
        elif '32' in os.environ['FLOAT_MODE']:
            trainer = Trainer(devices=NUM_GPUS, accelerator="gpu", precision=32)
    else:
        from pytorch_lightning.strategies import DeepSpeedStrategy
        
        DEEPSPEED_CFG = {
            "zero_allow_untested_optimizer":True,
            "zero_optimization":{
                "stage":2,
                "contiguous_gradients":True,
                "overlap_comm":True,
                "allgather_partitions":True,
                "reduce_scatter":True,
                "allgather_bucket_size":200000000,
                "reduce_bucket_size":200000000,
                "sub_group_size":1000000000000
            },
            "activation_checkpointing":{
                "partition_activations":False,
                "cpu_checkpointing":False,
                "contiguous_memory_optimization":False,
                "synchronize_checkpoint_boundary":False
            },
            "aio":{
                "block_size":1048576,
                "queue_depth":8,
                "single_submit":False,
                "overlap_events":True,
                "thread_count":1
            },
            "gradient_clipping": 1.0,
            "gradient_accumulation_steps": 1,
        }
        if NUM_GPUS == 1:
            DEEPSPEED_CFG['zero_optimization'] = {
                "stage":1, # saves some VRAM
                "contiguous_gradients":False,
                "overlap_comm":False,
                "allgather_partitions":False,
                "reduce_scatter":False,
                "allgather_bucket_size":200000000,
                "reduce_bucket_size":200000000,
                "sub_group_size":1000000000000
            }

        if os.environ['FLOAT_MODE'] == 'fp16':
            DEEPSPEED_CFG["fp16"] = {
                "fp16": True,
                "enabled": True,
                "loss_scale": 0,
                "initial_scale_power": 12,
                "loss_scale_window": 1000,
                "hysteresis": 2,
                "min_loss_scale": 1
            }
            trainer = Trainer(strategy=DeepSpeedStrategy(config=DEEPSPEED_CFG), devices=NUM_GPUS, accelerator="gpu", precision=16)
            
        elif os.environ['FLOAT_MODE'] == 'bf16':
            DEEPSPEED_CFG["bf16"] = {
                "enabled": True
            }
            trainer = Trainer(strategy=DeepSpeedStrategy(config=DEEPSPEED_CFG), devices=NUM_GPUS, accelerator="gpu", precision='bf16')

        elif '32' in os.environ['FLOAT_MODE']:
            trainer = Trainer(strategy=DeepSpeedStrategy(config=DEEPSPEED_CFG), devices=NUM_GPUS, accelerator="gpu", precision=32)

        print(trainer._strategy.config)
    
    trainer.run(m_cfg, train_dataset, valid_dataset, tconf)
