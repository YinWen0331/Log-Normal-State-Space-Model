import os
NUM_GPUS = int(os.environ['NUM_GPUS'])
USE_WANDB = (int(os.environ['USE_WANDB']) == 1)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from torch.utils.data.dataloader import DataLoader
import torch
from tqdm.auto import tqdm
import logging
import datetime
import math
from pytorch_lightning.lite import LightningLite
import gc

logger = logging.getLogger(__name__)

torch.backends.cudnn.benchmark = True
if os.environ['FLOAT_MODE'] == 'fp32':
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
else:
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True

class TrainerConfig:
    batch_size = 64
    learning_rate = 4e-4
    betas = (0.9, 0.99)
    eps = 1e-8
    grad_norm_clip = 1.0
    warmup_tokens = 0
    final_tokens = 0
    epoch_save_frequency = 0
    epoch_save_path = 'trained-'
    num_workers = 0  # for DataLoader

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

from src.model import LLM, LLMConfig

class Trainer(LightningLite):

    def get_run_name(self):
        raw_model = self.model.module if hasattr(
            self.model, "module") else self.model
        cfg = raw_model.config
        run_name = str(cfg.vocab_size) + '-' + str(cfg.ctx_len) + '-' + str(cfg.n_layer) + '-' + str(cfg.n_embd)
        return run_name

    def run(self, m_cfg, train_dataset, test_dataset, config):
        self.cuda_id = int(str(self.device).strip('cuda:'))
        print('[0]')
        model = LLM(LLMConfig(train_dataset.vocab_size, train_dataset.ctx_len,
                        n_layer=m_cfg.n_layer, n_embd=m_cfg.n_embd))
        print('[1]')
        with torch.no_grad():
            if m_cfg.LOAD_MODEL:
                print('loading', m_cfg.MODEL_NAME)
                m2 = torch.load(m_cfg.MODEL_NAME + '.pth', map_location='cpu')
                model.load_state_dict(m2)
                del m2
        model.to(self.device)

        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config
        self.avg_loss = -1
        self.EPOCH_BEGIN = m_cfg.EPOCH_BEGIN

        self.steps = self.EPOCH_BEGIN * (len(self.train_dataset) // (config.batch_size // NUM_GPUS))

        if self.cuda_id == 0:
            log_file = open("train.txt", "a")
            valid_log_file = open("valid.txt", "a")
            if USE_WANDB:
                print('logging to wandb... (comment it if you don\'t have wandb)')
                import wandb # comment this if you don't have wandb
                cfg = model.config
                for k in config.__dict__:
                    setattr(cfg, k, config.__dict__[k]) # combine cfg
                wandb.init(project="LNSSM", name=self.get_run_name() + '-' + datetime.datetime.today().strftime('%Y-%m-%d-%H-%M-%S'), config=cfg, save_code=False)

        model, config = self.model, self.config
        raw_model = model.module if hasattr(self.model, "module") else model
        optimizer = raw_model.configure_optimizers(config)
        model, optimizer = self.setup(model, optimizer)
        # print('[3]')

        def run_epoch(split):
            is_train = split == 'train'
            model.train(is_train)
            data = self.train_dataset if is_train else self.test_dataset
            data.idx_begin = self.steps * config.batch_size + 1
            data.cuda_id = self.cuda_id
            if is_train:
                loader = DataLoader(
                    data,
                    shuffle=False,
                    pin_memory=True,
                    batch_size=config.batch_size // NUM_GPUS,
                    num_workers=config.num_workers
                )
            else:
                loader = DataLoader(
                    data,
                    shuffle=False,
                    pin_memory=True,
                    batch_size=1,
                    num_workers=config.num_workers
                )                
            loader = self.setup_dataloaders(loader)

            pbar = tqdm(enumerate(loader), total=len(loader), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}') if is_train else enumerate(loader)

            gc.collect()
            torch.cuda.empty_cache()

            total_loss = 0
            count = 0

            for it, (x, y) in pbar:
                if is_train:
                    with torch.set_grad_enabled(True):
                        loss = model(x, y)
                else:
                    with torch.no_grad():

                        loss = model(x, y)

                if os.environ['DEEPSPEED'] == '0':
                    all_loss = [loss.clone()]
                else:
                    all_loss = [loss.clone() for _ in range(NUM_GPUS)]
                    torch.distributed.all_gather(all_loss, loss)

                avg_step_loss = sum([l.item() for l in all_loss]) / NUM_GPUS

                if is_train:
                    model.zero_grad()
                    self.backward(loss)
                    optimizer.step()

                    self.tokens += (y >= 0).sum()

                    # Learning rate scheduling
                    lr_final_factor = config.lr_final / config.learning_rate
                    if self.tokens < config.warmup_tokens:
                        lr_mult = lr_final_factor + (1 - lr_final_factor) * float(self.tokens) / float(config.warmup_tokens)
                        progress = 0
                    else:
                        progress = float(self.tokens - config.warmup_tokens) / float(max(1, config.final_tokens - config.warmup_tokens))
                        lr_mult = lr_final_factor if progress >= 1 else math.exp(math.log(lr_final_factor) * pow(progress, 1))
                    lr = config.learning_rate * lr_mult

                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr

                    self.lr = lr
                    self.steps += 1

                    if USE_WANDB and self.cuda_id == 0:
                        wandb.log({"loss": avg_step_loss}, step=self.steps)

                    if self.avg_loss < 0:
                        self.avg_loss = avg_step_loss
                    else:
                        factor = 1 / (it + 1)
                        self.avg_loss = self.avg_loss * (1.0 - factor) + avg_step_loss * factor

                    pbar.set_description(f"miniE {epoch+1+self.EPOCH_BEGIN} s {self.steps} prog {progress*100.0:.2f}% : ppl {math.exp(self.avg_loss):.6f} loss {self.avg_loss:.6f} lr {lr:e}")
                else:
                    total_loss += avg_step_loss
                    count += 1

            if not is_train and count > 0:
                
                test_avg_loss = total_loss / count
                test_ppl = math.exp(test_avg_loss)
                if self.cuda_id == 0:
                    print(f"[Test] Epoch {epoch+1+self.EPOCH_BEGIN} - avg loss: {test_avg_loss:.6f}, ppl: {test_ppl:.4f}")
                    valid_log_file.write(f'{epoch+1+self.EPOCH_BEGIN} {test_avg_loss:.6f} {test_ppl:.4f} {datetime.datetime.now()} {epoch+1} \n')
                    valid_log_file.flush()
                    if USE_WANDB:
                        wandb.log({"test/loss": test_avg_loss, "test/ppl": test_ppl}, step=self.steps)

        self.tokens = 0  # counter used for learning rate decay
        for epoch in range(99999999):

            run_epoch('train')
            if (epoch + 1) % 10== 0:
                run_epoch('test')

            if math.isnan(self.avg_loss):
                exit(0)

            if self.cuda_id == 0:
                log_file.write(f'{epoch+1+self.EPOCH_BEGIN} {self.avg_loss:.6f} {math.exp(self.avg_loss):.4f} {self.lr:.8f} {datetime.datetime.now()} {epoch+1} \n')
                log_file.flush()

                if (self.config.epoch_save_frequency > 0 and epoch % self.config.epoch_save_frequency == 0) or (epoch == config.max_epochs - 1):
                    raw_model = self.model.module if hasattr(self.model, "module") else self.model
                    torch.save(raw_model.state_dict(), self.config.epoch_save_path + str(epoch+1+self.EPOCH_BEGIN) + '.pth')
