import math, os
import numpy as np
import logging
import torch
import torch.nn as nn
from torch.nn import functional as F
try:
    from deepspeed.ops.adam import FusedAdam
except:
    pass 

logger = logging.getLogger(__name__)

class L2Wrap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, loss, y):
        ctx.save_for_backward(y)
        return loss
    @staticmethod
    def backward(ctx, grad_output):
        y = ctx.saved_tensors[0]
        # to encourage the logits to be close to 0
        factor = 1e-4 / (y.shape[0] * y.shape[1])
        maxx, ids = torch.max(y, -1, keepdim=True)
        gy = torch.zeros_like(y)
        gy.scatter_(-1, ids, maxx * factor)
        return (grad_output, gy)

NUM_HEAD = 4 ## 
EMBD = 512 ## 
from torch.utils.cpp_extension import load
LNSSM_cuda = load(name="LNSSM", sources=["cuda/LNSSM_op.cpp", f"cuda/LNSSM_cuda.cu"],
                    verbose=True, extra_cuda_cflags=["-res-usage", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization", f"-D_N_={NUM_HEAD / EMBD}"])

class LNSSM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, B, T, C, H, q, k, v, a):
        with torch.no_grad():
            assert q.dtype == torch.bfloat16
            assert k.dtype == torch.bfloat16
            assert v.dtype == torch.bfloat16
            assert a.dtype == torch.bfloat16

            ctx.B = B
            ctx.T = T
            ctx.C = C
            ctx.H = H
            assert q.is_contiguous()
            assert k.is_contiguous()
            assert v.is_contiguous()
            assert a.is_contiguous()

            ea = (-torch.exp(a.float())).contiguous()
            eea = (torch.exp(ea)).contiguous()
            ctx.save_for_backward(q, k, v, eea, ea)
            S = torch.empty((B, T, C), device=q.device, dtype=torch.bfloat16, memory_format=torch.contiguous_format) 
            Z = torch.empty((B, T, C), device=q.device, dtype=torch.bfloat16, memory_format=torch.contiguous_format)
            LNSSM_cuda.forward(B, T, C, H, q, k, v, eea, S, Z)
            return S, Z

    @staticmethod
    def backward(ctx, gS, gZ):
        with torch.no_grad():
            assert gS.dtype == torch.bfloat16
            assert gZ.dtype == torch.bfloat16
            B = ctx.B
            T = ctx.T
            C = ctx.C
            H = ctx.H
            assert gS.is_contiguous()
            assert gZ.is_contiguous()
            q, k, v, eea, ea = ctx.saved_tensors
            gq = torch.empty((B, T, C), device=gS.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format) 
            gk = torch.empty((B, T, C), device=gS.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format) 
            gv = torch.empty((B, T, C), device=gS.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format) 
            ga = torch.empty((B, C), device=gS.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format) 
            LNSSM_cuda.backward(B, T, C, H, q, k, v, eea, ea, gS, gZ, gq, gk, gv, ga)
            ga = torch.sum(ga, 0)

            return (None, None, None, None, gq, gk, gv, ga)

def RUN_CUDA(B, T, C, H, q, k, v, a):
    return LNSSM.apply(B, T, C, H, q.bfloat16().contiguous(), k.bfloat16().contiguous(), v.bfloat16().contiguous(), a.bfloat16().contiguous())



def Init(model, args):  

    for mm in model.modules():
        if "RecursiveScriptModule" in str(type(mm)):
            if mm.original_name not in ["Linear"]:
                continue
            ww = None
            for name, param in mm.named_parameters():
                if name == "weight":
                    ww = param
        else:
            m = mm
            if not isinstance(m, (nn.Linear, nn.Embedding)):
                continue
            ww = m.weight
        with torch.no_grad():
            name = "[unknown weight]"
            for name, parameter in model.named_parameters():  
                if id(ww) == id(parameter):
                    break

            shape = ww.shape
            gain = 1.0
            scale = 1.0  

            if isinstance(m, nn.Embedding):
                gain = math.sqrt(max(shape[0], shape[1]))
                if shape[0] == args.vocab_size and shape[1] == args.n_embd:  
                    scale = 1e-4
                else:
                    scale = 0

            if isinstance(m, nn.Linear):
                if shape[0] > shape[1]:
                    gain = math.sqrt(shape[0] / shape[1])
                if shape[0] == args.vocab_size and shape[1] == args.n_embd:  
                    scale = 0.5

            if hasattr(m, "scale_init"):
                scale = m.scale_init

            gain *= scale
            if scale == -999:
                nn.init.eye_(ww)
            elif gain == 0:
                nn.init.zeros_(ww)
            elif gain > 0:
                nn.init.orthogonal_(ww, gain=gain)
            else:
                nn.init.normal_(ww, mean=0.0, std=-scale)

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.norm(p=2, dim=-1, keepdim=True) / (x.shape[-1] ** 0.5)
        return self.scale * (x / (norm + self.eps))

class TimeMix(torch.jit.ScriptModule):
    def __init__(self, config, layer_id):
        super().__init__()
        self.layer_id = layer_id
        self.ctx_len = config.ctx_len
        self.n_embd = config.n_embd
        self.n_head = NUM_HEAD
        attn_sz = config.n_embd

        with torch.no_grad(): # fancy init
            ratio_0_to_1 = (layer_id / (config.n_layer - 1)) # 0 to 1
            ratio_1_to_almost0 = (1.0 - (layer_id / config.n_layer)) # 1 to ~0

            # fancy time_decay
            decay_speed = torch.ones(attn_sz)
            for h in range(attn_sz):
                decay_speed[h] = -5 + 8 * (h / (attn_sz-1)) ** (0.7 + 1.3 * ratio_0_to_1)
            self.time_decay = nn.Parameter(decay_speed)

            # token shift
            x = torch.ones(1, 1, config.n_embd)
            for i in range(config.n_embd):
                x[0, 0, i] = i / config.n_embd
            self.time_mix_q = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
            self.time_mix_k = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
            self.time_mix_v = nn.Parameter(torch.pow(x, ratio_1_to_almost0) + 0.3 * ratio_0_to_1)
            self.time_mix_g = nn.Parameter(torch.pow(x, 0.5 * ratio_1_to_almost0))
            
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        self.query = nn.Linear(config.n_embd, attn_sz, bias=False)
        self.key = nn.Linear(config.n_embd, attn_sz, bias=False)
        self.value = nn.Linear(config.n_embd, attn_sz, bias=False)
        self.gate = nn.Linear(config.n_embd, attn_sz, bias=False)

        self.output = nn.Linear(attn_sz, config.n_embd, bias=False)

        self.norm = RMSNorm(attn_sz)

        self.query.scale_init = 0
        self.key.scale_init = 0
        self.gate.scale_init = 0
        self.output.scale_init = 0

    @torch.jit.script_method
    def jit_func(self, x):

        xx = self.time_shift(x)
        xq = x * self.time_mix_q + xx * (1 - self.time_mix_q)
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xv = x * self.time_mix_v + xx * (1 - self.time_mix_v)
        xg = x * self.time_mix_g + xx * (1 - self.time_mix_g)

        q = self.query(xq)
        k = self.key(xk)
        v = self.value(xv)
        g = self.gate(xg)

        g = torch.sigmoid(g)

        return g, q, k, v

    def forward(self, x):
        B, T, C = x.size() 

        g, q, k, v = self.jit_func(x) 
        q = self.norm(q)
        k = self.norm(k)

        k = torch.exp(k)
        q = torch.exp(q)

        H = self.n_head
        head_dim = C // H

        S, Z = RUN_CUDA(B, T, C, H, q, k, v, self.time_decay)

        q_heads = q.view(B, T, H, head_dim)
        Z_heads = Z.view(B, T, H, head_dim)
        Z_scalar = torch.sum(q_heads * Z_heads, dim=-1, keepdim=True)

        S_heads = S.view(B, T, H, head_dim)
        att_heads = S_heads / Z_scalar

        att = att_heads.view(B, T, C)
        att = g * att

        att = self.output(att)
        return att



class ChannelMix(torch.jit.ScriptModule):
    def __init__(self, config, layer_id):
        super().__init__()
        self.layer_id = layer_id
        hidden_sz = 4 * config.n_embd
        self.linear1 = nn.Linear(config.n_embd, hidden_sz, bias=False)
        self.linear2 = nn.Linear(hidden_sz, config.n_embd, bias=False)

    @torch.jit.script_method
    def forward(self, x):
        x = self.linear1(x)
        x = F.gelu(x)
        x = self.linear2(x)

        return x
    
class LLMConfig:
    def __init__(self, vocab_size, ctx_len, **kwargs):
        self.vocab_size = vocab_size
        self.ctx_len = ctx_len
        for k, v in kwargs.items():
            setattr(self, k, v)


class Block(nn.Module):
    def __init__(self, config, layer_id):
        super().__init__()
        self.config = config
        self.layer_id = layer_id

        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)

        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(config.n_embd)

        self.att = TimeMix(config, layer_id)

        self.ffn = ChannelMix(config, layer_id)

    def forward(self, x):
        if self.layer_id == 0:
            x = self.ln0(x)        
        x = x + self.att(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class LLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.step = 0
        self.config = config

        self.emb = nn.Embedding(config.vocab_size, config.n_embd)

        self.blocks = nn.Sequential(*[Block(config, i)
                                    for i in range(config.n_layer)])

        self.ln_out = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)



        self.ctx_len = config.ctx_len

        try:
            if os.environ['LOAD_MODEL'] == str(False):
                Init(self, config) 
        except:
            pass

        logger.info("number of parameters: %e", sum(p.numel()
                    for p in self.parameters()))

    def get_ctx_len(self):
        return self.ctx_len

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear)):
            module.weight.data.normal_(mean=0.0, std=0.01)
        if isinstance(module, (nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=1e-5)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def configure_optimizers(self, train_config):
        no_decay = set()

        for mn, m in self.named_modules():  # here we disable weight_decay
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name
                no_decay.add(fpn)

        param_dict = {pn: p for pn, p in self.named_parameters()}
        optim_groups = [
            {"params": [param_dict[pn]
                        for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]

        try:
            optimizer = FusedAdam(optim_groups, lr=train_config.learning_rate, betas=train_config.betas, eps=train_config.eps, bias_correction=True, adam_w_mode=False, weight_decay=0, amsgrad=False)
        except:
            print('\n\nDeepSpeed not found. Using torch optimizer instead (probably slower)\n\n')
            optimizer = torch.optim.Adam(optim_groups, lr=train_config.learning_rate, betas=train_config.betas, eps=train_config.eps)

        return optimizer

    def forward(self, idx, targets=None):
        idx = idx.to(self.emb.weight.device)

        self.step += 1
        B, T = idx.size()
        # assert T <= self.ctx_len, "Cannot forward, because len(input) > model ctx_len."

        x = self.emb(idx)
        x = self.blocks(x)
        x = self.ln_out(x)

        x = self.head(x)

        loss = None

        # loss calculation

        # training
        if targets is not None:
            loss = F.cross_entropy(x.view(-1, x.size(-1)), targets.to(x.device).view(-1))
            
        # fine-tuning  
        # if targets is not None:
        #     x_last = x[:, -1, :]          # shape: (B, vocab_size)
        #     target_last = targets[:, -1]  # shape: (B,)
        #     loss = F.cross_entropy(x_last, target_last)
        return L2Wrap.apply(loss, x)
