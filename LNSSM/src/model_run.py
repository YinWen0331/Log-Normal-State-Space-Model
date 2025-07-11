import types
import copy
import torch
import math, os
from torch.nn import functional as F
import torch.nn as nn


DEBUG_TIME = False 

if os.environ['RUN_DEVICE'] == 'cuda':
    NUM_HEAD = 4 ## 
    EMBD = 512 ## 

    from torch.utils.cpp_extension import load
    LNSSM_cuda = load(name="LNSSM", sources=["cuda/LNSSM_op.cpp", f"cuda/LNSSM_cuda.cu"],
                    verbose=True, extra_cuda_cflags=["-res-usage", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization", f"-D_N_={NUM_HEAD/EMBD}"])

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

############################################################################################################

CFG = types.SimpleNamespace()

class ChannelMix(nn.Module):
    def __init__(self, layer_id):
        super().__init__()
        self.layer_id = layer_id

        hidden_sz = 4 * CFG.n_embd
        self.linear1 = nn.Linear(CFG.n_embd, hidden_sz, bias=False)
        self.linear2 = nn.Linear(hidden_sz, CFG.n_embd, bias=False)

    def forward(self, x):

        x = self.linear1(x)
        x = F.gelu(x)
        x = self.linear2(x)
        return x

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.norm(p=2, dim=-1, keepdim=True) / (x.shape[-1] ** 0.5)
        return self.scale * (x / (norm + self.eps))

class TimeMix(nn.Module):
    def __init__(self, layer_id):
        super().__init__()
        self.layer_id = layer_id
        self.time_decay = nn.Parameter(torch.ones(CFG.n_embd))

        self.n_head = 4
        self.time_shift = nn.ZeroPad2d((0,0,1,-1))
        self.time_mix_q = nn.Parameter(torch.ones(1,1,CFG.n_embd))
        self.time_mix_k = nn.Parameter(torch.ones(1,1,CFG.n_embd))
        self.time_mix_v = nn.Parameter(torch.ones(1,1,CFG.n_embd))
        self.time_mix_g = nn.Parameter(torch.ones(1,1,CFG.n_embd))

        self.query = nn.Linear(CFG.n_embd, CFG.n_embd, bias=False)
        self.key = nn.Linear(CFG.n_embd, CFG.n_embd, bias=False)
        self.value = nn.Linear(CFG.n_embd, CFG.n_embd, bias=False)
        self.gate = nn.Linear(CFG.n_embd, CFG.n_embd, bias=False)

        self.norm = RMSNorm(CFG.n_embd)

        self.output = nn.Linear(CFG.n_embd, CFG.n_embd, bias=False)

    def forward(self, x):
        B, T, C = x.size()

        xx = self.time_shift(x)
        xq = x * self.time_mix_q + xx * (1 - self.time_mix_q)
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xv = x * self.time_mix_v + xx * (1 - self.time_mix_v)
        xg = x * self.time_mix_g + xx * (1 - self.time_mix_g)

        q = self.query(xq)
        k = self.key(xk)
        v = self.value(xv)
        g = self.gate(xg)

        q_norm = self.norm(q)
        k_norm = self.norm(k)

        q_exp = torch.exp(q_norm)
        k_exp = torch.exp(k_norm)

        H = self.n_head
        g = torch.sigmoid(g)
        S, Z = RUN_CUDA(B, T, C, H, q_exp, k_exp, v, self.time_decay)

        head_dim = C // H
        q_heads = q_exp.view(B, T, H, head_dim)
        Z_heads = Z.view(B, T, H, head_dim)
        Z_scalar = torch.sum(q_heads * Z_heads, dim=-1, keepdim=True) 


        S_heads = S.view(B, T, H, head_dim)
        att_heads = S_heads / Z_scalar
        att = att_heads.view(B, T, C)
        att = g * att
        att = self.output(att)
        return att

class Block(nn.Module):
    def __init__(self, layer_id):
        super().__init__()
        self.layer_id = layer_id

        self.ln1 = nn.LayerNorm(CFG.n_embd)
        self.ln2 = nn.LayerNorm(CFG.n_embd)
        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(CFG.n_embd)

        self.att = TimeMix(layer_id)

        self.ffn = ChannelMix(layer_id)

    def forward(self, x):
        if self.layer_id == 0:
            x = self.ln0(x)
        x = x + self.att(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class LLM(nn.Module):
    def __init__(self, MODEL_NAME, RUN_DEVICE, vocab_size, n_layer, n_embd, ctx_len):
        global CFG
        super().__init__()

        CFG.RUN_DEVICE = RUN_DEVICE
        CFG.vocab_size = vocab_size
        CFG.n_layer = n_layer
        CFG.n_embd = n_embd
        CFG.ctx_len = ctx_len

        print('\nloading LLM', MODEL_NAME)

        self.emb = nn.Embedding(vocab_size, n_embd)

        self.blocks = nn.Sequential(*[Block(i) for i in range(n_layer)])

        self.ln_out = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)

        self.ctx_len = ctx_len
        self.eval()
        self.load_state_dict(torch.load(MODEL_NAME + '.pth'))
        self.eval()

    def forward(self, idx):
        B, T = idx.size()
        # assert T <= self.ctx_len, "Cannot forward, because len(input) > model ctx_len."
        
        x = self.emb(idx)
        x = self.blocks(x)
        x = self.ln_out(x)

        x = self.head(x)        

        return x
