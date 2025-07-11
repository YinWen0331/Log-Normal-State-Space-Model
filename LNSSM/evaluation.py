from tqdm import tqdm
import numpy as np
import os
import json
import torch


np.set_printoptions(precision=4, suppress=True, linewidth=200)


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['FLOAT_MODE'] = 'bf16'
os.environ['RUN_DEVICE'] = 'cuda'
RUN_DEVICE = os.environ['RUN_DEVICE']
from src.model_run import LLM
from src.model import LLM, LLMConfig
from src.utils import TOKENIZER

WORD_NAME = ['slimpajama_tokenizer.json', 'slimpajama_tokenizer.json']
UNKNOWN_CHAR = None
tokenizer = TOKENIZER(WORD_NAME, UNKNOWN_CHAR=UNKNOWN_CHAR)
max = 0


MODEL_NAME = f'trained-600'

ctx_len = 1024
n_layer = 6
n_embd = 512




model = LLM(MODEL_NAME, RUN_DEVICE, tokenizer.vocab_size, n_layer, n_embd, ctx_len).cuda()
if os.environ['FLOAT_MODE'] == 'fp16':
    model = model.half()
elif os.environ['FLOAT_MODE'] == 'bf16':
    model = model.bfloat16()


TEMPERATURE = 1.0
top_p = 0.7
top_p_newline = 0.9

for i in range(1,21):
    data_path = f"../data/babilong_1k_0_qa{i}_test_95.jsonl"
    lines = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            if 'text' in item:
                lines.append(item['text'])

    correct = 0
    total = 0

    for idx, full_text in enumerate(lines):
        if "The answer is " not in full_text:
            continue

        try:
            prefix, answer = full_text.rsplit("The answer is ", 1)
        except ValueError:
            continue

        context = prefix.strip() + "The answer is"
        answer = answer.strip().split()[0]
        ctx = tokenizer.tokenizer.encode(context)
        ctx_tensor = torch.tensor(ctx).unsqueeze(0).cuda()

        with torch.no_grad():
            out = model.forward(ctx_tensor)[0]
        logits = out[-1].float().detach().cpu().numpy()

        pred_token = tokenizer.sample_logits(logits, torch.tensor(ctx), ctx_len,
                                            temperature=TEMPERATURE,
                                            top_p_usual=top_p,
                                            top_p_newline=top_p_newline)
        pred_word = (tokenizer.tokenizer.decode([pred_token.item()]).strip().split()[0]).replace('Ġ', '')



        if pred_word.lower() == answer.lower():
            correct += 1
        # else:
        #     print(f"Predicted: {pred_word}")
        #     print(f"Answer: {answer}")
        total += 1

    accuracy = correct / total if total > 0 else 0
    if accuracy >= max:
        max = accuracy
        index = i
    output_str = f"\nAccuracy: {accuracy:.2%}, qa: {i}"
    print(output_str)
    with open("BABI_0_5%_qa.txt", "a") as f: 
        f.write(output_str + "\n")
print(f"\nAccuracy: {max:.2%}, qa: {index}")



