# LNSSM: Log-Normal State-Space Model

## Structure
```
./
├── LNSSM
    ├──src
        ├──model.py
        ├──model_run.py
        ├──trainer.py
        └──utils.py
    ├──cuda
    ├──tokenizer
    ├──train.py
    ├──finetune.py
    └──evaluation.py
├── data
    ├──training dataset
    ├──validation dataset
    ├──fine-tuning dataset
    └──evaluation dataset

└── README.md
```
## Preprocessing- Data Download
Please download the file throught google drive link below and unzip to ./data:
- [Google Link](https://drive.google.com/drive/folders/19KwnEuuf3sfST8nIfoRlUZX6r8GnEGnx?usp=drive_link)

## How to Run
- Enter Folder **LNSSM**
### Training
- Setup **datasets** and **hyper-parameter** in **train.py**
    ```
        # datasets
        datafile = "../data/corpus" 
        valid_datafile = "../data/SlimPajama-1B_validation.jsonl"

        # hyper-parameter
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
    ```
- Run the bash command:
    ```
    python3 train.py
    ```
### Fine-tuning
- Setup **loss calculation** in **./src/model.py**
    ```
        # fine-tuning  
        if targets is not None:
            x_last = x[:, -1, :]          # shape: (B, vocab_size)
            target_last = targets[:, -1]  # shape: (B,)
            loss = F.cross_entropy(x_last, target_last)
    ```
- Setup **datasets** in **finetune.py**
    ```
        # datasets
        train_dataset = PromptDataset("../data/babilong_1k_0_fewshot.jsonl", ctx_len, tokenizer, vocab_size=tokenizer.vocab_size)
    ```
- Run the bash command:
    ```
    python3 finetune.py
    ```
### Evaluation
- Setup **datasets** and **model** in **finetune.py**
    ```
        # datasets
        data_path = f"../data/babilong_1k_0_qa{i}_test_95.jsonl"

        # model
        MODEL_NAME = f'trained-600'    
    ```
- Run the bash command:
    ```
    python3 evaluation.py
    ```
