# Standard
import os
from tqdm import tqdm

# PIP
import torch
from sklearn.metrics import accuracy_score, matthews_corrcoef, f1_score

# Custom
from config import Config
from c_dataset import CustomDataModule
from module import CustomModule

CHECKPOINT_PATH = f'./checkout/filtering-epoch=0038-val_loss=0.26.ckpt'

cfg = Config()
device = torch.device("cuda:0")

model = CustomModule.load_from_checkpoint(
    CHECKPOINT_PATH,
    strict=False,
)
model.to(device)
model.eval()

file_list = os.listdir(cfg.TWITTER_DATASET_DIR)
file_list = sorted(file_list)
file_list = [f[:-4] for f in file_list if f.endswith('.csv')]

for idx in range(len(file_list)):
    symbol = file_list[idx]
    print(idx, symbol)
    result_dict = {}

    data_module = CustomDataModule(
        cfg=cfg,
        batch_size=cfg.batch_size,
        time_delta=cfg.time_delta,
        option='total',
        idx=idx,
    )

    for date, x, y in tqdm(data_module.test_dataset):
        x = x.unsqueeze(0)
        x = x.to(device)

        output = model(
            input_ids=x,
            output_hidden_states=True,
        )

        # last layer
        output = output.hidden_states[-1]

        # batch 0's 512 sequence vector
        output = output[0]

        if date not in result_dict:
            result_dict[date] = []

        result_dict[date].extend(output.tolist())

    max_len = 20
    with open(f'{cfg.LM_FEATURE_DIR}/{symbol}.csv', 'w') as feature_file:
        feature_file.write(f'date,{",".join([str(i) for i in range(max_len * 768)])}\n')
        for date in result_dict:
            features = [str(tmp) for tmp in result_dict[date][:max_len * 768]]
            features = ','.join(features)
            feature_file.write(f'{date},{features}\n')
