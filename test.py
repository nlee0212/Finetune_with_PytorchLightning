# Standard

# PIP
import torch
from sklearn.metrics import accuracy_score, matthews_corrcoef, f1_score

# Custom
from config import Config
from c_dataset import CustomDataModule
from module import CustomModule

epoch = '0165'
val_loss = '7.16'
CHECKPOINT_PATH = f'./checkout/finetune-epoch={epoch}-val_loss={val_loss}.ckpt'

cfg = Config()

data_module = CustomDataModule(
    cfg=cfg,
    batch_size=cfg.batch_size,
    num_workers=cfg.NUM_WORKERS,
    time_delta=cfg.time_delta,
    option='test',
)

model = CustomModule.load_from_checkpoint(
    cfg.CHECKPOINT_PATH,
    strict=False,
    map_location="cuda:0",
    learning_rate=cfg.learning_rate,
    optimizer_name=cfg.optimizer,
    is_finetune=True,
    is_classification=True,
)
device = torch.device("cuda:0")
model.to(device)
model.eval()

count_dict = {
    'correct': 0,
    'wrong': 0,
}
probability_dict = {
    'correct': 0.0,
    'wrong': 0.0,
}

predict_list = []

with open('./test_result.csv', 'w') as test_csv:
    test_csv.write('real\tpredict\tprobaility\ttweet\n')
    for idx, (x, y) in enumerate(data_module.test_dataset):
        x = x.unsqueeze(0)
        x = x.to(device)

        y_hat = model(x)[0]
        predict = torch.argmax(y_hat, dim=0)  # 0 or 1
        probaility = y_hat[predict]  # 0.00 ~ 0.99

        predict_list.append(predict)

        idx = 'correct' if y == predict else 'wrong'
        count_dict[idx] += 1
        probability_dict[idx] += probaility

        test_csv.write(f'{y}\t{predict}\t{probaility}\t{x}\n')

data_length = len(data_module.test_dataset)
probability_dict['correct'] *= 100 / data_length
probability_dict['wrong'] *= 100 / data_length

print(f'Count - Correct: {count_dict["correct"]} / Wrong: {count_dict["wrong"]}')
print(f'Probability Avarage - Correct: {probability_dict["correct"]:.3f}% / Wrong: {probability_dict["wrong"]:.3f}%')

print(f'Acc: {accuracy_score(data_module.test_dataset.y, predict_list) * 100:.2f}%')
print(f'MCC: {matthews_corrcoef(data_module.test_dataset.y, predict_list):.4f}')
print(f'F1: {f1_score(data_module.test_dataset.y, predict_list):.4f}')
