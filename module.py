# Standard

# PIP
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import BertForSequenceClassification,BertTokenizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef

# Custom
import loss as c_loss


class CustomModule(pl.LightningModule):
    def __init__(
        self,
        d_model=1024,
        seq_len=30,
        learning_rate=1e-5,
        criterion_name='RMSE',
        optimizer_name='Adam',
        momentum=0.9,
        is_finetune=False,
        is_classification=False,
        is_estimation=False,
        **kwargs
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.is_finetune = is_finetune
        self.is_classification = is_classification
        self.is_estimation = is_estimation

        self.bert = BertForSequenceClassification.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        special_tokens_dict = {'additional_special_tokens': ['at_user','$ticker','$target_ticker']}
        num_added_toks = self.tokenizer.add_special_tokens(special_tokens_dict)
        self.bert.resize_token_embeddings(len(self.tokenizer)+num_added_toks)

        self.optimizer = self.get_optimizer(optimizer_name)

    def get_optimizer(self, optimizer_name):
        name = optimizer_name.lower()

        if name == 'SGD'.lower():
            return torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=self.momentum)
        if name == 'Adam'.lower():
            return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        if name == 'AdamW'.lower():
            return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)

        raise ValueError(f'{optimizer_name} is not on the custom optimizer list!')

    def forward(self, **kwargs):

        out = self.bert(**kwargs)

        return out

    def step(self, batch, batch_idx):
        data, labels = batch
        output = self(input_ids=data, labels=labels)

        # Transformers 4.0.0+
        loss = output.loss
        self.log('loss', float(loss))
        logits = output.logits
        print(logits)

        preds = logits.argmax(dim=-1)

        y_true = list(labels.cpu().numpy())
        y_pred = list(preds.cpu().numpy())

        return {
            'loss': loss,
            'y_true': y_true,
            'y_pred': y_pred,
        }

    def configure_optimizers(self):
        return self.optimizer

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        return self.step(batch, batch_idx)

    def epoch_end(self, outputs, state='train'):
        loss = torch.tensor(0, dtype=torch.float)
        loss_arr = []

        for i in outputs:
            loss_cpu = i['loss'].cpu().detach()
            loss += loss_cpu
            loss_arr.append(float(loss_cpu))
        loss = loss / len(outputs)

        y_true = []
        y_pred = []
        for i in outputs:
            y_true.extend(i['y_true'])
            y_pred.extend(i['y_pred'])

        y_true = [int(y) for y in y_true]
        y_pred = [int(y) for y in y_pred]

        print('y_true')
        print(f'TOTAL: {len(y_true)}')
        print(f'UP: {sum(y_true)}')
        print(f'DOWN: {len(y_true) - sum(y_true)}')

        print('y_pred')
        print(f'TOTAL: {len(y_pred)}')
        print(f'UP: {sum(y_pred)}')
        print(f'DOWN: {len(y_pred) - sum(y_pred)}')
            
        self.log(state+'_loss', float(loss), on_epoch=True, prog_bar=True)
        self.log(state+'_acc', accuracy_score(y_true, y_pred), on_epoch=True, prog_bar=True)
        self.log(state+'_precision', precision_score(y_true, y_pred), on_epoch=True, prog_bar=True)
        self.log(state+'_recall', recall_score(y_true, y_pred), on_epoch=True, prog_bar=True)
        self.log(state+'_f1', f1_score(y_true, y_pred), on_epoch=True, prog_bar=True)
        self.log(state+'_mcc', matthews_corrcoef(y_true, y_pred), on_epoch=True, prog_bar=True)

        if state=='test':
            with open('loss for filtering.txt','a') as f:
                for loss in loss_arr:
                    f.write(str(loss))
                    f.write('\t')
                f.write('\n')

        return {'loss': loss}
    
    def training_epoch_end(self, outputs):
        self.epoch_end(outputs, state='train')

    def validation_epoch_end(self, outputs):
        return self.epoch_end(outputs, state='val')

    def test_epoch_end(self, outputs):
        self.epoch_end(outputs, state='test')
