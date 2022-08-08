# Standard

# PIP
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
import seaborn as sns
import matplotlib.pyplot as plt

# Custom
from config import Config
from c_dataset import CustomDataModule
from module import CustomModule


tb_logger = TensorBoardLogger('logs/', name='timedelta1_stocknet_test')
# csv_logger = CSVLogger('./', name='pretrain', version='0'),

cfg = Config()

early_stop_callback = EarlyStopping(
    monitor='val_loss',
    min_delta=0.00,
    patience=100,
    verbose=False,
    mode='min'
)

checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    dirpath='./checkout/',
    filename='filtering-{epoch:04d}-{val_loss:.4f}',
    save_top_k=3,
    mode='min',
)

trainer = Trainer(
    gpus=[3],
    max_epochs=cfg.max_epochs,
    logger=tb_logger,
    # logger=csv_logger,
    progress_bar_refresh_rate=1,
    deterministic=True,
    precision=16,
    callbacks=[
        early_stop_callback,
        checkpoint_callback,
    ],
    num_sanity_val_steps=0,
    # accelerator='ddp'
)

ckpt = './checkout/finetune-epoch=0073-val_loss=0.05.ckpt'

filter_data_module = CustomDataModule(
    cfg=cfg,
    batch_size=cfg.batch_size,
    num_workers=cfg.num_workers,
    option='filter'
)

print('gathering test data (stocknet) ...')
test = filter_data_module.test_dataloader()
print('gathering test data (for filtering) ...')
filter_test = filter_data_module.filter_dataloader()


print(f'load model from checkpoint with path: {ckpt}')
model_test = CustomModule.load_from_checkpoint(ckpt)
model_test.is_filtering = True

print('Start Generating CE Loss graph')

print('\tStocknet Data')

trainer.test(model_test,test_dataloaders=test)

print('\tOur Data')

trainer.test(model_test,test_dataloaders=filter_test)

with open('loss for filtering.txt','r') as f:
    loss_lists = f.read().split('\n')
    stocknet_loss = loss_lists[0].split('\t')
    our_loss = loss_lists[1].split('\t')

    stocknet_loss = [float(loss) for loss in stocknet_loss]
    our_loss = [float(loss) for loss in our_loss]

    sns.distplot(stocknet_loss,color='blue',label='stocknet')
    sns.distplot(our_loss,color='orange',label='ours')

    plt.legend(title='Tweet Data Source')
    plt.title('CE Loss for each Tweet Data Source')
    plt.savefig('./ce_loss_graph')
