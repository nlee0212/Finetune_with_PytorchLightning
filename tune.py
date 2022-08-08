# Standard

# PIP
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.tune.integration.pytorch_lightning import TuneReportCallback

# Custom
from config import Config
from c_dataset import CustomDataModule
from module import CustomModule


ray.init(dashboard_host='0.0.0.0')


def train_tune(config, num_epochs=10, num_gpus=0):
    cfg = Config(config['seed'])

    tb_logger = TensorBoardLogger('logs/', name='hypertune')

    tune_report_callback = TuneReportCallback(
        {
            'loss': 'val_loss',
            'acc': 'val_acc',
        },
        on='validation_end',
    )

    data_module = CustomDataModule(
        cfg=cfg,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
    )
    train = data_module.train_dataloader()
    val = data_module.val_dataloader()
    test = data_module.test_dataloader()

    model = CustomModule(
        learning_rate=config['learning_rate'],
        criterion_name=cfg.criterion,
        optimizer_name=config['optimizer'],
        is_finetune=True,
        is_classification=True,
        is_estimation=False,
    )
    trainer = Trainer(
        gpus=num_gpus,
        max_epochs=cfg.max_epochs,
        logger=tb_logger,
        progress_bar_refresh_rate=1,
        deterministic=True,
        precision=16,
        callbacks=[
            tune_report_callback,
        ],
        num_sanity_val_steps=0,
        # accelerator = 'ddp2'
    )

    trainer.fit(model, train, val)


def tune_asha(num_samples=10, num_epochs=10):
    num_cpus = 1 
    num_gpus = 1

    config = {
        'seed': tune.randint(0, 100),
        'learning_rate': tune.loguniform(1e-8, 1e-5),
        # 'batch_size': tune.choice([16, 8]),
        'optimizer': tune.choice(['AdamW','SGD','Adam'])
    }

    scheduler = ASHAScheduler(
        max_t=num_epochs,
        grace_period=3,
        reduction_factor=2,
    )

    reporter = CLIReporter(
        parameter_columns=['seed', 'learning_rate', 'optimizer'],
        metric_columns=['loss', 'acc', 'training_iteration'],
    )

    analysis = tune.run(
        tune.with_parameters(
            train_tune,
            num_epochs=num_epochs,
            num_gpus=num_gpus,
        ),
        resources_per_trial={
            'cpu': num_cpus,
            'gpu': num_gpus,
        },
        metric='loss',
        mode='min',
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter,
        # resume=True,
        name='tune_asha',
    )

    print('Best hyperparameters found were: ', analysis.best_config)


tune_asha(
    num_samples=20,
    num_epochs=5,
)
