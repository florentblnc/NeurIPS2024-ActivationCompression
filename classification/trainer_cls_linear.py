import logging
from pytorch_lightning.utilities.cli import LightningCLI
from pytorch_lightning.loggers import TensorBoardLogger
from classification.model_linear import ClassificationModel
from dataloader.pl_dataset import ClsDataset
from classification.callback import LogActivationMemoryCallback

logging.basicConfig(level=logging.INFO)


class CLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_argument("--logger.save_dir", default='./runs')
        parser.add_argument("--logger.exp_name", default='test')
        parser.add_argument("--checkpoint", default=None)

    def instantiate_trainer(self, **kwargs):
        if 'fit' in self.config.keys():
            cfg = self.config['fit']
        elif 'validate' in self.config.keys():
            cfg = self.config['validate']
        else:
            cfg = self.config
        logger_name = cfg['logger']['exp_name'] + "_" + cfg['data']['name']
        if 'logger_postfix' in kwargs:
            logger_name += kwargs['logger_postfix']
            kwargs.pop('logger_postfix')
        logger = TensorBoardLogger(cfg['logger']['save_dir'], logger_name)
        kwargs['logger'] = logger
        
        # kwargs['accelerator']='gpu'
        # kwargs['devices']="auto"
        trainer = super(CLI, self).instantiate_trainer(**kwargs)
        return trainer


def run():
    cli = CLI(ClassificationModel, ClsDataset, run=False, save_config_overwrite=True)
    model = cli.model
    trainer = cli.trainer
    data = cli.datamodule

    log_activation_mem = True
    if log_activation_mem:
        # Add call back to log activation memory
        callback = LogActivationMemoryCallback(log_activation_mem=log_activation_mem)
        trainer.callbacks.append(callback)
    
    # logging.info(str(model))

    if cli.config['checkpoint'] is not None and cli.config['checkpoint'] != 'None':
        # trainer.validate(model, datamodule=data, ckpt_path=cli.config['checkpoint'])
        trainer.fit(model, data, ckpt_path=cli.config['checkpoint'])
    else:
        # trainer.validate(model, datamodule=data)
        trainer.fit(model, data)
    # trainer.validate(model, datamodule=data)


run()
