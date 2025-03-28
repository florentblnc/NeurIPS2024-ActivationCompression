from argparse import Namespace
import json
import logging
from pytorch_lightning.utilities.cli import LightningCLI
from pytorch_lightning.loggers import TensorBoardLogger
from classification.model import ClassificationModel
from dataloader.pl_dataset import ClsDataset
from classification.callback import LogActivationMemoryCallback

logging.basicConfig(level=logging.INFO)


class CLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_argument("--logger.save_dir", default="./runs")
        parser.add_argument("--logger.exp_name", default="test")
        parser.add_argument("--checkpoint", default=None)

    def instantiate_trainer(self, **kwargs):
        if "fit" in self.config.keys():
            cfg = self.config["fit"]
        elif "validate" in self.config.keys():
            cfg = self.config["validate"]
        else:
            cfg = self.config

        logger_name = cfg["logger"]["exp_name"] + "_" + cfg["data"]["name"]
        if "logger_postfix" in kwargs:
            logger_name += kwargs["logger_postfix"]
            kwargs.pop("logger_postfix")

        logger = TensorBoardLogger(cfg["logger"]["save_dir"], logger_name)
        kwargs["logger"] = logger

        # Force Trainer to use CPU if GPU is unavailable
        kwargs["accelerator"] = "cpu"  # Ensures it runs on GPU if available, otherwise CPU

        trainer = super(CLI, self).instantiate_trainer(**kwargs)
        return trainer


def run():
    cli = CLI(ClassificationModel, ClsDataset, run=False, save_config_overwrite=True, subclass_mode_model=True)

    print("\n✅ Loaded CLI config, now instantiating model manually...")

    model_args = cli.config["model"]["init_args"]

    # Convert Namespace to dictionary if necessary
    if isinstance(model_args, Namespace):
        model_args = vars(model_args)

    print("\n🔹 Final Model Args:")
    print(json.dumps(model_args, indent=4))

    # Manually create model instance
    model = ClassificationModel(**model_args)

    trainer = cli.trainer
    data = cli.datamodule

    print("\n🚀 Training Started\n\n")

    # ✅ Enable activation memory logging (only visible if using GPU)
    log_activation_mem = True
    if log_activation_mem:
        trainer.callbacks.append(LogActivationMemoryCallback(log_activation_mem=True))

    # Check if a checkpoint is specified
    if cli.config.get("checkpoint") not in [None, "None"]:
        trainer.fit(model, data, ckpt_path=cli.config["checkpoint"])
    else:
        trainer.fit(model, data)
        print("\n✅ Training complete.")

    # ✅ Run validation at the end and print accuracy
    print("\n🔍 Running final validation...\n")
    val_result = trainer.validate(model, datamodule=data)

    if val_result:
        final_acc = val_result[0].get("val/acc", val_result[0].get("acc", "N/A"))
        print(f"\n🎯 Final Validation Accuracy: {final_acc:.4f}\n")

if __name__ == "__main__":
    run()

