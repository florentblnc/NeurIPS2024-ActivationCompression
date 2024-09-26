
# Notes on Experiment Setup Changes

**Important**: Setup B is only available for experiments involving CIFAR-10, CIFAR-100, and ImageNet.

Each experimental script file contains `setup` variable, which specifies the setup to be applied for the experiment:
- For **Setup A**, use the default script already provided. (Example of a script using Setup A: `scripts/HOSVD_with_var_compression/mcunet/c10_mcunet_var0.8.sh`)
- For **Setup B**, the model's checkpoint has been pre-trained on half of the dataset being tested. (Example of a script using Setup B: `scripts/HOSVD_with_var_compression/mcunet/imagenet_mcunet_var0.8.sh`)

# Logging Activation Memory

To log the activation memory for each experiment, set the flag `log_activation_mem` to `True` in the `run()` function of either `trainer_cls.py` (for convolutional models) or `trainer_cls_linear.py` (for transformer models).

The output will be saved in the experiment’s result folder. (For example:
`runs/setupA/mcunet/flowers102/HOSVD/var0.8/HOSVD_l2_var0.8_full_pretrain_imagenet_flowers102/version_0/activation_memory_MB.log`  for experiment that finetunes 2 last convolutional layers of MCUNet using HOSVD with $\varepsilon=0.8$ following setup A and FLowers102 dataset)

You can use the `read_result.py` script to summarize the results into a single `.xlsx` file.