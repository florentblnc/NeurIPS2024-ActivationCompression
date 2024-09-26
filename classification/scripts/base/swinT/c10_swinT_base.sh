setup="A"
dataset="cifar10"
num_classes="10"
usr_group_kl="full_pretrain_imagenet"

# usr_group_kl=15.29
# load_args="--model.load pretrained_ckpts/swinT/c10_epoch=24-val-acc=0.809.ckpt"

general_config_args="--config configs/swinT_config.yaml"
logger_args="--logger.save_dir runs/setup$setup/swinT/$dataset/base"
data_args="--data.setup $setup --data.name $dataset --data.data_dir data/$dataset --data.train_workers 24 --data.val_workers 24 --data.partition 1 --data.usr_group data/$dataset/usr_group_${usr_group_kl}.npy"
trainer_args="--trainer.max_epochs 50"
model_args="--model.setup $setup --model.set_bn_eval True --model.use_sgd True --model.learning_rate 0.05 --model.num_classes $num_classes --model.momentum 0 --model.anneling_steps 50 --model.scheduler_interval epoch --trainer.gradient_clip_val 2.0"
seed_args="--seed_everything 233"

common_args="$general_config_args $trainer_args $data_args $model_args $load_args $logger_args $seed_args"

echo $common_args

python trainer_cls_linear.py ${common_args} --logger.exp_name base_l2_${usr_group_kl} --model.num_of_finetune 2
python trainer_cls_linear.py ${common_args} --logger.exp_name base_l4_${usr_group_kl} --model.num_of_finetune 4