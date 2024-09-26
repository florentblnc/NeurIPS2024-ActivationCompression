setup="B"
dataset="imagenet"
num_classes="1000"
var="0.8"
usr_group_kl=13.10
load_args="--model.load pretrained_ckpts/res18/pretrain_13.10_imagenet/version_0/checkpoints/epoch=179-val-acc=0.753.ckpt"

# # Set this variable if want to resume training
# checkpoint="--checkpoint xxx"

general_config_args="--config configs/resnet18_config.yaml"
logger_args="--logger.save_dir runs/setup$setup/resnet18/$dataset/SVD/var$var"
data_args="--data.setup $setup --data.name $dataset --data.data_dir data/$dataset --data.train_workers 5 --data.val_workers 5 --data.partition 1 --data.usr_group data/$dataset/usr_group_${usr_group_kl}.npy --data.batch_size 64"
trainer_args="--trainer.max_epochs 90 --trainer.gradient_clip_val 2.0"
model_args="--model.setup $setup --model.explained_variance_threshold $var --model.with_SVD True --model.set_bn_eval True --model.use_sgd True --model.learning_rate 0.005 --model.lr_warmup 4 --model.num_classes $num_classes --model.momentum 0.9 --model.anneling_steps 90 --model.scheduler_interval epoch"
seed_args="--seed_everything 233"

common_args="$general_config_args $trainer_args $data_args $model_args $load_args $logger_args $seed_args $checkpoint"

echo $common_args
# There are 20 convolutional layers in resnet18
python trainer_cls.py ${common_args} --logger.exp_name SVD_l2_var${var}_${usr_group_kl} --model.num_of_finetune 2
python trainer_cls.py ${common_args} --logger.exp_name SVD_l4_var${var}_${usr_group_kl} --model.num_of_finetune 4