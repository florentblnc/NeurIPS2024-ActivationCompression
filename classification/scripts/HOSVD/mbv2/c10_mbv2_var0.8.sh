setup="A"
dataset="cifar10"
num_classes="10"
var="0.8"
usr_group_kl="full_pretrain_imagenet"

general_config_args="--config configs/mbv2_config.yaml"
data_args="--data.setup $setup --data.name $dataset --data.data_dir data/$dataset --data.train_workers 8 --data.val_workers 8 --data.partition 1 --data.usr_group data/$dataset/usr_group_${usr_group_kl}.npy"
trainer_args="--trainer.max_epochs 5"
model_args="--model.setup $setup --model.explained_variance_threshold $var --model.with_HOSVD True --model.set_bn_eval True --model.use_sgd True --model.learning_rate 0.05 --model.num_classes $num_classes --model.momentum 0 --model.anneling_steps 50 --model.scheduler_interval epoch"
seed_args="--seed_everything 233"

# ✅ REMOVE --logger.save_dir and --logger.exp_name
common_args="$general_config_args $trainer_args $data_args $model_args $seed_args"

echo $common_args

python trainer_cls.py ${common_args} --model.num_of_finetune 0
python trainer_cls.py ${common_args} --model.num_of_finetune 6
