export OMP_NUM_THREADS=4
DEVICES=$1
Master_port=$2
echo "Runs on GPU: [$DEVICES], port [$Master_port]"
DevArray=(${DEVICES//,/ })
NumGpus=${#DevArray[@]}
LABEL=res_nonPretrained_CIFAR100_B50_T10_teacher_final_loss
TODAY=$(date +"%Y%m%d")
TODAYDAY=${TODAY:6:8}
Week=$((((10#$TODAYDAY-1)/7)+1))
SAVE_PATH="results/dev/continual_trans/"${TODAY:0:6}"/week_"$Week"/"$TODAY"_"$LABEL
echo "SAVE at: "$SAVE_PATH

Config_stage_pre=options/continual_trans/non_pretrained/stage0/cifar100_stage0_deit_teacher_final_1.yaml

# Yes I know the pretraining on CIFAR100 (first 50 classes) is cumbersome, maybe there is a more simple way to train a ViT on cifar100.
CUDA_VISIBLE_DEVICES=$DEVICES python -m torch.distributed.launch --nproc_per_node=$NumGpus --master_port=$Master_port --use_env inclearn \
  --options $Config_stage_pre \
  options/data/cifar100.yaml \
  --initial-increment 50 \
  --increment 5 \
  --label $LABEL\
  --data-path ./datasets \
  --save task \
  --workers 4 \
  --force_from_stage0 \
  --only_stage0 \

mkdir $SAVE_PATH"/for_resume"
cp $SAVE_PATH"/net_0_task_0.pth" $SAVE_PATH"/for_resume"
Config_stage_pre=options/continual_trans/non_pretrained/stage0/cifar100_stage0_deit_teacher_final_2.yaml

python runs/non_pretrained/cifar_pretrain/edit_phrase1_pth.py --path $SAVE_PATH"/for_resume"
CUDA_VISIBLE_DEVICES=$DEVICES python -m torch.distributed.launch --nproc_per_node=$NumGpus --master_port=$Master_port --use_env inclearn \
  --options $Config_stage_pre \
  options/data/cifar100.yaml \
  --initial-increment 50 \
  --increment 5 \
  --label $LABEL"_resume"\
  --data-path ./datasets \
  --save task \
  --workers 4 \
  --force_from_stage0 \
  --only_stage0 \
  --resume $SAVE_PATH"/for_resume/modified_chkpt"