# run on 213
export OMP_NUM_THREADS=4
DEVICES=$1
Master_port=$2
echo "Runs on GPU: [$DEVICES], port [$Master_port]"
DevArray=(${DEVICES//,/ })
NumGpus=${#DevArray[@]}
LABEL=res_nonPretrained_imageNetSub_B50_T10_try5
TODAY=$(date +"%Y%m%d")
TODAYDAY=${TODAY:6:8}
Week=$((((10#$TODAYDAY-1)/7)+1))
SAVE_PATH="results/dev/continual_trans/"${TODAY:0:6}"/week_"$Week"/"$TODAY"_"$LABEL
echo "SAVE at: "$SAVE_PATH

Config_stage0=options/continual_trans/non_pretrained/stage0/imagenet100_stage0_deit_APG.yaml
Config_incremental=options/continual_trans/non_pretrained/imagenet_sub/imagenet100_nonPretrained_incremental.yaml

echo -e "using configs: \nConfig_stage0\nConfig_incremental"

CUDA_VISIBLE_DEVICES=$DEVICES python -m torch.distributed.launch --nproc_per_node=$NumGpus --master_port=$Master_port --use_env inclearn \
  --options $Config_stage0 \
  options/data/imagenet100.yaml \
  --initial-increment 50 \
  --increment 5 \
  --label $LABEL\
  --data-path ./datasets \
  --save task \
  --resume chkpts/my_deit_B50_85.5_no_cls \
  --workers 4 \
  --force_from_stage0 \
  --only_stage0


mkdir $SAVE_PATH"/for_resume"
cp $SAVE_PATH"/net_0_task_0.pth" $SAVE_PATH"/for_resume"


CUDA_VISIBLE_DEVICES=$DEVICES python -m torch.distributed.launch --nproc_per_node=$NumGpus --master_port=$Master_port --use_env inclearn \
  --options $Config_incremental \
  options/data/imagenet100.yaml \
  --initial-increment 50 \
  --increment 5 \
  --label $LABEL"_resume"\
  --data-path ./datasets \
  --save task \
  --resume $SAVE_PATH"/for_resume" \
  --workers 4