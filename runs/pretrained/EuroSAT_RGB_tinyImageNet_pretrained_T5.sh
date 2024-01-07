export OMP_NUM_THREADS=4
DEVICES=$1
Master_port=$2
echo "Runs on GPU: [$DEVICES], port [$Master_port]"
DevArray=(${DEVICES//,/ })
NumGpus=${#DevArray[@]}
LABEL=EuroSAT_RGB_tinyImageNet_pretrained_T5

TODAY=$(date +"%Y%m%d")
TODAYDAY=${TODAY:6:8}
Week=$((((10#$TODAYDAY-1)/7)+1))
SAVE_PATH="results/dev/continual_trans/"${TODAY:0:6}"/week_"$Week"/"$TODAY"_"$LABEL
echo "SAVE at: "$SAVE_PATH

Config_stage1=options/continual_trans/pretrained/satellite_tinyImageNet_pretrained_vitbase.yaml

CUDA_VISIBLE_DEVICES=$DEVICES python -m torch.distributed.launch --nproc_per_node=$NumGpus --master_port=$Master_port --use_env inclearn \
  --options $Config_stage1 \
  options/data/EuroSAT_RGB.yaml \
  --initial-increment 2 \
  --increment 2 \
  --label $LABEL\
  --data-path ./datasets \
  --save task \
  --workers 4