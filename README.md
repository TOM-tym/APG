# When Prompt-based Incremental Learning Does Not Meet Strong Pretraining (ICCV2023)

Official PyTorch implementation of our ICCV2023 paper “When Prompt-based Incremental Learning Does Not Meet Strong
Pretraining”[[paper]]()

Codes will be released soon.

## Environments

- Python: 3.8
- PyTorch: 1.10
- See requirements.txt for more details

## Data preparation

### Download Datasets

- CIFAR100 dataset will be downloaded automatically to `data_path` specified
  in `CIFAR100/options/data/cifar100_3orders.yaml`.

- ImageNet100 dataset cannot be downloaded automatically, please download it
  from [image-net.org](https://image-net.org/download.php).
  Place the dataset in `data_path` specified in `ImageNet/options/data/imagenet100_1order.yaml`.

- ImageNet-R, EuroSAT and RESISC45 datasets: TODO

In order to conduct incremental training, we also need to put imagenet split file `train_100.txt`, `val_100.txt` into
the data path. Symbolic link is recommended:

```
ln -s ImageNet/imagenet_split/train_100.txt data_path/imagenet1k/train_100.txt
ln -s ImageNet/imagenet_split/val_100.txt data_path/imagenet1k/val_100.txt
```

### Path structure

After downloading, the dataset should be organized like this

```
data_path
│  
│──imagenet1k
│   │
│   └───train
│       │   n01440764
│       │   n01443537 
│       │   ...
│   │
│   └───val
│       │   n01440764
│       │   n01443537
│       │   ...
│   │   
│   │ train_100.txt
│   │ train_900.txt
│   │ val_100.txt 
│   
└
```

## Training

### Non-pretrained incremental learning

This setting is the default setting of class-incremental learning.
In this setting, the network is trained from scratch.

- Step 1:
Download weights from the following
[link](https://drive.google.com/drive/folders/1DRpbNpkJ2lwIPtO_PF-mFV_kKIYKeHgt?usp=drive_link) and put them in 'chkpts'
folder.

- Step 2:
To reproduce the results of our method in Table 1 & 2:

```
runs/non_pretrained/run_nonPretrained_imageNetSub_B50_T10.sh
```

### Notes:
Since this setting is not using any pretrained weights, we treat the first task as the pretraining tasks.
For ViT we use, it is difficult to train the model from scratch to match the first-task performance as the CNN (ResNets)
. 
So we use an ResNet teacher to assist the first-task training. 
The resnet teacher can be trained under repos like
[PODNet](https://github.com/arthurdouillard/incremental_learning.pytorch) or our previous work
[Imagine](https://github.com/TOM-tym/Learn-to-Imagine))

For simplicity, we provide the trained ViT weights on the first task which can be found in the link above.
Running the script above will load such weights.
If you want to train the ViT on the first task, you can run
```
runs/non_pretrained/run_nonPretrained_imageNetSub_B50_teacher.sh
```
After training, please put the checkpoint on the chkpts folder and modify the path in the script above.


### Pretrained incremental learning

In order to compare our method with other prompt-based methods, we also conduct experiments with pretrained weights
(ImageNet21k & TinyImageNet pretrained).

(TODO)

## Todo list

-[ ] Non-pretrained incremental learning on CIFAR100
-[ ] Pretrained incremental learning on ImageNetR
-[ ] Pretrained incremental learning on EuroSAT and RESISC45

## Acknowledgement

- This repository is heavily based
  on [incremental_learning.pytorch](https://github.com/arthurdouillard/incremental_learning.pytorch)
  by [arthurdouillard](https://github.com/arthurdouillard).


- If you use this paper/code in your research, please consider citing us:

```
@inproceedings{tang2022learning,
  title={When Prompt-based Incremental Learning Does Not Meet Strong Pretraining},
  author={Tang, Yu-Ming and Peng, Yi-Xing and Zheng, Wei-Shi},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision (ICCV)},
  year={2023}
}
```

## Notes

- This is a project for over one year and it is maintained by myself alone, and I am trying my best to clean up the codes. 
So please understand and forgive me for the delay in releasing the code. (If you find and bugs, please let me know :D)
- At the moment, the codes for non-pretrained incremental learning is not released yet. However, if you're in hurry to
  reimplement this part, you can replace the pretrianed weight and use the incremental config like
  `options/continual_trans/non_pretrained/imagenet_sub/imagenet100_nonPretrained_incremental.yaml` (the hyper-parameters
  may not be the optimal ones)
- The full code may be released in the middle of November, maybe after the CVPR deadline. (Sorry for the delay again)