# When Prompt-based Incremental Learning Does Not Meet Strong Pretraining (ICCV2023)

Official PyTorch implementation of our ICCV2023 paper “When Prompt-based Incremental Learning Does Not Meet Strong
Pretraining”[[paper]](https://arxiv.org/abs/2308.10445)

## Environments

- Python: 3.8
- PyTorch: 1.10
- See requirements.txt for more details

## Data preparation

### Download Datasets

- **CIFAR100** dataset will be downloaded automatically to the directory specified by `--data-path`.

- **ImageNet100** dataset cannot be downloaded automatically, please download it
  from [image-net.org](https://image-net.org/download.php).
  Place the dataset the directory specified by `--data-path`.

  In order to conduct incremental training, we also need to put imagenet split file `train_100.txt`, `val_100.txt` into
  the data path. Symbolic link is recommended:
  ```
  ln -s ImageNet/imagenet_split/train_100.txt data_path/imagenet/train_100.txt
  ln -s ImageNet/imagenet_split/val_100.txt data_path/imagenet/val_100.txt
  ```
 
- **ImageNet-R** is from[The Many Faces of Robustness](https://github.com/hendrycks/imagenet-r?tab=readme-ov-file). 
  Since the official dataset didn't provide the train/test split, you can use our split in the [datalink](https://drive.google.com/drive/folders/18GdcsKIxx2MdbdjjblHXE3nqXXgjIiew?usp=drive_link).
  Place the dataset the directory specified by `--data-path`.
  
- **EuroSAT** is from the official github [repo](https://github.com/phelber/EuroSAT), you can use the version in the [datalink](https://drive.google.com/drive/folders/18GdcsKIxx2MdbdjjblHXE3nqXXgjIiew?usp=drive_link) for convenience.
  Place the dataset the directory specified by `--data-path`.

- **RESISC45** is from the paper ([url](https://arxiv.org/abs/1703.00121)), you can use the version in the [datalink](https://drive.google.com/drive/folders/18GdcsKIxx2MdbdjjblHXE3nqXXgjIiew?usp=drive_link). Place the dataset the directory specified by `--data-path`.

### Path structure

After downloading, the dataset should be organized like this

```
datasets
│  
│──imagenet
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
│──cifar-100-python
│   │ ...
│ 
│──my-imagenet-r 
│   │ ...
│ 
│──my-EuroSAT_RGB
│   │ ...
│ 
│──my-NWPU-RESISC45
│   │ ...
└
```

## Training

### Non-pretrained incremental learning

#### ImageNet-100
This setting is the default setting of class-incremental learning.
In this setting, the network is trained from scratch.

- Step 1:
Download weights from the following
[link](https://drive.google.com/drive/folders/1DRpbNpkJ2lwIPtO_PF-mFV_kKIYKeHgt?usp=drive_link) and put them in 'chkpts'
folder.

- Step 2:
  To reproduce the results of our method in Table 1 & 2:

  ```
  bash runs/non_pretrained/run_nonPretrained_imageNetSub_B50_T10.sh
  ```

##### Notes for step1:
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
bash runs/non_pretrained/imagnet_pretrain/run_nonPretrained_imageNetSub_B50_teacher.sh
```
After training, please put the checkpoint on the chkpts folder and modify the path in the script above (line28).

#### CIFAR100

- Step 1:
Download weights from the following
[link](https://drive.google.com/drive/folders/1DRpbNpkJ2lwIPtO_PF-mFV_kKIYKeHgt?usp=drive_link) and put them in 'chkpts'
folder.

- Step 2:
  To reproduce the results of our method in Table 3:

  ```
  bash runs/non_pretrained/run_nonPretrained_cifar100_B50_T10.sh
  ```
##### Notes for step1:
Like the Imagenet, here is the script for training the first stage.
  ```
  runs/non_pretrained/cifar_pretrain/run_nonPretrained_CIFAR100_B50_teacher.sh
  ```

### Pretrained incremental learning
In order to compare our method with other prompt-based methods, we also conduct experiments with pretrained weights
(ImageNet21k & TinyImageNet pretrained).
#### Download the pretrained checkpoints
Download weights from the following
[link](https://drive.google.com/drive/folders/1DRpbNpkJ2lwIPtO_PF-mFV_kKIYKeHgt?usp=drive_link) and put them in 'chkpts'
folder.
We use the pretrained-weight(deit_base_patch16_224-b5f2ef4d.pth) from the Deit [repo](https://github.com/facebookresearch/deit).
We use the tiny-ImageNet weight from this [repo](https://github.com/ehuynh1106/TinyImageNet-Transformers).
#### ImageNet-R
- To reproduce the results of our method in Table 5:
  
  (ImageNet-pretrained)
  ```
  bash runs/pretrained/ImageNetR_pretrained_vitbase_T10.sh
  ```
- To reproduce the results of our method with tiny-imagenet pretrained in Table A4 (in the Appendix):
  
  (tinyImageNet-pretrained)
  ```
  bash runs/pretrained/ImageNetR_tinyImageNet_pretrained_vitbase_T10.sh
  ```
 
#### CIFAR100

- To reproduce the results of our method in Table 4:

  (ImageNet-pretrained)
  ```
  bash runs/pretrained/CIFAR100_pretrained_vitbase_T10.sh
  ```
- To reproduce the results of our method with tiny-imagenet pretrained in Table 4 (in the Appendix):

  (tinyImageNet-pretrained)
  ```
  bash runs/pretrained/CIFAR100_tinyImageNet_pretrained_vitbase_T10.sh
  ```

#### EuroSAT and RESISC45
- To reproduce the results of our method with ImageNet pretrained in Table A2 (in the Appendix):

  (EuroSAT_RGB)
  ```
  bash runs/pretrained/EuroSAT_RGB_pretrained_T5.sh
  ```
  
  (NWPU-RESISC45)
  ```
  bash runs/pretrained/NWPU-RESISC45_pretrained_T10.sh
  ```
  
- Further, with tiny-imagenet pretrained:

  (EuroSAT_RGB)
  ```
  bash runs/pretrained/EuroSAT_RGB_tinyImageNet_pretrained_T5.sh
  ```
  
  (NWPU-RESISC45)
  ```
  bash runs/pretrained/NWPU-RESISC45_tinyImageNet_pretrained_T10.sh
  ```
  
## Todo list

-[X] Non-pretrained incremental learning on CIFAR100
-[X] Pretrained incremental learning on ImageNetR
-[X] Pretrained incremental learning on EuroSAT and RESISC45

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