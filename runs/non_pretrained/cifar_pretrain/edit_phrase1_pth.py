import os
import torch
import argparse


def main(path):
    pth = os.path.join(path, 'net_0_task_0.pth')
    print(pth)
    current_pth = torch.load(pth)
    current_pth['epoch'] = 0
    current_pth['type'] = 'teacher_pretrain'

    new_path = os.path.join(path, 'modified_chkpt',)
    if not os.path.exists(new_path):
        os.makedirs(new_path)
    torch.save(current_pth, os.path.join(path, 'modified_chkpt', 'net_0_task_0.pth'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser("script for modified the chkpt", )
    parser.add_argument("--path", type=str, required=True)
    args = parser.parse_args()
    main(args.path)
