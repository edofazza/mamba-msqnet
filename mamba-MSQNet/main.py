import os
import torch
import string
import random
import numpy as np
from utils.utils import read_config
#from fvcore.nn import FlopCountAnalysis
#from fvcore.nn import flop_count_table
import time


def main(args):
    os.environ['HF_HOME'] = './.cache'
    if (args.seed >= 0):
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
        print("[INFO] Setting SEED: " + str(args.seed))
    else:
        print("[INFO] Setting SEED: None")

    if (torch.cuda.is_available() == False): print("[WARNING] CUDA is not available.")

    print("[INFO] Found", str(torch.cuda.device_count()), "GPU(s) available.", flush=True)
    device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")
    print("[INFO] Device type:", str(device), flush=True)

    config = dict()
    config['path_dataset'] = '.'  # MODIFIED, substituted get_config
    if args.dataset == "animalkingdom":
        dataset = 'AnimalKingdom'
    else:
        dataset = string.capwords(args.dataset)
    path_data = os.path.join(config['path_dataset'], dataset)
    print("[INFO] Dataset path:", path_data, flush=True)

    from datasets.datamanager import DataManager
    manager = DataManager(args, path_data)
    class_list = list(manager.get_act_dict().keys())
    num_classes = len(class_list)

    # training data
    train_transform = manager.get_train_transforms()
    train_loader = manager.get_train_loader(train_transform)
    print("[INFO] Train size:", str(len(train_loader.dataset)), flush=True)

    # val or test data
    val_transform = manager.get_test_transforms()
    val_loader = manager.get_test_loader(val_transform)
    print("[INFO] Test size:", str(len(val_loader.dataset)), flush=True)

    # evaluation metric
    if args.dataset in ['animalkingdom']:
        from torchmetrics.classification import MultilabelAveragePrecision
        eval_metric = MultilabelAveragePrecision(num_labels=num_classes, average='micro')
        eval_metric_string = 'Multilabel Average Precision'
        # criterion or loss
        import torch.nn as nn
        criterion = nn.BCEWithLogitsLoss()
    else:
        print('[INFO] No such dataset:', args.dataset)
        return

    # model
    model_args = (train_loader, val_loader, criterion, eval_metric, class_list, args.test_every, args.distributed,
                  device, args.videomamba_version)
    if args.model == 'videomambaclipinitvideoguidemultilayermamba':
        from models.videomambaclipinitvideoguidemultilayermamba import VideoMambaCLIPInitVideoGuideMultiLayerMambaExecutor
        executor = VideoMambaCLIPInitVideoGuideMultiLayerMambaExecutor(*model_args)
        """s = time.time()
        flops = FlopCountAnalysis(executor.model, torch.rand(1, 16, 3, 224, 224).to('cuda', non_blocking=True))
        print(time.time() - s)
        print(flop_count_table(flops, max_depth=1))"""
    else:
        print('[INFO] No such model:', args.model)
        return

    executor.train(args.epoch_start, args.epochs)
    eval = executor.test()
    torch.save(executor.model.state_dict(), 'mamba_msqnet.pth')
    print("[INFO] " + eval_metric_string + ": {:.2f}".format(eval * 100), flush=True)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Training script for action recognition")
    parser.add_argument("--seed", default=1, type=int, help="Seed for Numpy and PyTorch. Default: -1 (None)")
    parser.add_argument("--epoch_start", default=0, type=int, help="Epoch to start learning from, used when resuming")
    parser.add_argument("--epochs", default=100, type=int, help="Total number of epochs")
    parser.add_argument("--dataset", default="volleyball",
                        help="Dataset: volleyball, hockey, charades, ava, animalkingdom")
    parser.add_argument("--model", default="convit", help="Model: convit, query2label")
    parser.add_argument("--total_length", default=10, type=int, help="Number of frames in a video")
    parser.add_argument("--batch_size", default=32, type=int, help="Size of the mini-batch")
    parser.add_argument("--id", default="", help="Additional string appended when saving the checkpoints")
    parser.add_argument("--checkpoint", default="", help="location of a checkpoint file, used to resume training")
    parser.add_argument("--num_workers", default=4, type=int,
                        help="Number of torchvision workers used to load data (default: 8)")
    parser.add_argument("--test_every", default=5, type=int, help="Test the model every this number of epochs")
    parser.add_argument("--gpu", default="0", type=str, help="GPU id in case of multiple GPUs")
    parser.add_argument("--distributed", default=False, type=bool, help="Distributed training flag")
    parser.add_argument("--test_part", default=6, type=int, help="Test partition for Hockey dataset")
    parser.add_argument("--zero_shot", default=False, type=bool, help="Zero-shot or Fully supervised")
    parser.add_argument("--split", default=1, type=int, help="Split 1: 50:50, Split 2: 75:25")
    parser.add_argument("--train", default=False, type=bool, help="train or test")
    parser.add_argument("--videomamba_version", default='m', type=str, help="m / s / t")
    args = parser.parse_args()

    main(args)
