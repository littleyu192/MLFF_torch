import argparse
import os
import random
import signal
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
import warnings

from model.dp import DP
from model.MLFF import MLFFNet
from optimizer.GKF import GKFOptimizer
from optimizer.LKF import LKFOptimizer
from pre_data.data_loader_2type import get_torch_data
from trainer import *

import parameters as pm


parser = argparse.ArgumentParser(description="PyTorch MLFF Training")
parser.add_argument(
    "--datatype",
    default="float64",
    type=str,
    help="Datatype and Modeltype default float64",
)
parser.add_argument(
    "-j",
    "--workers",
    default=4,
    type=int,
    metavar="N",
    help="number of data loading workers (default: 4)",
)
parser.add_argument(
    "--epochs", default=30, type=int, metavar="N", help="number of total epochs to run"
)
parser.add_argument(
    "--start-epoch",
    default=1,
    type=int,
    metavar="N",
    help="manual epoch number (useful on restarts)",
)
parser.add_argument(
    "-b",
    "--batch-size",
    default=16,
    type=int,
    metavar="N",
    help="mini-batch size (default: 1), this is the total "
    "batch size of all GPUs on the current node when "
    "using Data Parallel or Distributed Data Parallel",
)
parser.add_argument(
    "--lr",
    "--learning-rate",
    default=0.001,
    type=float,
    metavar="LR",
    help="initial learning rate",
    dest="lr",
)
parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
parser.add_argument(
    "--wd",
    "--weight-decay",
    default=1e-4,
    type=float,
    metavar="W",
    help="weight decay (default: 1e-4)",
    dest="weight_decay",
)
parser.add_argument(
    "-p",
    "--print-freq",
    default=10,
    type=int,
    metavar="N",
    help="print frequency (default: 10)",
)
parser.add_argument(
    "-r",
    "--resume",
    dest="resume",
    action="store_true",
    help="resume the latest checkpoint",
)
parser.add_argument(
    "-s" "--store-path",
    default="default",
    type=str,
    metavar="STOREPATH",
    dest="store_path",
    help="path to store checkpoints (default: 'default')",
)
parser.add_argument(
    "-e",
    "--evaluate",
    dest="evaluate",
    action="store_true",
    help="evaluate model on validation set",
)
parser.add_argument(
    "--world-size",
    default=-1,
    type=int,
    help="number of nodes for distributed training",
)
parser.add_argument(
    "--rank", default=-1, type=int, help="node rank for distributed training"
)
parser.add_argument(
    "--dist-url",
    default="tcp://localhost:23456",
    type=str,
    help="url used to set up distributed training",
)
parser.add_argument(
    "--dist-backend", default="nccl", type=str, help="distributed backend"
)
parser.add_argument(
    "--seed", default=None, type=int, help="seed for initializing training. "
)
parser.add_argument("--magic", default=2022, type=int, help="Magic number. ")
parser.add_argument("--gpu", default=None, type=int, help="GPU id to use.")
parser.add_argument(
    "--dp", dest="dp", action="store_true", help="Whether to use DP, default False."
)
parser.add_argument(
    "--multiprocessing-distributed",
    action="store_true",
    help="Use multi-processing distributed training to launch "
    "N processes per node, which has N GPUs. This is the "
    "fastest way to use PyTorch for either single node or "
    "multi node data parallel training",
)
# parser.add_argument("--dummy", action="store_true", help="use fake data to benchmark")
parser.add_argument(
    "-n", "--net-cfg", default="DeepMD_cfg_dp_kf", type=str, help="Net Arch"
)
parser.add_argument("--act", default="sigmoid", type=str, help="activation kind")
parser.add_argument(
    "--opt", default="ADAM", type=str, help="optimizer type: LKF, GKF, ADAM, SGD"
)
parser.add_argument(
    "--Lambda", default=0.98, type=float, help="KFOptimizer parameter: Lambda."
)
parser.add_argument(
    "--nue", default=0.99870, type=float, help="KFOptimizer parameter: Nue."
)
parser.add_argument(
    "--blocksize", default=10240, type=int, help="KFOptimizer parameter: Blocksize."
)
parser.add_argument(
    "--nselect", default=24, type=int, help="KFOptimizer parameter: Nselect."
)
parser.add_argument(
    "--groupsize", default=6, type=int, help="KFOptimizer parameter: Groupsize."
)

best_loss = 1e10


def destory_process(signum, frame):
    signame = signal.Signals(signum).name
    print(
        f"Signal handler called with signal {signame} ({signum}): destory dist process"
    )
    dist.destroy_process_group()


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        warnings.warn(
            "You have chosen to seed training. "
            "This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! "
            "You may see unexpected behavior when restarting "
            "from checkpoints."
        )

    if args.gpu is not None:
        warnings.warn(
            "You have chosen a specific GPU. This will completely "
            "disable data parallelism."
        )

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    if not os.path.exists(args.store_path):
        os.mkdir(args.store_path)

    if torch.cuda.is_available():
        ngpus_per_node = torch.cuda.device_count()
    else:
        ngpus_per_node = 1
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_loss
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank,
        )
        signal.signal(signal.SIGINT, destory_process)

    if torch.cuda.is_available():
        if args.gpu:
            device = torch.device("cuda:{}".format(args.gpu))
        else:
            device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Create dataset
    train_data_path = pm.train_data_path
    train_dataset = get_torch_data(train_data_path)

    valid_data_path = pm.test_data_path
    val_dataset = get_torch_data(valid_data_path, False)

    if args.datatype == "float32":
        training_type = torch.float32  # training type is weights type
    else:
        training_type = torch.float64

    # create model
    if args.dp:
        davg, dstd, ener_shift = train_dataset.get_stat()
        stat = [davg, dstd, ener_shift]
        model = DP(args.net_cfg, args.act, device, stat, args.magic)
        model = model.to(training_type)
    else:
        model = MLFFNet(device, training_type)

    if not torch.cuda.is_available() and not torch.backends.mps.is_available():
        print("using CPU, this will be slow")
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if torch.cuda.is_available():
            if args.gpu is not None:
                torch.cuda.set_device(args.gpu)
                model.cuda(args.gpu)
                # When using a single GPU per process and per
                # DistributedDataParallel, we need to divide the batch size
                # ourselves based on the total number of GPUs of the current node.
                args.batch_size = int(args.batch_size / ngpus_per_node)
                args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
                model = torch.nn.parallel.DistributedDataParallel(
                    model, device_ids=[args.gpu], find_unused_parameters=False
                )
            else:
                model.cuda()
                # DistributedDataParallel will divide and allocate batch_size to all
                # available GPUs if device_ids are not set
                model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        model = model.to(device)
    else:
        model = model.cuda()

    # define loss function (criterion), optimizer, and learning rate scheduler
    criterion = nn.MSELoss().to(device)

    if args.opt == "LKF":
        optimizer = LKFOptimizer(
            model.parameters(),
            args.Lambda,
            args.nue,
            args.blocksize,
        )
    elif args.opt == "GKF":
        optimizer = GKFOptimizer(
            model.parameters(), args.Lambda, args.nue, device, training_type
        )
    elif args.opt == "ADAM":
        optimizer = optim.Adam(model.parameters(), args.lr)
    elif args.opt == "SGD":
        optimizer = optim.SGD(
            model.parameters(),
            args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    else:
        print("Unsupported optimizer!")
    # """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    # scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

    # optionally resume from a checkpoint
    if args.resume:
        file_name = os.path.join(args.store_path, "best.pth.tar")
        if os.path.isfile(file_name):
            print("=> loading checkpoint '{}'".format(file_name))
            if args.gpu is None:
                checkpoint = torch.load(file_name)
            elif torch.cuda.is_available():
                # Map model to be loaded to specified single gpu.
                loc = "cuda:{}".format(args.gpu)
                checkpoint = torch.load(file_name, map_location=loc)

            args.start_epoch = checkpoint["epoch"]
            best_loss = checkpoint["best_loss"]
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            # scheduler.load_state_dict(checkpoint["scheduler"])
            print(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    file_name, checkpoint["epoch"]
                )
            )
        else:
            print("=> no checkpoint found at '{}'".format(file_name))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset, shuffle=False, drop_last=True
        )
    else:
        train_sampler = None
        val_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        sampler=val_sampler,
    )

    if args.evaluate:
        # validate(val_loader, model, criterion, args)
        valid(val_loader, model, criterion, device, args)
        return

    if not args.multiprocessing_distributed or (
        args.multiprocessing_distributed and args.rank % ngpus_per_node == 0
    ):
        train_log = os.path.join(args.store_path, "epoch_train.dat")

        f_train_log = open(train_log, "w")
        f_train_log.write(
            "epoch\t loss\t RMSE_Etot\t RMSE_Ei\t RMSE_F\t real_lr\t time\n"
        )

        valid_log = os.path.join(args.store_path, "epoch_valid.dat")
        f_valid_log = open(valid_log, "w")
        f_valid_log.write("epoch\t loss\t RMSE_Etot\t RMSE_Ei\t RMSE_F\n")

    for epoch in range(args.start_epoch, args.epochs + 1):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        if args.opt == "LKF" or args.opt == "GKF":
            real_lr = args.lr
            loss, loss_Etot, loss_Force, loss_Ei, epoch_time = train_KF(
                train_loader, model, criterion, optimizer, epoch, device, args
            )
        else:
            loss, loss_Etot, loss_Force, loss_Ei, epoch_time, real_lr = train(
                train_loader, model, criterion, optimizer, epoch, args.lr, device, args
            )
        # evaluate on validation set
        vld_loss, vld_loss_Etot, vld_loss_Force, vld_loss_Ei = valid(
            val_loader, model, criterion, device, args
        )

        if not args.multiprocessing_distributed or (
            args.multiprocessing_distributed and args.rank % ngpus_per_node == 0
        ):
            f_train_log = open(train_log, "a")
            f_train_log.write(
                "%d %e %e %e %e %e %s\n"
                % (epoch, loss, loss_Etot, loss_Ei, loss_Force, real_lr, epoch_time)
            )
            f_valid_log = open(valid_log, "a")
            f_valid_log.write(
                "%d %e %e %e %e\n"
                % (epoch, vld_loss, vld_loss_Etot, vld_loss_Ei, vld_loss_Force)
            )

        # scheduler.step()

        # remember best loss and save checkpoint
        is_best = vld_loss < best_loss
        best_loss = min(loss, best_loss)

        if not args.multiprocessing_distributed or (
            args.multiprocessing_distributed and args.rank % ngpus_per_node == 0
        ):
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "best_loss": best_loss,
                    "optimizer": optimizer.state_dict(),
                    # "scheduler": scheduler.state_dict(),
                },
                is_best,
                "checkpoint.pth.tar",
                args.store_path,
            )

    if args.distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
