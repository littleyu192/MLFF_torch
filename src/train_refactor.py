import argparse
from fileinput import filename
import os
import random
import shutil
import time
import warnings
from enum import Enum

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Subset
from torch.autograd import Variable

from model.dp import DP
from model.MLFF import MLFFNet

from optimizer.kalmanfilter import (
    GKalmanFilter,
    LKalmanFilter,
    SKalmanFilter,
    L1KalmanFilter,
)
from optimizer.LKF import LKFOptimizer
from optimizer.GKF import GKFOptimizer
from optimizer.KFWrapper import KFOptimizerWrapper

import math
import sys

sys.path.append(os.getcwd())
import parameters as pm

codepath = os.path.abspath(sys.path[0])
sys.path.append(codepath + "/pre_data")
sys.path.append(codepath + "/..")
from data_loader_2type import MovementDataset, get_torch_data
from scalers import DataScalers

model_names = sorted(
    name
    for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
)

parser = argparse.ArgumentParser(description="PyTorch MLFF Training")
parser.add_argument(
    "--datatype", default="float64", type=str, help="Datatype default float64"
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
    default=0,
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
    default=0.1,
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
    "--dp", default=True, type=bool, help="Weather to use DP, default True."
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
parser.add_argument("--net-cfg", default="DeepMD_cfg_dp_kf", type=str, help="Net Arch")
parser.add_argument("--act", default="sigmoid", type=str, help="activation kind")
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

best_acc1 = 0
best_loss = 1e10


def main():
    args = parser.parse_args()

    # ================================================

    # ================================================

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
    global best_acc1
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
    # create model
    if args.dp:
        davg, dstd, ener_shift = train_dataset.get_stat(image_num=10)
        stat = [davg, dstd, ener_shift]
        model = DP(args.net_cfg, args.act, device, stat, args.magic)
    else:
        model = MLFFNet(device)

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
        model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion), optimizer, and learning rate scheduler
    criterion = nn.MSELoss().to(device)

    # TODO: set optimizer

    # optimizer = torch.optim.SGD(
    #     model.parameters(),
    #     args.lr,
    #     momentum=args.momentum,
    #     weight_decay=args.weight_decay,
    # )
    optimizer = LKFOptimizer(
        model.parameters(), args.Lambda, args.nue, args.blocksize, device
    )

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
        # train_log = args.store_path + "epoch_train.dat"
        # import ipdb;ipdb.set_trace()
        f_train_log = open(train_log, "w")
        f_train_log.write("epoch\t loss\t RMSE_Etot\t RMSE_Ei\t RMSE_F\t time\n")

        valid_log = os.path.join(args.store_path, "epoch_valid.dat")
        f_valid_log = open(valid_log, "w")
        f_valid_log.write("epoch\t loss\t RMSE_Etot\t RMSE_Ei\t RMSE_F\n")

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        loss, loss_Etot, loss_Force, loss_Ei, epoch_time = train_LKF(
            train_loader, model, criterion, optimizer, epoch, device, args
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
                "%d %e %e %e %e %s\n"
                % (epoch, loss, loss_Etot, loss_Ei, loss_Force, epoch_time)
            )
            f_valid_log = open(valid_log, "a")
            f_valid_log.write(
                "%d %e %e %e %e\n"
                % (epoch, vld_loss, vld_loss_Etot, vld_loss_Ei, vld_loss_Force)
            )

        # scheduler.step()

        # remember best loss and save checkpoint
        is_best = loss < best_loss
        best_loss = min(loss, best_loss)

        if not args.multiprocessing_distributed or (
            args.multiprocessing_distributed and args.rank % ngpus_per_node == 0
        ):
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "best_loss": best_loss,
                    "optimizer": optimizer.state_dict(),
                    # "scheduler": scheduler.state_dict(),
                },
                is_best,
                "checkpoint" + str(epoch) + ".pth.tar",
                args.store_path,
            )


# def train(train_loader, model, criterion, optimizer, epoch, device, args):
#     batch_time = AverageMeter("Time", ":6.3f")
#     data_time = AverageMeter("Data", ":6.3f")
#     losses = AverageMeter("Loss", ":.4e")
#     losses = AverageMeter("Etot", ":.4e")
#     losses = AverageMeter("Force", ":.4e")
#     losses = AverageMeter("Ei", ":.4e")
#     top1 = AverageMeter("Acc@1", ":6.2f")
#     top5 = AverageMeter("Acc@5", ":6.2f")
#     progress = ProgressMeter(
#         len(train_loader),
#         [batch_time, data_time, losses, top1, top5],
#         prefix="Epoch: [{}]".format(epoch),
#     )

#     # switch to train mode
#     model.train()

#     end = time.time()
#     for i, (images, target) in enumerate(train_loader):
#         # measure data loading time
#         data_time.update(time.time() - end)

#         # move data to the same device as model
#         images = images.to(device, non_blocking=True)
#         target = target.to(device, non_blocking=True)

#         # compute output
#         output = model(images)
#         loss = criterion(output, target)

#         # measure accuracy and record loss
#         acc1, acc5 = accuracy(output, target, topk=(1, 5))
#         losses.update(loss.item(), images.size(0))
#         top1.update(acc1[0], images.size(0))
#         top5.update(acc5[0], images.size(0))

#         # compute gradient and do SGD step
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         # measure elapsed time
#         batch_time.update(time.time() - end)
#         end = time.time()

#         if i % args.print_freq == 0:
#             progress.display(i + 1)


def train_LKF(train_loader, model, criterion, optimizer, epoch, device, config):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4e", Summary.AVERAGE)
    loss_Etot = AverageMeter("Etot", ":.4e", Summary.ROOT)
    loss_Force = AverageMeter("Force", ":.4e", Summary.ROOT)
    loss_Ei = AverageMeter("Ei", ":.4e", Summary.ROOT)
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, loss_Etot, loss_Force, loss_Ei],
        prefix="Epoch: [{}]".format(epoch),
    )

    KFOptWrapper = KFOptimizerWrapper(
        model, optimizer, config.nselect, config.groupsize, config.distributed, "torch"
    )

    # switch to train mode
    model.train()

    end = time.time()
    for i, sample_batches in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if config.datatype == "float64":
            Ei_label = Variable(
                sample_batches["output_energy"][:, :, :].double().to(device)
            )
            Force_label = Variable(
                sample_batches["output_force"][:, :, :].double().to(device)
            )  # [40,108,3]
            if pm.dR_neigh:
                dR = Variable(
                    sample_batches["input_dR"].double().to(device), requires_grad=True
                )
                dR_neigh_list = Variable(
                    sample_batches["input_dR_neigh_list"].to(device)
                )
                Ri = Variable(
                    sample_batches["input_Ri"].double().to(device), requires_grad=True
                )
                Ri_d = Variable(sample_batches["input_Ri_d"].to(device))
            else:
                Egroup_label = Variable(
                    sample_batches["input_egroup"].double().to(device)
                )
                input_data = Variable(
                    sample_batches["input_feat"].double().to(device), requires_grad=True
                )
                dfeat = Variable(
                    sample_batches["input_dfeat"].double().to(device)
                )  # [40,108,100,42,3]
                egroup_weight = Variable(
                    sample_batches["input_egroup_weight"].double().to(device)
                )
                divider = Variable(sample_batches["input_divider"].double().to(device))

        elif config.datatype == "float32":
            Ei_label = Variable(
                sample_batches["output_energy"][:, :, :].float().to(device)
            )
            Force_label = Variable(
                sample_batches["output_force"][:, :, :].float().to(device)
            )  # [40,108,3]
            if pm.dR_neigh:
                dR = Variable(
                    sample_batches["input_dR"].float().to(device), requires_grad=True
                )
                dR_neigh_list = Variable(
                    sample_batches["input_dR_neigh_list"].to(device)
                )
                Ri = Variable(
                    sample_batches["input_Ri"].double().to(device), requires_grad=True
                )
                Ri_d = Variable(sample_batches["input_Ri_d"].to(device))
            else:
                Egroup_label = Variable(
                    sample_batches["input_egroup"].float().to(device)
                )
                input_data = Variable(
                    sample_batches["input_feat"].float().to(device), requires_grad=True
                )
                dfeat = Variable(
                    sample_batches["input_dfeat"].float().to(device)
                )  # [40,108,100,42,3]
                egroup_weight = Variable(
                    sample_batches["input_egroup_weight"].float().to(device)
                )
                divider = Variable(sample_batches["input_divider"].float().to(device))

        Etot_label = torch.sum(Ei_label, dim=1)
        neighbor = Variable(
            sample_batches["input_nblist"].int().to(device)
        )  # [40,108,100]
        natoms_img = Variable(sample_batches["natoms_img"].int().to(device))

        if config.dp:
            Etot_predict, Ei_predict, Force_predict = model(
                Ri, Ri_d, dR_neigh_list, natoms_img, None, None
            )
        else:
            Etot_predict, Ei_predict, Force_predict = model(
                input_data, dfeat, neighbor, natoms_img, egroup_weight, divider
            )

        loss_F_val = criterion(Force_predict, Force_label)
        loss_Etot_val = criterion(Etot_predict, Etot_label)
        loss_Ei_val = criterion(Ei_predict, Ei_label)
        loss_val = loss_F_val + loss_Etot_val

        if config.dp:
            kalman_inputs = [Ri, Ri_d, dR_neigh_list, natoms_img, None, None]
        else:
            kalman_inputs = [
                input_data,
                dfeat,
                neighbor,
                natoms_img,
                egroup_weight,
                divider,
            ]

        KFOptWrapper.update_energy(kalman_inputs, Etot_label)
        KFOptWrapper.update_force(kalman_inputs, Force_label)

        batch_size = Ri.shape[0]
        # measure accuracy and record loss
        losses.update(loss_val.item(), batch_size)
        loss_Etot.update(loss_Etot_val.item(), batch_size)
        loss_Ei.update(loss_Ei_val.item(), batch_size)
        loss_Force.update(loss_F_val.item(), batch_size)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.print_freq == 0:
            progress.display(i + 1)

    progress.display_summary(["Training Set:"])
    return losses.avg, loss_Etot.root, loss_Force.root, loss_Ei.root, batch_time.sum


def valid(val_loader, model, criterion, device, args):
    def run_validate(loader, base_progress=0):
        end = time.time()
        for i, sample_batches in enumerate(loader):
            i = base_progress + i
            if args.datatype == "float64":
                Ei_label = Variable(
                    sample_batches["output_energy"][:, :, :].double().to(device)
                )
                Force_label = Variable(
                    sample_batches["output_force"][:, :, :].double().to(device)
                )  # [40,108,3]
                if pm.dR_neigh:
                    dR = Variable(
                        sample_batches["input_dR"].double().to(device),
                        requires_grad=True,
                    )
                    dR_neigh_list = Variable(
                        sample_batches["input_dR_neigh_list"].to(device)
                    )
                    Ri = Variable(
                        sample_batches["input_Ri"].double().to(device),
                        requires_grad=True,
                    )
                    Ri_d = Variable(sample_batches["input_Ri_d"].to(device))
                else:
                    Egroup_label = Variable(
                        sample_batches["input_egroup"].double().to(device)
                    )
                    input_data = Variable(
                        sample_batches["input_feat"].double().to(device),
                        requires_grad=True,
                    )
                    dfeat = Variable(
                        sample_batches["input_dfeat"].double().to(device)
                    )  # [40,108,100,42,3]
                    egroup_weight = Variable(
                        sample_batches["input_egroup_weight"].double().to(device)
                    )
                    divider = Variable(
                        sample_batches["input_divider"].double().to(device)
                    )

            elif args.datatype == "float32":
                Ei_label = Variable(
                    sample_batches["output_energy"][:, :, :].float().to(device)
                )
                Force_label = Variable(
                    sample_batches["output_force"][:, :, :].float().to(device)
                )  # [40,108,3]
                if pm.dR_neigh:
                    dR = Variable(
                        sample_batches["input_dR"].float().to(device),
                        requires_grad=True,
                    )
                    dR_neigh_list = Variable(
                        sample_batches["input_dR_neigh_list"].to(device)
                    )
                    Ri = Variable(
                        sample_batches["input_Ri"].double().to(device),
                        requires_grad=True,
                    )
                    Ri_d = Variable(sample_batches["input_Ri_d"].to(device))
                else:
                    Egroup_label = Variable(
                        sample_batches["input_egroup"].float().to(device)
                    )
                    input_data = Variable(
                        sample_batches["input_feat"].float().to(device),
                        requires_grad=True,
                    )
                    dfeat = Variable(
                        sample_batches["input_dfeat"].float().to(device)
                    )  # [40,108,100,42,3]
                    egroup_weight = Variable(
                        sample_batches["input_egroup_weight"].float().to(device)
                    )
                    divider = Variable(
                        sample_batches["input_divider"].float().to(device)
                    )

            Etot_label = torch.sum(Ei_label, dim=1)
            neighbor = Variable(
                sample_batches["input_nblist"].int().to(device)
            )  # [40,108,100]
            natoms_img = Variable(sample_batches["natoms_img"].int().to(device))

            if args.dp:
                Etot_predict, Ei_predict, Force_predict = model(
                    Ri, Ri_d, dR_neigh_list, natoms_img, None, None
                )
            else:
                Etot_predict, Ei_predict, Force_predict = model(
                    input_data, dfeat, neighbor, natoms_img, egroup_weight, divider
                )

            loss_F_val = criterion(Force_predict, Force_label)
            loss_Etot_val = criterion(Etot_predict, Etot_label)
            loss_Ei_val = criterion(Ei_predict, Ei_label)
            loss_val = loss_F_val + loss_Etot_val

            # measure accuracy and record loss
            batch_size = Ri.shape[0]
            losses.update(loss_val.item(), batch_size)
            loss_Etot.update(loss_Etot_val.item(), batch_size)
            loss_Ei.update(loss_Ei_val.item(), batch_size)
            loss_Force.update(loss_F_val.item(), batch_size)
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i + 1)

    batch_time = AverageMeter("Time", ":6.3f", Summary.NONE)
    losses = AverageMeter("Loss", ":.4e", Summary.AVERAGE)
    loss_Etot = AverageMeter("Etot", ":.4e", Summary.ROOT)
    loss_Force = AverageMeter("Force", ":.4e", Summary.ROOT)
    loss_Ei = AverageMeter("Ei", ":.4e", Summary.ROOT)
    progress = ProgressMeter(
        len(val_loader)
        + (
            args.distributed
            and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset))
        ),
        [batch_time, losses, loss_Etot, loss_Force, loss_Ei],
        prefix="Test: ",
    )

    # switch to evaluate mode
    model.eval()

    run_validate(val_loader)
    if args.distributed:
        losses.all_reduce()
        loss_Etot.all_reduce()
        loss_Force.all_reduce()
        loss_Ei.all_reduce()

    if args.distributed and (
        len(val_loader.sampler) * args.world_size < len(val_loader.dataset)
    ):
        aux_val_dataset = Subset(
            val_loader.dataset,
            range(len(val_loader.sampler) * args.world_size, len(val_loader.dataset)),
        )
        aux_val_loader = torch.utils.data.DataLoader(
            aux_val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
        )
        run_validate(aux_val_loader, len(val_loader))

    progress.display_summary(["Test Set:"])

    return losses.avg, loss_Etot.root, loss_Force.root, loss_Ei.root


def save_checkpoint(state, is_best, filename, prefix):
    filename = os.path.join(prefix, filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(prefix, "best.pth.tar"))


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3
    ROOT = 4


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f", summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.root = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.root = self.avg**0.5

    def all_reduce(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count
        self.root = self.avg**0.5

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ""
        if self.summary_type is Summary.NONE:
            fmtstr = ""
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = "{name} {avg:.3f}"
        elif self.summary_type is Summary.SUM:
            fmtstr = "{name} {sum:.3f}"
        elif self.summary_type is Summary.COUNT:
            fmtstr = "{name} {count:.3f}"
        elif self.summary_type is Summary.ROOT:
            fmtstr = "{name} {root:.3f}"
        else:
            raise ValueError("invalid summary type %r" % self.summary_type)

        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def display_summary(self, entries=[" *"]):
        entries += [meter.summary() for meter in self.meters]
        print(" ".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


if __name__ == "__main__":
    main()
