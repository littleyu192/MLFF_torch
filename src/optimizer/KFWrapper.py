import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
import time, math
import numpy as np
import torch.distributed as dist
import random
import pandas as pd
class KFOptimizerWrapper:
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        atoms_selected: int,
        atoms_per_group: int,
        is_distributed: bool = False,
        distributed_backend: str = "torch",  # torch or horovod
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.atoms_selected = atoms_selected  # 24
        self.atoms_per_group = atoms_per_group  # 6
        self.is_distributed = is_distributed
        self.distributed_backend = distributed_backend

    def update_energy(
        self, inputs: list, Etot_label: torch.Tensor, update_prefactor: float = 1
    ) -> None:
        Etot_predict, _, _ = self.model(
            inputs[0],
            inputs[1],
            inputs[2],
            inputs[3],
            inputs[4],
            inputs[5],
            is_calc_f=False,
        )
        natoms_sum = inputs[3][0, 0]
        self.optimizer.set_grad_prefactor(natoms_sum)

        self.optimizer.zero_grad()
        bs = Etot_label.shape[0]
        error = Etot_label - Etot_predict
        error = error / natoms_sum
        mask = error < 0

        error = error * update_prefactor
        error[mask] = -1 * error[mask]
        error = error.mean()

        if self.is_distributed:
            if self.distributed_backend == "horovod":
                import horovod as hvd

                error = hvd.torch.allreduce(error)
            elif self.distributed_backend == "torch":
                dist.all_reduce(error)
                error /= dist.get_world_size()

        Etot_predict = update_prefactor * Etot_predict
        Etot_predict[mask] = -update_prefactor * Etot_predict[mask]

        Etot_predict.sum().backward()
        error = error * math.sqrt(bs)
        self.optimizer.step(error)
        return Etot_predict

    def update_force(
        self, inputs: list, Force_label: torch.Tensor, update_prefactor: float = 1
    ) -> None:
        natoms_sum = inputs[3][0, 0]
        bs = Force_label.shape[0]
        self.optimizer.set_grad_prefactor(natoms_sum * self.atoms_per_group * 3)

        index = self.__sample(self.atoms_selected, self.atoms_per_group, natoms_sum)

        for i in range(index.shape[0]):
            self.optimizer.zero_grad()
            Etot_predict, Ei_predict, force_predict = self.model(
                inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5]
            )
            error_tmp = Force_label[:, index[i]] - force_predict[:, index[i]]
            error_tmp = update_prefactor * error_tmp
            mask = error_tmp < 0
            error_tmp[mask] = -1 * error_tmp[mask]
            error = error_tmp.mean() / natoms_sum

            if self.is_distributed:
                if self.distributed_backend == "horovod":
                    import horovod as hvd

                    error = hvd.torch.allreduce(error)
                elif self.distributed_backend == "torch":
                    dist.all_reduce(error)
                    error /= dist.get_world_size()

            tmp_force_predict = force_predict[:, index[i]] * update_prefactor
            tmp_force_predict[mask] = -update_prefactor * tmp_force_predict[mask]

            # In order to solve a pytorch bug, reference: https://github.com/pytorch/pytorch/issues/43259
            (tmp_force_predict.sum() + Etot_predict.sum() * 0).backward()
            error = error * math.sqrt(bs)
            self.optimizer.step(error)
            return Etot_predict, Ei_predict, force_predict

    def __sample(
        self, atoms_selected: int, atoms_per_group: int, natoms: int
    ) -> np.ndarray:
        if atoms_selected % atoms_per_group:
            raise Exception("divider")
        index = range(natoms)
        res = np.random.choice(index, atoms_selected).reshape(-1, atoms_per_group)
        return res

    '''
    description: do predict 
    return {*}
    '''    
    def valid(
        self, inputs: list, Etot_label: torch.Tensor, update_prefactor: float = 1
    ) -> None:
        Etot_predict, Ei_predict, Force_predict = self.model(
                inputs[0],
                inputs[1],
                inputs[2],
                inputs[3],
                inputs[4],
                inputs[5],
                is_calc_f=True,
            )
        return Etot_predict, Ei_predict, Force_predict

    # """
    # @Description :
    # calculate kpu by etot
    # @Returns     :
    # @Author       :wuxingxing
    # """
    def cal_kpu_etot(
        self, inputs: list, Etot_label: torch.Tensor, update_prefactor: float = 1
    ) -> None:
        Etot_predict, Ei_predict, Force_predict = self.model(
                inputs[0],
                inputs[1],
                inputs[2],
                inputs[3],
                inputs[4],
                inputs[5],
                is_calc_f=False,
            )
        natoms_sum = inputs[3][0, 0]
        self.optimizer.set_grad_prefactor(natoms_sum)
        self.optimizer.zero_grad()
        (Etot_predict / natoms_sum).backward()
        etot_kpu = self.optimizer.cal_kpu()
        self.optimizer.step(None)
        return etot_kpu, Etot_label, Etot_predict

    # """
    # @Description :
    # calculate kpu by force
    # 1. random select 50% atoms
    # 2. force_x, force_y, force_z of each atom do backward() then calculat its kpu
    # 3. force kpu = mean of these kpus
    # @Returns     :
    # @Author       :wuxingxing
    # """
    def cal_kpu_force(
        self, inputs: list, Force_label: torch.Tensor, update_prefactor: float = 1
    ) -> None:
        """
        randomly generate n different nums of int type in the range of [start, end)
        """
        def get_random_nums(start, end, n):
            random.seed(2022)
            numsArray = set()
            while len(numsArray) < n:
                numsArray.add(random.randint(start, end-1))
            return list(numsArray)

        natoms_sum = inputs[3][0, 0]
        self.optimizer.set_grad_prefactor(1) #natoms_sum * self.atoms_per_group * 3
        natom_list = get_random_nums(0, natoms_sum, int(0.5*natoms_sum))

        column_name=["atom_index", "kpu_x", "kpu_y", "kpu_z", "f_x", "f_y", "f_z", "f_x_pre", "f_y_pre", "f_z_pre"]
        force_kpu = pd.DataFrame(columns=column_name)
        for i in natom_list:
            force_x_y_z_kpu = []
            force_x_y_z_kpu.append(i)
            for j in range(3):
                self.optimizer.zero_grad()
                Etot_predict, Ei_predict, Force_predict = self.model(inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5], is_calc_f=True)
                #xyz
                (Force_predict[0][i][j] + Force_predict.sum() * 0 + Etot_predict.sum() * 0).backward()
                # Force_predict[0][i][j].backward()
                f_kpu = self.optimizer.cal_kpu()
                force_x_y_z_kpu.append(float(f_kpu))
                self.optimizer.step(None)
            force_x_y_z_kpu.extend([float(Force_label[0][i][0]),float(Force_label[0][i][1]), float(Force_label[0][i][2])])
            force_x_y_z_kpu.extend([float(Force_predict[0][i][0]),float(Force_predict[0][i][1]), float(Force_predict[0][i][2])])
            force_kpu.loc[i]=force_x_y_z_kpu
        
        #["atoms", "f_avg_lab", "f_avg_pre", "f_x_norm", "f_y_norm", "f_z_norm", "f_kpu"] loss is mse
        f_x_norm, f_y_norm, f_z_norm = torch.norm(Force_predict.squeeze(0),dim=0) #tensor shape [natoms_sum , 3] use random selected atoms
        return natoms_sum, Force_label, Force_predict, Ei_predict,\
                Force_label.abs().mean(), Force_predict.abs().mean(), f_x_norm, f_y_norm, f_z_norm, \
                    ((force_kpu['kpu_x'].abs()+force_kpu['kpu_y'].abs()+force_kpu['kpu_z'].abs())/3).mean(), \
                force_kpu

