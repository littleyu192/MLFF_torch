from torch import nn
import torch
from torch.autograd import Function
import op

class CalculateForce(Function):
    @staticmethod
    def forward(ctx, list_neigh, dE, Ri_d, F):
        dims = list_neigh.shape
        batch_size = dims[0]
        natoms = dims[1]
        neigh_num = dims[2] * dims[3]
        ctx.save_for_backward(list_neigh, dE, Ri_d)
        op.calculate_force(list_neigh, dE, Ri_d, batch_size, natoms, neigh_num, F)
        return F

    @staticmethod
    def backward(ctx, grad_output):
        inputs = ctx.saved_tensors
        list_neigh = inputs[0]
        dE = inputs[1]
        Ri_d = inputs[2]
        dims = list_neigh.shape
        batch_size = dims[0]
        natoms = dims[1]
        neigh_num = dims[2] * dims[3]
        grad = torch.zeros_like(dE)
        op.calculate_force_grad(list_neigh, Ri_d, grad_output, batch_size, natoms, neigh_num, grad)
        return (None, grad, None, None)


class CalculateVirialForce(Function):
    @staticmethod
    def forward(ctx, list_neigh, dE, Rij, Ri_d):
        dims = list_neigh.shape
        batch_size = dims[0]
        natoms = dims[1]
        neigh_num = dims[2] * dims[3]
        ctx.save_for_backward(list_neigh, dE, Rij, Ri_d)
        virial = torch.zeros(batch_size, 9, dtype=dE.dtype, device=dE.device)
        atom_virial = torch.zeros(batch_size, natoms, 9, dtype=dE.dtype, device=dE.device)
        op.calculate_virial_force(list_neigh, dE, Rij, Ri_d, batch_size, natoms, neigh_num, virial, atom_virial)
        return virial

    @staticmethod
    def backward(ctx, grad_output):
        inputs = ctx.saved_tensors
        list_neigh = inputs[0]
        dE = inputs[1]
        Rij = inputs[2]
        Ri_d = inputs[3]
        dims = list_neigh.shape
        batch_size = dims[0]
        natoms = dims[1]
        neigh_num = dims[2] * dims[3]
        grad = torch.zeros_like(dE)
        op.calculate_virial_force_grad(list_neigh, Rij, Ri_d, grad_output, batch_size, natoms, neigh_num, grad)
        return (None, grad, None, None, None)

class CalculateDR(Function):
    @staticmethod
    def forward(ctx, xyz_scater, maxNeighborNum, ntype, list_neigh):
        dims = xyz_scater.shape
        batch_size = dims[0]
        natoms = dims[1]
        embedding_net_output_dim = dims[3]
        ctx.save_for_backward(xyz_scater, list_neigh)
        DR = torch.empty(batch_size, natoms, embedding_net_output_dim * 16, device=xyz_scater.device, dtype=xyz_scater.dtype)
        op.calculate_DR(xyz_scater, batch_size, natoms, maxNeighborNum, ntype, embedding_net_output_dim, DR)
        return DR

    @staticmethod
    def backward(ctx, grad_output):
        inputs = ctx.saved_tensors
        xyz_scater = inputs[0]
        list_neigh = inputs[1]
        dims = xyz_scater.shape
        batch_size = dims[0]
        natoms = dims[1]
        embedding_net_output_dim = dims[3]

        # hard code 
        neigh_num = 100
        ntype = int(list_neigh.shape[2] / neigh_num)

        # ======================================================
        # naive impl
        # tmp = torch.zeros(batch_size, natoms, 4, embedding_net_output_dim, embedding_net_output_dim, 16, device=xyz_scater.device, dtype=xyz_scater.dtype)

        # for i in range(4):
        #     for j in range(embedding_net_output_dim):
        #         tmp[:, :, i, j, j, :] +=  0.01 * xyz_scater[:, :, i, :16]
        #         if j < 16:
        #             tmp[:, :, i, j, :, j] += 0.01 * xyz_scater[:, :, i, :]

        # tmp = tmp.reshape(batch_size, natoms, 100, 400)
        # grad = torch.matmul(tmp, grad_output.reshape(batch_size, natoms, 400, 1)).reshape(batch_size, natoms, 4, 25)
        # ======================================================


        # ======================================================
        # vecterize
        # tmpA = torch.matmul(grad_output.reshape(batch_size, natoms, 25, 16), 0.01 * xyz_scater[:, :, :, :16].transpose(-2, -1))  # 25 x 4
        # tmpB = torch.matmul(grad_output.reshape(batch_size, natoms, 25, 16).transpose(-2, -1), 0.01 * xyz_scater.transpose(-2, -1))  # 16 x 4
        # tmpA[:, :, :16] += tmpB

        # tmpA = tmpA.transpose(-2, -1)        
        # grad = tmpA
        # ======================================================


        grad = torch.empty(batch_size, natoms, 4, embedding_net_output_dim, device=xyz_scater.device, dtype=xyz_scater.dtype)
        op.calculate_DR_grad(xyz_scater, batch_size, natoms, neigh_num, ntype, embedding_net_output_dim, grad_output, grad)

        return (grad, None, None, None)
