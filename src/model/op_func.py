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
        op.calculate_force_grad(
            list_neigh, Ri_d, grad_output, batch_size, natoms, neigh_num, grad
        )
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
        atom_virial = torch.zeros(
            batch_size, natoms, 9, dtype=dE.dtype, device=dE.device
        )
        op.calculate_virial_force(
            list_neigh,
            dE,
            Rij,
            Ri_d,
            batch_size,
            natoms,
            neigh_num,
            virial,
            atom_virial,
        )
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
        op.calculate_virial_force_grad(
            list_neigh, Rij, Ri_d, grad_output, batch_size, natoms, neigh_num, grad
        )
        return (None, grad, None, None, None)


class CalculateDR(Function):
    @staticmethod
    def forward(ctx, xyz_scater, maxNeighborNum, ntype):
        dims = xyz_scater.shape
        batch_size = dims[0]
        natoms = dims[1]
        embedding_net_output_dim = dims[3]
        neigh_num_ntype = torch.Tensor([maxNeighborNum, ntype])
        ctx.save_for_backward(xyz_scater, neigh_num_ntype)
        DR = torch.empty(
            batch_size,
            natoms,
            embedding_net_output_dim * 16,
            device=xyz_scater.device,
            dtype=xyz_scater.dtype,
        )
        op.calculate_DR(
            xyz_scater,
            batch_size,
            natoms,
            maxNeighborNum,
            ntype,
            embedding_net_output_dim,
            DR,
        )
        return DR

    @staticmethod
    def backward(ctx, grad_output):
        inputs = ctx.saved_tensors
        xyz_scater = inputs[0]
        neigh_num_ntype = inputs[1]

        neigh_num = int(neigh_num_ntype[0])
        ntype = int(neigh_num_ntype[1])

        # ======================================================
        # naive impl
        # dims = xyz_scater.shape
        # batch_size = dims[0]
        # natoms = dims[1]
        # embedding_net_output_dim = dims[3]
        # tmp = torch.zeros(batch_size, natoms, 4, embedding_net_output_dim, embedding_net_output_dim, 16, device=xyz_scater.device, dtype=xyz_scater.dtype)

        # for i in range(4):
        #     for j in range(embedding_net_output_dim):
        #         tmp[:, :, i, j, j, :] +=  0.01 * 0.01 * xyz_scater[:, :, i, :16]
        #         if j < 16:
        #             tmp[:, :, i, j, :, j] += 0.01 * 0.01 * xyz_scater[:, :, i, :]

        # tmp = tmp.reshape(batch_size, natoms, 100, 400)
        # grad = torch.matmul(tmp, grad_output.reshape(batch_size, natoms, 400, 1)).reshape(batch_size, natoms, 4, 25)
        # ======================================================

        # ======================================================
        # vecterize
        # dims = xyz_scater.shape
        # batch_size = dims[0]
        # natoms = dims[1]
        # embedding_net_output_dim = dims[3]
        # tmpA = torch.matmul(grad_output.reshape(batch_size, natoms, 25, 16), 0.01 * 0.01 * xyz_scater[:, :, :, :16].transpose(-2, -1))  # 25 x 4
        # tmpB = torch.matmul(grad_output.reshape(batch_size, natoms, 25, 16).transpose(-2, -1), 0.01 * 0.01 * xyz_scater.transpose(-2, -1))  # 16 x 4
        # tmpA[:, :, :16] += tmpB

        # grad = tmpA.transpose(-2, -1)
        # ======================================================

        grad = CalculateDRGrad.apply(xyz_scater, grad_output, neigh_num, ntype)

        return (grad, None, None, None)


class CalculateDRGrad(Function):
    @staticmethod
    def forward(ctx, xyz_scater, grad_output, maxNeighborNum, ntype):
        dims = xyz_scater.shape
        batch_size = dims[0]
        natoms = dims[1]
        embedding_net_output_dim = dims[3]
        neigh_num_ntype = torch.Tensor([maxNeighborNum, ntype])
        ctx.save_for_backward(xyz_scater, grad_output, neigh_num_ntype)

        grad = torch.empty(
            batch_size,
            natoms,
            4,
            embedding_net_output_dim,
            device=xyz_scater.device,
            dtype=xyz_scater.dtype,
        )

        op.calculate_DR_grad(
            xyz_scater,
            batch_size,
            natoms,
            maxNeighborNum,
            ntype,
            embedding_net_output_dim,
            grad_output,
            grad,
        )
        return grad

    @staticmethod
    def backward(ctx, grad_second):
        inputs = ctx.saved_tensors
        xyz_scater = inputs[0]
        grad_output = inputs[1]
        neigh_num_ntype = inputs[2]  # store on host
        
        dims = xyz_scater.shape
        batch_size = dims[0]
        natoms = dims[1]
        embedding_net_output_dim = dims[3]

        grad_output = grad_output.reshape(
            batch_size, natoms, embedding_net_output_dim, 16
        )

        # ===============================================
        # naive impl
        # dtmpA_dgradoutput = torch.zeros(
        #     batch_size,
        #     natoms,
        #     4,
        #     embedding_net_output_dim,
        #     embedding_net_output_dim,
        #     16,
        #     device=xyz_scater.device,
        #     dtype=xyz_scater.dtype,
        # )
        # for i in range(embedding_net_output_dim):
        #     for j in range(16):
        #         dtmpA_dgradoutput[:, :, :, i, i, j] += 0.0001 * xyz_scater[:, :, :, j]
        #         dtmpA_dgradoutput[:, :, :, j, i, j] += 0.0001 * xyz_scater[:, :, :, i]

        # dtmpA_dxyzscater = torch.zeros(
        #     batch_size,
        #     natoms,
        #     4,
        #     embedding_net_output_dim,
        #     4,
        #     embedding_net_output_dim,
        #     device=xyz_scater.device,
        #     dtype=xyz_scater.dtype,
        # )

        # for i in range(4):
        #     for j in range(embedding_net_output_dim):
        #         dtmpA_dxyzscater[:, :, i, :16, i, j] += 0.0001 * grad_output[:, :, j, :]
        #         if j < 16:
        #             dtmpA_dxyzscater[:, :, i, :16, i, j] += 0.0001 * grad_output[:, :, :16, j]

        # for i in range(4):
        #     for j in range(16):
        #         dtmpA_dxyzscater[:, :, i, 16:, i, j] += 0.0001 * grad_output[:, :, 16:, j]

        # derror_dxyzscatter = torch.matmul(
        #     grad_second.reshape(batch_size, natoms, 1, 4 * embedding_net_output_dim),
        #     dtmpA_dxyzscater.reshape(
        #         batch_size,
        #         natoms,
        #         4 * embedding_net_output_dim,
        #         embedding_net_output_dim * 4,
        #     ),
        # )
        # derror_dgradoutput = torch.matmul(
        #     grad_second.reshape(batch_size, natoms, 1, 4 * embedding_net_output_dim),
        #     dtmpA_dgradoutput.reshape(
        #         batch_size,
        #         natoms,
        #         4 * embedding_net_output_dim,
        #         embedding_net_output_dim * 16,
        #     ),
        # )
        # ======================================================

        # ======================================================
        # vecterize
        # dgrad_xyz_scater = torch.matmul(
        #     grad_second[:, :, :, :16], 0.0001 * grad_output.transpose(-2, -1)
        # )
        # tmp = torch.matmul(grad_second, 0.0001 * grad_output)
        # dgrad_xyz_scater[:, :, :, :16] += tmp

        # dgrad_gradoutput = torch.matmul(
        #     grad_second.transpose(-2, -1), 0.0001 * xyz_scater[:, :, :, :16]
        # ) + torch.matmul(
        #     0.0001 * xyz_scater.transpose(-2, -1), grad_second[:, :, :, :16]
        # )
        # return (
        #     dgrad_xyz_scater.reshape(batch_size, natoms, 4, embedding_net_output_dim),
        #     dgrad_gradoutput.reshape(
        #         batch_size, natoms, embedding_net_output_dim * 16
        #     ),
        #     None,
        #     None,
        # )
        # ======================================================

        scale = 1.0 / (neigh_num_ntype[0] * neigh_num_ntype[1])
        scale = scale * scale

        dgrad_xyz_scater = torch.empty(
            batch_size,
            natoms,
            4,
            embedding_net_output_dim,
            dtype=xyz_scater.dtype,
            device=xyz_scater.device,
        )

        dgrad_gradoutput = torch.empty(
            batch_size,
            natoms,
            embedding_net_output_dim * 16,
            dtype=xyz_scater.dtype,
            device=xyz_scater.device,
        )

        op.calculate_DR_second_grad(
            batch_size,
            natoms,
            scale,
            embedding_net_output_dim,
            xyz_scater,
            grad_output,
            grad_second,
            dgrad_xyz_scater,
            dgrad_gradoutput,
        )

        return (
            dgrad_xyz_scater,
            dgrad_gradoutput,
            None,
            None,
        )
