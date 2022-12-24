import numpy as np
import os
from torch.utils.data import Dataset
import torch
import yaml


class MovementDataset(Dataset):
    def __init__(self, data_path):
        super(MovementDataset, self).__init__()
        self.davg = np.load(os.path.join(data_path, "davg.npy"))
        self.dstd = np.load(os.path.join(data_path, "dstd.npy"))

        self.dirs = []

        for current_dir, child_dir, child_file in os.walk(data_path):
            if len(child_dir) == 0 and "Ri.npy" in child_file:
                self.dirs.append(current_dir)

        self.dirs = sorted(self.dirs)

        self.__compute_stat_output(10, 1e-3)

    def __load_data(self, path):

        data = {}

        data["Force"] = -1 * np.load(os.path.join(path, "Force.npy"))
        data["Ei"] = np.load(os.path.join(path, "Ei.npy"))
        data["ListNeighbor"] = np.load(os.path.join(path, "ListNeighbor.npy"))
        data["Ri"] = np.load(os.path.join(path, "Ri.npy"))
        data["Ri_d"] = np.load(os.path.join(path, "Ri_d.npy"))
        data["ImageAtomNum"] = np.load(os.path.join(path, "ImageAtomNum.npy")).reshape(
            -1
        )

        return data

    def __getitem__(self, index):

        file_path = self.dirs[index]
        data = self.__load_data(file_path)
        return data

    def __len__(self):
        return len(self.dirs)

    def __compute_stat_output(self, image_num=10, rcond=1e-3):

        data = self.__getitem__(0)

        self.ener_shift = []
        natoms_sum = data["ImageAtomNum"][0]
        natoms_per_type = data["ImageAtomNum"][1:]

        for i in range(image_num):
            data = self.__getitem__(i)
            tmp = data["Ei"].reshape(-1, natoms_sum)
            if i == 0:
                energy = tmp
            else:
                energy = np.concatenate([energy, tmp], axis=0)

        energy = np.reshape(energy, (-1, natoms_sum, 1))
        # natoms_sum = 0
        # for ntype in range(self.ntypes):
        #     energy_ntype = energy[:, natoms_sum:natoms_sum+natoms_per_type[ntype]]
        #     natoms_sum += natoms_per_type[ntype]
        #     energy_sum = energy_ntype.sum(axis=1)
        #     energy_one = np.ones_like(energy_sum) * natoms_per_type[ntype]
        #     ener_shift, _, _, _ = np.linalg.lstsq(energy_one, energy_sum, rcond=rcond)
        #     self.ener_shift.append(ener_shift[0, 0])
        # energy_ntype = energy[:, natoms_sum:natoms_sum+natoms_per_type[ntype]]
        # natoms_sum += natoms_per_type[ntype]
        energy_sum = energy.sum(axis=1)
        energy_avg = np.average(energy_sum)
        # energy_one = np.ones_like(energy_sum) * natoms_per_type[ntype]
        ener_shift, _, _, _ = np.linalg.lstsq(
            [natoms_per_type], [energy_avg], rcond=rcond
        )
        self.ener_shift = ener_shift.tolist()

    def get_stat(self):
        return self.davg, self.dstd, self.ener_shift


def main():

    with open("config.yaml", "r") as yamlfile:
        config = yaml.load(yamlfile, Loader=yaml.FullLoader)
    #     print("Read Config successful")
    # import ipdb; ipdb.set_trace()
    dataset = MovementDataset("./train")

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    for i, sample_batches in enumerate(dataloader):
        # import ipdb;ipdb.set_trace()

        print(sample_batches["Force"].shape)
        print(sample_batches["Ei"].shape)
        print(sample_batches["ListNeighbor"].shape)
        print(sample_batches["Ri"].shape)
        print(sample_batches["Ri_d"].shape)
        print(sample_batches["ImageAtomNum"].shape)


if __name__ == "__main__":
    main()
