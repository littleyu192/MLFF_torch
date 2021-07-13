import numpy as np
import re
import pandas as pd
import os
from torch.utils.data import Dataset, DataLoader
import torch


def get_data_property(filename, valid_ratio):
    # data loader
    f = open(filename, 'r')
    lines = f.readlines()

    # the atom number in one image
    lines0 = lines[0]
    atoms_number_in_one_image = int(re.findall(r"(.*) atom", lines0)[0])
    print("atoms number:", atoms_number_in_one_image)

    # number of all the samples
    number_of_samples = 0
    for lines in lines:
        if "-----------" in lines:
            number_of_samples = number_of_samples + 1
    train_samples = int(number_of_samples * (1 - valid_ratio))
    print("all samples:", number_of_samples)
    print("train samples:", train_samples)
    valid_samples = number_of_samples - train_samples
    return atoms_number_in_one_image, train_samples


def get_data(filename, atoms_number_in_one_image, train_data_file, valid_data_file, train_samples):
    # the generated data, delete them if exist already
    # if os.path.exists(train_data_file):
    #     os.remove(train_data_file)
    # if os.path.exists(valid_data_file):
    #     os.remove(valid_data_file)

    f = open(filename, 'r')
    lines = f.readlines()
    atoms_position_all = []
    atoms_force_all = []
    Etot = []
    Ep = []
    Ek = []
    for (num, value) in enumerate(lines):
        if value.find("atoms") != -1:
            # print(value)
            Etot.append(eval(value.split(",")[4].split()[3]))
            Ep.append(eval(value.split(",")[4].split()[4]))
            Ek.append(eval(value.split(",")[4].split()[5]))
        elif lines[num].startswith(" Position"):
            atom_position = []
            for i in range(1, atoms_number_in_one_image + 1):
                position = lines[num + i].split(' ')
                position = [x for x in position if x != '']
                isotope = [int(x) for x in position[0]]  # 同位素个数
                position_xyz = [float(x) for x in position[1:4]]
                # print(position)
                atom_position.append(position_xyz[0])
                atom_position.append(position_xyz[1])
                atom_position.append(position_xyz[2])
            atoms_position_all.append(atom_position)
        elif lines[num].startswith(" Force"):
            atom_force = []
            for i in range(1, atoms_number_in_one_image + 1):
                force = lines[num + i].split(' ')
                force = [x for x in force if x != '']
                force_xyz = [float(x) for x in force[1:4]]
                atom_force.append(force_xyz[0])
                atom_force.append(force_xyz[1])
                atom_force.append(force_xyz[2])
            atoms_force_all.append(atom_force)

    df_x = pd.DataFrame(atoms_position_all)
    df_y1 = pd.DataFrame(atoms_force_all)
    df_y2 = pd.DataFrame(Etot, columns=['Etot'])
    df_y = df_y1.join(df_y2)
    df = df_x.join(df_y, lsuffix='_x', rsuffix='_y')
    df_train = df[:train_samples]
    df_valid = df[train_samples:]
    df_train.to_csv(train_data_file, index=False)
    df_valid.to_csv(valid_data_file, index=False)
    # x_train=atoms_position_all[:train_samples]
    # x_valid=atoms_position_all[train_samples:]
    # y_train=atoms_force_all[:train_samples]
    # y_valid=atoms_force_all[train_samples:]
    #
    # torch.set_printoptions(precision=15)   #打印显示15个小数位
    # x_train=torch.tensor(x_train,dtype=torch.float64)
    # y_train=torch.tensor(y_train,dtype=torch.float64)
    # x_valid=torch.tensor(x_valid,dtype=torch.float64)
    # y_valid=torch.tensor(y_valid,dtype=torch.float64)
    #
    # torch_dataset_train=Data.TensorDataset(x_train,y_train)
    # with open(train_data, 'a') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(torch_dataset_train)
    # torch_dataset_valid=Data.TensorDataset(x_valid,y_valid)
    # with open(valid_data, 'a') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(torch_dataset_valid)
    # return torch_dataset_train, torch_dataset_valid


class MovementDataset(Dataset):

    def __init__(self, file_path):
        super(MovementDataset, self).__init__()
        data = pd.read_csv(file_path)
        features = [str(x) + "_x" for x in range(324)]
        labels_Fi = [str(x) + "_y" for x in range(324)]
        # labels.append("Etot")
        self.features = data[features]
        self.labels_Fi = data[labels_Fi]
        self.labels_Etot = data["Etot"]

    def __getitem__(self, index):
        dic = {
            'input': torch.from_numpy(np.array(self.features[index:index + 1])[0]).float(),
            'output_Fi': torch.from_numpy(np.array(self.labels_Fi[index:index + 1])[0]).float(),
            'output_Etot': torch.from_numpy(np.array(self.labels_Etot[index:index + 1])).float()
        }
        return dic

    def __len__(self):
        return self.features.shape[0]


def main():
    filename = './train_data/MOVEMENT'
    valid_ratio = 0.2
    train_data_file = './train_data.csv'
    valid_data_file = './valid_data.csv'
    atoms_number_in_one_image, train_samples = get_data_property(filename, valid_ratio)
    if not os.path.exists(train_data_file):
        print("==========generate train and valid data============")
        get_data(filename, atoms_number_in_one_image, train_data_file, valid_data_file, train_samples)

if __name__ == "__main__":
    main()