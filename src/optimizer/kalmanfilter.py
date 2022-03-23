import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import random
import math
import parameters as pm


class GKalmanFilter(nn.Module):

    def __init__(self, model, kalman_lambda, kalman_nue, device):
        super(GKalmanFilter, self).__init__()
        self.kalman_lambda = kalman_lambda
        self.kalman_nue = kalman_nue
        self.device = device
        self.model = model
        self.natoms = pm.natoms
        self.natoms_sum = sum(pm.natoms)
        self.n_select = 24
        self.Force_group_num = 6
        self.n_select_eg = 12
        self.Force_group_num_eg = 3
        self.__init_P()

    def __init_P(self):
        param_num = 0
        self.weights_index = [param_num]
        for name, param in self.model.named_parameters():
            param_num += param.data.nelement()
            self.weights_index.append(param_num)
        self.P = torch.eye(param_num).to(self.device)
    
    def __update(self, H, error, weights):
        '''
        1. get the Kalman Gain Matrix
        '''
        A = 1 / (self.kalman_lambda + torch.matmul(torch.matmul(H.T, self.P), H))
        K = torch.matmul(self.P, H)
        K = torch.matmul(K, A)
        '''
        2. update weights
        '''
        weights += K * error
        i = 0
        for name, param in self.model.named_parameters():
            start_index = self.weights_index[i]
            end_index = self.weights_index[i+1]
            param.data = weights[start_index:end_index].reshape(param.data.T.shape).T
            i += 1
        '''
        3. update P
        '''
        self.P = (1 / self.kalman_lambda) * (self.P - torch.matmul(torch.matmul(K, H.T), self.P))
        self.P = (self.P + self.P.T) / 2
        '''
        4. update kalman_lambda
        '''
        self.kalman_lambda = self.kalman_nue * self.kalman_lambda + 1 - self.kalman_nue

        for name, param in self.model.named_parameters():
                param.grad.detach_()
                param.grad.zero_()

        torch.cuda.empty_cache()
    
    def __get_random_index(self, Force_label, n_select, Force_group_num):
        total_atom=0
        atom_list=[0]
        select_list=[]
        random_index=[[[],[]] for i in range(math.ceil(n_select/Force_group_num))]

        for i in range(len(pm.natoms)):
            total_atom+=pm.natoms[i]
            atom_list.append(total_atom)
        random_list=list(range(total_atom))
        for i in range(n_select):
            select_list.append(random_list.pop(random.randint(0, total_atom-i-1)))

        tmp=0
        tmp2=0
        Force_shape = Force_label.shape[0] * Force_label.shape[1]
        tmp_1=list(range(Force_label.shape[0]))
        tmp_2=select_list
        for i in tmp_1:
            for k in tmp_2:
                random_index[tmp][0].append(i)
                random_index[tmp][1].append(k)
                tmp2+=1
                if tmp2%Force_group_num==0:
                    tmp+=1
                if tmp2==Force_shape:
                    break
        return random_index

    def update_energy(self, inputs, Etot_label,  update_prefactor = 1):
        time_start = time.time()
        Etot_predict, Ei_predict, force_predict = self.model(inputs[0], inputs[1], inputs[2], inputs[3], inputs[4])
        errore = Etot_label.item() - Etot_predict.item()
        errore = errore / self.natoms_sum
        if errore < 0:
            errore = - update_prefactor * errore
            (-1.0 * Etot_predict).backward()
        else:
            errore = update_prefactor * errore
            Etot_predict.backward()
        
        i = 0
        for name, param in self.model.named_parameters():
            if i == 0:
                H = (param.grad / self.natoms_sum).T.reshape(param.grad.nelement(), 1)
                weights = param.data.T.reshape(param.data.nelement(),1)
            else:
                H = torch.cat((H, (param.grad/self.natoms_sum).T.reshape(param.grad.nelement(), 1)))
                weights = torch.cat((weights, param.data.T.reshape(param.data.nelement(), 1))) #!!!!waring, should use T
            i += 1
        self.__update(H, errore, weights)
        time_end = time.time()
        print("update Energy time:", time_end - time_start, 's')
        

    def update_force(self, inputs, Force_label, update_prefactor = 1):
        time_start = time.time()
        
        '''
        now we begin to group
        NOTICE! for every force, we should repeat calculate Fatom
        because every step the weight will update, and the same dfeat/dR and de/df will get different Fatom
        '''
        random_index=self.__get_random_index(Force_label, self.n_select, self.Force_group_num)

        for index_i in range(len(random_index)):  # 4
            error = 0
            for index_ii in range(len(random_index[index_i][0])): # 6
                for j in range(3):
                    #error = 0 #if we use group , it should not be init
                    Etot_predict, Ei_predict, force_predict = self.model(inputs[0], inputs[1], inputs[2], inputs[3], inputs[4])
                    force_predict.requires_grad_(True)
                    error_tmp = (Force_label[random_index[index_i][0][index_ii]][random_index[index_i][1][index_ii]][j] - force_predict[random_index[index_i][0][index_ii]][random_index[index_i][1][index_ii]][j])
                    if error_tmp < 0:
                        error_tmp = - update_prefactor * error_tmp
                        error += error_tmp
                        (-1.0 * force_predict[random_index[index_i][0][index_ii]][random_index[index_i][1][index_ii]][j]).backward(retain_graph=True)
                    else:
                        error +=  update_prefactor * error_tmp
                        (force_predict[random_index[index_i][0][index_ii]][random_index[index_i][1][index_ii]][j]).backward(retain_graph=True)

            num = len(random_index[index_i][0])
            error = (error / (num * 3.0)) / self.natoms_sum
            
            tmp_grad = 0
            i = 0
            for name, param in self.model.named_parameters():
                if i == 0:
                    tmp_grad = param.grad
                    if tmp_grad == None: #when name==bias, the grad will be None
                        tmp_grad = torch.tensor([0.0])
                    H = ((tmp_grad /(num * 3.0)) / self.natoms_sum).T.reshape(tmp_grad.nelement(), 1)
                    weights = param.data.T.reshape(param.data.nelement(), 1)
                else:
                    tmp_grad=param.grad
                    if tmp_grad==None: #when name==bias, the grad will be None
                        tmp_grad=torch.tensor([0.0])
                    H = torch.cat((H, ((tmp_grad / (num*3.0)) / self.natoms_sum).T.reshape(tmp_grad.nelement(), 1)))
                    weights = torch.cat((weights, param.data.T.reshape(param.data.nelement(), 1)))  # !!!!waring, should use T
                i += 1
            
            self.__update(H, error, weights)

        torch.cuda.empty_cache()
        time_end = time.time()
        print("update Force time:", time_end - time_start, 's')


    def update_egroup(self, inputs, Egroup_label, update_prefactor=0.1):

        random_index=self.__get_random_index(Egroup_label, self.n_select_eg, self.Force_group_num_eg)

        for index_i in range(len(random_index)):
            error = 0
            for index_ii in range(len(random_index[index_i][0])):

                Egroup_predict = self.model.get_egroup(inputs[3], inputs[4]) #egroup_weights, divider

                error_tmp = update_prefactor*(Egroup_label[random_index[index_i][0][index_ii]][random_index[index_i][1][index_ii]][0] - Egroup_predict[random_index[index_i][0][index_ii]][random_index[index_i][1][index_ii]][0])
                if error_tmp < 0:
                    error_tmp = -1.0 * error_tmp
                    error += error_tmp
                    (update_prefactor*(-1.0 * Egroup_predict[random_index[index_i][0][index_ii]][random_index[index_i][1][index_ii]][0])).backward(retain_graph=True)
                else:
                    error += error_tmp
                    (update_prefactor*(Egroup_predict[random_index[index_i][0][index_ii]][random_index[index_i][1][index_ii]][0])).backward(retain_graph=True)

            num=len(random_index[index_i][0])
            error=error/num

            tmp_grad=0
            i = 0
            for name, param in self.model.named_parameters():
                if i == 0:
                    tmp_grad=param.grad
                    if tmp_grad==None:
                        tmp_grad=torch.tensor([0.0])
                    H = (tmp_grad / num).T.reshape(tmp_grad.nelement(), 1)
                    weights = param.data.T.reshape(param.data.nelement(), 1)
                else:
                    tmp_grad=param.grad
                    if tmp_grad==None:
                        tmp_grad=torch.tensor([0.0])
                    H = torch.cat((H, (tmp_grad / num).T.reshape(tmp_grad.nelement(), 1)))
                    weights = torch.cat((weights, param.data.T.reshape(param.data.nelement(), 1)))
                i += 1
            self.__update(H, error, weights)

        torch.cuda.empty_cache()


class LKalmanFilter(nn.Module):

    def __init__(self, model, kalman_lambda, kalman_nue, device):
        super(LKalmanFilter, self).__init__()
        self.kalman_lambda = kalman_lambda
        self.kalman_nue = kalman_nue
        self.device = device
        self.model = model
        self.natoms = pm.natoms
        self.natoms_sum = sum(pm.natoms)
        self.n_select = 24
        self.Force_group_num = 6
        self.n_select_eg = 12
        self.Force_group_num_eg = 3
        self.__init_P()

    def __init_P(self):
        self.P = []
        for name, param in self.model.named_parameters():
            param_num = param.data.nelement()
            print(name, param_num)
            self.P.append(torch.eye(param_num).to(self.device))
        self.weights_num = len(self.P)
    
    def __update(self, H, error, weights):

        tmp = 0
        for i in range(self.weights_num):
            tmp = tmp + (self.kalman_lambda + torch.matmul(torch.matmul(H[i].T, self.P[i]), H[i]))
        
        A = 1 / tmp

        for i in range(self.weights_num):
            '''
            1. get the Kalman Gain Matrix
            '''
            K = torch.matmul(self.P[i], H[i])
            K = torch.matmul(K, A)

            '''
            2. update weights
            '''
            weights[i] += K * error
        
            '''
            3. update P
            '''
            self.P[i] = (1 / self.kalman_lambda) * (self.P[i] - torch.matmul(torch.matmul(K, H[i].T), self.P[i]))
            self.P[i] = (self.P[i] + self.P[i].T) / 2
        '''
        4. update kalman_lambda
        '''
        self.kalman_lambda = self.kalman_nue * self.kalman_lambda + 1 - self.kalman_nue

        i = 0
        for name, param in self.model.named_parameters():
            param.data = weights[i].reshape(param.data.T.shape).T
            i += 1

        for name, param in self.model.named_parameters():
                param.grad.detach_()
                param.grad.zero_()

        torch.cuda.empty_cache()
    
    def __get_random_index(self, Force_label, n_select, Force_group_num):
        total_atom=0
        atom_list=[0]
        select_list=[]
        random_index=[[[],[]] for i in range(math.ceil(n_select/Force_group_num))]

        for i in range(len(pm.natoms)):
            total_atom+=pm.natoms[i]
            atom_list.append(total_atom)
        random_list=list(range(total_atom))
        for i in range(n_select):
            select_list.append(random_list.pop(random.randint(0, total_atom-i-1)))

        tmp=0
        tmp2=0
        Force_shape = Force_label.shape[0] * Force_label.shape[1]
        tmp_1=list(range(Force_label.shape[0]))
        tmp_2=select_list
        for i in tmp_1:
            for k in tmp_2:
                random_index[tmp][0].append(i)
                random_index[tmp][1].append(k)
                tmp2+=1
                if tmp2%Force_group_num==0:
                    tmp+=1
                if tmp2==Force_shape:
                    break
        return random_index

    def update_energy(self, inputs, Etot_label):
        time_start = time.time()
        Etot_predict, Ei_predict, force_predict = self.model(inputs[0], inputs[1], inputs[2], inputs[3], inputs[4])
        errore = Etot_label.item() - Etot_predict.item()
        errore = errore / self.natoms_sum
        if errore < 0:
            errore = -1.0 * errore
            (-1.0 * Etot_predict).backward()
        else:
            Etot_predict.backward()
        
        weights = []
        H = []
        for name, param in self.model.named_parameters():
            H.append((param.grad / self.natoms_sum).T.reshape(param.grad.nelement(), 1))
            weights.append(param.data.T.reshape(param.data.nelement(),1))
        self.__update(H, errore, weights)
        time_end = time.time()
        print("update Energy time:", time_end - time_start, 's')
        

    def update_force(self, inputs, Force_label):
        time_start = time.time()
        random_index=self.__get_random_index(Force_label, self.n_select, self.Force_group_num)

        for index_i in range(len(random_index)):  # 4
            error = 0
            for index_ii in range(len(random_index[index_i][0])): # 6
                for j in range(3):
                    #error = 0 #if we use group , it should not be init
                    Etot_predict, Ei_predict, force_predict = self.model(inputs[0], inputs[1], inputs[2], inputs[3], inputs[4])
                    force_predict.requires_grad_(True)
                    error_tmp = (Force_label[random_index[index_i][0][index_ii]][random_index[index_i][1][index_ii]][j] - force_predict[random_index[index_i][0][index_ii]][random_index[index_i][1][index_ii]][j])
                    if error_tmp < 0:
                        error_tmp = -1.0 * error_tmp
                        error += error_tmp
                        (-1.0 * force_predict[random_index[index_i][0][index_ii]][random_index[index_i][1][index_ii]][j]).backward(retain_graph=True)
                    else:
                        error += error_tmp
                        (force_predict[random_index[index_i][0][index_ii]][random_index[index_i][1][index_ii]][j]).backward(retain_graph=True)

            num = len(random_index[index_i][0])
            error = (error / (num * 3.0)) / self.natoms_sum
            
            tmp_grad = 0
            i = 0
            weights = []
            H = []
            for name, param in self.model.named_parameters():
                if i == 0:
                    tmp_grad = param.grad
                    if tmp_grad == None: #when name==bias, the grad will be None
                        tmp_grad = torch.tensor([0.0])
                    H.append(((tmp_grad /(num * 3.0)) / self.natoms_sum).T.reshape(tmp_grad.nelement(), 1))
                    weights.append(param.data.T.reshape(param.data.nelement(), 1))
                else:
                    tmp_grad=param.grad
                    if tmp_grad==None: #when name==bias, the grad will be None
                        tmp_grad=torch.tensor([0.0])
                    H.append(((tmp_grad / (num*3.0)) / self.natoms_sum).T.reshape(tmp_grad.nelement(), 1))
                    weights.append(param.data.T.reshape(param.data.nelement(), 1))
                i += 1
            self.__update(H, error, weights)

        torch.cuda.empty_cache()
        time_end = time.time()
        print("update Force time:", time_end - time_start, 's')


class SKalmanFilter(nn.Module):

    def __init__(self, model, kalman_lambda, kalman_nue, device):
        super(SKalmanFilter, self).__init__()
        self.kalman_lambda = kalman_lambda
        self.kalman_nue = kalman_nue
        self.device = device
        self.model = model
        self.natoms = pm.natoms
        self.natoms_sum = sum(pm.natoms)
        self.n_select = 24
        self.Force_group_num = 6
        self.n_select_eg = 12
        self.Force_group_num_eg = 3
        self.block_size = 2048  # 2048 3072 4096
        self.__init_P()

    def __init_P(self):
        param_sum = 0
        for _, param in self.model.named_parameters():
            param_num = param.data.nelement()
            param_sum += param_num
        
        self.block_num = math.ceil(param_sum / self.block_size)
        self.last_block_size = param_sum % self.block_size
        self.padding_size = self.block_size - self.last_block_size

        P = torch.eye(self.block_size).to(self.device)
        # self.P dims ---> block_num * block_size * block_size
        self.P = P.unsqueeze(dim=0).repeat(self.block_num, 1, 1)
        self.weights_num = len(self.P)
    
    def __get_weights_H(self, divider):
        i = 0
        for _, param in self.model.named_parameters():
            if i==0:
                weights = param.data.reshape(-1, 1)
                H = (param.grad / divider).T.reshape(-1, 1)
            else:
                weights = torch.cat([weights, param.data.reshape(-1, 1)], dim=0)
                tmp = (param.grad / divider).T.reshape(-1, 1)
                H = torch.cat([H, tmp], dim=0)
            i += 1
        # padding weight to match P dims
        if self.last_block_size != 0:
            weights = torch.cat([weights, torch.zeros(self.padding_size, 1).to(self.device)], dim=0)
            H = torch.cat([H, torch.zeros(self.padding_size, 1).to(self.device)], dim=0)
        return weights, H
    
    def __update_weights(self, weights):

        weights = weights.reshape(-1)
        
        i = 0
        param_index = 0
        for _, param in self.model.named_parameters():
            param_num = param.data.nelement()
            param.data = weights[param_index:param_index+param_num].reshape(param.data.T.shape).T
            i += 1
            param_index += param_num
    

    def __update(self, H, error, weights):

        H = H.reshape(self.block_num, -1, 1)
        weights = weights.reshape(self.block_num, -1, 1)
        
        tmp = self.kalman_lambda + torch.matmul(torch.matmul(H.transpose(1, 2), self.P), H)

        A = 1 / tmp.sum(dim=0)

        '''
        1. get the Kalman Gain Matrix
        '''
        K = torch.matmul(self.P, H)
        K = torch.matmul(K, A)

        '''
        2. update weights
        '''
        weights = weights + K * error

        '''
        3. update P
        '''
        self.P = (1 / self.kalman_lambda) * (self.P - torch.matmul(torch.matmul(K, H.transpose(1, 2)), self.P))
        self.P = (self.P + self.P.transpose(1, 2)) / 2

        '''
        4. update kalman_lambda
        '''
        self.kalman_lambda = self.kalman_nue * self.kalman_lambda + 1 - self.kalman_nue

        self.__update_weights(weights)

        for name, param in self.model.named_parameters():
                param.grad.detach_()
                param.grad.zero_()

        torch.cuda.empty_cache()
    
    def __get_random_index(self, Force_label, n_select, Force_group_num):
        total_atom=0
        atom_list=[0]
        select_list=[]
        random_index=[[[],[]] for i in range(math.ceil(n_select/Force_group_num))]

        for i in range(len(pm.natoms)):
            total_atom+=pm.natoms[i]
            atom_list.append(total_atom)
        random_list=list(range(total_atom))
        for i in range(n_select):
            select_list.append(random_list.pop(random.randint(0, total_atom-i-1)))

        tmp=0
        tmp2=0
        Force_shape = Force_label.shape[0] * Force_label.shape[1]
        tmp_1=list(range(Force_label.shape[0]))
        tmp_2=select_list
        for i in tmp_1:
            for k in tmp_2:
                random_index[tmp][0].append(i)
                random_index[tmp][1].append(k)
                tmp2+=1
                if tmp2%Force_group_num==0:
                    tmp+=1
                if tmp2==Force_shape:
                    break
        return random_index

    def update_energy(self, inputs, Etot_label):
        time_start = time.time()
        Etot_predict, Ei_predict, force_predict = self.model(inputs[0], inputs[1], inputs[2], inputs[3], inputs[4])
        errore = Etot_label.item() - Etot_predict.item()
        errore = errore / self.natoms_sum
        if errore < 0:
            errore = -1.0 * errore
            (-1.0 * Etot_predict).backward()
        else:
            Etot_predict.backward()
        
        weights, H = self.__get_weights_H(self.natoms_sum)
        self.__update(H, errore, weights)
        time_end = time.time()
        print("update Energy time:", time_end - time_start, 's')
        

    def update_force(self, inputs, Force_label):
        time_start = time.time()
        random_index=self.__get_random_index(Force_label, self.n_select, self.Force_group_num)

        for index_i in range(len(random_index)):  # 4
            error = 0
            Etot_predict, Ei_predict, force_predict = self.model(inputs[0], inputs[1], inputs[2], inputs[3], inputs[4])
            force_predict.requires_grad_(True)
            for index_ii in range(len(random_index[index_i][0])): # 6
                for j in range(3):
                    #error = 0 #if we use group , it should not be init
                    error_tmp = (Force_label[random_index[index_i][0][index_ii]][random_index[index_i][1][index_ii]][j] - force_predict[random_index[index_i][0][index_ii]][random_index[index_i][1][index_ii]][j])
                    if error_tmp < 0:
                        error_tmp = -1.0 * error_tmp
                        error += error_tmp
                        (-1.0 * force_predict[random_index[index_i][0][index_ii]][random_index[index_i][1][index_ii]][j]).backward(retain_graph=True)
                    else:
                        error += error_tmp
                        (force_predict[random_index[index_i][0][index_ii]][random_index[index_i][1][index_ii]][j]).backward(retain_graph=True)

            num = len(random_index[index_i][0])
            error = (error / (num * 3.0)) / self.natoms_sum
            
            # tmp_grad = 0
            # i = 0
            # weights = []
            # H = []

            weights, H = self.__get_weights_H(num * 3.0)
            # for name, param in self.model.named_parameters():
            #     if i == 0:
            #         tmp_grad = param.grad
            #         if tmp_grad == None: #when name==bias, the grad will be None
            #             tmp_grad = torch.tensor([0.0])
            #         H.append(((tmp_grad /(num * 3.0)) / self.natoms_sum).T.reshape(tmp_grad.nelement(), 1))
            #         weights.append(param.data.T.reshape(param.data.nelement(), 1))
            #     else:
            #         tmp_grad=param.grad
            #         if tmp_grad==None: #when name==bias, the grad will be None
            #             tmp_grad=torch.tensor([0.0])
            #         H.append(((tmp_grad / (num*3.0)) / self.natoms_sum).T.reshape(tmp_grad.nelement(), 1))
            #         weights.append(param.data.T.reshape(param.data.nelement(), 1))
            #     i += 1
            self.__update(H, error, weights)

        torch.cuda.empty_cache()
        time_end = time.time()
        print("update Force time:", time_end - time_start, 's')


