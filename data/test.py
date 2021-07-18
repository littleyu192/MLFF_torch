import prepare
import parameters as pm
import numpy as np
import ipdb

class MinMaxScaler:
    ''' a*x +b = x_scaled like sklearn's MinMaxScaler
        note cp.atleast_2d and self.a[xmax==xmin] = 0
    '''
    def __init__(self, feature_range=(0,1)):
        self.fr = feature_range
        self.a = 0
        self.b = 0


    def fit_transform(self, x):

        if len(x) == 0:
            self.a = 0
            self.b = 0
            return x
        x = np.atleast_2d(x)
        xmax = x.max(axis=0)
        xmin = x.min(axis=0)
        self.a = (self.fr[1] -self.fr[0]) / (xmax-xmin)
        self.a[xmax-xmin<=0.1] = 10
        
        self.a[xmax<0.01] = 1
        self.a[xmax==xmin] = 0  # important !!!
        self.b = self.fr[0] - self.a*xmin
        # self.a = np.ones(xmax.shape)
        # self.b = np.zeros(xmin.shape)
        return self.transform(x)


    def transform(self, x):
        x = np.atleast_2d(x)
        return self.a*x +self.b


    def inverse_transform(self, y):
        y = np.atleast_2d(y)
        return (y -self.b) /self.a


class DataScaler:


    def __init__(self):
        self.feat_scaler=MinMaxScaler(feature_range=(0,1))
        self.feat_a = None
        self.engy_scaler=MinMaxScaler(feature_range=(0,1))
        self.engy_a = None

    
    def get_scaler(self, f_feat, f_ds, b_save=True):
        
        from prepare import r_feat_csv

        itypes,feat,engy = r_feat_csv(f_feat)
        print('=DS.get_scaler ', f_feat, 'feat.shape, feat.dtype', feat.shape, feat.dtype)
        
        _ = self.feat_scaler.fit_transform(feat)
        _ = self.engy_scaler.fit_transform(engy)
        
        feat_b      = self.feat_scaler.transform(cp.zeros((1, feat.shape[1])))    
        self.feat_a = self.feat_scaler.transform(cp.ones((1, feat.shape[1]))) - feat_b
        engy_b      = self.engy_scaler.transform(0)
        self.engy_a = self.engy_scaler.transform(1) - engy_b

    
    def pre_feat(self, feat):
        return self.feat_scaler.transform(feat)


def get_scalers(f_feat, f_ds, b_save=True):
    itypes,feat,engy = prepare.r_feat_csv(f_feat)
    scalers = {}
    feat_as = {}
    engy_as = {}
    print('=DS.get_scaler ', f_feat, 'feat.shape, feat.dtype', feat.shape, feat.dtype)
    print('=DS.get_scaler ', f_feat, 'engy.shape, feat.dtype', engy.shape, engy.dtype)
    print('=DS.get_scaler ', f_feat, 'itypes.shape, feat.dtype', itypes.shape, itypes.dtype)
    dsnp = []
    for i in range(pm.ntypes):
        itype = pm.atomType[i]
        scalers[itype] = DataScaler()
        subfeat = feat[itypes == itype]
        subengy = engy[itypes == itype]
        scalers[itype].feat_scaler.fit_transform(subfeat)
        scalers[itype].engy_scaler.fit_transform(subengy)
        feat_b = scalers[itype].feat_scaler.transform(np.zeros((1, subfeat.shape[1])))
        engy_b = scalers[itype].engy_scaler.transform(np.zeros((1, subengy.shape[1])))
        feat_as[itype] = scalers[itype].\
                                feat_scaler.transform(np.ones((1, subfeat.shape[1]))) - feat_b
        engy_as[itype] = scalers[itype].\
                                engy_scaler.transform(np.ones((1, subengy.shape[1]))) - engy_b

        feat_scaler = scalers[itype].feat_scaler
        engy_scaler = scalers[itype].engy_scaler
        dsnp.append(np.array(feat_scaler.fr))
        dsnp.append(np.array(feat_scaler.a))
        dsnp.append(np.array(feat_scaler.b))
        dsnp.append(np.array(feat_as[itype]))
        dsnp.append(np.array(engy_scaler.fr))
        dsnp.append(np.array(engy_scaler.a))
        dsnp.append(np.array(engy_scaler.b))
        dsnp.append(np.array(engy_as[itype]))
    dsnp = np.array(dsnp)
    np.save("ds.npy", dsnp)
    return scalers


def main():
    # 计算scale变换的参数
    scalers = get_scalers(pm.f_train_feat, pm.f_data_scaler)
    ipdb.set_trace()

if __name__ == '__main__':
    main()
    