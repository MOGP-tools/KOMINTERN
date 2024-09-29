import numpy as np
import pickle
import pandas as pd
# import h5py
import copy

class XsSetGrids:

    def __init__( self, multiparam=None, support='grid',
                  pickle_name=None, cc_dict={} ):
        self.multiparam = multiparam
        self.labels = []
        self.xs_dict = {}
        self.cc_dict = cc_dict
        self.support = support
        self.nonzero_keys = []
        if pickle_name is not None:
            dico = pickle.load(open(pickle_name, 'rb'))
            for key, value in dico.items():
                setattr(self, key, value)
        self.keys = np.array(list(self.xs_dict.keys()))

    def unpack_cr( self, cr_position=4, cr_position_cc=3 ):
        dict_assemblies = {}
        for i_cr, cr_value in enumerate(self.multiparam['cr']):
            xs_set = copy.deepcopy(self)
            xs_temp, cc_temp = list(self.xs_dict.values())[0], list(self.cc_dict.values())[0]
            xs_slice, cc_slice = list(xs_temp.shape), list(cc_temp.shape)
            for i in range(len(xs_slice)):
                if i != cr_position:
                    xs_slice[i] = slice(None)
                else:
                    xs_slice[i] = i_cr
            for i in range(len(cc_slice)):
                if i != cr_position_cc:
                    cc_slice[i] = slice(None)
                else:
                    cc_slice[i] = i_cr
            for key, xs in self.xs_dict.items():
                xs_set.xs_dict[key] = xs[tuple(xs_slice)]
            for key, cc in self.cc_dict.items():
                xs_set.cc_dict[key] = cc[tuple(cc_slice)]
            dict_assemblies[cr_value] = xs_set
        return dict_assemblies

    def to_snaps( self ):
        snaps = {}
        for key, snap in self.xs_dict.items():
            medium, iso, label = key.split('_')
            if medium not in snaps:
                snaps[medium] = {}
            snaps[medium][iso + label] = snap.reshape(np.product(snap.shape))
        return snaps

    def to_dataframe( self, remove_zeros=False ):
        if self.support == 'grid':
            xs_temp = list(self.xs_set.xs_dict.values())[0]
            df = pd.DataFrame(np.zeros((len(self.xs_dict.keys()), np.product(xs_temp.shape))),
                              index=self.keys)
            for key, xs in self.xs_dict.items():
                df.loc[key, :] = xs.reshape(xs_temp.shape)
            if remove_zeros:
                df = df.loc[~(df < 1E-7).all(axis=1)]
        else:
            data = []
            for key, xs_list in self.xs_dict.items():
                medium, iso, rea, g, g_out, ani = key.split('_')
                cc_key = medium + '_' + iso
                for id_point, xs in xs_list:
                    new_line = id_point.copy()
                    new_line.update({'key': key, 'xs': xs, 'xs_pred': 0., 'cc': self.cc_dict[cc_key][id_point['bu']]})
                    data.append(new_line)
            df = pd.append(data, ignore_index=True)
            df = df[['key', 'tf', 'tm', 'br', 'bu', 'cr', 'cc', 'xs', 'xs_pred']]
        return df

    def to_numpy( self, remove_zeros=False ):
        xs_temp = list(self.xs_dict.values())[0]
        array = np.zeros((len(self.keys), np.product(xs_temp.shape)))
        for i, xs in enumerate(self.xs_dict.values()):
            array[i, :] = xs.reshape(np.product(xs_temp.shape))
        if remove_zeros:
            mask = ~(array < 1E-6).all(axis=1)
            array, self.nonzero_keys = array[mask], self.keys[mask]
        return array

    def to_hdf5( self, data_name, remove_zeros=False, save_keys=True ):
        array = self.to_numpy(remove_zeros=remove_zeros)
        h5f = h5py.File('data/' + data_name + '.h5', 'w')
        h5f.create_dataset(data_name, data=array)
        if save_keys:
            pickle.dump(self.keys, open('data/' + data_name + '_keys.pickle', 'wb'))

    def cc_to_hdf5( self, data_name ):
        cc_temp = list(self.cc_dict.values())[0]
        array = np.zeros((len(self.cc_dict.keys()), np.product(cc_temp.shape)))
        for i, cc in enumerate(self.cc_dict.values()):
            array[i, :] = cc.reshape(np.product(xs_temp.shape))
        h5f = h5py.File('data/' + data_name + '_cc.h5', 'w')
        h5f.create_dataset(data_name, data=array)
        pickle.dump(list(self.cc_dict.keys()), open('data/' + data_name + '_cckeys.pickle', 'wb'))

    def save( self, file_name ):
        dico = {'multiparam': self.multiparam, 'labels': self.labels, 'cc_dict': self.cc_dict, 'xs_dict': self.xs_dict}
        pickle.dump(dico, open(file_name, 'wb'))

    def load( self, file_name ):
        if isinstance(file_name, str):
            dico = pickle.load(open(file_name, 'rb'))
        elif isinstance(file_name, dict):
            dico = file_name
        for key, value in dico.items():
            setattr(self, key, value)



