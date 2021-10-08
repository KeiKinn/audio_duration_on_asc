from torch.utils.data import Dataset
import h5py


class DCASE2020(Dataset):
    def __init__(self, hdf5_path, backbone):
        self.hdf5_path = hdf5_path
        self.backbone = backbone

    def __len__(self):
        hf = h5py.File(self.hdf5_path, 'r')
        res = hf['audio_name'].len()
        hf.close()
        return res

    def __getitem__(self, index):
        """Get input and target data of an audio clip.
        """
        with h5py.File(self.hdf5_path, 'r') as hf:
            audio_name = hf['audio_name'][index].decode()
            label = hf['label'][index]
            logmel = hf['logmel'][index]
            if "baseline" == self.backbone:
                return {'audio_name': audio_name, 'label': label, 'logmel': logmel}
            elif "dresnet" == self.backbone:
                delta= hf['delta'][index]
                delta_delta = hf['delta-delta'][index]
                return {'audio_name': audio_name, 'label': label, 'logmel': logmel, 'delta': delta, 'delta-delta': delta_delta}
