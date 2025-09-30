from scipy.io import loadmat
import torch
from torch.utils.data import TensorDataset
import yaml
from easydict import EasyDict as edict
from torch_cluster import radius_graph
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import Batch, Data




if __name__ == '__main__':
    with open('config.yaml', 'r') as f:
        config = edict(yaml.safe_load(f))

    # LOADING
    pilots, channel = loadmat('./datasets/pilots_ris_16_len_8_30dBm.mat'), loadmat('./datasets/Q_ris_16_len_8_30dBm.mat')
    pilots = torch.from_numpy(pilots['Y_k_all_real']).permute(3, 2, 0, 1).to(torch.float32)# (n_samples, K, 2N, U*len_pilot)
    pilots = pilots.reshape(pilots.shape[0], pilots.shape[1], -1)
    pilots = (pilots - pilots.mean()) / (pilots.std() + 1e-12)
    pilots = pilots[:-1]

    channel = torch.from_numpy(channel['Q_k_all']).permute(3, 2, 0, 1)# （n_samples, K, N * U, MG**2*Gr)
    channel = torch.stack([channel.real, channel.imag], dim=2).to(torch.float32)# （n_samples, K, 2, N * U, MG**2*Gr)
    Q = channel[:-1]
    # TRAIN
    num_train_samples = int(10e3)

    stats = {'mean': Q[:num_train_samples].mean(),
             'std': Q[:num_train_samples].std()}
    Q = (Q - stats['mean']) / stats['std']

    train_dataset = TensorDataset(pilots[:num_train_samples], Q[:num_train_samples])

    print('save train dataset..\n')
    torch.save((train_dataset, stats),
               f'../datasets/'
               f'ris_elements{config.system.ris.n_elements_per_block * config.system.ris.n_blocks}_'
               f'pilot_len{config.system.pilot_length}_'
               f'train.pt')

    # TEST
    test_dataset = TensorDataset(pilots[num_train_samples:], Q[num_train_samples:])

    print('save test dataset..')
    torch.save((test_dataset, stats),
               f'../datasets/'
               f'ris_elements{config.system.ris.n_elements_per_block * config.system.ris.n_blocks}_'
               f'pilot_len{config.system.pilot_length}_'
               f'test.pt')