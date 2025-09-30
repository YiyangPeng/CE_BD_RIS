import torch
import numpy as np
import yaml
from easydict import EasyDict as edict
from tqdm import tqdm
from torch_geometric.seed import seed_everything
from model_transformer import Net
from torch.utils.data import DataLoader

def MSE_for_channel_est(Q_pred, Q_true):
    # Step 1: Combine real/imaginary parts to form complex tensors
    Q_pred_complex = Q_pred[:, :, 0, :, :] + 1j * Q_pred[:, :, 1, :, :]  # (B, K, N*U, MG**2*Gr)
    Q_true_complex = Q_true[:, :, 0, :, :] + 1j * Q_true[:, :, 1, :, :]  # (B, K, N*U, MG**2*Gr)

    # Step 2: Compute squared Frobenius norm of the error for each user and sample
    error_norm_sq = torch.abs(Q_pred_complex - Q_true_complex) ** 2  # # (B, K, N*U, MG**2*Gr)
    mse_per_entry = error_norm_sq.sum(dim=(2, 3))  # Sum over N*U, MG**2*Gr -> (B, K)

    # Step 3: Average over all users and samples
    averaged_mse = mse_per_entry.mean()  # Scalar

    return averaged_mse

def NMSE_for_channel_est(Q_pred, Q_true):
    # Step 1: Combine real/imag to complex
    Q_pred_complex = Q_pred[:, :, 0, :, :] + 1j * Q_pred[:, :, 1, :, :]  # (B, K, N*U, MG**2*Gr)
    Q_true_complex = Q_true[:, :, 0, :, :] + 1j * Q_true[:, :, 1, :, :]  # (B, K, N*U, MG**2*Gr)

    # Step 2: Compute squared Frobenius norms
    error_norm_sq = torch.abs(Q_pred_complex - Q_true_complex) ** 2  # |G - Ĝ|²
    error_power = error_norm_sq.sum(dim=(1, 2, 3))  # Sum over N*U, MG**2*Gr -> (B, )

    true_power = torch.abs(Q_true_complex) ** 2  # |G|²
    true_power = true_power.sum(dim=(1, 2, 3))  # (B, )

    # Step 3: Compute NMSE for each user and sample
    nmse_per_entry = error_power / true_power # (B, )

    # Step 4: Average over all users and samples
    averaged_nmse = nmse_per_entry.mean(dim=(0)) # (, )

    return averaged_nmse

def trainer(config, datasets):
    # Initialization
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Net(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.train.lr, weight_decay=0)
    # Loader
    loaders = {split: DataLoader(
        datasets[split], batch_size=config.train.batch_size, shuffle=(split == 'train')
    ) for split in ['train', 'test']}
    mean, std = datasets['stats'].values()

    @torch.no_grad()
    def eval(tag):
        model.eval()
        nmse = []
        for data in tqdm(loaders[tag]):
            pilots, Q = data
            Q_pred = model(x=pilots.to(device))
            nmse.append(
                NMSE_for_channel_est(Q_pred.to('cpu') * std + mean, Q * std + mean))
        return sum(nmse) / len(nmse)


    for epoch in range(config.train.max_epochs):
        train_loss = []
        for data in tqdm(loaders['train']):
            pilots, Q = data
            # TRAINING
            model.train()
            Q_pred = model(x=pilots.to(device))
            optimizer.zero_grad()
            loss = MSE_for_channel_est(Q_pred, Q.to(device))
            loss.backward()
            optimizer.step()
            train_loss.append(loss.detach().item())
        train_loss_mean = np.mean(train_loss)
        print(train_loss_mean)

        # Evaluation on test dataset
        nmse_train, nmse_test = eval('train'), eval('test')

        print(f"Epoch {epoch+1:03d}: "
              f"Train Metirc (NMSE) = {nmse_train:.6f} \n"
              f"Test Metric (NMSE) = {nmse_test:.6f}")


if __name__ == '__main__':
    with open('config.yaml', 'r') as f:
        config = edict(yaml.safe_load(f))

    seed_everything(seed=41)
    datasets = {}
    datasets['train'], stats = torch.load(f'../datasets/'
               f'ris_elements{config.system.ris.n_elements_per_block * config.system.ris.n_blocks}_'
               f'pilot_len{config.system.pilot_length}'
               f'_train.pt', weights_only=False)

    datasets['test'], _ = torch.load(f'../datasets/'
               f'ris_elements{config.system.ris.n_elements_per_block * config.system.ris.n_blocks}_'
               f'pilot_len{config.system.pilot_length}'
               f'_test.pt', weights_only=False)
    datasets['stats'] = stats

    trainer(config, datasets)

