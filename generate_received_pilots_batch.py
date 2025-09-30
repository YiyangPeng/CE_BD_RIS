import numpy as np
from numpy.linalg import inv
from .generate_channel import func_P

def generate_pilot_matrix(K, U, L):
    """
    Generate (K*U) x L pilot matrix with:
    - |X[i,j]| = 1
    - X_k X_k^H = L*I_U (antenna orthogonality)
    - X_k X_k'^H = 0 (user orthogonality)
    """
    if L % U != 0:
        raise ValueError("L must be multiple of U for this design")
    if L % K != 0:
        raise ValueError("L must be multiple of K for this design")
    if L % (K*U) != 0:
        raise ValueError("L must be multiple of K*U for this design")

    # Total number of orthogonal sequences needed
    N = K * U

    # Create N orthogonal sequences of length L using DFT
    n = np.arange(N)
    k = n.reshape(-1, 1)
    X = np.exp(-1j * 2 * np.pi * n * k / N)

    # If L > N, repeat columns while maintaining orthogonality
    if L > N:
        reps = L // N
        remainder = L % N
        X = np.tile(X, (1, reps))
        if remainder > 0:
            X = np.hstack([X, X[:, :remainder]])

    return X

def generate_received_pilots_batch(params_system, PL, channels, len_pilot, Pu_dBm=15, num_samples=100, noise_power_dbm=-100, SNR_db=20):
    (N, Gr, MG, K, U) = params_system
    # Unpack the channels
    channel_bs_ris, channel_ris_user = channels

    # Initialization
    Y0 = 1 / 50
    sigma = np.sqrt(10 ** ((noise_power_dbm - 30) / 10))
    Pu = 10 ** ((Pu_dBm - 30) / 10)
    # SNR = 10 ** (SNR_db / 10)
    # Pu = SNR * sigma ** 2 / (len_pilot * K * PL.mean())
    len_frame = K*U
    M = Gr * MG
    num = M * (MG + 1) // 2
    num_g = MG * (MG + 1) // 2

    P = func_P(MG)

    # Orthogonal pilot sequences
    X = generate_pilot_matrix(K, U, len_frame)

    # Initialize received pilots
    Y_k_all = np.zeros((N, U * len_pilot, K, num_samples), dtype=complex)
    Y_k_all_1 = np.zeros((N, U * len_pilot, K, num_samples), dtype=complex)
    Q_k_all = np.zeros((N * U, MG ** 2 * Gr, K, num_samples), dtype=complex)
    phy_t = np.zeros((MG ** 2 * Gr, len_pilot, num_samples), dtype=complex)
    Q_k_pred = np.zeros((N * U, MG ** 2 * Gr, K, num_samples), dtype=complex)
    #Q_k_pred_1 = np.zeros((N * U, MG ** 2 * Gr, K, num_samples), dtype=complex)

    for i_sample in range(num_samples):
        # Get channels for this sample
        G = channel_bs_ris[..., i_sample]
        H_k_all = channel_ris_user[..., i_sample]
        H_k = H_k_all.reshape(M, U * K, order='F')
        #H_d = channel_bs_user[..., i_sample]

        # Generate Phi, noise and received pilots for this sample
        B = np.zeros((M, M, len_pilot), dtype=complex)
        Phy_t = np.zeros((M, M, len_pilot), dtype=complex)
        N_t = np.zeros((N, len_frame, len_pilot), dtype=complex)
        Y_t = np.zeros((N, len_frame, len_pilot), dtype=complex)

        for i_pilot in range(len_pilot):
            b = np.random.randn(num, 1)
            N_t[:, :, i_pilot] = sigma * 1 / np.sqrt(2) * (
                        np.random.randn(N, len_frame) + 1j * np.random.randn(N, len_frame))

            # Construct B, Phy and Y_t
            for i_g in range(Gr):
                P_b_g = P @ b[num_g * i_g: num_g * (i_g + 1)]
                B[MG * i_g:MG * (i_g + 1), MG * i_g:MG * (i_g + 1), i_pilot] = P_b_g.reshape(MG, MG)
                # B[MG * i_g:MG * (i_g + 1), MG * i_g:MG * (i_g + 1), i_pilot] = np.random.randn(MG, MG)
            Phy_t[:, :, i_pilot] = inv(Y0 * np.eye(M) + 1j * B[:, :, i_pilot]) @ (Y0 * np.eye(M) - 1j * B[:, :, i_pilot])
            Y_t[:, :, i_pilot] = np.sqrt(Pu) * (G @ Phy_t[:, :, i_pilot] @ H_k) @ X + N_t[:, :, i_pilot]

        # Generate total received pilots (decorrelation)
        Y_k_t = np.zeros((N, U, K, len_pilot), dtype=complex)
        Y_k_t_1 = np.zeros((N, U, K, len_pilot), dtype=complex)
        N_tilde = np.zeros((N, U, K, len_pilot), dtype=complex)
        #Y_k_t_test = np.zeros((N, U, K, len_pilot), dtype=complex)
        for i_pilot in range(len_pilot):
            for i_k in range(K):
                Y_k_t[:, :, i_k, i_pilot] = (1 / len_frame) * Y_t[:, :, i_pilot] @ X[i_k * U:(i_k + 1) * U, :].conj().T
                #Y_k_t_test[:, :, i_k, i_pilot] = np.sqrt(Pu) * (G @ Phy_t[:, :, i_pilot] @ H_k[:, i_k * U:(i_k + 1) * U]) + (1 / len_frame) * N_t[:, :, i_pilot] @ X[i_k * U:(i_k + 1) * U, :].conj().T

        for i_k in range(K):
            for i_g in range(Gr):
                Q_k_all[:, MG ** 2 * i_g:MG ** 2 * (i_g + 1), i_k, i_sample] = np.kron(
                    H_k[MG * i_g:MG * (i_g + 1), i_k * U:(i_k + 1) * U].T,
                    G[:, MG * i_g:MG * (i_g + 1)])

        for i_pilot in range(len_pilot):
            for i_g in range(Gr):
                Phy_t_vec = Phy_t[MG * i_g:MG * (i_g + 1), MG * i_g:MG * (i_g + 1), i_pilot].ravel()
                phy_t[MG ** 2 * i_g:MG ** 2 * (i_g + 1), i_pilot, i_sample] = Phy_t_vec

        for i_pilot in range(len_pilot):
            for i_k in range(K):
                Y_k_t_1[:, :, i_k, i_pilot] = np.sqrt(Pu) * (Q_k_all[:, :, i_k, i_sample] @ phy_t[:, i_pilot, i_sample]).reshape(N, U, order='F') + (1 / len_frame) * N_t[:, :, i_pilot] @ X[i_k * U:(i_k + 1) * U, :].conj().T
                N_tilde[:, :, i_k, i_pilot] = (1 / len_frame) * N_t[:, :, i_pilot] @ X[i_k * U:(i_k + 1) * U, :].conj().T

        Y_k_all[:, :, :, i_sample] = Y_k_t.transpose(0, 1, 3, 2).reshape(N, U * len_pilot, K) # Reshape to (N, U*len_pilot, K, num_samples)
        Y_k_all_1[:, :, :, i_sample] = Y_k_t_1.transpose(0, 1, 3, 2).reshape(N, U * len_pilot, K) # Reshape to (N, U*len_pilot, K, num_samples)

        # Do LS estimation
        #Y_k = np.zeros((N * U, len_pilot, K), dtype=complex)
       # for i_k in range(K):
           # Y_k[:, :, i_k] = np.sqrt(Pu) * Q_k_all[:, :, i_k, i_sample] @ phy_t[:, :, i_sample] + N_tilde[:, :, i_k, :].reshape(N*U, len_pilot, order='F')
            #Q_k_pred[:, :, i_k, i_sample] = (1/np.sqrt(Pu)) * Y_k[:, :, i_k] @ (phy_t[:, :, i_sample].conj().T @ np.linalg.pinv(phy_t[:, :, i_sample] @ phy_t[:, :, i_sample].conj().T))
            #Q_k_pred_1[:, :, i_k, i_sample] = Q_k_all[:, :, i_k, i_sample] + (1/np.sqrt(Pu)) * N_tilde[:, :, i_k, :].reshape(N*U, len_pilot, order='F') @ (phy_t[:, :, i_sample].conj().T @ np.linalg.pinv(phy_t[:, :, i_sample] @ phy_t[:, :, i_sample].conj().T))

    Y_k_all_real = np.concatenate([Y_k_all.real, Y_k_all.imag], axis=0)

    return Y_k_all, Y_k_all_real


def generate_received_pilots_batch_miso(params_system, channels, len_pilot, Pu_dBm=15, num_samples=100, noise_power_dbm=-100, scale_factor_db=100):
    (N, Gr, MG, K) = params_system
    # Unpack the channels
    channel_bs_ris, channel_ris_user = channels

    # Initialization
    Y0 = 1 / 50
    sigma = np.sqrt(10 ** ((noise_power_dbm - 30) / 10))
    Pu = 10 ** ((Pu_dBm - 30) / 10)
    len_frame = K
    M = Gr * MG
    num = M * (MG + 1) // 2
    num_g = MG * (MG + 1) // 2

    P = func_P(MG)

    # Orthogonal pilot sequence
    j, k = np.meshgrid(np.arange(K), np.arange(K))
    dft_matrix = np.exp(-2j * np.pi * j * k / K) / np.sqrt(K)
    X = np.sqrt(K) * dft_matrix  # Scale by sqrt(K)

    # Initialize received pilots
    Y_k_all = np.zeros((N, len_pilot, K, num_samples), dtype=complex)
    Y_k_all_1 = np.zeros((N, len_pilot, K, num_samples), dtype=complex)
    Q_k_all = np.zeros((N, MG ** 2 * Gr, K, num_samples), dtype=complex)
    phy_t = np.zeros((MG ** 2 * Gr, len_pilot, num_samples), dtype=complex)
    #phy_tilde_t = np.zeros((MG ** 2 * Gr + 1, len_pilot, num_samples), dtype=complex)
    #Q_tilde_k_all = np.zeros((N, MG ** 2 * Gr + 1, K, num_samples), dtype=complex)

    for i_sample in range(num_samples):
        # Get channels for this sample
        G = channel_bs_ris[..., i_sample]
        H_k = channel_ris_user[..., i_sample]
        #H_d = channel_bs_user[..., i_sample]

        # Generate Phi, noise and received pilots for this sample
        B = np.zeros((M, M, len_pilot), dtype=complex)
        Phy_t = np.zeros((M, M, len_pilot), dtype=complex)
        N_t = np.zeros((N, len_frame, len_pilot), dtype=complex)
        Y_t = np.zeros((N, len_frame, len_pilot), dtype=complex)

        for i_pilot in range(len_pilot):
            b = np.random.randn(num, 1)
            N_t[:, :, i_pilot] = sigma * 1 / np.sqrt(2) * (
                        np.random.randn(N, len_frame) + 1j * np.random.randn(N, len_frame))

            # Construct B, Phy and Y_t
            for i_g in range(Gr):
                P_b_g = P @ b[num_g * i_g: num_g * (i_g + 1)]
                B[MG * i_g:MG * (i_g + 1), MG * i_g:MG * (i_g + 1), i_pilot] = P_b_g.reshape(MG, MG)
            Phy_t[:, :, i_pilot] = inv(Y0 * np.eye(M) + 1j * B[:, :, i_pilot]) @ (Y0 * np.eye(M) - 1j * B[:, :, i_pilot])
            Y_t[:, :, i_pilot] = np.sqrt(Pu) * (G @ Phy_t[:, :, i_pilot] @ H_k) @ X + N_t[:, :, i_pilot]

        # Generate total received pilots (decorrelation)
        y_k_t = np.zeros((N, K, len_pilot), dtype=complex)
        y_k_t_1 = np.zeros((N, K, len_pilot), dtype=complex)
        for i_pilot in range(len_pilot):
            for i_k in range(K):
                y_k_t[:, i_k, i_pilot] = (1 / len_frame) * Y_t[:, :, i_pilot] @ X[i_k, :].conj().T


        for i_k in range(K):
            for i_g in range(Gr):
                Q_k_all[:, MG ** 2 * i_g:MG ** 2 * (i_g + 1), i_k, i_sample] = np.kron(
                    channel_ris_user[MG * i_g:MG * (i_g + 1), i_k, i_sample].T,
                    channel_bs_ris[:, MG * i_g:MG * (i_g + 1), i_sample])
            # Construct Q_tilde_k including bs_user channel
            #Q_tilde_k_all[:, 0, i_k, i_sample] = channel_bs_user[:, i_k, i_sample]
            #Q_tilde_k_all[:, 1:MG ** 2 * Gr + 1, i_k, i_sample] = Q_k_all[:, :, i_k, i_sample]

        for i_pilot in range(len_pilot):
            for i_g in range(Gr):
                Phy_t_vec = Phy_t[MG * i_g:MG * (i_g + 1), MG * i_g:MG * (i_g + 1), i_pilot].ravel()
                phy_t[MG ** 2 * i_g:MG ** 2 * (i_g + 1), i_pilot, i_sample] = Phy_t_vec
            #phy_tilde_t[0, i_pilot, i_sample] = 1
            #phy_tilde_t[1:MG ** 2 * Gr + 1, i_pilot, i_sample] = phy_t[:, i_pilot, i_sample]

        for i_pilot in range(len_pilot):
            for i_k in range(K):
                y_k_t_1[:, i_k, i_pilot] = np.sqrt(Pu) * (Q_k_all[:, :, i_k, i_sample] @ phy_t[:, i_pilot, i_sample]) + (1 / len_frame) * N_t[:, :, i_pilot] @ X[i_k, :].conj().T

        for i_k in range(K):
            Y_k_all[:, :, :, i_sample] = np.transpose(y_k_t, (0, 2, 1))  # Reshape to (N, len_pilot, K, num_samples)
            Y_k_all_1[:, :, :, i_sample] = np.transpose(y_k_t_1, (0, 2, 1))  # Reshape to (N, len_pilot, K, num_samples)

    Y_k_all_real = np.concatenate([Y_k_all.real, Y_k_all.imag], axis=0)

    return Y_k_all, Y_k_all_real

def generate_received_pilots_batch_no_decorr(params_system, channels, len_pilot, Pu_dBm=15, num_samples=100, noise_power_dbm=-100):
    (N, Gr, MG, K, U) = params_system
    # Unpack the channels
    channel_bs_ris, channel_ris_user = channels

    # Initialization
    Y0 = 1 / 50
    sigma = np.sqrt(10 ** ((noise_power_dbm - 30) / 10))
    Pu = 10 ** ((Pu_dBm - 30) / 10)
    len_frame = K*U
    M = Gr * MG
    num = M * (MG + 1) // 2
    num_g = MG * (MG + 1) // 2

    P = func_P(MG)

    # Orthogonal pilot sequences
    X = generate_pilot_matrix(K, U, len_frame)

    # Initialize received pilots
    Y_k_all = np.zeros((N, len_frame, len_pilot, K, num_samples), dtype=complex)
    Y_k_all_1 = np.zeros((N, U * len_pilot, K, num_samples), dtype=complex)
    Q_k_all = np.zeros((N * U, MG ** 2 * Gr, K, num_samples), dtype=complex)
    phy_t = np.zeros((MG ** 2 * Gr, len_pilot, num_samples), dtype=complex)

    for i_sample in range(num_samples):
        # Get channels for this sample
        G = channel_bs_ris[..., i_sample]
        H_k_all = channel_ris_user[..., i_sample]
        H_k = H_k_all.reshape(M, U * K, order='F')

        # Generate Phi, noise and received pilots for this sample
        B = np.zeros((M, M, len_pilot), dtype=complex)
        Phy_t = np.zeros((M, M, len_pilot), dtype=complex)
        N_t = np.zeros((N, len_frame, len_pilot), dtype=complex)
        Y_t = np.zeros((N, len_frame, len_pilot), dtype=complex)

        for i_pilot in range(len_pilot):
            #b = np.random.randn(num, 1)
            N_t[:, :, i_pilot] = sigma * 1 / np.sqrt(2) * (
                        np.random.randn(N, len_frame) + 1j * np.random.randn(N, len_frame))

            # Construct B, Phy and Y_t
            for i_g in range(Gr):
                #P_b_g = P @ b[num_g * i_g: num_g * (i_g + 1)]
                #B[MG * i_g:MG * (i_g + 1), MG * i_g:MG * (i_g + 1), i_pilot] = P_b_g.reshape(MG, MG)
                B[MG * i_g:MG * (i_g + 1), MG * i_g:MG * (i_g + 1), i_pilot] = np.random.randn(MG, MG)
            Phy_t[:, :, i_pilot] = inv(Y0 * np.eye(M) + 1j * B[:, :, i_pilot]) @ (Y0 * np.eye(M) - 1j * B[:, :, i_pilot])
            Y_t[:, :, i_pilot] = np.sqrt(Pu) * (G @ Phy_t[:, :, i_pilot] @ H_k) @ X + N_t[:, :, i_pilot]

        for i_k in range(K):
            Y_k_all[:, :, :, i_k, i_sample] = Y_t # Reshape to (N, len_frame, len_pilot, K, num_samples)


    Y_k_all_real = np.concatenate([Y_k_all.real, Y_k_all.imag], axis=0)

    return Y_k_all, Y_k_all_real