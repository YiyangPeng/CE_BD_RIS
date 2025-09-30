import numpy as np
from numpy.linalg import inv


def func_P(MG):
    num = MG * (MG + 1) // 2
    P = np.zeros((MG ** 2, num))

    for i in range(1, MG + 1):
        for j in range(1, MG + 1):
            if i >= j:
                P[MG * (i - 1) + j - 1, i - 1 + (2 * MG - j) * (j - 1) // 2] = 1
            else:
                P[MG * (i - 1) + j - 1, j - 1 + (2 * MG - i) * (i - 1) // 2] = 1
    return P
def generate_user_locations(K):
    # Define locations
    user_x = 5 + 10 * np.random.rand(K, 1)  # x ∈ [5,15]
    user_y = -5 + 10 * np.random.rand(K, 1)  # y ∈ [-5,5]
    user_z = -5 * np.ones((K, 1))  # z = -5
    user_pos = np.hstack([user_x, user_y, user_z])  # K×3 matrix of user positions

    # user_x = np.random.rand(2, 1)
    # user_x[0] = 5
    # user_x[1] = 5.1
    # user_y = np.random.rand(2, 1)
    # user_y[0] = -5
    # user_y[1] = -6
    # user_z = -5 * np.ones((2, 1))
    # user_pos = np.hstack([user_x, user_y, user_z])  # K×3 matrix of user positions

    return user_pos


def generate_channels(params_system, ris_idx, num_samples=100):
    (N, Gr, MG, K, U) = params_system

    # Initialization
    M = ris_idx[0] * ris_idx[1]
    K_TI = 10 ** (20 / 10)
    K_IU = 10 ** (20 / 10)
    PL0 = 1e-3
    #channel_bs_user = np.zeros((N, K, num_samples), dtype=complex)
    channel_bs_ris = np.zeros((N, M, num_samples), dtype=complex)
    channel_ris_user = np.zeros((M, U, K, num_samples), dtype=complex)
    set_location_user = np.zeros((K, 3, num_samples))
    PL_all = np.zeros((K, num_samples))

    # Define locations and distance between BS and RIS
    RIS_pos = np.array([0, 10, 0])
    BS_pos = np.array([60, -60, 0])
    d1 = np.linalg.norm(BS_pos - RIS_pos)  # Distance BS→RIS

    # Generate all user locations
    num_user_locations = int(2e3) # todo 12 or 12k``
    all_user_locations = [generate_user_locations(K) for _ in range(num_user_locations)]

    for i_sample in range(num_samples):

        # Select user location for this sample (round-robin assignment)
        loc_idx = i_sample % num_user_locations
        user_pos = all_user_locations[loc_idx]
        set_location_user[:, :, i_sample] = user_pos

        # Path loss model (include scale factor)
        d2 = np.linalg.norm(user_pos - RIS_pos[None, :], axis=1)  # Distance RIS→Each User (K×1 vector)
        #d3 = np.linalg.norm(user_pos - BS_pos, axis=1)  # Distance BS→Each User (K×1 vector)
        epsilon1 = 2.5
        #epsilon2 = 3.8

        PL_T = PL0 * d1 ** (-epsilon1) # BS→RIS path loss
        PL_R = PL0 * d2 ** (-epsilon1) # RIS→User path loss (K×1 vector)
        PL_all[:, i_sample] = PL_T * PL_R  # total path losses

        # ----------------TX-RIS channel (G)----------------------
        # Calculate angles from BS to RIS
        sinElevation_Tx = 0  # Elevation angle (BS→RIS)
        sinElevation_IT_cosAzimuth_IT = (BS_pos[0] - RIS_pos[0]) / d1
        sinElevation_IT_sinAzimuth_IT = (BS_pos[1] - RIS_pos[1]) / d1

        # LoS channel
        ind_x = np.arange(0, ris_idx[0]).reshape(-1, 1)
        ind_y = np.arange(0, ris_idx[1]).reshape(-1, 1)
        ind_Tx = np.arange(0, N).reshape(-1, 1)
        alpha = 1
        a_Tx = np.sqrt(1 / N) * np.exp(1j * np.pi * ind_Tx * np.sin(sinElevation_Tx))  # BS array response
        a_x = np.sqrt(1 / ris_idx[0]) * np.exp(1j * np.pi * ind_x * sinElevation_IT_cosAzimuth_IT)
        a_y = np.sqrt(1 / ris_idx[1]) * np.exp(1j * np.pi * ind_y * sinElevation_IT_sinAzimuth_IT)
        a = np.kron(a_x, a_y)
        H_LoS_TI = np.sqrt(M * N) * alpha * a_Tx @ a.T

        # NLoS channel
        H_NLoS_TI = np.sqrt(1 / 2) * (np.random.randn(N, M) + 1j * np.random.randn(N, M))
        G = np.sqrt(K_TI / (K_TI + 1)) * H_LoS_TI + np.sqrt(1 / (K_TI + 1)) * H_NLoS_TI
        channel_bs_ris[:, :, i_sample] = np.sqrt(PL_T) * G  # RIS-BS channel

        # -----------------RIS-User channel-------------------
        # Calculate angles from BS to RIS
        sinElevation_Rx = (user_pos[:, 2] - RIS_pos[2]) / d2  # Elevation angle (users→RIS)
        sinElevation_UI_cosAzimuth_UI = (RIS_pos[0] - user_pos[:, 0]) / d2
        sinElevation_IT_sinAzimuth_IT = (RIS_pos[1] - user_pos[:, 1]) / d2

        # Los channel
        H_LoS_IU = np.zeros((M, U, K), dtype=complex)
        ind_Rx = np.arange(0, U).reshape(-1, 1)
        alpha = 1
        for k in range(K):
            a_Rx = np.sqrt(1 / U) * np.exp(1j * np.pi * ind_Rx * sinElevation_Rx[k])
            a_x = np.sqrt(1 / ris_idx[0]) * np.exp(1j * np.pi * ind_x * sinElevation_UI_cosAzimuth_UI[k])
            a_y = np.sqrt(1 / ris_idx[1]) * np.exp(1j * np.pi * ind_y * sinElevation_IT_sinAzimuth_IT[k])
            a = np.kron(a_x, a_y)
            H_LoS_IU[:, :, k] = np.sqrt(M * U) * alpha * a @ a_Rx.T

        # NLoS channel
        H_NLoS_IU = np.sqrt(1 / 2) * (np.random.randn(M, U, K) + 1j * np.random.randn(M, U, K))
        H_IU = np.sqrt(K_IU / (K_IU + 1)) * H_LoS_IU + np.sqrt(1 / (K_IU + 1)) * H_NLoS_IU
        for k in range(K):
            channel_ris_user[:, :, k, i_sample] = np.sqrt(PL_R[k]) * H_IU[:, :, k]  # users-RIS channel

        # -----------------BS-User channel-------------------
        #PL_d = PL0 * d3 ** (-epsilon2)  # BS→user path loss

        # NLoS channel
        #H_d_all = np.sqrt(1 / 2) * (np.random.randn(N, K) + 1j * np.random.randn(N, K))
        #for k in range(K):
            #channel_bs_user[:, k, i_sample] = np.sqrt(PL_d[k]) * H_d_all[:, k]

    #channel_bs_user = np.zeros((N, K, num_samples), dtype=complex)

    # Combine all channels into a tuple
    channels = (channel_bs_ris, channel_ris_user)


    return channels, PL_all, set_location_user

def generate_cascaded_channels(params_system, channels, num_samples):
    (N, Gr, MG, K, U) = params_system
    # Unpack the channels
    channel_bs_ris, channel_ris_user = channels

    Q_k_all = np.zeros((N * U, MG ** 2 * Gr, K, num_samples), dtype=complex)
    #Q_tilde_k_all = np.zeros((N, MG**2 * Gr + 1, K, num_samples), dtype=complex)

    # Construct cascaded channels Q_k for all users
    for i_sample in range(num_samples):

        G = channel_bs_ris[..., i_sample]
        H_k_all = channel_ris_user[..., i_sample]
        H_k = H_k_all.reshape(MG * Gr, U * K, order='F')

        for i_k in range(K):
            for i_g in range(Gr):
                Q_k_all[:, MG**2 * i_g:MG**2 * (i_g + 1), i_k, i_sample] = np.kron(
                    H_k[MG * i_g:MG * (i_g + 1), i_k * U:(i_k + 1) * U].T, G[:, MG * i_g:MG * (i_g + 1)])
            # Construct Q_tilde_k including bs_user channel
            #Q_tilde_k_all[:, 0, i_k, i_sample] = channel_bs_user[:, i_k, i_sample]
            #Q_tilde_k_all[:, 1:MG**2 * Gr + 1, i_k, i_sample] = Q_k_all[:, :, i_k, i_sample]

    return Q_k_all

def generate_channels_miso(params_system, ris_idx, num_samples=100):
    (N, Gr, MG, K) = params_system

    # Initialization
    Nk = 1
    M = ris_idx[0] * ris_idx[1]
    K_TI = 10 ** (20 / 10)
    K_IU = 10 ** (20 / 10)
    PL0 = 1e-3
    #channel_bs_user = np.zeros((N, K, num_samples), dtype=complex)
    channel_bs_ris = np.zeros((N, M, num_samples), dtype=complex)
    channel_ris_user = np.zeros((M, K, num_samples), dtype=complex)
    set_location_user = np.zeros((K, 3, num_samples))
    PL_all = np.zeros((K, num_samples))

    # Define locations and distance between BS and RIS
    RIS_pos = np.array([-10, 0, 0])
    BS_pos = np.array([20, -20, 0])
    d1 = np.linalg.norm(BS_pos - RIS_pos)  # Distance BS→RIS

    for i_sample in range(num_samples):

        # Generate user locations for this sample
        user_pos = generate_user_locations(K)
        set_location_user[:, :, i_sample] = user_pos

        # Path loss model (include scale factor)
        d2 = np.linalg.norm(user_pos - RIS_pos, axis=1)  # Distance RIS→Each User (K×1 vector)
        #d3 = np.linalg.norm(user_pos - BS_pos, axis=1)  # Distance BS→Each User (K×1 vector)
        epsilon1 = 2.5
        #epsilon2 = 3.8

        PL_T = PL0 * d1 ** (-epsilon1) # BS→RIS path loss
        PL_R = PL0 * d2 ** (-epsilon1) # RIS→User path loss (K×1 vector)
        PL_all[:, i_sample] = PL_T * PL_R  # total path losses

        # ----------------TX-RIS channel (G)----------------------
        # Calculate angles from BS to RIS
        sinElevation_Tx = 0  # Elevation angle (BS→RIS)
        sinElevation_IT_cosAzimuth_IT = (BS_pos[0] - RIS_pos[0]) / d1
        sinElevation_IT_sinAzimuth_IT = (BS_pos[1] - RIS_pos[1]) / d1

        # LoS channel
        ind_x = np.arange(0, ris_idx[0]).reshape(-1, 1)
        ind_y = np.arange(0, ris_idx[1]).reshape(-1, 1)
        ind_Tx = np.arange(0, N).reshape(-1, 1)
        alpha = 1
        a_Tx = np.sqrt(1 / N) * np.exp(1j * np.pi * ind_Tx * np.sin(sinElevation_Tx))  # BS array response
        a_x = np.sqrt(1 / ris_idx[0]) * np.exp(1j * np.pi * ind_x * sinElevation_IT_cosAzimuth_IT)
        a_y = np.sqrt(1 / ris_idx[1]) * np.exp(1j * np.pi * ind_y * sinElevation_IT_sinAzimuth_IT)
        a = np.kron(a_x, a_y)
        H_LoS_TI = np.sqrt(M * N) * alpha * a_Tx @ a.T

        # NLoS channel
        H_NLoS_TI = np.sqrt(1 / 2) * (np.random.randn(N, M) + 1j * np.random.randn(N, M))
        G = np.sqrt(K_TI / (K_TI + 1)) * H_LoS_TI + np.sqrt(1 / (K_TI + 1)) * H_NLoS_TI
        channel_bs_ris[:, :, i_sample] = np.sqrt(PL_T) * G  # RIS-BS channel

        # -----------------RIS-User channel-------------------
        # Calculate angles from BS to RIS
        sinElevation_Rx = (user_pos[:, 2] - RIS_pos[2]) / d2  # Elevation angle (users→RIS)
        sinElevation_UI_cosAzimuth_UI = (RIS_pos[0] - user_pos[:, 0]) / d2
        sinElevation_IT_sinAzimuth_IT = (RIS_pos[1] - user_pos[:, 1]) / d2

        # Los channel
        H_LoS_IU = np.zeros((M, K), dtype=complex)
        ind_Rx = np.arange(0, Nk).reshape(-1, 1)
        alpha = 1
        a_Rx = np.sqrt(1 / K) * np.exp(1j * np.pi * ind_Rx * sinElevation_Rx.reshape(-1, 1))
        for k in range(K):
            a_x = np.sqrt(1 / ris_idx[0]) * np.exp(1j * np.pi * ind_x * sinElevation_UI_cosAzimuth_UI[k])
            a_y = np.sqrt(1 / ris_idx[1]) * np.exp(1j * np.pi * ind_y * sinElevation_IT_sinAzimuth_IT[k])
            a = np.kron(a_x, a_y)
            H_LoS_IU[:, k] = np.sqrt(M * K) * alpha * a @ a_Rx[k].T

        # NLoS channel
        H_NLoS_IU = np.sqrt(1 / 2) * (np.random.randn(M, K) + 1j * np.random.randn(M, K))
        H_IU = np.sqrt(K_IU / (K_IU + 1)) * H_LoS_IU + np.sqrt(1 / (K_IU + 1)) * H_NLoS_IU
        for k in range(K):
            channel_ris_user[:, k, i_sample] = np.sqrt(PL_R[k]) * H_IU[:, k]  # users-RIS channel

        # -----------------BS-User channel-------------------
        #PL_d = PL0 * d3 ** (-epsilon2)  # BS→user path loss

        # NLoS channel
        #H_d_all = np.sqrt(1 / 2) * (np.random.randn(N, K) + 1j * np.random.randn(N, K))
        #for k in range(K):
            #channel_bs_user[:, k, i_sample] = np.sqrt(PL_d[k]) * H_d_all[:, k]

    #channel_bs_user = np.zeros((N, K, num_samples), dtype=complex)

    # Combine all channels into a tuple
    channels = (channel_bs_ris, channel_ris_user)


    return channels, PL_all, set_location_user

def generate_cascaded_channels_miso(params_system, channels, num_samples):
    (N, Gr, MG, K) = params_system
    # Unpack the channels
    channel_bs_ris, channel_ris_user = channels

    Q_k_all = np.zeros((N, MG**2 * Gr, K, num_samples), dtype=complex)
    #Q_tilde_k_all = np.zeros((N, MG**2 * Gr + 1, K, num_samples), dtype=complex)

    # Construct cascaded channels Q_k for all users
    for i_sample in range(num_samples):
        for i_k in range(K):
            for i_g in range(Gr):
                Q_k_all[:, MG**2 * i_g:MG**2 * (i_g + 1), i_k, i_sample] = np.kron(
                    channel_ris_user[MG * i_g:MG * (i_g + 1), i_k, i_sample].T, channel_bs_ris[:, MG * i_g:MG * (i_g + 1), i_sample])
            # Construct Q_tilde_k including bs_user channel
            #Q_tilde_k_all[:, 0, i_k, i_sample] = channel_bs_user[:, i_k, i_sample]
            #Q_tilde_k_all[:, 1:MG**2 * Gr + 1, i_k, i_sample] = Q_k_all[:, :, i_k, i_sample]

    return Q_k_all