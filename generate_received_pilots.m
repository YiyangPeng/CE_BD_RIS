function [Y_k_all_real,Q_k_all] = generate_received_pilots(G_all,H_RI,Gr,MG,len_pilot,Pu,SNR)

% Initialization
Y0 = 1/50;
[N,~,~,~] = size(G_all);
[U,M,n_samples,K] = size(H_RI);
H_k_all = reshape(permute(H_RI, [2 1 4 3]), [M, U*K, n_samples]);

noise_power_dbm = -100;
sigma = sqrt(10^((noise_power_dbm - 30)/10));
num = M*(MG + 1)/2;
num_g = MG*(MG + 1)/2;
len_frame = K*U;

P = func_P(MG);

% Orthogonal pilot sequences
X = generate_pilots(K, U, len_frame);

% Generate received pilots
% n_samples = 2000;
Q_k_all = zeros(N*U, MG^2*Gr, K, n_samples);
Y_k_all = zeros(N, U*len_pilot, K, n_samples);
Y_k_all_1 = zeros(N, U*len_pilot, K, n_samples);
phy_t = zeros(MG^2*Gr, len_pilot, n_samples);

for i_sample = 1 : n_samples

    G = G_all(:,:,i_sample);
    H_k = H_k_all(:,:,i_sample);
    
    % Generate Phi, noise and received pilots for each sample
    N_t = zeros(N,len_frame,len_pilot);
    B = zeros(M,M,len_pilot);
    Phy_t = zeros(M,M,len_pilot);
    Y_t = zeros(N,len_frame,len_pilot);

    
    for i_pilot = 1 : len_pilot
        b = randn(num,1);
        N_t(:,:,i_pilot) = sigma*1/sqrt(2)*(randn(N,len_frame) + 1j*randn(N,len_frame));

        % Construct B, Phy and Y_t
        for i_g = 1 : Gr    
            P_b_g = P*b(1 + num_g*(i_g - 1) : num_g + num_g*(i_g - 1), 1);
            B(1 + MG*(i_g - 1) : MG + MG*(i_g - 1), 1 + MG*(i_g - 1) : MG + MG*(i_g - 1), i_pilot) = reshape(P_b_g, MG, MG);
        end
        Phy_t(:,:,i_pilot) = (Y0*eye(M) + 1j*B(:,:,i_pilot)) \ (Y0*eye(M) - 1j*B(:,:,i_pilot));
        % Phy_t(:,:,i_pilot) = hadamard(M) / sqrt(M);
        Y_t(:,:,i_pilot) = sqrt(Pu)*(G*Phy_t(:,:,i_pilot)*H_k)*X + N_t(:,:,i_pilot);
    end

    % Generate total received pilots (decorrelation)
    Y_k_t = zeros(N, U, K, len_pilot);
    Y_k_t_1 = zeros(N, U, K, len_pilot);
    Y_k_t_test = zeros(N, U, K, len_pilot);
    for i_pilot = 1 : len_pilot
        for i_k = 1 : K
            Y_k_t(:,:,i_k,i_pilot) = 1/len_frame*Y_t(:,:,i_pilot)*X(1 + U*(i_k - 1) : U + U*(i_k - 1),:)';
            Y_k_t_test(:,:,i_k,i_pilot) = sqrt(Pu)*(G*Phy_t(:,:,i_pilot)*H_k(:, 1 + U*(i_k - 1) : U + U*(i_k - 1))) + 1/len_frame*N_t(:,:,i_pilot)*X(1 + U*(i_k - 1) : U + U*(i_k - 1),:)';
        end
    end
    
    % Generate cascaded channels
    for i_k = 1 : K
        for i_g = 1 : Gr
            H_k_g = H_k(1 + MG*(i_g - 1) : MG + MG*(i_g - 1), 1 + U*(i_k - 1) : U + U*(i_k - 1));
            G_g = G(:, 1 + MG*(i_g - 1) : MG + MG*(i_g - 1));
            Q_k_all(:, 1 + MG^2*(i_g - 1) : MG^2 + MG^2*(i_g - 1), i_k, i_sample) = kron(H_k_g.', G_g);
        end
    end

    % Convert Phy_t to phy_t
    for i_pilot = 1 : len_pilot
        for i_g = 1 : Gr
            Phy_t_vec = reshape(Phy_t(1 + MG*(i_g - 1) : MG + MG*(i_g - 1), 1 + MG*(i_g - 1) : MG + MG*(i_g - 1), i_pilot), [], 1);
            phy_t(1 + MG^2*(i_g - 1) : MG^2 + MG^2*(i_g - 1), i_pilot, i_sample) = Phy_t_vec;
        end
    end

    % Calculate Y_k_t in another way
    for i_pilot = 1 : len_pilot
        for i_k = 1 : K
            term1 = sqrt(Pu) * reshape(Q_k_all(:, :, i_k, i_sample) * phy_t(:, i_pilot, i_sample), [N, U]);
            term2 = (1 / len_frame) * N_t(:, :, i_pilot) * X(1 + U*(i_k - 1) : U + U*(i_k - 1), :)';
            Y_k_t_1(:, :, i_k, i_pilot) = term1 + term2;
        end
    end
    
    % Generate outputs (reshape to (N, U*len_pilot, K, num_samples))
    Y_k_all(:, :, :, i_sample) = reshape(permute(Y_k_t, [1 2 4 3]), [N, U*len_pilot, K]);
    Y_k_all_1(:, :, :, i_sample) = reshape(permute(Y_k_t_1, [1 2 4 3]), [N, U*len_pilot, K]);
end
Y_k_all_real = [real(Y_k_all); imag(Y_k_all)]; % (2*N, U*len_pilot, K, num_samples)

end

