clear;

%%----------------------system parameters---------------------------
% N = 4; %% number of antennas at BS
% K = 4; %% number of users
% U = 1; %% number of antennas at each user
M_ind = [4,4];
M = M_ind(1)*M_ind(2); %% number of RIS elements
MG = 4; %% group size
Gr = M/MG;

load('H_IT_16_4.mat','H_IT');
load('H_RI_16_4.mat','H_RI'); % (M, U, n_samples, K)

%%----------------------simulation parameters---------------------------
len_pilot = 8;
Pu_dBm = 30;
Pu = 10^((Pu_dBm - 30) / 10);
SNR_db = 20;
SNR = 10^(SNR_db / 10);

% Extend H_IT to G_all （N, M, n_samples)
repetitions = 12000;
new_depth = size(H_IT, 3) * repetitions;
G_all = zeros(size(H_IT, 2), size(H_IT, 1), new_depth + 1);
for i = 1:size(H_IT, 3)
    start_idx = (i - 1)*repetitions + 1;
    end_idx = i*repetitions;
    G_all(:,:,start_idx:end_idx) = repmat(H_IT(:,:,i).', [1, 1, repetitions]);
end
G_all(:,:,end) = G_all(:,:,end - 1);

% Generate all received pilots and cascaded channels (2*N, U*len_pilot, K, num_samples) (N*U, MG^2*Gr, K, n_samples)
[Y_k_all_real, Q_k_all] = generate_received_pilots(G_all,H_RI,Gr,MG,len_pilot,Pu,SNR);

% Get the number of samples 
num_samples = size(Y_k_all_real, 4);

% Generate one random permutation of [1:num_samples]
perm_idx = randperm(num_samples);

% Apply the same permutation along the 4th dimension
Y_k_all_real = Y_k_all_real(:,:,:,perm_idx);
Q_k_all      = Q_k_all(:,:,:,perm_idx);

% save('pilots_ris_16_len_8_30dBm','Y_k_all_real')
% save('Q_ris_16_len_8_30dBm','Q_k_all')

% figure;
% time = 1:1:12001;
% Q_k_all_reshaped = reshape(Q_k_all(:,:,1,:), 4*64, 12001);
% norm_vector = vecnorm(Q_k_all_reshaped, 2, 1)';
% plot(time,norm_vector);hold on
% 
% Q_k_all_reshaped = reshape(Q_k_all(:,:,2,:), 4*64, 12001);
% norm_vector = vecnorm(Q_k_all_reshaped, 2, 1)';
% plot(time,norm_vector);hold on
% 
% Q_k_all_reshaped = reshape(Q_k_all(:,:,3,:), 4*64, 12001);
% norm_vector = vecnorm(Q_k_all_reshaped, 2, 1)';
% plot(time,norm_vector);hold on
% 
% Q_k_all_reshaped = reshape(Q_k_all(:,:,4,:), 4*64, 12001);
% norm_vector = vecnorm(Q_k_all_reshaped, 2, 1)';
% plot(time,norm_vector);hold off
% legend('UE1','UE2','UE3','UE4')
