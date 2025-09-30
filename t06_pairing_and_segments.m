clear

n_samples_G = 1;
n_samples_H = 1;
n_users = 4;
H_IT = zeros(16,4,n_samples_G); % (Rx_elements,Tx_elements,n_samples)
H_RI = zeros(1,16,12001,n_users); % (Rx_elements,Tx_elements,n_samples,n_users)

s = qd_simulation_parameters;                           % Set up simulation parameters
s.show_progress_bars = 0;                               % Disable progress bars
s.use_absolute_delays = 1;                              % Include delay of the LOS path
s.center_frequency = 2.8e10;
s.samples_per_meter = 600;

%% BS-RIS channel
 
a_BS = qd_arrayant('3gpp-3d',4,1,s.center_frequency);     % ULA with vertical polarization
a_RIS = qd_arrayant('3gpp-3d',4,4,s.center_frequency);

l = qd_layout(s);                                       % Create new QuaDRiGa layout

l.no_tx = 1;                                            
l.tx_position = [ -50 ; 50 ; 25 ];                    % Outdoor BS
l.tx_array = a_BS;
% l.tx_position(:,2) = [ 30 ; 0; 10 ];                     % Indoor BS

l.no_rx = 1;                                            % Two MTs
l.rx_array = a_RIS;
% l.rx_track(1,1) = qd_track('linear', 0.5 );             % Linear track with 20 cm length
l.rx_track(1,1).name = 'Rx1';                           % Set the MT1 name
l.rx_track(1,1).scenario = {'3GPP_38.901_UMa_LOS'};  % Two Scenarios

l.rx_position = [ 10 ; 5 ; 2 ];                      % Start position of the MT2 track
% interpolate_positions( l.rx_track, s.samples_per_meter );  % Interpolate positions

% l.rx_track(1,2).segment_index = [1 3000];                  % Set segments
% l.rx_track(1,2).scenario = {'WINNER_UMa_C2_LOS'};
% l.visualize;

for i_sample = 1 : n_samples_G
    cb = l.init_builder;                                    % Initialize builder
    gen_parameters( cb );                                   % Generate small-scale-fading
    c = get_channels( cb );                                 % Get channel coefficients
    cn = merge( c );

    % freq = 0;
    % pilot_grid = (0:100 - 1)/100;
    % for i = 1:34
    %     freq = freq +
    %     cn.coeff(1,1,i)*exp(-1j*2*pi*cn.delay(1,1,i)*pilot_grid*2e9);
    % end
    % H_IT_100 = cn.fr(2e9,100);

    % Combine the channel coefficients
    H_IT(:,:,i_sample) = cn.fr(2e9,1);
end

%% RIS-User channels

a_UE = qd_arrayant('patch'); 

ll = qd_layout(s);                                       % Create new QuaDRiGa layout
ll.no_tx = 1; 
ll.tx_position = [ 10 ; 5 ; 2 ];                    % RIS 
ll.tx_array = a_RIS;

ll.no_rx = 4;                                            
ll.rx_array = a_UE;

ll.rx_track(1,1) = qd_track('linear',20,0);              % Linear track, 20 m length
ll.rx_track(1,1).initial_position  = [0 ; 0 ; 1.5];            % Start point
ll.rx_track(1,1).scenario          = {'QuaDRiGa_Industrial_LOS'};  % Scenarios
ll.rx_track(1,1).name = 'UE1';                         % Set the MT2 name
interpolate( ll.rx_track(1,1), 'distance', 1/s.samples_per_meter, [],[],1  ); % Set sampling intervals
ll.rx_track(1,1).segment_index     = [1 3001 6001 9001];             % Segments

ll.rx_track(1,2) = qd_track('linear',20,0);              % Linear track, 20 m length
ll.rx_track(1,2).initial_position  = [0 ; 0.5 ; 1.5];            % Start point
ll.rx_track(1,2).scenario          = {'QuaDRiGa_Industrial_LOS'};  % Scenarios
ll.rx_track(1,2).name = 'UE2';                         % Set the MT2 name
interpolate( ll.rx_track(1,2), 'distance', 1/s.samples_per_meter, [],[],1  ); % Set sampling intervals
ll.rx_track(1,2).segment_index     = [1 3001 6001 9001];             % Segments

ll.rx_track(1,3) = qd_track('linear',20,0);              % Linear track, 20 m length
ll.rx_track(1,3).initial_position  = [0 ; 1 ; 1.5];            % Start point
ll.rx_track(1,3).scenario          = {'QuaDRiGa_Industrial_LOS'};  % Scenarios
ll.rx_track(1,3).name = 'UE3';                         % Set the MT2 name
interpolate( ll.rx_track(1,3), 'distance', 1/s.samples_per_meter, [],[],1  ); % Set sampling intervals
ll.rx_track(1,3).segment_index     = [1 3001 6001 9001]; 

ll.rx_track(1,4) = qd_track('linear',20,0);              % Linear track, 20 m length
ll.rx_track(1,4).initial_position  = [0 ; 1.5 ; 1.5];            % Start point
ll.rx_track(1,4).scenario          = {'QuaDRiGa_Industrial_LOS'};  % Scenarios
ll.rx_track(1,4).name = 'UE4';                         % Set the MT2 name
interpolate( ll.rx_track(1,4), 'distance', 1/s.samples_per_meter, [],[],1  ); % Set sampling intervals
ll.rx_track(1,4).segment_index     = [1 3001 6001 9001]; 

for i_sample = 1 : n_samples_H
    cbb = ll.init_builder;                                    % Initialize builder
    gen_parameters( cbb );                                   % Generate small-scale-fading
    c = get_channels( cbb );                                 % Get channel coefficients
    cn = merge( c );                                        % Combine the channel coefficients
end

for i_user = 1 : n_users
    H_RI(1,:,:,i_user) = squeeze(cn(1,i_user).fr(2e6,1)); % 1 * M * snapshots * n_users
end

% ll.visualize;

% save('H_RI_16_4',"H_RI")
% save('H_IT_16_4',"H_IT")
figure;
time = 1:1:12001;
H_RI_reshaped = reshape(H_RI(:,:,:,1), 16, 12001);
norm_vector = vecnorm(H_RI_reshaped, 2, 1)';
plot(time,norm_vector);hold on

H_RI_reshaped = reshape(H_RI(:,:,:,2), 16, 12001);
norm_vector = vecnorm(H_RI_reshaped, 2, 1)';
plot(time,norm_vector);hold on

H_RI_reshaped = reshape(H_RI(:,:,:,3), 16, 12001);
norm_vector = vecnorm(H_RI_reshaped, 2, 1)';
plot(time,norm_vector);hold on

H_RI_reshaped = reshape(H_RI(:,:,:,4), 16, 12001);
norm_vector = vecnorm(H_RI_reshaped, 2, 1)';
plot(time,norm_vector);hold off
legend('UE1','UE2','UE3','UE4')