clear all;
close all;
rng(2002, "combRecursive");
% Used the paper for the following parameters with same variable names.
M = 64; % number of Tx antennas.
L = 10; % number of LIS antennas. L=100;
K = 8; % Number of users
Ns = 1; % number of data streams.
%% opts is used to specify set of options that control the neural Net
opts.Num_paths = 4;
opts.fc = 60*10^9; %carrier frequency
opts.BW = 4*10^9; %bandwidth
opts.fs = opts.BW;  % sampling frequency  fs=2M(BW)
opts.snr_param = [30]; % SNR dB.
opts.Nreal = 30; % number of realizations.
opts.Nchannels = 1; % number of channel matrices for the input data, make it 100 for better sampling the data space.
opts.fixedUsers = 0;
opts.fixedChannelGain = 0;
opts.noiseLevelHdB_CE = [ 10 15 20 25]; % dB.
opts.inputH = 1;
opts.inputRy = 0;
timeGenerate = tic;

Num_paths = opts.Num_paths;
Nch = opts.Nchannels;
Nreal = opts.Nreal;
N = Nreal*Nch*K;
Z = repmat(struct('channel_dc',zeros(M,1,K)),1,N );
snr = db2pow(opts.snr_param);

F = dftmtx(M);

jjHB = 1;
jjCE = 1;
jjA2 = 1;
X = eye(M); % pilot data.
X2 = eye(M*L);
V = eye(L); % reflect beamforming data.
doaMismatchList = linspace(0,10,Nch);
for nch = 1:Nch
    %% Generate channels
        % H, G.
        [H, At, Ar, DoA, AoA, beta, ~] = direct_channel(L,M,Num_paths,opts.fs,opts.fc,1,1);
        paramH(nch,1).At = At;
        paramH(nch,1).Ar = Ar;
        paramH(nch,1).DoA = DoA;
        paramH(nch,1).AoA = AoA;
        paramH(nch,1).beta = beta;
        [h_lis, At, Ar, DoA, AoA, beta, ~] = cascaded_channel(1,L,Num_paths,opts.fs,opts.fc,1,K);
        paramH_LIS(nch,1).At = At;
        paramH_LIS(nch,1).Ar = Ar;
        paramH_LIS(nch,1).DoA = DoA;
        paramH_LIS(nch,1).AoA = AoA;
        paramH_LIS(nch,1).beta = beta;
        % H DC.
        [h_dc, At, Ar, DoA, AoA, beta, delay] = cascaded_channel(1,M,Num_paths,opts.fs,opts.fc,1,K); 
        paramH_DC(nch,1).At = At;
        paramH_DC(nch,1).Ar = Ar;
        paramH_DC(nch,1).DoA = DoA;
        paramH_DC(nch,1).AoA = AoA;
        paramH_DC(nch,1).beta = beta;
    G = zeros(M,L,K);
    for kk = 1:K
        G(:,:,kk) = H* diag(h_lis(:,1,kk));
    end
    %% Channel estimation
    timeGenerate = tic;
    for nr = 1:Nreal
        snrIndex_CE = ceil(nr/(Nreal/size(opts.noiseLevelHdB_CE,2)));
        snrChannel = opts.noiseLevelHdB_CE(snrIndex_CE);
        S = 1/sqrt(2)*(randn(K,M) + 1i*randn(K,M));
        for kk = 1:K % number of users.
            y_dc(kk,:) = awgn( h_dc(:,1,kk)'*X, snrChannel,'measured'  ); % direct channel data.
            h_dc_e(:,kk) = (y_dc(kk,:)*pinv(X))'; % direct channel LS.
            
            vG = []; h_dc_kron = [];
            for p = 1:L % for each LIS components. estimate cascaded channel
                v = V(:,p);
                vG = [vG v'*G(:,:,kk)'];
                h_dc_kron = [h_dc_kron h_dc(:,1,kk)'];
            end
            
            y_cc(:,:,kk) = reshape(awgn( (h_dc_kron + vG )*X2  ,snrChannel,'measured'),[M,L]);

            
            %% test.
            R_dc(:,:,nr,kk) = reshape(y_dc(kk,:),[sqrt(M) sqrt(M)]); % (direct channel) input to neuralNet training
            R_cc(:,:,nr,kk) = y_cc(:,:,kk); % (cascaded channel) input to neuralNet training
        end
    end
    %% Channel Estimation. Training data for A3
    for kk = 1:K % input-output pair of the DL model.
        for nr = 1:Nreal
            NN{1,1}.X_dc(:,:,1,jjCE) = real(R_dc(:,:,nr,kk)); % input.
            NN{1,1}.X_dc(:,:,2,jjCE) = imag(R_dc(:,:,nr,kk)); % input.
            NN{1,1}.X_cc(:,:,1,jjCE) = real(R_cc(:,:,nr,kk)); 
            NN{1,1}.X_cc(:,:,2,jjCE) = imag(R_cc(:,:,nr,kk));
         
            channel_dc = h_dc(:,1,kk); % output. direct channel
            channel_cc = G(:,:,kk);% output. cascaded channel
            NN{1,1}.Y_dc(jjCE,:) = [real(channel_dc(:)); imag(channel_dc(:))]; % output
            NN{1,1}.Y_cc(jjCE,:) = [real(channel_cc(:)); imag(channel_cc(:))]; % output.
            
            Z(1,jjCE).h_dc = h_dc;
            Z(1,jjCE).G = G;
            jjCE = jjCE + 1;
            
            keepIndex(jjCE) = [nch]; 
        end
    end
    nch;
end 
timeGenerate = toc(timeGenerate);

%% Network..
for kkt = 1
    fprintf(2,['Train MLP \n'])
    
    fprintf(2,['Train SFCNN \n'])    
    fprintf(2,['Train NeuralNet{' num2str(kkt) '} \n'])
    [NN{1,kkt}.net_dc] = Train(NN{1,kkt}.X_dc,NN{1,kkt}.Y_dc,0.00021);
    fprintf(2,['Train ChannelNet_CC{' num2str(kkt) '} \n'])
    [NN{1,kkt}.net_cc] = Train(NN{1,kkt}.X_cc,NN{1,kkt}.Y_cc,0.00000211);
    
end


