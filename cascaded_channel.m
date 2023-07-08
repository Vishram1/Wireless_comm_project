function [H, At, Ar, DOA, AOA, BETA, delay] = cascaded_channel(Nt,Nr,L,fs,fc,M,Num_users)
delay = [1.780e-08,9.59e-09,7.336e-09,4.7302621e-09,3.1204e-09,8.529540e-09,3.9759e-10,1.1872e-08,8.1484e-09,7.6631e-10];

lambda_c = 3e8/fc; % wavelength
dt = lambda_c/2;
dr = dt;
lambda = zeros(M);
H = zeros(Nr,Nt,Num_users,M);
DOA = zeros(Num_users, M);
AOA = zeros(Num_users, L);
BETA = zeros(Num_users, L);

for u=1:1:Num_users
    beta(1:L) = exp(1i*2*pi*rand(1,L));
    AoA = 2*rand(1,L) - 1;
    for m = 1 :  M
        f(m) = fc + fs/M*(m-1-(M-1)/2);
        lambda(m) = 3e8/f(m);
        for l = 1 : L
            At(:,l,u) = 0;
            Ar(:,l,u) = Converter(AoA(l),Nr,dr,lambda(1));
            H(:,:,u,m) = H(:,:,u,m) + beta(l)*exp(-1i*2*pi*f(m)*delay(l))* Ar(:,l,u);
        end
        H(:,:,u,m) = sqrt(Nt*Nr)*H(:,:,u,m);
        
    end
    
    DOA(u,:) = 0;
    AOA(u,:) = AoA;
    BETA(u,:) = beta;
end