function [H, At, Ar, DOA, AOA, BETA, delay] = direct_channel(Nt,Nr,L,fs,fc,M,Num_users)
delay = [1.78e-08,9.5529e-09,7.35536e-09,4.7321e-09,3.1329004e-09,8.540e-09,3.9129e-10,1.13972e-08,8.14984e-09,7.6662931e-10];

lambda_c = 3e8/fc; % wavelength
dt = lambda_c/2;
dr = dt;
H = zeros(Nr,Nt,Num_users,M);
lambda = zeros(M);
DOA = zeros(Num_users, L);
AOA = zeros(Num_users, L);
BETA = zeros(Num_users, L);

for u=1:1:Num_users
    beta(1:L) = exp(1i*2*pi*rand(1,L));
    x = randperm(Nt);
    y = x(1:L);
    DoA_index = reshape(y,L,1);
    set_t = (-(Nt-1)/2:1:(Nt-1)/2)/(Nt/2);
    DoA = ((2/Nt)*rand(1,L) - 1/Nt) + set_t(DoA_index);
    AoA = 2*rand(1,L) - 1;
    for m = 1 :  M
        f(m) = fc + fs/M*(m-1-(M-1)/2);
        lambda(m) = 3e8/f(m);
        for l = 1 : L
            At(:,l,u) = Converter(DoA(l),Nt,dt,lambda(1));
            Ar(:,l,u) = Converter(AoA(l),Nr,dr,lambda(1));
            H(:,:,u,m) = H(:,:,u,m) + beta(l)*exp(-1i*2*pi*f(m)*delay(l))* Ar(:,l,u)* At(:,l,u)';
        end
        H(:,:,u,m) = sqrt(Nt*Nr)*H(:,:,u,m);      
    end
    DOA(u,:) = DoA;
    AOA(u,:) = AoA;
    BETA(u,:) = beta;
end