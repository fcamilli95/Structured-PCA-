clear;
close all;
clc;

%In this script we implement the Bayes-optimal AMP (BAMP) for the quartic
%potential. The algorithm is very similar to the standard AMP with
%rotationally invariant noise, with the observations matrix replaced by its
%pre-processed version. In order to simplify and speed up things a little
%bit we import the Onsagers and other parameters directly from the state
%evolution. Hence this code must be executed only after the state evolution
%has been run (see the file loaded). We suggest reading
%ZF_AMP_single_quartic.m first.

tic

rng(14091);



% alphagrid = 0 : 0.05 : 1;
alphagrid = sqrt(4.1^2);

epsl = sqrt(0.5); % correlation of initialization

n = 8000;

ntrials = 50;
niter = 10;

load SE_u0_epssqrt0dot5_alpha4dot1_Ons.mat;

%for GTEx related universality experiments, for instance, de-comment the
%following two lines:
%load gene_skin_cov.mat; 
%V_gene,D_gene]=eig(cov_gene);

max_it = niter; %maximum number of iterations, it matches that of SE. 

%parameters of the quartic potential spectral density
u = 0;
gamma = (8-9*u+sqrt(64-144*u+108*u.^2-27*u.^3))/27;

%regularization to avoid dividing by 0
if u == 1
    a2 = 1;
else
    a2 = (sqrt(u.^2+12*gamma)-u)./(6*gamma);
end


tol = 1e-8;

scal_all = zeros(niter, ntrials, length(alphagrid));
overlap = zeros(niter, ntrials, length(alphagrid));
MSE = zeros(niter, ntrials, length(alphagrid));

for j = 1 : length(alphagrid)
    
    alpha = alphagrid(j);
    SigmaMATall = zeros(niter, niter, ntrials);
    DeltaMATall = zeros(3*niter, 3*niter, ntrials);
    muSEexpall = zeros(niter, ntrials);
    muSEall = zeros(niter, ntrials);
    sigmaexpall = zeros(niter, ntrials);
    tildemuexpall = zeros(3*niter, ntrials);
    
    for i = 1 : ntrials
        
        fprintf('alpha=%f, trial #%d\n', alpha, i);

        %coefficients of the preprocessing polynomial
        c1 = u * alpha;
        c2 = -gamma * alpha^2;
        c3 = gamma * alpha;
        
        %generating signal
        x = sign(rand(n, 1)-0.5);
        normx = sqrt(sum(x.^2));
        x = sqrt(n)*x/normx;
        
        %generating eigenvalues fo the noise
        eigdiag = zeros(1, n);
        
        for j1 = 1 : n
                        
            p = rand(1, 1);
            
            eigmin = -2*sqrt(a2)+10^(-6);
            eigmax = 2*sqrt(a2)-10^(-6);
            
            rhofun = @(y) (u+gamma*(2*a2+y.^2)).*sqrt(4*a2-y.^2)/(2*pi);
            intval = integral(rhofun,-2*sqrt(a2),eigmin);
            
            if intval > p
                fprintf('Something wrong with eigmin\n');
                return;
            end

            rhofun = @(y) (u+gamma*(2*a2+y.^2)).*sqrt(4*a2-y.^2)/(2*pi);
            intval = integral(rhofun,-2*sqrt(a2),eigmax);
            
            if intval < p
                fprintf('Something wrong with eigmax\n');
                return;
            end
            
            while eigmax-eigmin > tol
                eigval = (eigmax+eigmin)/2;
                rhofun = @(y) (u+gamma*(2*a2+y.^2)).*sqrt(4*a2-y.^2)/(2*pi);
                intval = integral(rhofun,-2*sqrt(a2),eigval);
            
                if intval < p
                    eigmin = eigval;
                else
                    eigmax = eigval;
                end
            end
            
            eigdiag(j1) = (eigmax+eigmin)/2;
            
        end
        
        Lambda = diag(eigdiag);%diagonal matrix of the noise
%         De-comment to have rotationally invariant noise
%         G = randn(n,n);
%         A = (G + G') / sqrt(2*n);
%         
%         [W, D] = eig(A);

%Hereby we work with Hadamard matrices instead of orthogonal Haar matrices,
%in virtue of universality this should still match the SE. If you want to
%work still with Haar matrices, just generate a Wigner matrix and take its
%eigenbasis, as done in BAMP_sestic.m. For other universality experiments
%we suggest to have a look at the code in BAMP_sestic.m .

        W = diag(sign(rand(1, n)-0.5)) * hadamard(n)/sqrt(n); 

        %W = V_gene; for GTEx database. Seel also BAMP_sestic.m for further
        %examples.


        Y = alpha/n * (x * x') + W * Lambda * W';

                
        % initialization 
        v0 = epsl * x + sqrt(1-epsl^2) * randn(n, 1);
        normv0 = sqrt(sum(v0.^2));
        u_init = sqrt(n) * v0/normv0;
        
        uAMP = zeros(n, niter+1);
        fAMP = zeros(n, niter);
        tildemuexp = zeros(3*niter, 1);
        tildesigmaexp = zeros(3*niter, 1);
        DeltaMAT = zeros(3*niter, 3*niter);
        Phi = zeros(niter+1, niter+1);
        scal = zeros(niter, 1);
        uAMP(:, 1) = u_init;
        scal_all(1, i, j) = (sum(uAMP(:, 1).* x))^2/sum(x.^2)/sum(uAMP(:, 1).^2);
        
        MSE(1, i, j) = 1/n^2 * ( sum(x.^2)^2 + sum(uAMP(:, 1).^2)^2 - 2 * (sum(uAMP(:, 1).* x))^2 )/2;
        fprintf('Iteration %d, MSE=%f, overlap=%f\n', 1, MSE(1, i, j), scal_all(1, i, j));

        for j1 = 1 : niter-1 %iterations
            fAMP(:, j1) = (c1*Y+c2*Y^2+c3*Y^3) * uAMP(:, j1) - uAMP(:, 1:j1) * Onsager_correct(j1, 1:j1)'; %local field
            uAMP(:, j1+1) = tanh(muSE(j1)/sigma2SE(j1)*fAMP(:, j1)); %AMP iterate
            
            scal_all(j1+1, i, j) = (sum(uAMP(:, j1+1).* x))^2/sum(x.^2)/sum(uAMP(:, j1+1).^2); %rescaled overlap
            MSE(j1+1, i, j) = 1/n^2 * ( sum(x.^2)^2 + sum(uAMP(:, j1+1).^2)^2 - 2 * (sum(uAMP(:, j1+1).* x))^2 )/2; %MSE
            
            fprintf('Iteration %d, MSE=%f, overlap=%f\n', j1+1, MSE(j1+1, i, j), scal_all(j1+1, i, j));
        end   
    end
end

save AMP_had_n8192_eps0dot5_u0_alpha4dot1_more alphagrid scal_all MSE;

toc