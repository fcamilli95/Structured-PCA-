clear;
close all;
clc;

%In this script we implement the Bayes-optimal AMP (BAMP) for the power six
%potential. The algorithm is very similar to the standard AMP with
%rotationally invariant noise, with the observations matrix replaced by its
%pre-processed version. In order to simplify and speed up things a little
%bit we import the Onsagers and other parameters directly from the state
%evolution. Hence this code must be executed only after the state evolution
%has been run (see the file loaded). We suggest reading
%ZF_AMP_single_sestic.m first.

tic

rng(78232);

epsl = sqrt(0.9); % correlation of initialization
%n = 8192; %for Hadamard
%n = 1024; %for CIFAR
n=8000; %for gene



ntrials = 50;
niter = 8;



load SE_sestic_BAMP2_9.mat;

%Here we carry out a universality experiment with one of the two GTEx
%datasets, using the eigenvectors of its covariance instead of Haar
%matrices.

load gene_skin_cov.mat;


%load cat.mat;
%cov_CIFAR = data;

%[V_CIFAR, D_CIFAR] = eig(cov_CIFAR);

[V_gene,D_gene]=eig(cov_gene);



max_it = niter; %maximum number of iterations 

%parameters of the sepctral density fo the power six potential 
xi=27/80;
a2=2/3;

tol = 1e-8;

scal_all = zeros(niter, ntrials, length(alphagrid));
overlap = zeros(niter, ntrials, length(alphagrid));
MSE = zeros(niter, ntrials, length(alphagrid));

for j = 1 : length(alphagrid)
    alpha = alphagrid(j);
    SigmaMATall = zeros(niter, niter, ntrials);
    DeltaMATall = zeros(5*niter, 5*niter, ntrials);
    muSEexpall = zeros(niter, ntrials);
    muSEall = zeros(niter, ntrials);
    sigmaexpall = zeros(niter, ntrials);
    tildemuexpall = zeros(5*niter, ntrials);
  
    for i = 1 : ntrials
        
        fprintf('\nalpha=%f, trial #%d\n', alpha, i);
        %coefficients of the pre-processing polynomial
        c1 = 0;
        c2 = -xi * alpha^2;
        c3 = 0;
        c4 = -xi * alpha^2;
        c5 = xi * alpha;
        
        %generating the signal
        x = sign(rand(n, 1)-0.5);
        normx = sqrt(sum(x.^2));
        x = sqrt(n)*x/normx;
        
        %sampling noise eigenvalues
        eigdiag = zeros(1, n);

        for j1 = 1 : n
                        
            p = rand(1, 1);
            
            eigmin = -2*sqrt(a2)+10^(-6);
            eigmax = 2*sqrt(a2)-10^(-6);
            
            rhofun = @(y) xi.*(6*a2^2+2*a2*y.^2+y.^4).*sqrt(4*a2-y.^2)/(2*pi);
            intval = integral(rhofun,-2*sqrt(a2),eigmin);
            
            if intval > p
                fprintf('Something wrong with eigmin\n');
                return;
            end

            rhofun = @(y) xi.*(6*a2^2+2*a2*y.^2+y.^4).*sqrt(4*a2-y.^2)/(2*pi);
            intval = integral(rhofun,-2*sqrt(a2),eigmax);
            
            if intval < p
                fprintf('Something wrong with eigmax\n');
                return;
            end
            
            while eigmax-eigmin > tol
                eigval = (eigmax+eigmin)/2;
                rhofun = @(y) xi.*(6*a2^2+2*a2*y.^2+y.^4).*sqrt(4*a2-y.^2)/(2*pi);
                intval = integral(rhofun,-2*sqrt(a2),eigval);
            
                if intval < p
                    eigmin = eigval;
                else
                    eigmax = eigval;
                end
            end
            
            eigdiag(j1) = (eigmax+eigmin)/2;
            
        end
        
        Lambda = diag(eigdiag);%diagonal noise matrix

        %generating Haar matrices via diagonlization of Wigner matrix, this
        %corresponds to the ordinary theory in the paper.
        %G = randn(n,n);
        %A = (G + G') / sqrt(2*n);
        %[W, D] = eig(A);

        %W = diag(sign(rand(1, n)-0.5)) * hadamard(n)/sqrt(n); %For
        %W = V_CIFAR;
        W = V_gene;
        %universality

        Y = alpha/n * (x * x') + W * Lambda * W';  %observations      
                
        % initialization 
        v0 = epsl * x + sqrt(1-epsl^2) * randn(n, 1);
        normv0 = sqrt(sum(v0.^2));
        u_init = sqrt(n) * v0/normv0;
        
        uAMP = zeros(n, niter+1);
        fAMP = zeros(n, niter);
        tildemuexp = zeros(5*niter, 1);
        tildesigmaexp = zeros(5*niter, 1);
        DeltaMAT = zeros(5*niter, 5*niter);
        Phi = zeros(niter+1, niter+1);
        scal = zeros(niter, 1);
        uAMP(:, 1) = u_init;
        scal_all(1, i, j) = (sum(uAMP(:, 1).* x))^2/sum(x.^2)/sum(uAMP(:, 1).^2);
        
        
        MSE(1, i, j) = 1/n^2 * ( sum(x.^2)^2 + sum(uAMP(:, 1).^2)^2 - 2 * (sum(uAMP(:, 1).* x))^2 )/2;
        fprintf('Iteration %d, MSE=%f, overlap=%f\n', 1, MSE(1, i, j), scal_all(1, i, j));

        for j1 = 1 : niter-1 %iterations
            fAMP(:, j1) = (c2.*Y^2+c4.*Y^4+c5.*Y^5) * uAMP(:, j1) - uAMP(:, 1:j1) * Onsager_correct(j1, 1:j1)';%local fields
            uAMP(:, j1+1) = tanh(muSE(j1)/sigma2SE(j1)*fAMP(:, j1)); %AMP iterate

            scal_all(j1+1, i, j) = (sum(uAMP(:, j1+1).* x))^2/sum(x.^2)/sum(uAMP(:, j1+1).^2);   %rescaled overlap
            MSE(j1+1, i, j) = 1/n^2 * ( sum(x.^2)^2 + sum(uAMP(:, j1+1).^2)^2 - 2 * (sum(uAMP(:, j1+1).* x))^2 )/2; %MSE
            fprintf('Iteration %d, MSE=%f, overlap=%f\n', j1+1, MSE(j1+1, i, j), scal_all(j1+1, i, j));
        end    
    end
end

save AMP_sestic_skin2_9.mat alphagrid scal_all MSE;

toc