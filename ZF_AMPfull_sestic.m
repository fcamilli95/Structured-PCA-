clear;
close all;
clc;

%This script implements the AMP designed for rotationally invariant noises
%in https://arxiv.org/pdf/2008.11892.pdf. The algorithm requires to import
%the free cumulants. It is very close to "ZF_AMP_single_sestic.m", so we
%strongly suggest to have a look at that first. The only difference is that
%here we employ a "full memory denoiser", that depends on all the past
%iterates. the algorithm thus differs from the single step memory only in one
%point that is later highlighted.


tic

% alphagrid = 0 : 0.05 : 1;
alphagrid = 2.9;

epsl = sqrt(0.9); % correlation of initialization

a2=2./3.;
xi=27./80.;

tol = 1e-8;

n = 8000;

ntrials = 50;
niter = 10;

max_it = niter;

load freecum_sestic.mat;

freecum = kdouble'; % free cumulants (starting from the 1st)
    
scal_all = zeros(niter, length(alphagrid), ntrials);
scal_allend = zeros(length(alphagrid), ntrials);
MSE = zeros(niter, ntrials, length(alphagrid));

for j = 1 : length(alphagrid)
    
    alpha = alphagrid(j);
    SigmaMATall = zeros(niter, niter, ntrials);
    DeltaMATall = zeros(niter, niter, ntrials);
    muSEexpall = zeros(niter, ntrials);
    muSEall = zeros(niter, ntrials);
    sigmaexpall = zeros(niter, ntrials);
  
    for i = 1 : ntrials
        
        fprintf('\nalpha=%f, trial #%d\n', alpha, i);

        x = sign(rand(n, 1)-0.5);
        normx = sqrt(sum(x.^2));
        x = sqrt(n)*x/normx;
        
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
        
        Lambda = diag(eigdiag);
        G = randn(n,n);
        A = (G + G') / sqrt(2*n);
        
        [W, D] = eig(A);

        Y = alpha/n * (x * x') + W * Lambda * W';        
                
        % initialization 
        v0 = epsl * x + sqrt(1-epsl^2) * randn(n, 1);
        normv0 = sqrt(sum(v0.^2));
        u_init = sqrt(n) * v0/normv0;
        
        uAMP = zeros(n, niter+1);
        fAMP = zeros(n, niter);
        muSE = zeros(niter, 1);
        muSEexp = zeros(niter, 1);
        sigmaSE = zeros(niter, 1);
        sigmaexp = zeros(niter, 1);
        SigmaMAT = zeros(niter, niter);
        DeltaMAT = zeros(niter, niter);
        Phi = zeros(niter+1, niter+1);
        scal = zeros(niter, 1);
        
        muSE(1) = alpha * epsl;
        sigmaSE(1) = freecum(2);
        SigmaMAT(1, 1) = sigmaSE(1);
        
        uAMP(:, 1) = u_init;
        scal(1) = (sum(uAMP(:, 1).* x))^2/sum(x.^2)/sum(uAMP(:, 1).^2);
        MSE(1, i, j) = 1/n^2 * ( sum(x.^2)^2 + sum(uAMP(:, 1).^2)^2 - 2 * (sum(uAMP(:, 1).* x))^2 )/2;
        fprintf('Iteration %d, scal=%f\n', 1, scal(1));
        
        DeltaMAT(1, 1) = (uAMP(:, 1)' * uAMP(:, 1))/n;
        
        b11 = freecum(1);
        
        fAMP(:, 1) = Y * uAMP(:, 1) - b11 * uAMP(:, 1);
        muSEexp(1) = sum(fAMP(:, 1).* x)/n; 
        sigmaexp(1) = sum((fAMP(:, 1) - sum(fAMP(:, 1).* x)/n * x).^2)/n; 
        uAMP(:, 2) = tanh(muSE(1)/sigmaSE(1)*fAMP(:, 1));
        MSE(2, i, j) = 1/n^2 * ( sum(x.^2)^2 + sum(uAMP(:, 1).^2)^2 - 2 * (sum(uAMP(:, 1).* x))^2 )/2;
        scal(2) = (sum(uAMP(:, 2).* x))^2/sum(x.^2)/sum(uAMP(:, 2).^2);
        
        fprintf('Iteration %d, scal=%f\n', 2, scal(2));
        
        Phi(2, 1) = muSE(1)/sigmaSE(1) * ( 1 - mean((tanh(muSE(1)/sigmaSE(1)*fAMP(:, 1))).^2));
        DeltaMAT(1, 2) = (uAMP(:, 2)' * uAMP(:, 1))/n;
        DeltaMAT(2, 2) = (uAMP(:, 2)' * uAMP(:, 2))/n;
        DeltaMAT(2, 1) = DeltaMAT(1, 2);


        for jj = 2 : niter-1
            
           Phired = Phi(1:jj, 1:jj); 
           Deltared = DeltaMAT(1:jj, 1:jj);
           
           B = zeros(jj, jj);
           
           for ii = 0 : jj-1
               B = B + freecum(ii+1) * Phired^ii;
           end
           
           b = B(jj, 1:jj);
           
           fAMP(:, jj) = Y * uAMP(:, jj) - sum(repmat(b, n, 1) .* uAMP(:, 1:jj), 2);
           muSEexp(jj) = sum(fAMP(:, jj) .* x)/n;
           sigmaexp(jj) = sum(fAMP(:, jj) - sum(fAMP(:, jj).* x)/n * x).^2/n; 

           
           Sigmared = zeros(jj, jj);
           
           for i1 = 0 : 2*(jj-1)
               ThetaMAT = zeros(jj, jj);
               
               for i2 = 0 : i1
                   ThetaMAT = ThetaMAT + Phired^i2 * Deltared * (Phired')^(i1-i2);
               end
               
               Sigmared = Sigmared + freecum(i1+2) * ThetaMAT;
           end
           
           muSE(jj) = sqrt(abs(sum(fAMP(:, jj).^2)/n - Sigmared(jj, jj)));
           uAMP(:, jj+1) = tanh(muSE(1:jj)'/Sigmared*(fAMP(:, 1:jj)'))'; %HERE WE USE ALL THE PAST ITERATES,
           %that is where the past memory comes into play

           scal(jj+1) = (sum(uAMP(:, jj+1).* x))^2/sum(x.^2)/sum(uAMP(:, jj+1).^2);
           MSE(jj+1, i, j) = 1/n^2 * ( sum(x.^2)^2 + sum(uAMP(:, jj+1).^2)^2 - 2 * (sum(uAMP(:, jj+1).* x))^2 )/2;


           for i1 = 1 : jj+1
               DeltaMAT(i1, jj+1) = (uAMP(:, i1)' * uAMP(:, jj+1))/n;
               DeltaMAT(jj+1, i1) = DeltaMAT(i1, jj+1);
           end
           Phi(jj+1, 1:jj) = ( 1 - mean((tanh(muSE(1:jj)' / Sigmared * fAMP(:, 1:jj)')).^2)) * muSE(1:jj)' / Sigmared;
           %HERE WE USE ALL THE PAST ITERATES, that is where the past memory comes into play
           
           fprintf('Iteration %d, scal=%f, MSE=%g\n', jj+1, scal(jj+1), MSE(jj+1, i, j));
           
        end
        
        muSEexpall(:, i)=muSEexp;
        muSEall(:, i)=muSE;
        sigmaexpall(:, i)=sigmaexp;
        
        SigmaMATall(1:niter-1, 1:niter-1, i) = Sigmared;
        DeltaMATall(:, :, i) = DeltaMAT;
        
        scal_all(:, j, i) = scal;
        scal_allend(j, i) = scal(end);
        
    end
end

save ZFAMPfull2_9.mat alphagrid scal_all MSE;

toc