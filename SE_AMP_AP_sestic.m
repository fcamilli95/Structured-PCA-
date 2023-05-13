% Rademacher prior

%This script contains implements state evolution of the AMP-AP, Approximate
%Message Passing with Alternating Posterior, that is supposed to match the
%replica predicition. 

clear;
close all;
clc;

tic


alphagrid = 2 : 0.03 : 3.02;
%alphagrid = 2;
%alphagrid = 1.82 : 0.06: 5;

epsl = sqrt(0.9); % correlation of initialization

niter = 55;
Nmc = 10^7; % number of MonteCarlo trials to compute integrals

atolint = 1e-12;
rtolint = 1e-12;

tolinc = 0.00005;
tolsmall = 10^(-3);

scal_all = zeros(niter, length(alphagrid));
MSE = zeros(niter, length(alphagrid));
flag_all = zeros(niter, length(alphagrid));


max_it = niter;

load freecum_sestic.mat;

freecum = kdouble'; % free cumulants (starting from the 1st)
    
for j = 1 : length(alphagrid)
    
    alpha = alphagrid(j);     
    fprintf('alpha=%f\n', alpha);
            
    muSE = zeros(niter, 1);
    sigmaSE = zeros(niter, niter);
    DeltaMAT = zeros(niter, niter);
    Phi = zeros(niter+1, niter+1);
    scal = zeros(niter, 1);
        
    muSE(1) = alpha * epsl;
    sigmaSE(1, 1) = freecum(2);
    scal(1) = epsl;
    DeltaMAT(1, 1) = 1;
    MSE(1, j) = 1-epsl^2;
    
    fprintf('Iteration %d, scal=%f\n', 1, scal(1)^2);
    
    
        
    for jj = 2 : niter
        
        Sigmared = sigmaSE(1:jj-1, 1:jj-1);
        
        if det(Sigmared) < 0
            for kk = jj : niter
                scal(kk) = scal(jj-1);
                MSE(kk, j) = MSE(jj-1, j);
            end
            
            break;
        end
        
        Wvec = (mvnrnd(zeros(1, jj-1),Sigmared,Nmc))';
        vpd = muSE(1:jj-1)' / Sigmared;
    
            
        if mod(jj-1, 5) ~= 0 
        
            Phi(jj, jj-1) = 1;
            muSE(jj) = alpha * muSE(jj-1);
            DeltaMAT(1, jj) = muSE(jj)/alpha * epsl;
            DeltaMAT(jj, 1) = DeltaMAT(1, jj);
        
            for i1 = 1 : jj-1
        
                muR = muSE(1:i1);
                vpdR = muR' / Sigmared(1:i1, 1:i1);
                WvecR = Wvec(1:i1, :);
                
                if mod(i1, 5) ~= 0 
        
                    DeltaMAT(jj, i1+1) = muSE(jj-1) * muSE(i1) + Sigmared(jj-1, i1);
                    DeltaMAT(i1+1, jj) = DeltaMAT(jj, i1+1);
                else
                    
                    DeltaMAT(jj, i1+1) = 1/2*mean((muSE(jj-1)*ones(1, Nmc) + Wvec(end, :)) .* tanh((vpdR * muR)*ones(1, Nmc) + vpdR *  WvecR) ...
                        + (-muSE(jj-1)*ones(1, Nmc) + Wvec(end, :)) .* tanh(-(vpdR * muR)*ones(1, Nmc) + vpdR *  WvecR));
                    DeltaMAT(i1+1, jj) = DeltaMAT(jj, i1+1);
                end
            
            end
            
        else
            
            
            intval = 1-1/2*mean(tanh((vpd * muSE(1:jj-1))*ones(1, Nmc) + vpd * Wvec).^2 ...
                + tanh(-(vpd * muSE(1:jj-1))*ones(1, Nmc) + vpd *  Wvec).^2);
    
            Phi(jj, 1:jj-1) = vpd * intval;    
        
            muSE(jj) = alpha * 1/2*mean(tanh((vpd * muSE(1:jj-1))*ones(1, Nmc) + vpd *  Wvec) ...
                - tanh(-(vpd * muSE(1:jj-1))*ones(1, Nmc) + vpd *  Wvec));
            DeltaMAT(1, jj) = muSE(jj)/alpha * epsl;
            DeltaMAT(jj, 1) = DeltaMAT(1, jj);
        
            for i1 = 1 : jj-1
                muR = muSE(1:i1);
                vpdR = muR' / Sigmared(1:i1, 1:i1);
                WvecR = Wvec(1:i1, :);
                
                %alternating posteriors
                if mod(i1, 5) ~= 0
                    
                    DeltaMAT(jj, i1+1) = 1/2*mean(tanh((vpd * muSE(1:jj-1))*ones(1, Nmc) + vpd *  Wvec) .* (muR(end)*ones(1, Nmc) + WvecR(end, :)) ...
                        + tanh(-(vpd * muSE(1:jj-1))*ones(1, Nmc) + vpd *  Wvec) .* (-muR(end)*ones(1, Nmc) + WvecR(end, :)));
                    
                else
                    
                    DeltaMAT(jj, i1+1) = 1/2*mean(tanh((vpd * muSE(1:jj-1))*ones(1, Nmc) + vpd *  Wvec) .* tanh((vpdR * muR)*ones(1, Nmc) + vpdR *  WvecR) ...
                        + tanh(-(vpd * muSE(1:jj-1))*ones(1, Nmc) + vpd *  Wvec) .* tanh(-(vpdR * muR)*ones(1, Nmc) + vpdR *  WvecR));
                    
                end
                
                
                DeltaMAT(i1+1, jj) = DeltaMAT(jj, i1+1);
            end
        end
        
        Sigmared = zeros(jj, jj);
        Phired = Phi(1:jj, 1:jj);
        Deltared = DeltaMAT(1:jj, 1:jj);

        for i1 = 0 : 2*(jj-1)
            ThetaMAT = zeros(jj, jj);

            for i2 = 0 : i1
                ThetaMAT = ThetaMAT + Phired^i2 * Deltared * (Phired')^(i1-i2);
            end

            Sigmared = Sigmared + freecum(i1+2) * ThetaMAT;
        end

        sigmaSE(1:jj, 1:jj) = Sigmared;        
                
        scal(jj) = muSE(jj)/alpha/sqrt(DeltaMAT(jj, jj));
        
        MSE(jj, j) = (1 - 2 * (muSE(jj)/alpha)^2 + DeltaMAT(jj, jj)^2)/2;

        
        if mod(jj-1, 5) == 0 
            
            fprintf('Iteration %d, scal=%f, MSE=%f\n', jj, scal(jj)^2, MSE(jj, j));
        
        end

    end
    
    scal_all(:, j) = scal;
    

    
end

save SE_ZFmod_sestic_Radprior_epssqrt0dot9.mat scal_all alphagrid MSE;


%save SEspect_MP_c2_approx_aem12_rem12.mat scal_all alphagrid flag_all;

%save SEspect_MP_c2_alpha3.mat scal_all alphagrid;


toc