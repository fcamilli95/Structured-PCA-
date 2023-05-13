% Sparse Rademacher prior
%This script contains implements state evolution of the AMP-AP, Approximate
%Message Passing with Alternating Posterior, that is supposed to match the
%replica predicition. 

clear;
close all;
clc;

tic


alphagrid = 2 : 0.06 : 4.04;
%alphagrid = 3.5;
%alphagrid = 1.82 : 0.06: 5;

epsl = sqrt(0.5); % correlation of initialization

niter = 30;
Nmc = 10^7; % number of MonteCarlo trials to compute integrals

atolint = 1e-12;
rtolint = 1e-12;

tolinc = 0.00005;
tolsmall = 10^(-3);

scal_all = zeros(niter, length(alphagrid));
MSE = zeros(niter, length(alphagrid));
flag_all = zeros(niter, length(alphagrid));

rho = 0.3;

max_it = niter;

load freecum_u0.mat;

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
    
        %alternating posteriors    
        if mod(jj-1, 3) ~= 0 
        
            Phi(jj, jj-1) = 1;
            muSE(jj) = alpha * muSE(jj-1);
            DeltaMAT(1, jj) = muSE(jj)/alpha * epsl;
            DeltaMAT(jj, 1) = DeltaMAT(1, jj);
        
            for i1 = 1 : jj-1
        
                muR = muSE(1:i1);
                vpdR = muR' / Sigmared(1:i1, 1:i1);
                WvecR = Wvec(1:i1, :);
                
                %alternating posteriors
                if mod(i1, 3) ~= 0 
        
                    DeltaMAT(jj, i1+1) = muSE(jj-1) * muSE(i1) + Sigmared(jj-1, i1);
                    DeltaMAT(i1+1, jj) = DeltaMAT(jj, i1+1);
                else
                    
                    DeltaMAT(jj, i1+1) = mean(rho/2*(muSE(jj-1)*ones(1, Nmc)/sqrt(rho) + Wvec(end, :)) .* (sqrt(rho)*tanh(((vpdR * muR)*ones(1, Nmc)/sqrt(rho) + vpdR *  WvecR)/sqrt(rho)) ./ ...
                        (rho+(1-rho).*sech(((vpdR * muR)*ones(1, Nmc)/sqrt(rho) + vpdR *  WvecR)/sqrt(rho)).*exp(vpdR*muR/2/rho))) ...
                        + rho/2*(-muSE(jj-1)*ones(1, Nmc)/sqrt(rho) + Wvec(end, :)) .* (sqrt(rho)*tanh((-(vpdR * muR)*ones(1, Nmc)/sqrt(rho) + vpdR *  WvecR)/sqrt(rho)) ./ ...
                        (rho+(1-rho).*sech((-(vpdR * muR)*ones(1, Nmc)/sqrt(rho) + vpdR *  WvecR)/sqrt(rho)).*exp(vpdR*muR/2/rho))) ...
                        + (1-rho)*(muSE(jj-1)*zeros(1, Nmc) + Wvec(end, :)).* (sqrt(rho)*tanh(((vpdR * muR)*zeros(1, Nmc) + vpdR *  WvecR)/sqrt(rho)) ./ ...
                        (rho+(1-rho).*sech(((vpdR * muR)*zeros(1, Nmc) + vpdR *  WvecR)/sqrt(rho)).*exp(vpdR*muR/2/rho))));
                    DeltaMAT(i1+1, jj) = DeltaMAT(jj, i1+1);
                end
            
            end
            
        else
            
            intval = rho/2*mean( 1./(rho+(1-rho).*exp(muSE(1:jj-1)'/ Sigmared *muSE(1:jj-1)./2./rho).*sech(((vpd * muSE(1:jj-1))*ones(1, Nmc)/sqrt(rho) + vpd *  Wvec)/sqrt(rho))) ...
                   - rho.* tanh(((vpd * muSE(1:jj-1))*ones(1, Nmc)/sqrt(rho) + vpd *  Wvec)/sqrt(rho)).^2 ./...
                   (rho+(1-rho).*exp(muSE(1:jj-1)'/ Sigmared *muSE(1:jj-1)./2./rho).*sech(((vpd * muSE(1:jj-1))*ones(1, Nmc)/sqrt(rho) + vpd *  Wvec)/sqrt(rho))).^2) ...
                   +rho/2*mean( 1./(rho+(1-rho).*exp(muSE(1:jj-1)'/ Sigmared *muSE(1:jj-1)./2./rho).*sech((-(vpd * muSE(1:jj-1))*ones(1, Nmc)/sqrt(rho) + vpd *  Wvec)/sqrt(rho))) ...
                   - rho.* tanh((-(vpd * muSE(1:jj-1))*ones(1, Nmc)/sqrt(rho) + vpd *  Wvec)/sqrt(rho)).^2 ./...
                   (rho+(1-rho).*exp(muSE(1:jj-1)'/ Sigmared *muSE(1:jj-1)./2./rho).*sech((-(vpd * muSE(1:jj-1))*ones(1, Nmc)/sqrt(rho) + vpd *  Wvec)/sqrt(rho))).^2) ...
                   +(1-rho)*mean( 1./(rho+(1-rho).*exp(muSE(1:jj-1)'/ Sigmared *muSE(1:jj-1)./2./rho).*sech(((vpd * muSE(1:jj-1))*zeros(1, Nmc) + vpd *  Wvec)/sqrt(rho))) ...
                   - rho.* tanh(((vpd * muSE(1:jj-1))*zeros(1, Nmc) + vpd *  Wvec)/sqrt(rho)).^2 ./...
                   (rho+(1-rho).*exp(muSE(1:jj-1)'/ Sigmared *muSE(1:jj-1)./2./rho).*sech(((vpd * muSE(1:jj-1))*zeros(1, Nmc) + vpd *  Wvec)/sqrt(rho))).^2);
    
            Phi(jj, 1:jj-1) = muSE(1:jj-1)' / Sigmared * intval;    
            
            muSE(jj) = alpha * sqrt(rho)/2*mean((sqrt(rho)*tanh(((vpd * muSE(1:jj-1))*ones(1, Nmc)/sqrt(rho) + vpd *  Wvec)/sqrt(rho)) ./ ...
                        (rho+(1-rho).*sech(((vpd * muSE(1:jj-1))*ones(1, Nmc)/sqrt(rho) + vpd *  Wvec)/sqrt(rho)).*exp(vpd*muSE(1:jj-1)/2/rho))) ...
                - (sqrt(rho)*tanh((-(vpd * muSE(1:jj-1))*ones(1, Nmc)/sqrt(rho) + vpd *  Wvec)/sqrt(rho)) ./ ...
                        (rho+(1-rho).*sech((-(vpd * muSE(1:jj-1))*ones(1, Nmc)/sqrt(rho) + vpd *  Wvec)/sqrt(rho)).*exp(vpd*muSE(1:jj-1)/2/rho))));
            DeltaMAT(1, jj) = muSE(jj)/alpha * epsl;
            DeltaMAT(jj, 1) = DeltaMAT(1, jj);
        
            for i1 = 1 : jj-1
                muR = muSE(1:i1);
                vpdR = muR' / Sigmared(1:i1, 1:i1);
                WvecR = Wvec(1:i1, :);
        
                %alternating posteriors
                if mod(i1, 3) ~= 0
                    
                    DeltaMAT(jj, i1+1) = rho/2*mean((sqrt(rho)*tanh(((vpd * muSE(1:jj-1))*ones(1, Nmc)/sqrt(rho) + vpd *  Wvec)/sqrt(rho)) ./ ...
                        (rho+(1-rho).*sech(((vpd * muSE(1:jj-1))*ones(1, Nmc)/sqrt(rho) + vpd *  Wvec)/sqrt(rho)).*exp(vpd*muSE(1:jj-1)/2/rho))) .* (muR(end)*ones(1, Nmc)/sqrt(rho) + WvecR(end, :)) ...
                        + (sqrt(rho)*tanh((-(vpd * muSE(1:jj-1))*ones(1, Nmc)/sqrt(rho) + vpd *  Wvec)/sqrt(rho)) ./ ...
                        (rho+(1-rho).*sech((-(vpd * muSE(1:jj-1))*ones(1, Nmc)/sqrt(rho) + vpd *  Wvec)/sqrt(rho)).*exp(vpd*muSE(1:jj-1)/2/rho))) .* (-muR(end)*ones(1, Nmc)/sqrt(rho) + WvecR(end, :))) ...
                        + (1-rho) * mean( (sqrt(rho)*tanh(((vpd * muSE(1:jj-1))*zeros(1, Nmc) + vpd *  Wvec)/sqrt(rho)) ./ ...
                        (rho+(1-rho).*sech(((vpd * muSE(1:jj-1))*zeros(1, Nmc) + vpd *  Wvec)/sqrt(rho)).*exp(vpd*muSE(1:jj-1)/2/rho))) .* (muR(end)*zeros(1, Nmc) + WvecR(end, :)) );
                    
                else
                    
                    DeltaMAT(jj, i1+1) = rho/2*mean((sqrt(rho)*tanh(((vpd * muSE(1:jj-1))*ones(1, Nmc)/sqrt(rho) + vpd *  Wvec)/sqrt(rho)) ./ ...
                        (rho+(1-rho).*sech(((vpd * muSE(1:jj-1))*ones(1, Nmc)/sqrt(rho) + vpd *  Wvec)/sqrt(rho)).*exp(vpd*muSE(1:jj-1)/2/rho))) .* ...
                        (sqrt(rho)*tanh(((vpdR * muR)*ones(1, Nmc)/sqrt(rho) + vpdR *  WvecR)/sqrt(rho)) ./ ...
                        (rho+(1-rho).*sech(((vpdR * muR)*ones(1, Nmc)/sqrt(rho) + vpdR *  WvecR)/sqrt(rho)).*exp(vpdR*muR/2/rho))) ...
                        + (sqrt(rho)*tanh((-(vpd * muSE(1:jj-1))*ones(1, Nmc)/sqrt(rho) + vpd *  Wvec)/sqrt(rho)) ./ ...
                        (rho+(1-rho).*sech((-(vpd * muSE(1:jj-1))*ones(1, Nmc)/sqrt(rho) + vpd *  Wvec)/sqrt(rho)).*exp(vpd*muSE(1:jj-1)/2/rho))) .* ...
                        (sqrt(rho)*tanh((-(vpdR * muR)*ones(1, Nmc)/sqrt(rho) + vpdR *  WvecR)/sqrt(rho)) ./ ...
                        (rho+(1-rho).*sech((-(vpdR * muR)*ones(1, Nmc)/sqrt(rho) + vpdR *  WvecR)/sqrt(rho)).*exp(vpdR*muR/2/rho)))) ...
                        + (1-rho) * mean( (sqrt(rho)*tanh(((vpd * muSE(1:jj-1))*zeros(1, Nmc) + vpd *  Wvec)/sqrt(rho)) ./ ...
                        (rho+(1-rho).*sech(((vpd * muSE(1:jj-1))*zeros(1, Nmc) + vpd *  Wvec)/sqrt(rho)).*exp(vpd*muSE(1:jj-1)/2/rho))) .* ...
                        (sqrt(rho)*tanh(((vpdR * muR)*zeros(1, Nmc) + vpdR *  WvecR)/sqrt(rho)) ./ ...
                        (rho+(1-rho).*sech(((vpdR * muR)*zeros(1, Nmc) + vpdR *  WvecR)/sqrt(rho)).*exp(vpdR*muR/2/rho))) );
                    
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

        
        if mod(jj-1, 3) == 0 
            
            fprintf('Iteration %d, scal=%f, MSE=%f\n', jj, scal(jj)^2, MSE(jj, j));
        

        end

    end
    
    scal_all(:, j) = scal;
    
 
end

save SE_ZFmod_u0_Radpriorsparse_epssqrt0dot5.mat scal_all alphagrid MSE;



toc