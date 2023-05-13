clear;
close all;
clc;

% This script implements the state evolution recursion, with the single memory step 
% posterior mean denoiser, described in Section "Approximate message
% passing, optimally" of the Supporting Information. The code below is a
% re-adaptation of "SE_2_quartic.m". We strongly suggest to see that first.

tic

%parameters of the spectral density for the pure power 6 potential
xi=27./80.;
a2=2./3.;

%choose your alpha grid, or a single value
%alphagrid = 2 : 0.01 : 4;
%alphagrid = 2.3 : 0.3 : 3.8;
alphagrid = 2.9;

epsl = sqrt(0.9); % correlation of initialization

niter = 10;

tolint = -1e-12;
tolintpos = 1e-12;
tols = 1e-10;



tolinc = 1e-5;
%tolsmall = 10^(-3);

scal_all = zeros(niter, length(alphagrid));
MSE = zeros(niter, length(alphagrid));
flag_all = zeros(niter, length(alphagrid));

max_it = 6*niter+1;

load freecum_sestic.mat;

freecum = kdouble'; 

for j = 1 : length(alphagrid)
    
    alpha = alphagrid(j);     
    fprintf('alpha=%f, eps=%f\n', alpha, epsl);
    
    %coefficients of the optimal pre-processing matrix polynomial for the
    %pure power 6 potential
    c1 = 0;
    c2 = -xi .* alpha^2; 
    c3 =   0; 
    c4 = -xi .* alpha^2;
    c5 =  xi .* alpha;
    
    %initializations-------------------------------------------------------
    %----------------------------------------------------------------------
    muSE = zeros(niter, 1);
    SNRSE = zeros(niter, 1);
    sigma2SE = zeros(niter, 1);
    tildemuSE = zeros(5*niter, 1); 
    tildesigmaSE = zeros(5*niter, 5*niter); 
    DeltaMAT = zeros(5*niter, 5*niter); 
    PhiMAT = zeros(5*niter, 5*niter); 
    theta_coeff = zeros(niter, 5*niter); 
    psi_coeff = zeros(5*niter, 5*niter); 
    beta_coeff = zeros(5*niter, 5*niter);  
    varphi_coeff = zeros(5*niter, 1);
    Onsager_correct = zeros(niter, niter);
    
    scal_all(1, j) = epsl;
    MSE(1, j) = 1-epsl^2;
    
    fprintf('Iteration %d, scal=%f, MSE=%f\n', 1, scal_all(1, j)^2, MSE(1, j));
    
    tildemuSE(1) = alpha * epsl;
    tildesigmaSE(1, 1) = freecum(2);
    DeltaMAT(1, 1) = 1;
    
    PhiMAT(2, 1) = 1;
    b_Ons = freecum(1);
    DeltaMAT(2, 1) = b_Ons + tildemuSE(1)^2/alpha;
    DeltaMAT(1, 2) = DeltaMAT(2, 1);
    DeltaMAT(2, 2) = tildesigmaSE(1, 1) + b_Ons^2 + tildemuSE(1)^2 + ...
        2 * b_Ons * tildemuSE(1)^2/alpha;
    tildemuSE(2) = b_Ons * tildemuSE(1) + alpha * tildemuSE(1);
    
    PhiTMP = PhiMAT(1:2, 1:2);
    DeltaTMP = DeltaMAT(1:2, 1:2);
    
    %onsager coeffs matrix Btilde 
    Bmat = zeros(2, 2);
    for jj = 0 : 1
        Bmat = Bmat + freecum(jj+1) * PhiTMP^(jj);
    end
    
    %update sigma tilde 
    SigmaTMP = zeros(2, 2);
    for jj = 0 : 2 %building the matrix Sigma
        
        ThetaMAT = zeros(2, 2);
        for j2 = 0 : jj
            ThetaMAT = ThetaMAT + PhiTMP^j2 * DeltaTMP * (PhiTMP^(jj-j2))';
        end
        
        SigmaTMP = SigmaTMP + freecum(jj+2) * ThetaMAT;
    end
    
    if SigmaTMP(1, 1) ~= freecum(2)
        fprintf('SigmaTMP(1, 1) not matching!\n');
        return;
    end
    
    tildesigmaSE(1:2, 1:2) = SigmaTMP;
    
    %update of 3x3 matrices, we add two additional steps
    tildemuSE(3) = Bmat(2, 1) * tildemuSE(1) + Bmat(2, 2) * tildemuSE(2) + alpha * tildemuSE(2);
    PhiMAT(3, 1) = Bmat(2, 2);
    PhiMAT(3, 2) = 1;
    DeltaMAT(3, 1) = Bmat(2, 1) + tildemuSE(1) * tildemuSE(2)/alpha + Bmat(2, 2) * DeltaMAT(2, 1);
    DeltaMAT(3, 2) = Bmat(2, 1) * DeltaMAT(2, 1) + tildemuSE(2)^2/alpha + ...
        Bmat(2, 2) * DeltaMAT(2, 2) +tildesigmaSE(1, 2);
    DeltaMAT(3, 3) = tildesigmaSE(2, 2) + Bmat(2, 1)^2 + tildemuSE(2)^2 + ...
        Bmat(2, 2)^2 * DeltaMAT(2, 2) + 2 * Bmat(2, 2) * (tildesigmaSE(2, 1) + ...
        Bmat(2, 1) * DeltaMAT(1, 2)) + 2 * tildemuSE(2) * Bmat(2, 1) * tildemuSE(1) / alpha + ...
        2 * Bmat(2, 2) * tildemuSE(2)^2/alpha;
    DeltaMAT(1, 3) = DeltaMAT(3, 1);
    DeltaMAT(2, 3) = DeltaMAT(3, 2);
    
    PhiTMP = PhiMAT(1:3, 1:3);
    DeltaTMP = DeltaMAT(1:3, 1:3);
    BmatNEW = zeros(3, 3);
    
    for jj = 0 : 2
        BmatNEW = BmatNEW + freecum(jj+1) * PhiTMP^(jj);
    end
    
    for i1 = 1 : 2
        for j1 = 1 : 2
            if BmatNEW(i1, j1) ~= Bmat(i1, j1)
                fprintf('BmatNEW(%d, %d) not matching!\n', i1, j1);
                return;
            end
        end
    end
    
    Bmat = BmatNEW;
    
    SigmaTMP = zeros(3, 3);

    for jj = 0 : 4
        
        ThetaMAT = zeros(3, 3);
        for j2 = 0 : jj
            ThetaMAT = ThetaMAT + PhiTMP^j2 * DeltaTMP * ((PhiTMP^(jj-j2))');
        end
                
        SigmaTMP = SigmaTMP + freecum(jj+2) * ThetaMAT;
    end
    
    for i1 = 1 : 2
        for j1 = 1 : 2
            if SigmaTMP(i1, j1) ~= tildesigmaSE(i1, j1)
                fprintf('SigmaTMP(%d, %d) not matching!\n', i1, j1);
                return;
            end
        end
    end
    
    tildesigmaSE(1:3, 1:3) = SigmaTMP;

    
%4-th step----------------------------------------------------------------
    tildemuSE(4) = Bmat(3, 1) * tildemuSE(1) + Bmat(3, 2) * tildemuSE(2)+ Bmat(3, 3) * tildemuSE(3) + alpha * tildemuSE(3);
    
    PhiMAT(4, 1) = Bmat(3, 1:3)*PhiMAT(1:3, 1);
    PhiMAT(4, 2) = Bmat(3, 1:3)*PhiMAT(1:3, 2);
    PhiMAT(4, 3) = 1;

    DeltaMAT(4, 1) = tildemuSE(1) * tildemuSE(3)/alpha +Bmat(3, 1:3)*DeltaMAT(1:3, 1);
    DeltaMAT(4, 2) = tildesigmaSE(3,1)*PhiMAT(2,1)+ tildemuSE(2) * tildemuSE(3)/alpha+...
        Bmat(3, 1:3)*DeltaMAT(1:3, 2);
    DeltaMAT(4, 3) = PhiMAT(3,1:2) * tildesigmaSE(1:2,3)+ tildemuSE(3)^2/alpha+...
        Bmat(3, 1:3)*DeltaMAT(1:3, 3);
    %symmetrization
    DeltaMAT(1, 4)=DeltaMAT(4, 1);
    DeltaMAT(2, 4)=DeltaMAT(4, 2);
    DeltaMAT(3, 4)=DeltaMAT(4, 3);

    DeltaMAT(4, 4) = PhiMAT(4,1:3) * tildesigmaSE(1:3,3)+ tildemuSE(3)*tildemuSE(4)/alpha+...
        Bmat(3, 1:3)*DeltaMAT(1:3, 4);
    
    PhiTMP = PhiMAT(1:4, 1:4);
    DeltaTMP = DeltaMAT(1:4, 1:4);
    BmatNEW = zeros(4, 4);
    
    for jj = 0 : 3
        BmatNEW = BmatNEW + freecum(jj+1) * PhiTMP^(jj);
    end
    
    for i1 = 1 : 3
        for j1 = 1 : 3
            if BmatNEW(i1, j1) ~= Bmat(i1, j1)
                fprintf('BmatNEW(%d, %d) not matching!\n', i1, j1);
                return;
            end
        end
    end
    
    Bmat = BmatNEW;
    
    SigmaTMP = zeros(4, 4);

    for jj = 0 : 6
        
        ThetaMAT = zeros(4, 4);
        for j2 = 0 : jj
            ThetaMAT = ThetaMAT + PhiTMP^j2 * DeltaTMP * ((PhiTMP^(jj-j2))');
        end
                
        SigmaTMP = SigmaTMP + freecum(jj+2) * ThetaMAT;
    end
    
    for i1 = 1 : 3
        for j1 = 1 : 3
            if SigmaTMP(i1, j1) ~= tildesigmaSE(i1, j1)
                fprintf('SigmaTMP(%d, %d) not matching!\n', i1, j1);
                return;
            end
        end
    end
    
    tildesigmaSE(1:4, 1:4) = SigmaTMP;

%-------------------------------------------------------------------------


%5-th step----------------------------------------------------------------
    tildemuSE(5) = Bmat(4, 1:4) * tildemuSE(1:4,1) + alpha * tildemuSE(4);
    
    PhiMAT(5, 1) = Bmat(4, 1:4)*PhiMAT(1:4, 1);
    PhiMAT(5, 2) = Bmat(4, 1:4)*PhiMAT(1:4, 2);
    PhiMAT(5, 3) = Bmat(4, 1:4)*PhiMAT(1:4, 3);
    PhiMAT(5, 4) = 1;

    DeltaMAT(5, 1) = tildemuSE(1) * tildemuSE(4)/alpha +Bmat(4, 1:4)*DeltaMAT(1:4, 1);
    DeltaMAT(5, 2) = tildesigmaSE(4,1)*PhiMAT(2,1)+ tildemuSE(4) * tildemuSE(2)/alpha+...
        Bmat(4, 1:4)*DeltaMAT(1:4, 2);
    DeltaMAT(5, 3) = PhiMAT(3,1:2) * tildesigmaSE(1:2,4)+ tildemuSE(4) * tildemuSE(3)/alpha+...
        Bmat(4, 1:4) * DeltaMAT(1:4, 3);
    DeltaMAT(5, 4) = PhiMAT(4,1:3) * tildesigmaSE(1:3,4)+ tildemuSE(4)^2/alpha+...
        Bmat(4, 1:4) * DeltaMAT(1:4, 4);
    %symmetrization
    DeltaMAT(1, 5)=DeltaMAT(5, 1);
    DeltaMAT(2, 5)=DeltaMAT(5, 2);
    DeltaMAT(3, 5)=DeltaMAT(5, 3);
    DeltaMAT(4, 5)=DeltaMAT(5, 4);

    DeltaMAT(5, 5) = PhiMAT(5,1:4) * tildesigmaSE(1:4,4)+ tildemuSE(5)*tildemuSE(4)/alpha+...
        Bmat(4, 1:4)*DeltaMAT(1:4, 5);
    


    PhiTMP = PhiMAT(1:5, 1:5);
    DeltaTMP = DeltaMAT(1:5, 1:5);
    BmatNEW = zeros(5, 5);
    
    for jj = 0 : 4
        BmatNEW = BmatNEW + freecum(jj+1) * PhiTMP^(jj);
    end
    
    for i1 = 1 : 4
        for j1 = 1 : 4
            if BmatNEW(i1, j1) ~= Bmat(i1, j1)
                fprintf('BmatNEW(%d, %d) not matching!\n', i1, j1);
                return;
            end
        end
    end
    
    Bmat = BmatNEW;
    
    SigmaTMP = zeros(5, 5);

    for jj = 0 : 8
        
        ThetaMAT = zeros(5, 5);
        for j2 = 0 : jj
            ThetaMAT = ThetaMAT + PhiTMP^j2 * DeltaTMP * ((PhiTMP^(jj-j2))');
        end
        
        SigmaTMP = SigmaTMP + freecum(jj+2) * ThetaMAT;
    end
    
    for i1 = 1 : 4
        for j1 = 1 : 4
            if SigmaTMP(i1, j1) ~= tildesigmaSE(i1, j1)
                fprintf('SigmaTMP(%d, %d) not matching!\n', i1, j1);
                return;
            end
        end
    end
    
    tildesigmaSE(1:5, 1:5) = SigmaTMP;

%-------------------------------------------------------------------------



    psi_coeff(2, 1) = 1; %psi is alpha in paper
    psi_coeff(3, 1) = Bmat(2, 2);
    psi_coeff(3, 2) = 1;
    psi_coeff(4, 1) = Bmat(3, 2)+Bmat(2, 2)*Bmat(3, 3);
    psi_coeff(4, 2) = Bmat(3,3);
    psi_coeff(4, 3) = 1;
    psi_coeff(5, 1) = Bmat(4,2:4)*psi_coeff(2:4, 1);
    psi_coeff(5, 2) = Bmat(4,2:4)*psi_coeff(2:4, 2);
    psi_coeff(5, 3) = Bmat(4,4);
    psi_coeff(5, 4) = 1;

    beta_coeff(1, 1) = 1; %beta is beta in paper
    beta_coeff(2, 1) = Bmat(1, 1);
    beta_coeff(3, 1) = Bmat(2, 1) + Bmat(2, 2) * Bmat(1, 1);
    beta_coeff(4, 1) = Bmat(3, 1:3)*beta_coeff(1:3, 1);
    beta_coeff(5, 1) = Bmat(4, 1:4)*beta_coeff(1:4, 1);

    varphi_coeff(2) = tildemuSE(1); %varphi is gamma in paper
    varphi_coeff(3) = Bmat(2, 2) * tildemuSE(1) + tildemuSE(2);
    varphi_coeff(4) = tildemuSE(3)+Bmat(3, 2:3) * varphi_coeff(2:3); %varphi is gamma in paper
    varphi_coeff(5) = tildemuSE(4)+Bmat(4, 2:4) * varphi_coeff(2:4);

    flgdet = 0;
%end of initializations----------------------------------------------------
%--------------------------------------------------------------------------

    for t = 1 : niter-1 %SE iterations
        
        %resetting beta
        beta_coeff(5*t+1, t+1) = 1;
        
        %update \mu
        muSE(t) = c1 * (tildemuSE(5*t-4) + Bmat(5*t-4, 1:5*t-4) * varphi_coeff(1:5*t-4) ) + ...
            c2 * (tildemuSE(5*t-3) + Bmat(5*t-3, 1:5*t-3) * varphi_coeff(1:5*t-3) ) + ...
            c4 * (tildemuSE(5*t-1) + Bmat(5*t-1, 1:5*t-1) * varphi_coeff(1:5*t-1) )+...
            c5 * (tildemuSE(5*t) + Bmat(5*t, 1:5*t) * varphi_coeff(1:5*t) );%c3 is zero!
        
        %update \theta coefficients
        theta_coeff(t, 1:5*t) = c1 * Bmat(5*t-4, 1:5*t-4) * psi_coeff(1:5*t-4, 1:5*t) + ...
            c2 * Bmat(5*t-3, 1:5*t-3) * psi_coeff(1:5*t-3, 1:5*t) + ...
            c4 * Bmat(5*t-1, 1:5*t-1) * psi_coeff(1:5*t-1, 1:5*t)+...
            c5 * Bmat(5*t, 1:5*t) * psi_coeff(1:5*t, 1:5*t); %c3 is zero!
        theta_coeff(t, 5*t-4) = theta_coeff(t, 5*t-4) + c1;
        theta_coeff(t, 5*t-3) = theta_coeff(t, 5*t-3) + c2;
        theta_coeff(t, 5*t-2) = theta_coeff(t, 5*t-2) + c3;%c3 is zero!
        theta_coeff(t, 5*t-1) = theta_coeff(t, 5*t-1) + c4;
        theta_coeff(t, 5*t) = theta_coeff(t, 5*t) + c5;
        
        %Onsager coefficients
        Onsager_correct(t, 1:t) = c1 * Bmat(5*t-4, 1:5*t-4) * beta_coeff(1:5*t-4, 1:t) + ...
            c2 * Bmat(5*t-3, 1:5*t-3) * beta_coeff(1:5*t-3, 1:t) + ...
            c3 * 0 +...
            c4 * Bmat(5*t-1, 1:5*t-1) * beta_coeff(1:5*t-1, 1:t)+...
            c5 * Bmat(5*t, 1:5*t) * beta_coeff(1:5*t, 1:t);
        
        %Variance of a linear combination og Gaussian noises whose
        %covariance is \tilde{\Sigma}, with coefficients theta.
        sigma2SE(t) = theta_coeff(t, 1:5*t) * tildesigmaSE(1:5*t, 1:5*t) * theta_coeff(t, 1:5*t)'; %variance of a combination of gaussians
        
        %effetive SNR of a signal drawn from the prior w.r.t. a gaussian
        %noise with the above variance
        SNRSE(t) = sqrt(muSE(t)^2/sigma2SE(t));

        
        fun = @(x) 1/sqrt(2*pi) * exp(-x.^2/2) .* tanh(SNRSE(t)^2 + SNRSE(t) * x);
        int1 = integral(fun, -Inf, Inf);
        fun2 = @(x) 1/sqrt(2*pi) * exp(-x.^2/2) .* tanh(-SNRSE(t)^2 + SNRSE(t) * x);
        int2 = integral(fun2, -Inf, Inf);
        

        tildemuSE(5*t+1) = alpha/2 * (int1 - int2);
        DeltaMAT(5*t+1, 1) = epsl/2 * (int1 - int2);
        DeltaMAT(1, 5*t+1) = DeltaMAT(5*t+1, 1);
        
        %needed to update the DeltaMat(3t+1,3t'+1) that are covariances of
        %the "physical" quantities U_t and U_t', not of auxiliary AMP
        %The rest of the code is a simple re-adaptation of the code for the
        %quartic potential, the only difference is that every factor 3* in
        %from of time indices is replaced with 5*. Other minor differences
        %will be pointed out later.

        for t1 = 1 : t-1
            
            cov_tt1 = theta_coeff(t, 1:5*t) * tildesigmaSE(1:5*t, 1:5*t1) * theta_coeff(t1, 1:5*t1)';
            Sigma1 = [sigma2SE(t), cov_tt1; cov_tt1, sigma2SE(t1)];%bidimensional matrix
            invS = inv(Sigma1);
            
            if det(Sigma1) <= 0
                fprintf('Calculation of DeltaMAT(%d, %d), det of Sigma1 =%f\n', 5*t+1, 5*t1+1, det(Sigma1));
                flgdet = 1;
                break;
            end
            
            fun1 = @(x, y) 1/(2*pi*sqrt(det(Sigma1))) * exp(-1/2 * (( invS(1, 1) * x.^2 + invS(2, 2) * y.^2 + 2*invS(1, 2)*x.*y))) ...
                .* tanh(muSE(t)^2/sigma2SE(t) + abs(muSE(t))/sigma2SE(t) * x) ...
                .* tanh(muSE(t1)^2/sigma2SE(t1) + abs(muSE(t1))/sigma2SE(t1) * y);
            int1 = integral2(fun1, -Inf, Inf, -Inf, Inf);
            fun2 = @(x, y) 1/(2*pi*sqrt(det(Sigma1))) * exp(-1/2 * (( invS(1, 1) * x.^2 + invS(2, 2) * y.^2 + 2*invS(1, 2)*x.*y))) ...
                .* tanh(-muSE(t)^2/sigma2SE(t) + abs(muSE(t))/sigma2SE(t) * x) ...
                .* tanh(-muSE(t1)^2/sigma2SE(t1) + abs(muSE(t1))/sigma2SE(t1) * y);
            int2 = integral2(fun2, -Inf, Inf, -Inf, Inf);
            
            DeltaMAT(5*t+1, 5*t1+1) = (int1 + int2)/2;
            DeltaMAT(5*t1+1, 5*t+1) = DeltaMAT(5*t+1, 5*t1+1);
        end        
        

        fun = @(x) 1/sqrt(2*pi) * exp(-x.^2/2) .* (tanh(SNRSE(t)^2 + SNRSE(t) * x)).^2;
        int1 = integral(fun, -Inf, Inf);
        fun2 = @(x) 1/sqrt(2*pi) * exp(-x.^2/2) .* (tanh(-SNRSE(t)^2 + SNRSE(t) * x)).^2;
        int2 = integral(fun2, -Inf, Inf);
        DeltaMAT(5*t+1, 5*t+1) = (int1 + int2)/2;
        phi_int = muSE(t)/sigma2SE(t) * (1 - (int1 + int2)/2);
        
        PhiMAT(5*t+1, 1:5*t) = phi_int * theta_coeff(t, 1:5*t); %uses the update with the denoiser!
        
        
        scal_all(t+1, j) = tildemuSE(5*t+1)/alpha/sqrt(DeltaMAT(5*t+1, 5*t+1));
        
        MSE(t+1, j) = (1 - 2 * (tildemuSE(5*t+1)/alpha)^2 + DeltaMAT(5*t+1, 5*t+1)^2)/2;

        fprintf('Iteration %d, scal=%f, MSE=%f\n', t+1, scal_all(t+1, j)^2, MSE(t+1, j));
        
        %stop criterion
        if (abs(scal_all(t+1, j)^2-scal_all(t, j)^2) < tolinc) || flgdet == 1
            for kk = t+2 : niter
                scal_all(kk, j) = scal_all(t+1, j); 
                MSE(kk, j) = MSE(t+1, j);
            end
            
            break;
        end

        
        
        %update of the auxiliary part of the AMP
        for i = 1 : 5*t
            
            if mod(i, 5) ~= 1 %summations only mod 5, instead of mod 3
                
                cov_tt1 = theta_coeff(t, 1:5*t) * tildesigmaSE(i-1, 1:5*t)';
                Sigma1 = [sigma2SE(t), cov_tt1; cov_tt1, tildesigmaSE(i-1, i-1)];

                if det(Sigma1) < tolint
                    fprintf('yey 2 Calculation of DeltaMAT(%d, %d), det of Sigma1 =%f\n', 5*t+1, i, det(Sigma1));
                    flgdet = 1;
                    break;
                end
                
                if det(Sigma1) < tolintpos
                    
                    fun = @(x) 1/sqrt(2*pi) * exp(-x.^2/2) .* tanh(SNRSE(t)^2 + SNRSE(t) * x) .* ...
                        sqrt(tildesigmaSE(i-1, i-1)) .* x;
                    int1 = integral(fun, -Inf, Inf);
                    fun2 = @(x) 1/sqrt(2*pi) * exp(-x.^2/2) .* tanh(-SNRSE(t)^2 + SNRSE(t) * x) .* ...
                        sqrt(tildesigmaSE(i-1, i-1)) .* x;
                    int2 = integral(fun2, -Inf, Inf);
                    DeltaMAT(5*t+1, i) = (int1+int2)/2 + tildemuSE(5*t+1) * tildemuSE(i-1)/alpha + Bmat(i-1, 1:i-1) * DeltaMAT(5*t+1, 1:i-1)';
                    DeltaMAT(i, 5*t+1) = DeltaMAT(5*t+1, i);
                else
                
                    invS = inv(Sigma1);

                    fun1 = @(x, y) 1/(2*pi*sqrt(det(Sigma1))) * exp(-1/2 * (( invS(1, 1) * x.^2 + invS(2, 2) * y.^2 + 2*invS(1, 2)*x.*y))) ...
                        .* tanh(muSE(t)^2/sigma2SE(t) + muSE(t)/sigma2SE(t) * x) ...
                        .* y;
                    int1 = integral2(fun1, -Inf, Inf, -Inf, Inf);
                    fun2 = @(x, y) 1/(2*pi*sqrt(det(Sigma1))) * exp(-1/2 * (( invS(1, 1) * x.^2 + invS(2, 2) * y.^2 + 2*invS(1, 2)*x.*y))) ...
                        .* tanh(-muSE(t)^2/sigma2SE(t) + muSE(t)/sigma2SE(t) * x) ...
                        .* y;
                    int2 = integral2(fun2, -Inf, Inf, -Inf, Inf);
                    DeltaMAT(5*t+1, i) = (int1+int2)/2 + tildemuSE(5*t+1) * tildemuSE(i-1)/alpha + Bmat(i-1, 1:i-1) * DeltaMAT(5*t+1, 1:i-1)';
                    DeltaMAT(i, 5*t+1) = DeltaMAT(5*t+1, i);
                end
            end
        end
        
        if flgdet == 1
            for kk = t+2 : niter
                scal_all(kk, j) = scal_all(t+1, j); 
                MSE(kk, j) = MSE(t+1, j);
            end
            
            break;
        end
        
        
        PhiTMP = PhiMAT(1:5*t+1, 1:5*t+1);
        DeltaTMP = DeltaMAT(1:5*t+1, 1:5*t+1);
        BmatNEW = zeros(5*t+1, 5*t+1);
    
        for jj = 0 : 5*t
            BmatNEW = BmatNEW + freecum(jj+1) * PhiTMP^(jj);
        end
    
        for i1 = 1 : 5*t
            for j1 = 1 : 5*t
                if abs(BmatNEW(i1, j1) - Bmat(i1, j1)) > tols
                    fprintf('BmatNEW(%d, %d) not matching!\n', i1, j1);
                    return;
                end
            end
        end
    
        Bmat = BmatNEW;
    
        SigmaTMP = zeros(5*t+1, 5*t+1);

        for jj = 0 : 2*(5*t)
        
            ThetaMAT = zeros(5*t+1, 5*t+1);
            for j2 = 0 : jj
                ThetaMAT = ThetaMAT + PhiTMP^j2 * DeltaTMP * (PhiTMP^(jj-j2))';
            end
        
            SigmaTMP = SigmaTMP + freecum(jj+2) * ThetaMAT;
        
        end
    
        for i1 = 1 : 5*t
            for j1 = 1 : 5*t
                if abs(SigmaTMP(i1, j1) - tildesigmaSE(i1, j1)) > tols
                    fprintf('SigmaTMP(%d, %d) not matching!a\n', i1, j1);
                    return;
                end
            end
        end
    
        tildesigmaSE(1:5*t+1, 1:5*t+1) = SigmaTMP;
    
        for ell = 1 : 4 %instead of 1 : 2
        
            tildemuSE(5*t+1+ell) = alpha * tildemuSE(5*t+ell) + Bmat(5*t+ell, 1:5*t+ell) * tildemuSE(1:5*t+ell);
        
            PhiMAT(5*t+ell+1, 5*t+ell) = 1;

            for j1 = 1 : 5*t+ell-1
                PhiMAT(5*t+ell+1, j1) = Bmat(5*t+ell, 1:5*t+ell) * PhiMAT(1:5*t+ell, j1);
            end
        
            gamma_par = zeros(5*t+ell+1, 1);
        
            for t1 = 1 : t
            
                cov_tt1 = theta_coeff(t1, 1:5*t1) * tildesigmaSE(5*t+ell, 1:5*t1)';
                Sigma1 = [sigma2SE(t1), cov_tt1; cov_tt1, tildesigmaSE(5*t+ell, 5*t+ell)];
            
                invS = inv(Sigma1);
                
                if det(Sigma1) <= 0
                    fprintf('Calculation of gamma_par(%d), det of Sigma1 =%f\n', 3*t1+1, det(Sigma1));
                    flgdet = 1;
                    break;
                end
            
                fun1 = @(x, y) 1/(2*pi*sqrt(det(Sigma1))) * exp(-1/2 * (( invS(1, 1) * x.^2 + invS(2, 2) * y.^2 + 2*invS(1, 2)*x.*y))) ...
                    .* tanh(muSE(t1)^2/sigma2SE(t1) + muSE(t1)/sigma2SE(t1) * x) ...
                    .* y;
                int1 = integral2(fun1, -Inf, Inf, -Inf, Inf);
                fun2 = @(x, y) 1/(2*pi*sqrt(det(Sigma1))) * exp(-1/2 * (( invS(1, 1) * x.^2 + invS(2, 2) * y.^2 + 2*invS(1, 2)*x.*y))) ...
                    .* tanh(-muSE(t1)^2/sigma2SE(t1) + muSE(t1)/sigma2SE(t1) * x) ...
                    .* y;
                int2 = integral2(fun2, -Inf, Inf, -Inf, Inf);
                gamma_par(5*t1+1) = (int1 + int2)/2;
            end
            
            if flgdet == 1
                break;
            end

        
            for i = 1 : 5*t+ell+1
                if mod(i, 5) ~= 1
                    gamma_par(i) = tildesigmaSE(5*t+ell, i-1) + Bmat(i-1, 1:i-1) * gamma_par(1:i-1);
                end
            end
        
            for j1 = 1 : 5*t+ell+1
                DeltaMAT(5*t+ell+1, j1) = gamma_par(j1) + tildemuSE(5*t+ell) * tildemuSE(j1)/alpha  + ...
                    Bmat(5*t+ell, 1:5*t+ell) * DeltaMAT(1:5*t+ell, j1);
                DeltaMAT(j1, 5*t+ell+1) = DeltaMAT(5*t+ell+1, j1);
            end
            
            PhiTMP = PhiMAT(1:5*t+ell+1, 1:5*t+ell+1);
            DeltaTMP = DeltaMAT(1:5*t+ell+1, 1:5*t+ell+1);

            BmatNEW = zeros(5*t+ell+1, 5*t+ell+1);

            for jj = 0 : 5*t+ell
                BmatNEW = BmatNEW + freecum(jj+1) * PhiTMP^(jj);
            end

            for i1 = 1 : 5*t+ell
                for j1 = 1 : 5*t+ell
                    if abs(BmatNEW(i1, j1) - Bmat(i1, j1)) > tols
                        fprintf('BmatNEW(%d, %d) not matching!\n', i1, j1);
                        return;
                    end
                end
            end

            Bmat = BmatNEW;

            SigmaTMP = zeros(5*t+ell+1, 5*t+ell+1);

            for jj = 0 : 2*(5*t+ell)

                ThetaMAT = zeros(5*t+ell+1, 5*t+ell+1);
                for j2 = 0 : jj
                    ThetaMAT = ThetaMAT + PhiTMP^j2 * DeltaTMP * (PhiTMP^(jj-j2))';
                end

                SigmaTMP = SigmaTMP + freecum(jj+2) * ThetaMAT;
            end

            for i1 = 1 : 5*t+ell
                for j1 = 1 : 5*t+ell
                    if abs(SigmaTMP(i1, j1) - tildesigmaSE(i1, j1)) > tols
                        fprintf('SigmaTMP(%d, %d) not matching!\n', i1, j1);
                        return;
                    end
                end
            end

            tildesigmaSE(1:5*t+ell+1, 1:5*t+ell+1) = SigmaTMP;
            
            psi_coeff(5*t+ell+1, 5*t+ell) = 1;
            varphi_coeff(5*t+ell+1) = tildemuSE(5*t+ell);
            beta_coeff(5*t+ell+1,1:t+1) = Bmat(5*t+ell, 5*(1:t+1)-4);
            
            
            for i = 1 : 5*t+ell
                if mod(i, 5) ~= 1
                    for j1 = 1 : i-1
                        psi_coeff(5*t+ell+1, j1) = psi_coeff(5*t+ell+1, j1) + Bmat(5*t+ell, i) * psi_coeff(i, j1);
                    end
                    for j1 = 1 : ceil((i-1)/5)
                        beta_coeff(5*t+ell+1, j1) = beta_coeff(5*t+ell+1, j1) + Bmat(5*t+ell, i) * beta_coeff(i, j1);
                    end
                    varphi_coeff(5*t+ell+1) = varphi_coeff(5*t+ell+1) + Bmat(5*t+ell, i) * varphi_coeff(i);
                end
            end
        end
        
        if flgdet == 1
            for kk = t+2 : niter
                scal_all(kk, j) = scal_all(t+1, j); 
                MSE(kk, j) = MSE(t+1, j);
            end

            break;
        end
        
        
    end
    
    
end


save SE_sestic_BAMP2_9.mat Onsager_correct muSE sigma2SE MSE scal_all alphagrid;


toc