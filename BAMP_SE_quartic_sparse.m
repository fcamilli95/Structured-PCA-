clear;
close all;
clc;

% This script implements the state evolution recursion, with the single memory step 
% posterior mean denoiser, described in Section "Approximate message
% passing, optimally" of the Supporting Information.

tic

%parameters of the quartic potential
u = 0;
gamma = (8-9*u+sqrt(64-144*u+108*u.^2-27*u.^3))/27; %chosen to have unit variance of the spectral density
rho=0.5;

%alphagrid = 2 : 0.01 : 4;
alphagrid = 2 : 0.06 : 4.06;
%alphagrid = 2.6;

epsl = sqrt(0.9); % correlation of initialization, it can be lowered.

niter = 10; %maximum number of iterations with the available free cumulants, 
% it is already sufficient to reach a good convergence

%some tolerances used later
tolint = -1e-12;
tolintpos = 1e-12;
tols = 1e-10;
tolinc = 1e-6;

%allocation of memory
scal_all = zeros(niter, length(alphagrid));
MSE = zeros(niter, length(alphagrid));
flag_all = zeros(niter, length(alphagrid));

%maximum iterations of the auxiliary AMP, that need 3 times the steps of
%the true one
max_it = 4*niter+1;

load freecum_u0.mat; %load a file containing the free cumulants you need

freecum = kdouble'; % free cumulants (starting from the 1st)

for j = 1 : length(alphagrid)
    
    alpha = alphagrid(j);     
    fprintf('alpha=%f, eps=%f\n', alpha, epsl);
    
    %coeffieicnts of the preprocessing matrix polynomial
    c1 = u * alpha;
    c2 = -gamma * alpha^2;
    c3 = gamma * alpha;
    
    %initializations-------------------------------------------------------
    %----------------------------------------------------------------------
    muSE = zeros(niter, 1); %as in the paper, correlation between iterates of the true BAMP and ground truth 
    SNRSE = zeros(niter, 1);%an additional quantity useful to track, it acts as an effective SNR
    sigma2SE = zeros(niter, 1);%an additional quantity useful to track, it is an effective variance of a Gaussian noise
    tildemuSE = zeros(3*niter, 1); %as in the paper, correlation between iterates of the auxiliary AMP and ground truth
    tildesigmaSE = zeros(3*niter, 3*niter); %\tilde{\Sigma}, matrix, as in the paper, covariance between the Gaussian noises in the auxiliary AMP
    DeltaMAT = zeros(3*niter, 3*niter); %\tilde{\Delta}, matrix, correlation between iterates of the axiliary AMP
    PhiMAT = zeros(3*niter, 3*niter); %\Phi, matrix, needed to compute Onsager coefficients later.
    
    theta_coeff = zeros(niter, 3*niter); % as in the paper
    psi_coeff = zeros(3*niter, 3*niter); %psi is alpha in paper
    beta_coeff = zeros(3*niter, 3*niter); % as in the paper
    varphi_coeff = zeros(3*niter, 1);%varphi is \gamma in paper
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
    
    %Building \tilde{B} for Onsagers
    Bmat = zeros(2, 2);
    
    for jj = 0 : 1
        Bmat = Bmat + freecum(jj+1) * PhiTMP^(jj);
    end
    
    %Building \tilde{\Sigma}
    SigmaTMP = zeros(2, 2);
    for jj = 0 : 2
        
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
    
    %completing the remaining rows and columns of \tilde{\Phi} and
    %\tilde{\Delta}
    PhiMAT(3, 1) = Bmat(2, 2);
    PhiMAT(3, 2) = 1;
    DeltaMAT(3, 1) = Bmat(2, 1) + tildemuSE(1) * tildemuSE(2)/alpha + Bmat(2, 2) * DeltaMAT(2, 1);
    DeltaMAT(3, 2) = Bmat(2, 1) * DeltaMAT(2, 1) + tildemuSE(2)^2/alpha + ...
        Bmat(2, 2) * DeltaMAT(2, 2) + tildesigmaSE(1, 2);
    DeltaMAT(3, 3) = tildesigmaSE(2, 2) + Bmat(2, 1)^2 + tildemuSE(2)^2 + ...
        Bmat(2, 2)^2 * DeltaMAT(2, 2) + 2 * Bmat(2, 2) * (tildesigmaSE(2, 1) + ...
        Bmat(2, 1) * DeltaMAT(1, 2)) + 2 * tildemuSE(2) * Bmat(2, 1) * tildemuSE(1) / alpha + ...
        2 * Bmat(2, 2) * tildemuSE(2)^2/alpha;
    DeltaMAT(1, 3) = DeltaMAT(3, 1);
    DeltaMAT(2, 3) = DeltaMAT(3, 2);
    
    tildemuSE(3) = Bmat(2, 1) * tildemuSE(1) + Bmat(2, 2) * tildemuSE(2) + alpha * tildemuSE(2);


    PhiTMP = PhiMAT(1:3, 1:3);
    DeltaTMP = DeltaMAT(1:3, 1:3);

    %Completing Bmat with a row and a column
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
    
    %Completing \tilde{\Sigma} with a row and a column
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

    
    % complete initializations of the coefficients psi(alpha in paper),
    % beta (beta also in the paper) and varphi (gamma in the paper)
    psi_coeff(2, 1) = 1;
    psi_coeff(3, 1) = Bmat(2, 2);
    psi_coeff(3, 2) = 1;
    beta_coeff(1, 1) = 1;
    beta_coeff(2, 1) = Bmat(1, 1);
    beta_coeff(3, 1) = Bmat(2, 1) + Bmat(2, 2) * Bmat(1, 1);
    varphi_coeff(2) = tildemuSE(1);
    varphi_coeff(3) = Bmat(2, 2) * tildemuSE(1) + tildemuSE(2);
    
  
    flgdet = 0;
    %end of initializations------------------------------------------------
    %----------------------------------------------------------------------


    for t = 1 : niter-1 %beginning of the iterations
        %resetting the value of beta as prescribed
        beta_coeff(3*t+1, t+1) = 1;
        
        %compute \mu_t using all the previous steps of the auxiliary AMP
        muSE(t) = c1 * (tildemuSE(3*t-2) + Bmat(3*t-2, 1:3*t-2) * varphi_coeff(1:3*t-2) ) + ...
            c2 * (tildemuSE(3*t-1) + Bmat(3*t-1, 1:3*t-1) * varphi_coeff(1:3*t-1) ) + ...
            c3 * (tildemuSE(3*t) + Bmat(3*t, 1:3*t) * varphi_coeff(1:3*t) );

        %compute \theta using all the previous steps of the auxiliary AMP
        theta_coeff(t, 1:3*t) = c1 * Bmat(3*t-2, 1:3*t-2) * psi_coeff(1:3*t-2, 1:3*t) + ...
            c2 * Bmat(3*t-1, 1:3*t-1) * psi_coeff(1:3*t-1, 1:3*t) + ...
            c3 * Bmat(3*t, 1:3*t) * psi_coeff(1:3*t, 1:3*t);        
        theta_coeff(t, 3*t-2) = theta_coeff(t, 3*t-2) + c1;
        theta_coeff(t, 3*t-1) = theta_coeff(t, 3*t-1) + c2;
        theta_coeff(t, 3*t) = theta_coeff(t, 3*t) + c3;

        %combine previous beta's and \tilde{B} elements to get Onsager's
        %coefficients
        Onsager_correct(t, 1:t) = c1 * Bmat(3*t-2, 1:3*t-2) * beta_coeff(1:3*t-2, 1:t) + ...
            c2 * Bmat(3*t-1, 1:3*t-1) * beta_coeff(1:3*t-1, 1:t) + ...
            c3 * Bmat(3*t, 1:3*t) * beta_coeff(1:3*t, 1:t);
        
        %variance of the linear combination of noises with covariance
        %\tilde{\Sigma}
        sigma2SE(t) = theta_coeff(t, 1:3*t) * tildesigmaSE(1:3*t, 1:3*t) * theta_coeff(t, 1:3*t)';
        
        %effective SNR of a signal drawn from prior w.r.t. a Gaussian noise
        %with the above variance
        SNRSE(t) = muSE(t)/sqrt(sigma2SE(t));
        
        %For Rademacher
        %fun = @(x) 1/sqrt(2*pi) * exp(-x.^2/2) .* tanh(SNRSE(t)^2 + SNRSE(t) * x);
        %fun2 = @(x) 1/sqrt(2*pi) * exp(-x.^2/2) .* tanh(-SNRSE(t)^2 + SNRSE(t) * x);
        
        %For sparse Rademacher
        fun = @(z) 1/sqrt(2*pi) * exp(-z.^2/2) .* tanh(SNRSE(t)^2/rho + SNRSE(t) * z/sqrt(rho)) ./ (rho+...
         (1-rho) .* sech(SNRSE(t)^2/rho + SNRSE(t) * z/sqrt(rho)) .* exp(SNRSE(t)^2/2/rho));
        fun2 = @(z) 1/sqrt(2*pi) * exp(-z.^2/2) .* tanh(-SNRSE(t)^2/rho + SNRSE(t) * z/sqrt(rho)) ./ (rho+...
         (1-rho) .* sech(-SNRSE(t)^2/rho + SNRSE(t) * z/sqrt(rho)) .* exp(SNRSE(t)^2/2/rho));

        int1 = integral(fun, -Inf, Inf);
        int2 = integral(fun2, -Inf, Inf);
        
        %some updates
        tildemuSE(3*t+1) = alpha/2 * (int1 - int2)*rho; %Added rho for sparse case, same below
        DeltaMAT(3*t+1, 1) = epsl/2 * (int1 - int2)*rho;
        DeltaMAT(1, 3*t+1) = epsl/2 * (int1 - int2)*rho;
        %we need to fill many holes in DeltaMAT, i.e. \tilde{Delta}, yet:

        for t1 = 1 : t-1
        %covariance between two iterates at steps that are multiples of 3
            cov_tt1 = theta_coeff(t, 1:3*t) * tildesigmaSE(1:3*t, 1:3*t1) * theta_coeff(t1, 1:3*t1)';
            Sigma1 = [sigma2SE(t), cov_tt1; cov_tt1, sigma2SE(t1)];
            invS = inv(Sigma1);
            
            if det(Sigma1) <= 0
                fprintf('Calculation of DeltaMAT(%d, %d), det of Sigma1 =%f\n', 3*t+1, 3*t1+1, det(Sigma1));
                flgdet = 1;
                break;
            end
            %Hereby we compute bivariate Gaussian integrals to evaluate
            %some elements of DeltaMAT
            %For Rademacher
            %fun1 = @(x, y) 1/(2*pi*sqrt(det(Sigma1))) * exp(-1/2 * (( invS(1, 1) * x.^2 + invS(2, 2) * y.^2 + 2*invS(1, 2)*x.*y))) ...
                %.* tanh(muSE(t)^2/sigma2SE(t) + muSE(t)/sigma2SE(t) * x) ...
                %.* tanh(muSE(t1)^2/sigma2SE(t1) + muSE(t1)/sigma2SE(t1) * y);
            %fun2 = @(x, y) 1/(2*pi*sqrt(det(Sigma1))) * exp(-1/2 * (( invS(1, 1) * x.^2 + invS(2, 2) * y.^2 + 2*invS(1, 2)*x.*y))) ...
                %.* tanh(-muSE(t)^2/sigma2SE(t) + muSE(t)/sigma2SE(t) * x) ...
                %.* tanh(-muSE(t1)^2/sigma2SE(t1) + muSE(t1)/sigma2SE(t1) * y);

            %int1 = integral2(fun1, -Inf, Inf, -Inf, Inf);
            %int2 = integral2(fun2, -Inf, Inf, -Inf, Inf);

            %For sparse Rademacher
            fun0 = @(x, y) rho*1/(2*pi*sqrt(det(Sigma1))) * exp(-1/2 * (( invS(1, 1) * x.^2 + invS(2, 2) * y.^2 + 2*invS(1, 2)*x.*y))) ...
                .* tanh(muSE(t)/sigma2SE(t)/sqrt(rho) * x) ...
                ./ (rho+(1-rho).* sech(muSE(t)/sigma2SE(t)/sqrt(rho) * x) .*exp(muSE(t)^2/sigma2SE(t)/2/rho))...
                .* tanh( muSE(t1)/sigma2SE(t1)/sqrt(rho) * y)...
                ./ (rho+(1-rho).* sech( muSE(t1)/sigma2SE(t1)/sqrt(rho) *y).*exp(muSE(t1)^2/sigma2SE(t1)/2/rho));

            fun1 = @(x, y) rho*1/(2*pi*sqrt(det(Sigma1))) * exp(-1/2 * (( invS(1, 1) * x.^2 + invS(2, 2) * y.^2 + 2*invS(1, 2)*x.*y))) ...
               .* tanh(-muSE(t)^2/sigma2SE(t)/rho + muSE(t)/sigma2SE(t)/sqrt(rho) * x) ...
                ./ (rho+(1-rho).* sech(-muSE(t)^2/sigma2SE(t)/rho + muSE(t)/sigma2SE(t)/sqrt(rho) * x) .*exp(muSE(t)^2/sigma2SE(t)/2/rho))...
                .* tanh(-muSE(t1)^2/sigma2SE(t1)/rho + muSE(t1)/sigma2SE(t1)/sqrt(rho) * y)...
                ./ (rho+(1-rho).* sech(-muSE(t1)^2/sigma2SE(t1)/rho + muSE(t1)/sigma2SE(t1)/sqrt(rho) *y).*exp(muSE(t1)^2/sigma2SE(t1)/2/rho));
            
            fun2 = @(x, y) rho*1/(2*pi*sqrt(det(Sigma1))) * exp(-1/2 * (( invS(1, 1) * x.^2 + invS(2, 2) * y.^2 + 2*invS(1, 2)*x.*y))) ...
                .* tanh(muSE(t)^2/sigma2SE(t)/rho + muSE(t)/sigma2SE(t)/sqrt(rho) * x) ...
                ./ (rho+(1-rho).* sech(muSE(t)^2/sigma2SE(t)/rho + muSE(t)/sigma2SE(t)/sqrt(rho) * x) .*exp(muSE(t)^2/sigma2SE(t)/2/rho))...
                .* tanh(muSE(t1)^2/sigma2SE(t1)/rho + muSE(t1)/sigma2SE(t1)/sqrt(rho) * y)...
                ./ (rho+(1-rho).* sech(muSE(t1)^2/sigma2SE(t1)/rho + muSE(t1)/sigma2SE(t1)/sqrt(rho) *y).*exp(muSE(t1)^2/sigma2SE(t1)/2/rho));

            int0 = integral2(fun0, -Inf, Inf, -Inf, Inf);
            int1 = integral2(fun1, -Inf, Inf, -Inf, Inf);
            int2 = integral2(fun2, -Inf, Inf, -Inf, Inf);
            
            DeltaMAT(3*t+1, 3*t1+1) = (int1 + int2)/2*rho+(1-rho)*int0;
            DeltaMAT(3*t1+1, 3*t+1) = (int1 + int2)/2*rho+(1-rho)*int0;
        end        
        
        fun0= @(z) rho*1/sqrt(2*pi) * exp(-z.^2/2) .* tanh(SNRSE(t) * z/sqrt(rho)).^2 ./ (rho...
            +(1-rho) .* sech(SNRSE(t) * z/sqrt(rho)).*exp(SNRSE(t)^2/2/rho)).^2;
        
        fun = @(z) rho*1/sqrt(2*pi) * exp(-z.^2/2) .* tanh(SNRSE(t)^2/rho + SNRSE(t) * z/sqrt(rho)).^2 ./ (rho...
            +(1-rho) .* sech(SNRSE(t)^2/rho + SNRSE(t) * z/sqrt(rho)).*exp(SNRSE(t)^2/2/rho)).^2;
        
        fun2 = @(z) rho*1/sqrt(2*pi) * exp(-z.^2/2) .* tanh(-SNRSE(t)^2/rho + SNRSE(t) * z/sqrt(rho)).^2 ./ (rho+...
         (1-rho) .* sech(-SNRSE(t)^2/rho + SNRSE(t) * z/sqrt(rho)).*exp(SNRSE(t)^2/2/rho)).^2;

        int0 = integral(fun0, -Inf , Inf);
        int1 = integral(fun, -Inf, Inf);
        int2 = integral(fun2, -Inf, Inf);

        %update diagonal elements of DeltaMAT with indices that are multiples of 3
        DeltaMAT(3*t+1, 3*t+1) = rho*(int1 + int2)/2+(1-rho)*int0;
        


        fun3= @(z) 1/sqrt(2*pi) * exp(-z.^2/2) .* 1 ./ (rho...
            +(1-rho) .* sech(SNRSE(t)^2/rho + SNRSE(t) * z/sqrt(rho)).*exp(SNRSE(t)^2/2/rho));
        
        fun4= @(z) 1/sqrt(2*pi) * exp(-z.^2/2) .* 1 ./ (rho...
            +(1-rho) .* sech(SNRSE(t) * z/sqrt(rho)).*exp(SNRSE(t)^2/2/rho));


        int3 = integral(fun3, -Inf, Inf);
        int4 = integral(fun4,-Inf,Inf);



        %update row of PhiMAT
        phi_int = muSE(t)/sigma2SE(t) * (rho*int3+(1-rho)*int4 - rho*(int1 + int2)/2-(1-rho)*int0);
        PhiMAT(3*t+1, 1:3*t) = phi_int * theta_coeff(t, 1:3*t);
        
        %update overlaps and MSE
        scal_all(t+1, j) = tildemuSE(3*t+1)/alpha/sqrt(DeltaMAT(3*t+1, 3*t+1));
        MSE(t+1, j) = (1 - 2 * (tildemuSE(3*t+1)/alpha)^2 + DeltaMAT(3*t+1, 3*t+1)^2)/2;
        

        fprintf('Iteration %d, scal=%f, MSE=%f\n', t+1, scal_all(t+1, j)^2, MSE(t+1, j));
        
        %stopping criterion
        if (abs(scal_all(t+1, j)^2-scal_all(t, j)^2) < tolinc) || flgdet == 1
            for kk = t+2 : niter
                scal_all(kk, j) = scal_all(t+1, j); 
                MSE(kk, j) = MSE(t+1, j);
            end
            
            break;
        end

        
        
        
        %update of the remaining elements of DeltaMAT, which we may
        %consider as the auxiliary part of the AMP 
        for i = 1 : 3*t
            
            if mod(i, 3) ~= 1
                %covariance between two guassian noises, one at time 3t,
                %and one at time i-1 in the auxliary AMP
                cov_tt1 = theta_coeff(t, 1:3*t) * tildesigmaSE(i-1, 1:3*t)';
                Sigma1 = [sigma2SE(t), cov_tt1; cov_tt1, tildesigmaSE(i-1, i-1)];
                
                if det(Sigma1) < tolint
                    fprintf('Calculation of DeltaMAT(%d, %d), det of Sigma1 =%f\n', 3*t+1, i, det(Sigma1));
                    flgdet = 1;
                    break;
                end
                
                if det(Sigma1) < tolintpos %Irrilevante per ora...
                    fprintf('Tolintpos triggered. Calculation of DeltaMAT(%d, %d), det of Sigma1 =%f\n', 3*t+1, i, det(Sigma1));
                    fun = @(x) 1/sqrt(2*pi) * exp(-x.^2/2) .* tanh(SNRSE(t)^2 + SNRSE(t) * x) .* ...
                        sqrt(tildesigmaSE(i-1, i-1)) .* x;
                    int1 = integral(fun, -Inf, Inf);
                    fun2 = @(x) 1/sqrt(2*pi) * exp(-x.^2/2) .* tanh(-SNRSE(t)^2 + SNRSE(t) * x) .* ...
                        sqrt(tildesigmaSE(i-1, i-1)) .* x;
                    int2 = integral(fun2, -Inf, Inf);
                    DeltaMAT(3*t+1, i) = (int1+int2)/2 + tildemuSE(3*t+1) * tildemuSE(i-1)/alpha + Bmat(i-1, 1:i-1) * DeltaMAT(3*t+1, 1:i-1)';
                    DeltaMAT(i, 3*t+1) = DeltaMAT(3*t+1, i);
                    break;
                    
                else
                
                    invS = inv(Sigma1);

                    fun0 = @(x, y) sqrt(rho)*1/(2*pi*sqrt(det(Sigma1))) * exp(-1/2 * (( invS(1, 1) * x.^2 + invS(2, 2) * y.^2 + 2*invS(1, 2)*x.*y))) ...
                    .* tanh(muSE(t)/sigma2SE(t)/sqrt(rho) * x) ...
                    ./ (rho+(1-rho).* sech(muSE(t)/sigma2SE(t)/sqrt(rho) * x) .* exp(muSE(t)^2/sigma2SE(t)/2/rho))...
                    .* y;

                    fun1 = @(x, y) sqrt(rho)*1/(2*pi*sqrt(det(Sigma1))) * exp(-1/2 * (( invS(1, 1) * x.^2 + invS(2, 2) * y.^2 + 2*invS(1, 2)*x.*y))) ...
                    .* tanh(-muSE(t)^2/sigma2SE(t)/rho + muSE(t)/sigma2SE(t)/sqrt(rho) * x) ...
                    ./ (rho+(1-rho).* sech(-muSE(t)^2/sigma2SE(t)/rho + muSE(t)/sigma2SE(t)/sqrt(rho) * x) .*exp(muSE(t)^2/sigma2SE(t)/2/rho))...
                    .* y;

                    fun2 = @(x, y) sqrt(rho)*1/(2*pi*sqrt(det(Sigma1))) * exp(-1/2 * (( invS(1, 1) * x.^2 + invS(2, 2) * y.^2 + 2*invS(1, 2)*x.*y))) ...
                    .* tanh(muSE(t)^2/sigma2SE(t)/rho + muSE(t)/sigma2SE(t)/sqrt(rho) * x) ...
                    ./ (rho+(1-rho).* sech(muSE(t)^2/sigma2SE(t)/rho + muSE(t)/sigma2SE(t)/sqrt(rho) * x) .*exp(muSE(t)^2/sigma2SE(t)/2/rho))...
                    .* y;

                    int0 = integral2(fun0, -Inf, Inf, -Inf, Inf);
                    int1 = integral2(fun1, -Inf, Inf, -Inf, Inf);
                    int2 = integral2(fun2, -Inf, Inf, -Inf, Inf);
                    %the above terms arise when computing a generic element
                    %of DeltaMAT.

                    %updates of the remaining elements of DeltaMAT!
                    DeltaMAT(3*t+1, i) = rho*(int1+int2)/2+(1-rho).*int0 + tildemuSE(3*t+1) * tildemuSE(i-1)/alpha + Bmat(i-1, 1:i-1) * DeltaMAT(3*t+1, 1:i-1)';
                    DeltaMAT(i, 3*t+1) = DeltaMAT(3*t+1, i);
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
        
        
        PhiTMP = PhiMAT(1:3*t+1, 1:3*t+1);
        DeltaTMP = DeltaMAT(1:3*t+1, 1:3*t+1);
        
        %Adding a row and a column to Bmat for Onsagers int he next
        %iteration
        BmatNEW = zeros(3*t+1, 3*t+1);
        for jj = 0 : 3*t
            BmatNEW = BmatNEW + freecum(jj+1) * PhiTMP^(jj);
        end
    
        for i1 = 1 : 3*t
            for j1 = 1 : 3*t
                if abs(BmatNEW(i1, j1) - Bmat(i1, j1)) > tols
                    fprintf('BmatNEW(%d, %d) not matching!\n', i1, j1);
                    return;
                end
            end
        end
        Bmat = BmatNEW;
    

        %updating \tilde{Sigma}
        SigmaTMP = zeros(3*t+1, 3*t+1);
        for jj = 0 : 2*(3*t)
        
            ThetaMAT = zeros(3*t+1, 3*t+1);
            for j2 = 0 : jj
                ThetaMAT = ThetaMAT + PhiTMP^j2 * DeltaTMP * (PhiTMP^(jj-j2))';
            end
        
            SigmaTMP = SigmaTMP + freecum(jj+2) * ThetaMAT;
        
        end
    
        for i1 = 1 : 3*t
            for j1 = 1 : 3*t
                if abs(SigmaTMP(i1, j1) - tildesigmaSE(i1, j1)) > tols
                    fprintf('SigmaTMP(%d, %d) not matching!a\n', i1, j1);
                    return;
                end
            end
        end
    
        tildesigmaSE(1:3*t+1, 1:3*t+1) = SigmaTMP;
        
        for ell = 1 : 2
            %updating \tilde{\mu}
            tildemuSE(3*t+1+ell) = alpha * tildemuSE(3*t+ell) + Bmat(3*t+ell, 1:3*t+ell) * tildemuSE(1:3*t+ell);
            %updating \tilde{\Phi}
            PhiMAT(3*t+ell+1, 3*t+ell) = 1;

            for j1 = 1 : 3*t+ell-1
                PhiMAT(3*t+ell+1, j1) = Bmat(3*t+ell, 1:3*t+ell) * PhiMAT(1:3*t+ell, j1);
            end
            
            %final completion of DeltaMAT----------------------------------
            gamma_par = zeros(3*t+ell+1, 1);
            for t1 = 1 : t
                cov_tt1 = theta_coeff(t1, 1:3*t1) * tildesigmaSE(3*t+ell, 1:3*t1)';
                Sigma1 = [sigma2SE(t1), cov_tt1; cov_tt1, tildesigmaSE(3*t+ell, 3*t+ell)];
            
                invS = inv(Sigma1);
                
                if det(Sigma1) <= 0
                    fprintf('Calculation of gamma_par(%d), det of Sigma1 =%f\n', 3*t1+1, det(Sigma1));
                    flgdet = 1;
                    break;
                end
            
                fun0 = @(x, y) sqrt(rho)*1/(2*pi*sqrt(det(Sigma1))) * exp(-1/2 * (( invS(1, 1) * x.^2 + invS(2, 2) * y.^2 + 2*invS(1, 2)*x.*y))) ...
                    .* tanh( muSE(t1)/sigma2SE(t1)/sqrt(rho) * x) ...
                    ./ (rho+(1-rho).* sech(muSE(t1)/sigma2SE(t1)/sqrt(rho) * x) .*exp(muSE(t1)^2/sigma2SE(t1)/2/rho))...
                    .* y;

                fun1 = @(x, y) sqrt(rho)*1/(2*pi*sqrt(det(Sigma1))) * exp(-1/2 * (( invS(1, 1) * x.^2 + invS(2, 2) * y.^2 + 2*invS(1, 2)*x.*y))) ...
                    .* tanh(-muSE(t1)^2/sigma2SE(t1)/rho + muSE(t1)/sigma2SE(t1)/sqrt(rho) * x) ...
                    ./ (rho+(1-rho).* sech(-muSE(t1)^2/sigma2SE(t1)/rho + muSE(t1)/sigma2SE(t1)/sqrt(rho) * x) .*exp(muSE(t1)^2/sigma2SE(t1)/2/rho))...
                    .* y;

                fun2 = @(x, y) sqrt(rho)*1/(2*pi*sqrt(det(Sigma1))) * exp(-1/2 * (( invS(1, 1) * x.^2 + invS(2, 2) * y.^2 + 2*invS(1, 2)*x.*y))) ...
                    .* tanh(muSE(t1)^2/sigma2SE(t1)/rho + muSE(t1)/sigma2SE(t1)/sqrt(rho) * x) ...
                    ./ (rho+(1-rho).* sech(muSE(t1)^2/sigma2SE(t1)/rho + muSE(t1)/sigma2SE(t1)/sqrt(rho) * x) .*exp(muSE(t1)^2/sigma2SE(t1)/2/rho))...
                    .* y;

                
                
                %fun2 = @(x, y) 1/(2*pi*sqrt(det(Sigma1))) * exp(-1/2 * (( invS(1, 1) * x.^2 + invS(2, 2) * y.^2 + 2*invS(1, 2)*x.*y))) ...
                    %.* tanh(-muSE(t1)^2/sigma2SE(t1) + muSE(t1)/sigma2SE(t1) * x) ...
                    %.* y;

                int0 = integral2(fun0, -Inf, Inf, -Inf, Inf);
                int1 = integral2(fun1, -Inf, Inf, -Inf, Inf);
                int2 = integral2(fun2, -Inf, Inf, -Inf, Inf);
                gamma_par(3*t1+1) = rho*(int1 + int2)/2+(1-rho)*int0;
            end
            
            if flgdet == 1
                break;
            end

        
            for i = 1 : 3*t+ell+1
                if mod(i, 3) ~= 1
                    gamma_par(i) = tildesigmaSE(3*t+ell, i-1) + Bmat(i-1, 1:i-1) * gamma_par(1:i-1);
                end
            end
        
            for j1 = 1 : 3*t+ell+1
                DeltaMAT(3*t+ell+1, j1) = gamma_par(j1) + tildemuSE(3*t+ell) * tildemuSE(j1)/alpha  + ...
                    Bmat(3*t+ell, 1:3*t+ell) * DeltaMAT(1:3*t+ell, j1);
                DeltaMAT(j1, 3*t+ell+1) = gamma_par(j1) + tildemuSE(3*t+ell) * tildemuSE(j1)/alpha  + ...
                    Bmat(3*t+ell, 1:3*t+ell) * DeltaMAT(1:3*t+ell, j1);
            end
            %--------------------------------------------------------------

            %final completion of \tilde{B} and \tilde{\Sigma}--------------
            PhiTMP = PhiMAT(1:3*t+ell+1, 1:3*t+ell+1);
            DeltaTMP = DeltaMAT(1:3*t+ell+1, 1:3*t+ell+1);
            BmatNEW = zeros(3*t+ell+1, 3*t+ell+1);

            for jj = 0 : 3*t+ell
                BmatNEW = BmatNEW + freecum(jj+1) * PhiTMP^(jj);
            end

            for i1 = 1 : 3*t+ell
                for j1 = 1 : 3*t+ell
                    if abs(BmatNEW(i1, j1) - Bmat(i1, j1)) > tols
                        fprintf('BmatNEW(%d, %d) not matching!\n', i1, j1);
                        return;
                    end
                end
            end

            Bmat = BmatNEW;

            SigmaTMP = zeros(3*t+ell+1, 3*t+ell+1);

            for jj = 0 : 2*(3*t+ell)

                ThetaMAT = zeros(3*t+ell+1, 3*t+ell+1);
                for j2 = 0 : jj
                    ThetaMAT = ThetaMAT + PhiTMP^j2 * DeltaTMP * (PhiTMP^(jj-j2))';
                end

                SigmaTMP = SigmaTMP + freecum(jj+2) * ThetaMAT;
            end

            for i1 = 1 : 3*t+ell
                for j1 = 1 : 3*t+ell
                    if abs(SigmaTMP(i1, j1) - tildesigmaSE(i1, j1)) > tols
                        fprintf('SigmaTMP(%d, %d) not matching!\n', i1, j1);
                        return;
                    end
                end
            end

            tildesigmaSE(1:3*t+ell+1, 1:3*t+ell+1) = SigmaTMP;
            %--------------------------------------------------------------

            %--------------------------------------------------------------
            psi_coeff(3*t+ell+1, 3*t+ell) = 1; %updating alpha
            varphi_coeff(3*t+ell+1) = tildemuSE(3*t+ell); %updating gamma
            beta_coeff(3*t+ell+1,1:t+1) = Bmat(3*t+ell, 3*(1:t+1)-2); %updating beta
            
            for i = 1 : 3*t+ell
                if mod(i, 3) ~= 1
                    for j1 = 1 : i-1
                        psi_coeff(3*t+ell+1, j1) = psi_coeff(3*t+ell+1, j1) + Bmat(3*t+ell, i) * psi_coeff(i, j1);
                    end
                    for j1 = 1 : ceil((i-1)/3)
                        beta_coeff(3*t+ell+1, j1) = beta_coeff(3*t+ell+1, j1) + Bmat(3*t+ell, i) * beta_coeff(i, j1);
                    end
                    varphi_coeff(3*t+ell+1) = varphi_coeff(3*t+ell+1) + Bmat(3*t+ell, i) * varphi_coeff(i);
                end
            end
            %--------------------------------------------------------------
            
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

save SE_quartic_sparse_u0_rho0dot5.mat Onsager_correct muSE sigma2SE MSE scal_all alphagrid;



toc