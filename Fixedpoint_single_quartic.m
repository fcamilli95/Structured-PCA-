clear;
close all;
clc;



% This script implements the fixed point recursion for the single step memory AMP algorithm
% desgined first by Opper and Winther and then studied by Zhou Fan in
% https://arxiv.org/abs/2008.11892v5. The fixed point equations can be read
% in eq. (3.14) in the mentioned paper.
% Hereby we write it for the quartic potential, but with a small effort it can be converted to treat the 
% sestic potential too. 


niter = 2000;

%here \mu=0, so the potential is "maximally quartic"
u = 0;

gamma = (8-9*u+sqrt(64-144*u+108*u.^2-27*u.^3))/27;

%regularization to avoid dividing by 0.
if u == 1
    a2 = 1;
else
    a2 = (sqrt(u.^2+12*gamma)-u)./(6*gamma);
end

tol = 1e-11;




rho = 0.3;

%choose which alpha grid you want, or a single value.
%alphagrid = 1.52 : 0.06 : 4.28;
%alphagrid = 0.9;
alphagrid = 3;


%Here we introduce the parameters needed for the recursion of https://arxiv.org/pdf/2008.11892.pdf

Sigma_all = zeros(length(alphagrid), niter);
Delta_all = zeros(length(alphagrid), niter);
overlap_all = zeros(length(alphagrid), niter);
MSE_all = zeros(length(alphagrid), niter);


for j1 = 1 : length(alphagrid)
    alpha = alphagrid(j1);
    %initializations
    % proper initializations to converge to the sought fixed point, may be
    % changed a little.
    if alpha <= 1.7
        Sigma_all(j1, 1) = 0.62;
        Delta_all(j1, 1) = 0.7;
    else   
        Sigma_all(j1, 1) = 0.9999;
        Delta_all(j1, 1) = 0.999;
    end
    MSE_all(j1, 1) = (1-Delta_all(j1, 1)^2)/2;
    overlap_all(j1, 1) = sqrt(Delta_all(j1, 1));

%alpha = 1.2;

for i = 1 : niter%iterations
    argmmse = alpha^2 * Delta_all(j1, i)^2/ Sigma_all(j1, i);
    

    fun = @(x) 1/sqrt(2*pi)*exp(-x.^2/2) .* tanh(argmmse/rho+sqrt(argmmse/rho)*x)./ ...
        ( (1-rho) * exp(argmmse/2/rho) .* sech(argmmse/rho+sqrt(argmmse/rho)*x) + rho);    
    mmseval = 1-rho* integral(fun, -Inf, Inf);
    
    Delta_all(j1, i+1) = 1-mmseval;
    
    %inverting the Stieltjes transform numerically------------------------
    zval = alpha * Delta_all(j1, i) * (1-Delta_all(j1, i))/Sigma_all(j1, i);
    
    zmin = 2*sqrt(a2)+0.00001;
    zmax = 10000;

    fun = @(x) 1./(zmin-x) .* (u+gamma*(2*a2+x.^2)).*sqrt(4*a2-x.^2)/(2*pi);
    valmin = integral(fun,-2*sqrt(a2),2*sqrt(a2));
    
    if valmin < zval
        fprintf('Error in setting zmin!\n');
%         zmax = zmin;
%         zc = zmin;
        break;
    end
    
    fun = @(x) 1./(zmax-x) .* (u+gamma*(2*a2+x.^2)).*sqrt(4*a2-x.^2)/(2*pi);
    valmax = integral(fun,-2*sqrt(a2),2*sqrt(a2));
    
    if valmax > zval
        fprintf('Error in setting zmax!\n');
        break;
    end
    
    %inversion by dichotomy
    while (zmax-zmin) > tol
        zc = (zmax+zmin)/2;
%    zc
        fun = @(x) 1./(zc-x) .* (u+gamma*(2*a2+x.^2)).*sqrt(4*a2-x.^2)/(2*pi);
        valint = integral(fun,-2*sqrt(a2),2*sqrt(a2));
    
        if valint > zval
            zmin  = zc;
        else
            zmax = zc;
        end
    end
    
    fun = @(x) 1./((zc-x).^2) .* (u+gamma*(2*a2+x.^2)).*sqrt(4*a2-x.^2)/(2*pi);    
    valint = -integral(fun,-2*sqrt(a2),2*sqrt(a2));
    
    R1 = 1/valint + 1/zval^2;
    %derivative of the R transform appearing in (3.14) of https://arxiv.org/pdf/2008.11892.pdf
    %---------------------------------------------------------------------
    
    %updates of the values
    Sigma_all(j1, i+1) = R1*Delta_all(j1, i);
    MSE_all(j1, i+1) = (1-Delta_all(j1, i+1)^2)/2; 
    overlap_all(j1, i+1) = sqrt(Delta_all(j1, i+1));

   
    fprintf('Step %d, Sigma=%f, Delta=%f, MSE=%f, overlap=%f\n', ...
        i, Sigma_all(j1, i+1), Delta_all(j1, i+1), MSE_all(j1, i+1), overlap_all(j1, i+1)^2);
    
    %stopping criterion
    if abs(Sigma_all(j1, i+1)-Sigma_all(j1, i)) < 1e-9 || overlap_all(j1, i+1) < 1e-4 || overlap_all(j1, i+1) > 1-1e-7
        for k1 = i+2 : niter
            Delta_all(j1, k1) = Delta_all(j1, i+1);
            Sigma_all(j1, k1) = Sigma_all(j1, i+1);
            MSE_all(j1, k1) = MSE_all(j1, i+1);
            overlap_all(j1, k1) = overlap_all(j1, i+1);
        end
        break;
    end
    
            
    
    
end

end

save FPsparse_rho0dot3_u0.mat Sigma_all Delta_all MSE_all overlap_all alphagrid;

