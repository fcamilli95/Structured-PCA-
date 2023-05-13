clear;
close all;
clc;

%See fp_iteration_Rad.m first. The following is a simple re-adaptation of
%that. The fixed point iterations of https://arxiv.org/pdf/2008.11892.pdf
%depend on the ensemble only through the R-transform of the ensemble
%present in (3.14). Hence it is sufficient to take the previous code and
%modify the density of the ensemble.

niter = 2000;

%parameters entering the density.
a2=2.0/3.0;
xi=27.0/80.0;

tol = 1e-11;





%Choose the alpha grid you like, or a single value.
%alphagrid = 2.0 : 0.06 : 4.04;
%alphagrid = 0.8;
alphagrid = 1.9;


%introduce parameters needed for the recursion.
Sigma_all = zeros(length(alphagrid), niter);
Delta_all = zeros(length(alphagrid), niter);
overlap_all = zeros(length(alphagrid), niter);
MSE_all = zeros(length(alphagrid), niter);


for j1 = 1 : length(alphagrid)
    alpha = alphagrid(j1);
    fprintf('alpha=');
    disp(alpha)
    fprintf('\n')
    % proper initializations to converge to the sought fixed point, may be
    % changed a little.
    if alpha <= 1.7
        Sigma_all(j1, 1) = 0.45;
        Delta_all(j1, 1) = 0.5;
    else   
        Sigma_all(j1, 1) = 0.99;
        Delta_all(j1, 1) = 0.999;
    end
    MSE_all(j1, 1) = (1-Delta_all(j1, 1)^2)/2;
    overlap_all(j1, 1) = sqrt(Delta_all(j1, 1));


for i = 1 : niter %iterations
    argmmse = alpha^2 * Delta_all(j1, i)^2/ Sigma_all(j1, i);
    fun = @(x) 1/sqrt(2*pi)*exp(-x.^2/2) .* (1-tanh(argmmse+sqrt(argmmse)*x)).^2;
    fun2 = @(x) 1/sqrt(2*pi)*exp(-x.^2/2) .* (-1-tanh(-argmmse+sqrt(argmmse)*x)).^2;
    
    mmseval = (integral(fun, -Inf, Inf)+integral(fun2, -Inf, Inf))/2;
    
    
    Delta_all(j1, i+1) = 1-mmseval;
    
    %numerical inversion of the Stieltjes transform-----------------------
    zval = alpha * Delta_all(j1, i) * (1-Delta_all(j1, i))/Sigma_all(j1, i);
    zmin = 2*sqrt(a2)+0.00001;
    zmax = 10000;
    
    %notice that the density has changed here:
    fun = @(x) 1./(zmin-x) .* xi.*(6*a2^2+2*a2*x.^2+x.^4).*sqrt(4*a2-x.^2)/(2*pi);
    valmin = integral(fun,-2*sqrt(a2),2*sqrt(a2));
    
    if valmin < zval
        fprintf('Error in setting zmin!\n');
        break;
    end
    
    fun = @(x) 1./(zmax-x) .*xi.*(6*a2^2+2*a2*x.^2+x.^4).*sqrt(4*a2-x.^2)/(2*pi);
    valmax = integral(fun,-2*sqrt(a2),2*sqrt(a2));
    
    if valmax > zval
        fprintf('Error in setting zmax!\n');
        break;
    end
    
    %inversion by dichotomy
    while (zmax-zmin) > tol
        zc = (zmax+zmin)/2;
        fun = @(x) 1./(zc-x) .* xi.*(6*a2^2+2*a2*x.^2+x.^4).*sqrt(4*a2-x.^2)/(2*pi);
        valint = integral(fun,-2*sqrt(a2),2*sqrt(a2));
    
        if valint > zval
            zmin  = zc;
        else
            zmax = zc;
        end
    end
    
    fun = @(x) 1./((zc-x).^2) .* xi.*(6*a2^2+2*a2*x.^2+x.^4).*sqrt(4*a2-x.^2)/(2*pi);    
    valint = -integral(fun,-2*sqrt(a2),2*sqrt(a2));
    
    R1 = 1/valint + 1/zval^2;% derivative of the R-transform appearing in (3.14) of https://arxiv.org/pdf/2008.11892.pdf.
    %---------------------------------------------------------------------
    %updates
    Sigma_all(j1, i+1) = R1*Delta_all(j1, i);
    MSE_all(j1, i+1) = (1-Delta_all(j1, i+1)^2)/2; 
    overlap_all(j1, i+1) = sqrt(Delta_all(j1, i+1));

    
    fprintf('Step %d, Sigma=%f, Delta=%f, MSE=%f, overlap=%f\n', ...
        i, Sigma_all(j1, i+1), Delta_all(j1, i+1), MSE_all(j1, i+1), overlap_all(j1, i+1)^2);
    
    % stopping criterion
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

save FP_full_sestic.mat Sigma_all Delta_all MSE_all overlap_all alphagrid;

