# Structured-PCA-
The code refers to the paper https://arxiv.org/abs/2210.01237. In this paper we study the paradigmatic spiked matrix model of principal 
components analysis, where the rank-one signal is corrupted by additive noise. While the noise is typically taken from a Wigner matrix with 
independent entries, here the potential acting on the eigenvalues has a quadratic plus a quartic component. The quartic term induces strong 
correlations between the matrix elements, which makes the setting relevant for applications but analytically challenging. Our work provides 
the first characterization of the Bayes-optimal limits for inference in this model with structured noise. If the signal prior is rotational-invariant,
then we show that a spectral estimator is optimal. In contrast, for more general priors, the existing approximate message passing algorithm 
(AMP) falls short of achieving the information-theoretic limits, and we provide a justification for this sub-optimality. Finally, by generalizing 
the theory of Thouless-Anderson-Palmer equations, we cure the issue by proposing a novel AMP which matches the theoretical limits. Our 
information-theoretic analysis is based on the replica method, a powerful heuristic from statistical mechanics; instead, the novel AMP comes 
with a rigorous state evolution analysis tracking its performance in the high-dimensional limit. Even if we focus on a specific noise distribution, 
our methodology can be generalized to a wide class of trace ensembles, at the cost of more involved expressions.

The arXived version of the paper has recently undergone some changes. Please, be sure you are looking at the up to date version.
