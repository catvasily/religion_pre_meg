#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr, pointbiserialr, norm


# In[29]:


def bootes_multivariate(
    dA, dB, cA, cB, 
    k=1000, 
    do_fig=False, 
    save_fig=False, 
    save_path='./', 
    save_effect_sizes=False, 
    es_path='./effect_sizes.csv',
    save_g=False,
    g_path='./hedges_g.csv',
    alpha = 0.05,
    save_CI=False,
    CI_lower_path='./CI_lower.csv',
    CI_upper_path='./CI_upper.csv',
    save_bootstrap_means_A=False,
    bootstrap_means_A_path='./bootstrap_means_A.csv',
    save_bootstrap_means_B=False,
    bootstrap_means_B_path='./bootstrap_means_B.csv',
    condition_names=('Condition A', 'Condition B'),
    method='pearson',
    save_dependent=None,
    save_covariate=None
):
    """
    Compare the magnitude of correlations between covariates and dependent variables
    across two conditions, for multivariate inputs (dataframes). 
    The two conditions do not require same number of participants.
    Now supports Pearson, Spearman, and Point-Biserial correlation.
    
    Parameters:
        dA, dB: pd.DataFrame
            Dependent variable dataframes for conditions A and B. 
            Neuroimaging data in subjects x variables. One variable per column and one participant per row. 
            Make sure the input is dataframe (instead of pandas series or numpy array) if it is univariate. 
        cA, cB: pd.DataFrame
            Covariate dataframes for conditions A and B (subjects x variables). 
            Behavioural data in subjects x variables. One variable per column and one participant per row. 
            Make sure the input is dataframe (instead of pandas series or numpy array) if it is univariate. 
        k: int, optional
            Number of bootstrap cycles (default is 1000).
        do_fig: bool, optional
            Whether to generate histograms of bootstrap correlations (default is False).
        save_fig: bool, optional
            Whether to save the figures (default is False).
        save_path: str, optional
            Directory to save the figures if `save_fig` is True (default is './').
        save_effect_sizes: bool, optional
            Whether to save the effect size matrix as a CSV (default is False).
        es_path: str, optional
            File path to save the effect size matrix if `save_effect_sizes` is True (default is './effect_sizes.csv').
        save_g: bool, optional
            Whether to save the Hedges' g matrix as a CSV (default is False).
        g_path: str, optional
            File path to save the Hedges' g matrix if `save_g` is True (default is './hedges_g.csv').
        alpha: float, optional
            Significance level (default is 0.05).
        save_CI: bool, optional
            Whether to save both confidence interval matrixes as a CSV (default is False).
        CI_lower_path: str, optional
            File path to save the lower confidence interval matrix if `save_CI` is True (default is './CI_lower.csv').
        CI_upper_path: str, optional
            File path to save the upper confidence interval matrix if `save_CI` is True (default is './CI_upper.csv').
        save_bootstrap_means_A: bool, optional
            Whether to save the matrix of bootstrap means for Condition A as a CSV (default is False).
        bootstrap_means_A_path: str, optional
            File path to save the bootstrap means for Condition A (default is './bootstrap_means_A.csv').
        save_bootstrap_means_B: bool, optional
            Whether to save the matrix of bootstrap means for Condition B as a CSV (default is False).
        bootstrap_means_B_path: str, optional
            File path to save the bootstrap means for Condition B (default is './bootstrap_means_B.csv').
        condition_names: tuple, optional
            Names of the two conditions for labeling figures (default is ('Condition A', 'Condition B')).
        method: str, optional
            Correlation method to use ('pearson', 'spearman', or 'pointbiserial', default is 'pearson').
        save_dependent: list, optional
            List of dependent variable indices or column names for which to save figures (default is None, meaning all).
        save_covariate: list, optional
            List of covariate variable indices or column names for which to save figures (default is None, meaning all).
    
    Returns:
        effect_sizes: pd.DataFrame
            DataFrame of effect sizes (Cohen's d) for each dependent-covariate variable pairing.
        hedges_g: pd.DataFrame
            DataFrame of Hedges' g for each dependent-covariate variable pairing.
        CI_lower: pd.DataFrame
            DataFrame of lower confidence interval for each dependent-covariate variable pairing. 
        CI_upper: pd.DataFrame
            DataFrame of upper confidence interval for each dependent-covariate variable pairing.
        bootstrap_means_A: pd.DataFrame
            DataFrame of bootstrap means for Condition A, for each dependent-covariate variable pairing.
        bootstrap_means_B: pd.DataFrame
            DataFrame of bootstrap means for Condition B, for each dependent-covariate variable pairing.
    """

    # Validate input dimensions
    if not (len(dA) == len(cA) and len(dB) == len(cB)):
        raise ValueError("Dependent and covariate matrices must have the same number of rows within each condition.")
    
    nA, nB = len(dA), len(dB)  # Separate the sample sizes for the two conditions
    N = nA + nB  # Total sample size
    
    # Handle bootstrap sampling for each condition
    indices_A = np.random.randint(0, nA, size=(k, nA))
    indices_B = np.random.randint(0, nB, size=(k, nB))
    
    # Validate method
    if method not in ['pearson', 'spearman', 'pointbiserial']:
        raise ValueError("Invalid method. Choose 'pearson', 'spearman', or 'pointbiserial'.")

    # Check if point-biserial is used and warn if covariates are not binary
    if method == 'pointbiserial':
        if not (cA.isin([0, 1]).all().all() and cB.isin([0, 1]).all().all()):
            raise ValueError("Point-biserial correlation requires covariates to be binary (0 or 1).")
   
    # Initialize result matrices
    matrices = {
        'effect_sizes': pd.DataFrame(0, index=dA.columns, columns=cA.columns),
        'hedges_g': pd.DataFrame(0, index=dA.columns, columns=cA.columns),
        'CI_lower': pd.DataFrame(0, index=dA.columns, columns=cA.columns),
        'CI_upper': pd.DataFrame(0, index=dA.columns, columns=cA.columns),
        'bootstrap_means_A': pd.DataFrame(0, index=dA.columns, columns=cA.columns),
        'bootstrap_means_B': pd.DataFrame(0, index=dA.columns, columns=cA.columns),
    }

    
    # Warning checks
    if nA < 10 or nB < 10:
        print("Warning: You are using too few participants in one or both conditions.")
    if k > 1000:
        print("Warning: k > 1000 might not be necessary.")

    # Ensure save_path exists if saving figures
    if save_fig and not os.path.exists(save_path):
        os.makedirs(save_path)

    # Handle selective saving of figures
    if save_fig:
        if save_dependent is not None:
            if isinstance(save_dependent[0], int):
                save_dep_vars = dA.columns[save_dependent]
            elif isinstance(save_dependent[0], str):
                save_dep_vars = save_dependent
            else:
                raise ValueError("`save_dependent` should be a list of column indices or names.")
        else:
            save_dep_vars = dA.columns
        
        if save_covariate is not None:
            if isinstance(save_covariate[0], int):
                save_cov_vars = cA.columns[save_covariate]
            elif isinstance(save_covariate[0], str):
                save_cov_vars = save_covariate
            else:
                raise ValueError("`save_covariate` should be a list of column indices or names.")
        else:
            save_cov_vars = cA.columns
    else:
        save_dep_vars, save_cov_vars = [], []  # No saving required

    # Function to calculate correlation based on the chosen method
    def compute_corr(x, y, method):
        if method == 'pearson':
            return pearsonr(x, y)[0]
        elif method == 'spearman':
            return spearmanr(x, y)[0]
        elif method == 'pointbiserial':
            return pointbiserialr(x, y)[0]
        
    # Iterate over dependent and covariate variable pairs
    for dep_var in dA.columns:
        for cov_var in cA.columns:
            # Extract single variable vectors
            dA_var = dA[dep_var]
            dB_var = dB[dep_var]
            cA_var = cA[cov_var]
            cB_var = cB[cov_var]
            
            # Bootstrap correlations
            sA = [compute_corr(cA_var[idx], dA_var[idx], method) for idx in indices_A]
            sB = [compute_corr(cB_var[idx], dB_var[idx], method) for idx in indices_B]

            # Compute means of bootstrap samples for each group
            matrices['bootstrap_means_A'].at[dep_var, cov_var] = np.mean(sA)
            matrices['bootstrap_means_B'].at[dep_var, cov_var] = np.mean(sB)
            
            # Fisher's z-transformation
            zA = np.arctanh(sA)
            zB = np.arctanh(sB)
            
            # Calculate pooled standard deviation
            pooled_std = np.sqrt(((nA - 1) * np.std(zA)**2 + (nB - 1) * np.std(zB)**2) / (nA + nB - 2))
            
            # Compute effect size (Cohen's d)
            # d = abs(np.mean(zB) - np.mean(zA)) / pooled_std
            d = (np.mean(zB) - np.mean(zA)) / pooled_std  # no need to take absolute
            matrices['effect_sizes'].at[dep_var, cov_var] = d
            
            # Calculate Hedges's g
            g = d + 3 * d / (4 * N - 9)
            matrices['hedges_g'].at[dep_var, cov_var] = g
            
            # Calculate Var(g)
            var_g = N/(nA*nB) + (d**2)/(2*(N-3.94))
            
            # Compute Z for given alpha
            Z = norm.ppf(1 - alpha/2) 
            
            # Calculate confidence interval
            matrices['CI_lower'].at[dep_var, cov_var] = g - Z * np.sqrt(var_g)
            matrices['CI_upper'].at[dep_var, cov_var] = g + Z * np.sqrt(var_g)
            
            
            # Optional visualization for this pair
            if do_fig and save_fig and dep_var in save_dep_vars and cov_var in save_cov_vars:
                plt.figure(figsize=(8, 6))
                plt.hist(sA, bins=30, alpha=0.7, label=f'{condition_names[0]} ({dep_var} ~ {cov_var})')
                plt.hist(sB, bins=30, alpha=0.7, label=f'{condition_names[1]} ({dep_var} ~ {cov_var})')
                plt.axvline(np.mean(sA), color='blue', linestyle='dashed', linewidth=1)
                plt.axvline(np.mean(sB), color='orange', linestyle='dashed', linewidth=1)
                plt.legend()
                plt.title(f'Bootstrapped Correlations ({method.capitalize()})\nEffect Size: {effect_sizes.at[dep_var, cov_var]:.3f}')
                plt.xlabel('Correlation Coefficient')
                plt.ylabel('Frequency')
                
                # Save the figure
                fig_name = f'bootstrapped_corrs_{dep_var}_{cov_var}.png'
                plt.savefig(os.path.join(save_path, fig_name), transparent=False, facecolor='white')
                plt.close()
    
    # Save results if requested
    if save_effect_sizes:
        matrices["effect_sizes"].to_csv(es_path)
    if save_g:
        matrices["hedges_g"].to_csv(g_path)
    if save_CI:
        matrices["CI_lower"].to_csv(CI_lower_path)
        matrices["CI_upper"].to_csv(CI_upper_path)
    if save_bootstrap_means_A:
        matrices["bootstrap_means_A"].to_csv(bootstrap_means_A_path)
    if save_bootstrap_means_B:
        matrices["bootstrap_means_B"].to_csv(bootstrap_means_B_path)


    
    return (matrices['effect_sizes'], matrices['hedges_g'], matrices['CI_lower'],
            matrices['CI_upper'], matrices['bootstrap_means_A'], matrices['bootstrap_means_B'])

if __name__ == "__main__":
    # In[30]:


    # usage
    # example data
    n_subjects = 30
    np.random.seed(42)

    # dA = pd.DataFrame(np.random.rand(n_subjects, 3), columns=['tryDep1', 'tryDep2', 'tryDep3'])
    # dB = pd.DataFrame(np.random.rand(n_subjects, 3), columns=['tryDep1', 'tryDep2', 'tryDep3'])
    # cA = pd.DataFrame(np.random.rand(n_subjects, 2), columns=['tryCov1', 'tryCov2'])
    # cB = pd.DataFrame(np.random.rand(n_subjects, 2), columns=['tryCov1', 'tryCov2'])

    # we could also try with different sample sizes
    dA = pd.DataFrame(np.random.rand(30, 3), columns=['tryDep1', 'tryDep2', 'tryDep3'])
    dB = pd.DataFrame(np.random.rand(20, 3), columns=['tryDep1', 'tryDep2', 'tryDep3'])
    # cA = pd.DataFrame(np.random.rand(30, 2), columns=['tryCov1', 'tryCov2'])
    # cB = pd.DataFrame(np.random.rand(20, 2), columns=['tryCov1', 'tryCov2'])
    # Generate binary covariates
    cA = pd.DataFrame(np.random.choice([0, 1], size=(30, 2)), columns=['tryCov1', 'tryCov2'])
    cB = pd.DataFrame(np.random.choice([0, 1], size=(20, 2)), columns=['tryCov1', 'tryCov2'])


    # In[ ]:





    # In[32]:


    effect_sizes, hedges_g, CI_lower, CI_upper, bootstrap_means_A, bootstrap_means_B = bootes_multivariate(
        dA, dB, cA, cB,
        k=500,
        do_fig=False,
        save_fig=False,
        save_path='allResults/bootes_ES/testFigures',
        save_effect_sizes=True,
        es_path='allResults/bootes_ES/testFigures/effect_sizes.csv',
        save_g=True,
        g_path='allResults/bootes_ES/testFigures/hedges_g.csv',
        alpha=0.01,
        save_CI=True,
        CI_lower_path='allResults/bootes_ES/testFigures/CI_lower.csv',
        CI_upper_path='allResults/bootes_ES/testFigures/CI_upper.csv',
        save_bootstrap_means_A=True,
        bootstrap_means_A_path='allResults/bootes_ES/testFigures/bootstrap_means_A.csv',
        save_bootstrap_means_B=True,
        bootstrap_means_B_path='allResults/bootes_ES/testFigures//bootstrap_means_B.csv',
        condition_names=('AQ', 'SPQ'),
        method='pointbiserial',
        save_dependent=['tryDep1', 'tryDep3'],  # Specify dependent variable names
        save_covariate=[1]  # Specify covariate index
    )


    # In[33]:


    effect_sizes


    # In[34]:


    hedges_g


    # In[35]:


    CI_lower


    # In[36]:


    CI_upper


    # In[37]:


    bootstrap_means_A


    # In[38]:


    bootstrap_means_B


    # In[ ]:




