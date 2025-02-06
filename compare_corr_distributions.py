"""
Compare distributions of correlations between behavioral variable(s) and
MEG features for two groups of subjects, calculate effect sizes (i.e.
normalized differences in z-scores of correlations) and their confidence
intervals.

Internally, the work horse is Bonnie's `bootes_multivariate()` script
"""
import pandas as pd
from collections import namedtuple

import pls_analysis as pa 
from bootes_multivariate import bootes_multivariate

"""
Definition of a named tuple with results to be returned.
"""
CCD_Result = namedtuple('CCD_Result', ['effect_sizes', 'hedges_g', 
                        'CI_lower', 'CI_upper', 
                        'bootstrap_means_A', 'bootstrap_means_B'])

def compare_corr_distributions(ss, lstSID, subj2group):
    """
    Compare distributions of correlations between behavioral variable(s) and
    MEG features for two groups of subjects.

    NOTE:
        For small group sizes there will be bootsrap realizations when only
        one subject is picked up for the whole group sample. In this case
        the behavioral variable value will be constant, and correlation
        calc function will return a warning `ConstantInputWarning: An input array
        is constant; the correlation coefficient is not defined.`
        This problem should not affect reasonably large samples though.

    Args:
        ss (object): reference to this app object
        lstSID(list of str): list of subject IDs
        subj2group(dict): mapping subject ID -> group # 
        
    Returns:
        ccd_result(CCD_Result): a named tuple

    The `ccd_result` tuple has the following fields.

    effect_sizes: pd.DataFrame
        DataFrame of effect sizes (Cohen's d) for each dependent-covariate variable pairing.
        This is `nfeatures x 1` in our case; `nfeatures = nROI x ntime`
    hedges_g: pd.DataFrame
        DataFrame of Hedges' g for each dependent-covariate variable pairing.
        `nfeatures x 1`
    CI_lower: pd.DataFrame
        DataFrame of lower confidence interval for each dependent-covariate variable pairing. 
        `nfeatures x 1`
    CI_upper: pd.DataFrame
        DataFrame of upper confidence interval for each dependent-covariate variable pairing.
        `nfeatures x 1`
    bootstrap_means_A: pd.DataFrame
        DataFrame of bootstrap means for Condition A, for each dependent-covariate variable pairing.
        `nfeatures x 1`
    bootstrap_means_B: pd.DataFrame
        DataFrame of bootstrap means for Condition B, for each dependent-covariate variable pairing.
        `nfeatures x 1`

    """
    STEP = 'pls_analysis'
    subjects_info_csv = ss.data_host.get_subjects_info_csv()
    cfg = ss.args[STEP]
    id_col = cfg['sID_colname']
    sc_col = cfg['santa_clara_colname']

    # Get Santa-Clara scores for all participants
    df = pd.read_csv(subjects_info_csv, usecols=[id_col, sc_col],
                dtype = {id_col:'string',sc_col:'int64'})
    df = df[df[id_col].isin(lstSID)]

    # This was already checked when constructing subj2group dictionary
    # - but still:
    if len(df) != len(lstSID):
        print(f'len(df)={len(df)}, len(lstSID)={len(lstSID)}')
        raise ValueError(f'lstSID contains subject IDs not listed in {subjects_info_csv}')

    df_sc0, df_sc1 = get_group_dfs(df, id_col, sc_col, subj2group)

    # lst_dmat: a list of arrays of shape (nsubj, nfeatures) for each group
    # lst_sid: a list lists - sIDs of subjects included for each group
    lst_dmat, lst_sid = pa.collect_groups_for_event(ss, STEP, subj2group)

    if not all([d.shape[0] for d in lst_dmat]):
        raise ValueError('One of the groups is empty; please add more subjects')

    ngroups = len(lst_dmat)

    if ngroups != 2:
        raise ValueError('Comparison of only two groups is supported')

    lst_df_meg = [pd.DataFrame(a) for a in lst_dmat]
    nboot = cfg['compare_corrs']['num_boot']
    method = cfg['compare_corrs']['method']
    kwargs = cfg['compare_corrs']['kwargs']

    # Call Bonnie's script and return the results
    """
    # QQQQ make the small just for testing
    lst_df_meg = [df.iloc[:, :10] for df in lst_df_meg.copy()]
    # QQQQ--------------------------
    """
    res = bootes_multivariate(*lst_df_meg, df_sc0, df_sc1, k=nboot,
                                method = method, **kwargs)

    # Convert all dataframes to arrays and return
    # Note that DFs have shapes (nfeatures, 1), so we
    # are also removing an extra dimension
    return CCD_Result(*[df.to_numpy()[:,0] for df in res])

def get_group_dfs(df, id_col, sc_col, subj2group):
    """
    Split a dataframe in two - one for each group, and return dataframes
    for each group.

    Args:
        df(DataFrame): input dataframe with columns ID and score columns
        id_col(str): name of the ID column
        sc_col(str): name of the score column
        subj2group(dict): mapping subject ID -> group # 

    Returns:
        lst_df (tuple of DFs): dataframes for group0 and group1

    """
    # Assign groups to DataFrame
    df['Group'] = df[id_col].map(subj2group)

    # Split into two DataFrames
    df_group_0 = df[df['Group'] == 0][[sc_col]].reset_index(drop=True)
    df_group_1 = df[df['Group'] == 1][[sc_col]].reset_index(drop=True)

    return df_group_0, df_group_1

