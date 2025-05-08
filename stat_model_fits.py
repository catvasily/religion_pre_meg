"""
Fitting regression models for vectors of MEG responses for each ROI as
functions of group, belief scores or categories, age and gender.
"""
import numpy as np
import pandas as pd
import pickle
import statsmodels.formula.api as smf   # To use R-style (patsy) formulas
import statsmodels.api as sm            # statsmodel main package
import setup_utils as su
from plot_waveforms import adjust_signs

STEP = 'pls_analysis'

def stat_model_fits(ss):
    """
    Fits stat models (1 per ROI) based on features collected for the PLS analyses.
    For tasks 'henv_std_sm_4groups<1,2>img', 'henv_std_sm_santa<1,2>img' the features are 
    time courses averaged over a time interval for the quantity in question
    (i.e. STD of Hilbert envelope). The time interval used is defined by `cfg['pls_interval']`
    setting.

    Args:
        ss(obj): reference to the main application opject

    Returns:
        sm_fit_results(dict): dictionary roi -> <fit results object>. Also saves
            this dictionary to the step output .pkl (pickle) file

    """
    config = ss.args[STEP]

    # Get all basic info for all subjects; create the "subjects" dict
    # as usual, based on the requested subjects list and actually
    # available MEG data. This will fail if some of the requested
    # subjects data is missing
    subjects_info_csv = ss.data_host.get_subjects_info_csv()
    in_dir = ss.data_host.get_step_in_dir(STEP)

    # subjects: {'sid': ['YYMMDD', 'YYMMDD',...]}
    subjects = su.make_subject_dict(in_dir, slist = ss.args['subjects'])
    lstSID = list(subjects.keys())
    task = config['task']

    # Here 2nd arg to get_step_out_file() can be any valid path
    # because it is not used
    out_basename = ss.data_host.get_step_out_file(STEP, config['in_dir'])
    out_file = ss.data_host.get_step_out_dir(STEP) / out_basename

    # Read all non-MEG info
    dfX = get_X_data(config, subjects_info_csv, lstSID)

    # Rename the sid column from whatever it was to 'sid'
    dfX = dfX.rename(columns={config['sID_colname']: 'sid'})

    # Read all MEG info
    dfY = get_Y_data(ss)

    # Reorder dfX based on sid order in dfY
    dfX = dfX.set_index('sid').loc[dfY['sid']].reset_index()

    # Verify that 'sid' columns are identical
    assert dfX['sid'].equals(dfY['sid']), "SID columns do not match!"

    # Concatenate while merging the 'sid' column only once
    df_4sm_fit = pd.concat([dfY, dfX.drop(columns=['sid'])], axis=1)

    # Compile formulae:
    age = config['age_colname']
    gender = config['gender_colname']
    group = config["bias_colname"]
    believer = config['bin_belief_colname']
    sc = config['santa_clara_colname']

    # Rename columns to get easily readable results and avoid Q's
    df_4sm_fit = df_4sm_fit.rename(columns={age:'age', gender:'gender',
                        group:'group',believer:'believer', sc:'sc'})

    if task in ('henv_std_sm_4groups1img','henv_std_sm_4groups2img'):
        # Prepare the right side
        # NOTE: linear terms for believer and group will be added automatically;
        # intercept will be added automatically due to presence of categorical
        # variables
        rside = f' ~ age + C(gender) + C(believer)*C(group)'
    elif task in ('henv_std_sm_santa1img','henv_std_sm_santa2img'):
        # Prepare the right side
        # NOTE: linear terms for santa clara score and group will be added automatically;
        # intercept will be added automatically due to presence of categorical
        # variables
        rside = f' ~ age + C(gender) + sc*C(group)'
    else:
        raise NotImplementedError(f'SM fitting for task {task} is not yet implemented')

    sm_fit_results = {}
    signif_rois = []

    for roi in dfY.columns[1:]:     # Note that dfY's col #0  is 'sid'
        res = smf.ols(formula=f'Q("{roi}")' + rside, data=df_4sm_fit).fit()
        # -------------------------------------------------------------
        # Fit results object properties of interest:
        #   .params[<name>], where for categorical parameters names are:
        #       'C(gender)[T.M]', 'C(believer)[T.1]', 'C(group)[T.1]',
        #       'C(believer)[T.1]:C(group)[T.1]' - returns fitted value
        #       of the parameter
        #   .bse[<name>] - standard error of the parameter
        #   .f_pvalue - P-value for the F statistics
        #   .pvalues[<name>] - 2-tailed P-values for the t-statistics for
        #       fitted parms
        # -------------------------------------------------------------
        # print(res.summary())
        sm_fit_results[roi] = res

        if res.f_pvalue <=0.05:
            signif_rois.append(roi)

    if signif_rois:
        print('Stat-significant results obtained for ROIs:')
        print(signif_rois)
    else:
        print('No stat-significant ROIs found')

    # Save to pickle
    with open(out_file, "wb") as f:
        pickle.dump(sm_fit_results, f)

    return sm_fit_results


def get_X_data(cfg, subjects_info_csv, lstSID):
    """
    Prepare dataframe with independent (exog) subjects data: sid,
    age, gender, group, binary believer score, Santa-Clara score. Note that
    the actual column names are defined in the input .csv file and are
    rather convoluted. Those are changed later outside this function
    when constructing the statistical model.

    Args:
        cfg(dict): this step configuration dictionary
        subjects_info_csv (Pathlike): pathname of .csv file with subjects
            behavioral test results
        lstSID(list of str): list of subject IDs

    Returns:
        df(DataFrame): a Pandas dataframe with relevant subjects data. Bias
            column values are replaced with 0 when bias <0 (SCZ group), and with 1
            when bias > 0 (ASD group)

    """
    id_col = cfg['sID_colname']
    age_col = cfg['age_colname']
    gender_col = cfg['gender_colname']
    bias_col = cfg['bias_colname']
    believer_col = cfg['bin_belief_colname']
    sc_col = cfg['santa_clara_colname']

    # NOTE: using dtype = 'string' produces dtype = 'python[string]', which is not
    # recognized by statsmodels. So we are using str type directly
    df = pd.read_csv(subjects_info_csv,
            usecols=[id_col, age_col, gender_col, bias_col, believer_col, sc_col],
            dtype = {id_col:str, age_col:'int64', gender_col:str, bias_col:'float64',
                     believer_col:'int64', sc_col:'int64'})

    df = df[df[id_col].isin(lstSID)]

    if len(df) != len(lstSID):
        print(f'len(df)={len(df)}, len(lstSID)={len(lstSID)}')
        raise ValueError(f'lstSID contains subject IDs not listed in {subjects_info_csv}')

    # Replace bias column values with group #
    # Now SCZ will be group 0, ASD - group 1
    df[bias_col] = df[bias_col].apply(lambda x: 0 if x < 0 else 1)
    return df

def get_Y_data(ss):
    """
    Get dependent (endog) data for stat model fitting. The data is returned
    for an image ID specified by `ss.args['pls_analysis']['event_id']` for
    1-image tasks, and for a pair of image IDs specified in
    `ss.args['pls_analysis']['img_events']` for 2-image tasks. In the latter
    case the response variable will be a difference Y(img2) - Y(img1) with
    Y(...) being the measurement for a single image.

    Args:
        ss(obj): reference to the main application opject

    Returns:
        df(DataFrame): dataframe with columns: 'sid', ...<roi_names>...
            and rows containing the response values for each ROI for a
            given subject.

    """
    cfg = ss.args[STEP]
    task = cfg['task']

    if '1img' in task:
        return get_Y1_data(ss)

    if '2img' not in task:
        raise ValueError(f'Unrecognized or not implemented task: \'{task}\'')

    eID_org = cfg['event_id']   # Save the original eID setting, just in case
    lst_df = []

    for i in range(2):
        cfg['event_id'] = cfg['img_events'][i]
        lst_df.append(get_Y1_data(ss))

    cfg['event_id'] = eID_org   # Restore the eID in the config

    df2 = lst_df[1].set_index('sid')
    df1 = lst_df[0].set_index('sid')
    dfY = df2 - df1

    return dfY.reset_index()

def get_Y1_data(ss):
    """
    Get dependent (endog) data for stat model fitting. The data is returned
    for an image ID specified by `ss.args['pls_analysis']['event_id']`. 

    Args:
        ss(obj): reference to the main application opject

    Returns:
        df(DataFrame): dataframe with columns: 'sid', ...<roi_names>...
            and rows containing the response values for each ROI for a
            given subject.

    """
    # Import stuff inside the function to avoid circular references
    from pls_analysis import include_in_pls, read_selected_roi_time_courses

    cfg = ss.args[STEP]
    task = cfg['task']

    t0 = ss.args['src_rec']['epochs']['t_range'][0] # epoch's time origin

    if ('erf' in task) or task == ('compare_corrs_2groups1img'):
        SR = ss.args['src_erf']['target_sample_rate']
    elif 'henv' in task:
        SR = ss.args['src_hilbert']['target_sample_rate']

    # NOTE: We'll fail here with SR not defined for tasks not mentioned above,
    # which is the intention

    # Calculate index interval for PLS
    tstart, tend = cfg['pls_interval']
    tstart -= t0
    tend -= t0
    istart = int(SR*tstart)
    iend = int(SR*tend) + 1

    # Calculate index interval for sign adjustments
    # This only will be used for erf tasks
    tstart = ss.args['src_erf']['sign_adjust_interval'][0] - t0
    tend = ss.args['src_erf']['sign_adjust_interval'][1] - t0
    isign_start = int(SR*tstart)
    isign_end = int(SR*tend) + 1

    # If erf_power is True - use square of time courses
    erf_power = False if 'henv' in task else cfg['erf_power']

    if erf_power or ('henv' in task):
        cfg['adjust_signs'] = False

    files = su.files_to_process(ss, STEP)
    epoch0 = None       # Reference epoch for sign adjustment
    include_labels = cfg['include_labels']

    df = None

    for in_file, out_file in files:
        # Choose only files names containing _<eID>
        if not include_in_pls(cfg, in_file):
            continue

        # label_tcs is nepochs x nlabels x ntimes (for epoched data)
        # label_names (nlabels,) vector of ROI names
        # For ERF files nepochs = 2: 1st epoch is the evoked for condition,
        # 2nd epoch is STD
        label_tcs, label_names = read_selected_roi_time_courses(in_file,
                        include_labels = include_labels)[:2]

        if df is None:
            # Create an output dataframe with columns 'sid', ROI-names
            columns = ['sid'] + list(label_names)   # list() because label_names is a tuple
            df = pd.DataFrame(columns=columns)

        if erf_power:
            # NOTE: Only square the 1st epoch (the mean). The 2nd epoch
            # (STDs) will still be the STDs of the original tcs
            label_tcs[0] = label_tcs[0] * label_tcs[0]

        if cfg['adjust_signs']:
            # Adjust signs of label time courses to those of the 1st subject
            # added
            if epoch0 is None:
                epoch0 = label_tcs[0]
            else:
                # Adjust channels signs based on epoch0
                # 'e1' is the (nlabels x ntime) sign-adjusted epoch
                e1 = adjust_signs([epoch0, label_tcs[0]],
                        istart = isign_start, iend = isign_end)[1]
                # Update the ltc data
                label_tcs[0] = e1

        sid = su.fif_subject(in_file)

        if task in ('henv_std_sm_4groups1img','henv_std_sm_4groups2img',
                    'henv_std_sm_santa1img','henv_std_sm_santa2img'):
            # Calculate mean STD of the envelope over the PLS interval
            # One needs to square the time course first to get variances
            # then average those, then take a square root.
            ydata = label_tcs[1,:,istart:iend]   # nlabels x ntime
            ydata *= ydata
            ydata = np.sqrt(np.mean(ydata, axis = 1))
            df.loc[len(df)] = [sid] + list(ydata)
        else:
            raise NotImplementedError(f'SM fitting for task {task} is not yet implemented')

    # NOTE: using .astype('string') produces dtype = 'python[string]', which is not
    # recognized by statsmodels
    df['sid'] = df['sid'].astype(str)  # Otherwise it becomes "O" - object

    for c in columns[1:]:
        df[c] = df[c].astype('float64')

    return df

