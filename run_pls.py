"""
An interface for running PLS analysis in matlab.
"""
import subprocess
import scipy.io

def run_pls(lst_dmat, lst_nsubj, ncond, option, path_to_matlab_pls, args_mat = "pls_args.mat",
        res_mat = "pls_res.mat", return_precalculated_result = False):
    """
        This function packs arguments to a .mat file, executes
        matlab script 'call_pls.m' with the .mat file as an agrument,
        and passes the result back to python via another .mat file.
        Inside the `call_pls.m` the main PLS script 'pls_analysis.m'
        is executed.

        Note: matlab program should be included in the user's default
            serach path.

        Args:
            lst_dmat(list of ndarray): a list of 2D matrices, one matrix
                per each group of subjects. Each matrix is shaped
                `(nsubj_in_group, nfeatures)`
            lst_nsubj(list of int): a list of numbers of subjects in
                each group
            ncond (int): number of conditions in each group
            option(dict): a dictionary of PLS options; see 'pls_analysys.m'
                for desctiption
            path_to_matlab_pls (Pathlike): on a host computer, path to the source
                folder with matlab files for PLS analysis
            args_mat (Pathlike): .mat file name to pass arguments to matlab
            res_mat (Pathlike): .mat file name to receive results from 
                matlab
            return_precalculated_result (bool): if True, only load and return
                already calculated PLS results from file `res_mat`. In this case,
                values of all parameters except `res_mat` are not used and may
                be set arbitrarily.
                
        Returns:
            res(dict): a dictionary of results, converted from a matlab structure
                returned by 'pls_analysys.m'.

        **res** dictionary fields description copied from matlab comments::

         'method:                 PLS option                                   ',
         '                        1. Mean-Centering Task PLS                   ',
         '                        2. Non-Rotated Task PLS                      ',
         '                        3. Regular Behavior PLS                      ',
         '                        4. Multiblock PLS                            ',
         '                        5. Non-Rotated Behavior PLS                  ',
         '                        6. Non-Rotated Multiblock PLS                ',
         '                                                                     ',
         'u:                      Brainlv or Salience                          ',
         '                                                                     ',
         's:                      Singular values (nlv x 1) matrix             ',
         '                                                                     ',
         'v:                      Designlv or Behavlv                          ',
         '                                                                     ',
         'usc:                    Brainscores or Scalpscores                   ',
         '                                                                     ',
         'vsc:                    Designscores or Behavscores                  ',
         '                                                                     ',
         'TBv:                    Store Task / Bahavior v separately           ',
         '                                                                     ',
         'TBusc:                  Store Task / Bahavior usc separately         ',
         '                                                                     ',
         'TBvsc:                  Store Task / Bahavior vsc separately         ',
         '                                                                     ',
         'datamatcorrs_lst:       Correlation of behavior data with datamat.   ',
         '                        Only available in behavior PLS.              ',
         '                                                                     ',
         'lvcorrs:                Correlation of behavior data with usc,       ',
         '                        only available in behavior PLS.              ',
         '                                                                     ',
         'perm_result:            struct containing permutation result         ',
         '        num_perm:       number of permutation                        ',
         '        sp:             permuted singular value greater than observed',
         '        sprob:          sp normalized by num_perm (nlv x 1) matrix   ',
         '        permsamp:       permutation reorder sample                   ',
         '        Tpermsamp:      permutation reorder sample for multiblock PLS',
         '        Bpermsamp:      permutation reorder sample for multiblock PLS',
         '                                                                     ',
         'perm_splithalf:         struct containing permutation splithalf      ',
         '        num_outer_perm: permutation splithalf related                ',
         '        num_split:      permutation splithalf related                ',
         '        orig_ucorr:     permutation splithalf related                ',
         '        orig_vcorr:     permutation splithalf related                ',
         '        ucorr_prob:     permutation splithalf related                ',
         '        vcorr_prob      permutation splithalf related                ',
         '        ucorr_ul:       permutation splithalf related                ',
         '        ucorr_ll:       permutation splithalf related                ',
         '        vcorr_ul:       permutation splithalf related                ',
         '        vcorr_ll:       permutation splithalf related                ',
         '                                                                     ',
         'boot_result:            struct containing bootstrap result           ',
         '        num_boot:       number of bootstrap                          ',
         '        boot_type:      Set to 'nonstrat' if using Natasha's         ',
         '                        'nonstrat' bootstrap type; set to            ',
         '                        'strat' for conventional bootstrap.          ',
         '        nonrotated_boot: Set to 1 if using Natasha's Non             ',
         '                        Rotated bootstrap; set to 0 for              ',
         '                        conventional bootstrap.                      ',
         '        bootsamp:       bootstrap reorder sample                     ',
         '        bootsamp_4beh:  bootstrap reorder sample for behav PLS       ',
         '        compare_u:      compared salience or compared brain          ',
         '        u_se:           standard error of salience or brainlv        ',
         '        clim:           confidence level between 0 and 100.          ',
         '        distrib:        orig_usc or orig_corr distribution           ',
         '        prop:           orig_usc or orig_corr probability            ',
         '                                                                     ',
         '        following boot_result only available in task PLS:            ',
         '                                                                     ',
         '        usc2:           brain scores that are obtained from the      ',
         '                        mean-centered datamat                        ',
         '        orig_usc:       same as usc, with mean-centering on subj     ',
         '        ulusc:          upper boundary of orig_usc                   ',
         '        llusc:          lower boundary of orig_usc                   ',
         '        ulusc_adj:      percentile of orig_usc distribution with     ',
         '                        upper boundary of orig_usc                   ',
         '        llusc_adj:      percentile of orig_usc distribution with     ',
         '                        lower boundary of orig_usc                   ',
         '                                                                     ',
         '        following boot_result only available in behavior PLS:        ',
         '                                                                     ',
         '        orig_corr:      same as lvcorrs                              ',
         '        ulcorr:         upper boundary of orig_corr                  ',
         '        llcorr:         lower boundary of orig_corr                  ',
         '        ulcorr_adj:     percentile of orig_corr distribution with    ',
         '                        upper boundary of orig_corr                  ',
         '        llcorr_adj:     percentile of orig_corr distribution with    ',
         '                        lower boundary of orig_corr                  ',
         '        num_LowVariability_behav_boots: display numbers of low       ',
         '                        variability resampled hehavior data in       ',
         '                        bootstrap test                               ',
         '        badbeh:         display bad behav data that is caused by     ',
         '                        bad re-order (with 0 standard deviation)     ',
         '                        which will further cause divided by 0        ',
         '        countnewtotal:  count the new sample that is re-ordered      ',
         '                        for badbeh                                   ',
         '                                                                     ',
         'is_struct:              Set to 1 if running Non-Behavior             ',
         '                        Structure PLS; set to 0 for other PLS.       ',
         '                                                                     ',
         'bscan:                  Subset of conditions that are selected       ',
         '                        for behav block, only in Multiblock PLS.     ',
         '                                                                     ',
         'num_subj_lst            Number of subject list array, containing     ',
         '                        the number of subjects in each group.        ',
         '                                                                     ',
         'num_cond                Number of conditions in datamat_lst.         ',
         '                                                                     ',
         'stacked_designdata:     Stacked design contrast data for all         ',
         '                        the groups.                                  ',
         '                                                                     ',
         'stacked_behavdata:      Stacked behavior data for all the groups.    ',
         '                                                                     ',
         'other_input:            struct containing other input data           ',
         '        meancentering_type: Use Natasha's meancentering type         ',
         '                        if it is not 0.                              ',
         '        cormode:        Use Natasha's correlation mode if it         ',
         '                        is not 0.                                    ',

    """
    if return_precalculated_result == False:
        # matlab expects doubles everywhere by default
        lst_nsubj = [float(i) for i in lst_nsubj]
        ncond = float(ncond)
        option['num_perm'] = float(option['num_perm'])
        option['num_boot'] = float(option['num_boot'])

        scipy.io.savemat(args_mat,
            {   'lst_dmat': lst_dmat,
                'lst_nsubj': lst_nsubj,
                'ncond': ncond,
                'option': option})

        # Quote the args again because if not quoted matlab interprets file
        # names as variable names, not as strings
        args_mat = "'" + args_mat + "'"
        res_mat_quoted = "'" + res_mat + "'"
        path_quoted = "'" + path_to_matlab_pls + "'"
        cmd = "matlab -nojvm -nodisplay -batch \"call_pls({},{},{})\"".format(
                    args_mat,res_mat_quoted,path_quoted)

        subprocess.run(cmd, shell=True, check=True)

    res = scipy.io.loadmat(res_mat)['res']

    # Return a nested directory structure with everything properly unpacked:
    return unpack_pls_result(res)

def unpack_pls_result(res):
    """
    Recursively unpack raw PLS result object retrieved from .mat
    file.

    All field values are wrapped inside 1 x 1 matrix, that is real field
    value is `res[field][0,0]`.

    Further on, we have:

      scalar value is returned as a matrix 1 x 1, and should be further
      unwrapped

      matrix value is returned properly as a matrix - nothing else to do

      stucture value is returned again as a matrix 1 x 1, whose dtype.names property
      is not None

    Finally, we know that the sub-structures do not have fields that are structures.

    Args:
        res(dict): raw 'res' dictionary with PLS results

    Returns:
        unpacked(dict): 'res' dictionary with matrix(1,1) wrappings removed,
            matrix(1,1) scalars replaced with the scalar values themselves, and
            the same done inside the sub-directories of 'res'

    """
    fields = res.dtype.names
    unpacked = {}

    if fields is None:
        return unpacked     # Actually, should never happen

    for f in fields:
        val = res[f][0,0]

        if val.dtype.names is not None:
            unpacked[f] = unpack_pls_result(val)
            continue

        if val.shape == (1,1):          # This is a scalar value
            unpacked[f] = val[0,0]
        else:                           # This is truly a matrix
            unpacked[f] = val

    return unpacked

if __name__ == "__main__":
    # Test script
    import numpy as np

    # Generate data matrix (subjects x features) for each group
    ngroup= 5;                  # Number of groups
    num_subj = [11,9,12,13,14]; # Number of subjects in each group
    ncond = 1                   # Number of conditions in each group
    a = 1;                      # Scale: a = 0.5,2
    seed = 12345
    return_precalculated_result = False
    pls_path = '/project/6019337/vvakorin/sdata3/distrib/plscmd'

    option = dict(method = 1, num_perm = 500, num_boot = 500)
    lst_dmat = []

    if not return_precalculated_result:
        # Generate data matrix for each group (subjects x 10 features)
        rng = np.random.default_rng(seed = seed)
        for g in range(ngroup):
            tmp = (g+1)*np.ones((num_subj[g],3));
            tmp1 = a*tmp + rng.standard_normal(size = tmp.shape);     # A model with an increasing trend [1 2 3 4 5]
            tmp2 = - a*tmp + rng.standard_normal(size = tmp.shape);   # A model with a decreasing trend [-1 -2 -3 -4 -5];
            lst_dmat.append(np.hstack((tmp1, tmp2, rng.standard_normal(size = (num_subj[g],4)))));     

    # Run PLS analysis or load results
    res = run_pls(lst_dmat, num_subj, ncond, option, pls_path,
                return_precalculated_result = return_precalculated_result)

    # Check the result
    lv = 0;     # First latent variable
    print('');

    print(f'Singular values: {res["s"][:,0]}')      # Returned is nlv x 1 matrix
    print(f'P-values: {res["perm_result"]["sprob"][:,0]}\n')
    print('Brain saliences:')
    print(f'{res["u"]}\n')

    print('Design saliences (contrasts)')
    print('You should see a trend across 5 groups in the 1st (i.e. significant) one:');
    print(f'{res["v"]}\n')

    print('Z-scores (10 features). In the significant lv = 0:');
    print('1-6 features (trend): large z-scores (positive or negative)');
    print('7-10 features (noise): close to 0');
    print(f'{res["boot_result"]["compare_u"]}\n')

    print(f'Brain scores:\n{res["usc"]}\n')
    print(f'Design scores:\n{res["vsc"]}\n')

    """
    # Just to make sure that returned results are interpreted
    # correctly:
    # verify that the brain scores are indeed calculated as X*V
    # (in the tutorial paper's notation)
    rng = np.random.default_rng(seed = seed)
    for g in range(ngroup):
        tmp = (g+1)*np.ones((num_subj[g],3));
        tmp1 = a*tmp + rng.standard_normal(size = tmp.shape);     # A model with an increasing trend [1 2 3 4 5]
        tmp2 = - a*tmp + rng.standard_normal(size = tmp.shape);   # A model with a decreasing trend [-1 -2 -3 -4 -5];
        lst_dmat.append(np.hstack((tmp1, tmp2, rng.standard_normal(size = (num_subj[g],4)))));     

    X = np.vstack(lst_dmat)
    Lx = X @ res['u']

    if np.allclose(Lx[:,0], res['usc'][:,0]):
        print('Brain scores interpretation verifies OK')
    else:
        print('Brain scores are NOT X @ V!!')
    """



