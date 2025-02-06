
function res = call_pls(fin, fout, PATH_TO_PLS_PACKAGE)
    % -------------------------------------------------------------------
    % PLS analysis call wrapper 
    %
    % Args:
    %   fin(str): pathname of .mat file with arguments for pls_analysis()
    %       call
    %   fout(str): pathname of .mat file where the returned results
    %       structure will be saved
    %   PATH_TO_PLS_PACKAGE(str): as is - path to the folder with matlab
    %       source code for PLS analysis
    %
    % Returns:
    %   res(struct): a matlab structure with PLS analysis results - see
    %       description in pls_analysis.m
    % -------------------------------------------------------------------
    % PATH_TO_PLS_PACKAGE = '/project/6019337/vvakorin/sdata3/distrib/plscmd';

    addpath(PATH_TO_PLS_PACKAGE);

    args = load(fin);

    if ~iscell(args.lst_dmat)   % Happens when a list has just one element
        dmat{1}=args.lst_dmat
        args.lst_dmat = dmat
    end
    
    res = pls_analysis(args.lst_dmat, args.lst_nsubj, args.ncond, args.option);
    save(fout,'res');
    return

