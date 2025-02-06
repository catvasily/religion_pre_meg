function res = test_pls

addpath('/project/6019337/vvakorin/sdata3/distrib/plscmd');
% -------------------------------------------------------------------
%  Usage: result = pls_analysis(datamat_lst, num_subj_lst, num_cond, ...
%			[option])
%
%  Inputs:
%
%  datamat_lst  -  Datamat list cell array. Each cell stands for one
%	datamat (2-D Matrix), which is also referred as a group. All
%	datamats must be in the form of "subject in condition".
%
%  num_subj_lst  -  Number of subject list array, containing the number
%	of subjects in each group.
%
%  num_cond  -  Number of conditions in datamat_lst.
%
%  option  -  A struct of optional inputs. It can be:
% -------------------------------------------------------------------

% Generate data matrix (subjects x features) for each group
ngroup= 5; % Number of groups
num_subj = [11 9 12 13 14]; % Number of subjects in each group
a = 1; % a = 0.5,2


% Generate data matrix for each group (subjects x 10 features)
for g = 1:ngroup
    tmp = g*ones(num_subj(g),3);
    tmp1 = a*tmp + randn(size(tmp)); % A model with an increasing trend [1 2 3 4 5]
    tmp2 = - a*tmp + randn(size(tmp)); % A model with a decreasing trend [-1 -2 -3 -4 -5];
    dmat{g} = [tmp1 tmp2 randn(num_subj(g),4)]; 
end

% Run PLS analysis
option.method = 1;
option.num_perm = 500;
option.num_boot = 500;

res = pls_analysis(dmat,num_subj,1,option);
 

% Visualize results from PLS analysis
lv = 1; % First latent variable
disp('');

disp('Z-scores (10 features):');
disp('1-6 features (trend): large z-scores (positive or negative)');
disp('7-10 features (noise): close to 0');

u = res.boot_result.compare_u(:,lv);

disp(u');
disp('Contrast (you should see a trend across 5 groups):');

x = res.v(:,lv);

disp(x');

p = res.perm_result.sprob(lv);
disp(['p-value = ' num2str(p)]);

return
