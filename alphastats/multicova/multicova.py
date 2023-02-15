import numpy as np
import pandas as pd
import numba as nb
import numba_stats as nbs
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from scipy import stats
import statsmodels.api as sm
from collections import Counter
import multiprocessing
from joblib import Parallel, delayed
import random
import math
import swifter
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests
import os
from pathlib import Path


def get_std(x):
    """
    Function to calculate the sample standard deviation.
    """
    std_x = np.sqrt(np.sum((abs(x - x.mean())**2)/(len(x)-1)))
    return std_x


def perform_ttest(x, y, s0):
    """
    Function to perform a independent two-sample t-test including s0
    adjustment.
    Assumptions: Equal or unequal sample sizes with similar variances.
    """
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    # Get fold-change
    fc = mean_x-mean_y
    n_x = len(x)
    n_y = len(y)
    # pooled standard vatiation
    # assumes that the two distributions have the same variance
    sp = np.sqrt((((n_x-1)*get_std(x)**2) + ((n_y-1)*(get_std(y)**2)))/(n_x+n_y-2))
    # Get t-values
    tval = fc/(sp * (np.sqrt((1/n_x)+(1/n_y))))
    tval_s0 = fc/((sp * (np.sqrt((1/n_x)+(1/n_y))))+s0)
    # Get p-values
    pval = 2*(1-stats.t.cdf(np.abs(tval), n_x+n_y-2))
    pval_s0 = 2*(1-stats.t.cdf(np.abs(tval_s0), n_x+n_y-2))
    return [fc, tval, pval, tval_s0, pval_s0]


def permutate_vars(X, n_rand, seed=42):
    """
    Function to randomly permutate an array `n_rand` times.
    """
    x_perm_idx = generate_perms(n=len(X), n_rand=n_rand, seed=seed)
    X_rand = list()
    for i in np.arange(0, len(x_perm_idx)):
        X_rand.append([X[j] for j in x_perm_idx[i]])
    return X_rand


def workflow_ttest(df, c1, c2, s0=1, parallelize=False):
    """
    Function to perform a t-test on all rows in a data frame.
    c1 specifies the column names with intensity values of the first condition.
    c2 specifies the column names with intensity values of the second
    condition. s0 is the tuning parameter that specifies the minimum
    fold-change to be trusted.
    """
    if parallelize:
        res = df.swifter.progress_bar(False).apply(lambda row : perform_ttest(row[c1], row[c2], s0=s0), axis = 1)
    else:
        res = df.apply(lambda row : perform_ttest(row[c1], row[c2], s0=s0), axis = 1)

    res = pd.DataFrame(list(res), columns=['fc','tval','pval','tval_s0','pval_s0'])

    df_real = pd.concat([df, res], axis=1)

    return df_real


def workflow_permutation_tvals(df, c1, c2, s0=1, n_perm=2, parallelize=False):
    """
    Function to perform a t-test on all rows in a data frame based on the
    permutation of samples across conditions.
    c1 specifies the column names with intensity value sof the first condition.
    c2 specifies the column names with intensity value sof the second
    condition. s0 is the tuning parameter that specifies the minimum
    fold-change to be trusted. n_perm specifies the number of random
    permutations to perform.
    """
    all_c = c1 + c2
    all_c_rand = permutate_vars(all_c, n_rand=n_perm)
    res_perm = list()
    for i in np.arange(0,len(all_c_rand)):
        if parallelize:
            res_i = df.swifter.progress_bar(False).apply(lambda row : perform_ttest(row[all_c_rand[i][0:len(c1)]],
                                                                row[all_c_rand[i][len(c1):len(c1)+len(c2)]],
                                                                s0=s0),
                                                                axis = 1)
        else:
            res_i = df.apply(lambda row : perform_ttest(row[all_c_rand[i][0:len(c1)]],
                                                        row[all_c_rand[i][len(c1):len(c1)+len(c2)]],
                                                        s0=s0),
                                                        axis=1)
        res_i = pd.DataFrame(list(res_i), columns=['fc','tval','pval','tval_s0','pval_s0'])
        res_perm.append(list(np.sort(np.abs(res_i.tval_s0.values))))
    return res_perm


def get_tstat_cutoff(res_real, t_perm_avg, delta):
    # Extract all t-stats from the results of the 'real' data
    # and sort the absolute values.
    t_real = res_real.tval_s0
    t_real_abs = np.sort(np.abs(t_real))
    # print(t_real_abs)

    # Calculate the difference between the observed t-stat
    # and the average random t-stat for each rank position.
    t_diff = t_real_abs - t_perm_avg
    # print(t_diff)

    # Find the minimum t-stat value for which the difference
    # between the observed and average random t-stat is
    # larger than the selected delta.
    t_max = t_real_abs[t_diff > delta]
    if (t_max.shape[0] == 0):
        t_max = np.ceil(np.max(t_real_abs))
    else:
        t_max = np.min(t_max)
    return t_max


@nb.njit
def get_positive_count(res_real_tval_s0, t_cut):
    """
    Count number of tval_s0 in res_real that are above the t_cut threshold.
    """
    ts_sig = list()
    for r in res_real_tval_s0:
        r_abs = np.abs(r)
        if r_abs >= t_cut:
            ts_sig.append(r_abs)

    return len(ts_sig)


@nb.njit
def get_false_positive_count(res_perm, t_cut):
    """
    Get median number of tval_s0 in res_perm that are above the t_cut
    threshold.
    """
    ts_sig = list()
    for rp in res_perm:
        ts_abs = np.abs(rp)
        ts_max = list()
        for t in ts_abs:
            if t >= t_cut:
                ts_max.append(t)

        ts_sig.append(len(ts_max))

    res = np.median(np.array(ts_sig))

    return res


def get_pi0(res_real, res_perm):
    """
    Estimate pi0, the proportion of true null (unaffected) genes in the data
    set, as follows:
    (a) Compute q25; q75 = 25% and 75% points of the permuted d values
    (if p = # genes, B = # permutations, there are pB such d values).
    (b) Count the number of values d in the real dataset that are between q25
    and q75 (there are p values to test). pi0 = #d/0.5*p.
    (c) Let pi0 = min(pi0; 1) (i.e., truncate at 1).
    Documentation in: https://statweb.stanford.edu/~tibs/SAM/sam.pdf
    """
    t_real = res_real["tval_s0"]
    t_real_abs = np.sort(np.abs(t_real))

    t_perm = list()
    for ri in res_perm:
        for r in ri:
            t_perm.append(r)
    t_perm_abs = np.sort(np.abs(t_perm))

    t_perm_25 = np.percentile(t_perm_abs, 25)
    t_perm_75 = np.percentile(t_perm_abs, 75)

    n_real_in_range = np.sum((t_real_abs >= t_perm_25) & (t_real_abs <= t_perm_75))
    pi0 = n_real_in_range/(0.5*len(t_real_abs))

    # pi0 can maximally be 1
    pi0 = np.min([pi0, 1])

    return pi0


def get_fdr(n_pos, n_false_pos, pi0):
    """
    Compute the FDR by dividing the number of false positives by the number of
    true positives. The number of false positives are adjusted by pi0, the
    proportion of true null (unaffected) genes in the data set.
    """
    n = n_false_pos*pi0
    if n != 0:
        if n_pos != 0:
            fdr = n/n_pos
        else:
            fdr = 0
    else:
        if n_pos > 0:
            fdr = 0
        else:
            fdr = np.nan
    return fdr


def estimate_fdr_stats(res_real, res_perm, delta):
    """
    Helper function for get_fdr_stats_across_deltas.
    It computes the FDR and tval_s0 thresholds for a specified delta.
    The function returns a list of the following values:
    t_cut, n_pos, n_false_pos, pi0, n_false_pos_corr, fdr
    """
    perm_avg = np.mean(res_perm, axis=0)
    t_cut = get_tstat_cutoff(res_real, perm_avg, delta)
    n_pos = get_positive_count(res_real_tval_s0=np.array(res_real.tval_s0), t_cut=t_cut)
    n_false_pos = get_false_positive_count(np.array(res_perm), t_cut=t_cut)
    pi0 = get_pi0(res_real, res_perm)
    n_false_pos_corr = n_false_pos*pi0
    fdr = get_fdr(n_pos, n_false_pos, pi0)
    return [t_cut, n_pos, n_false_pos, pi0, n_false_pos_corr, fdr]


def get_fdr_stats_across_deltas(res_real, res_perm):
    """
    Wrapper function that starts with the res_real and res_perm to derive
    the FDR and tval_s0 thresholds for a range of different deltas.
    """
    res_stats = list()
    for d in np.arange(0, 10, 0.01):
        res_d = estimate_fdr_stats(res_real, res_perm, d)
        if np.isnan(res_d[5]):
            break
        else:
            res_stats.append(res_d)
    res_stats_df = pd.DataFrame(res_stats,
                                columns=['t_cut', 'n_pos', 'n_false_pos', 'pi0', 'n_false_pos_corr', 'fdr'])

    return(res_stats_df)


def get_tstat_limit(stats, fdr=0.01):
    """
    Function to get tval_s0 at the specified FDR.
    """
    t_limit = np.min(stats[stats.fdr <= fdr].t_cut)
    return(t_limit)


def annotate_fdr_significance(res_real, stats, fdr=0.01):
    t_limit = np.min(stats[stats.fdr <= fdr].t_cut)
    res_real['qval'] = [np.min(stats[stats.t_cut <= abs(x)].fdr) for x in res_real['tval_s0']]
    res_real['FDR' + str(int(fdr*100)) + '%'] = ["sig" if abs(x) >= t_limit else "non_sig" for x in res_real['tval_s0']]
    return(res_real)


def perform_ttest_getMaxS(fc, s, s0, n_x, n_y):
    """
    Helper function to get ttest stats for specified standard errors s.
    Called from within get_fdr_line.
    """
    # Get t-values
    tval = fc/s
    tval_s0 = fc/(s+s0)

    # Get p-values
    pval = 2*(1-stats.t.cdf(np.abs(tval), n_x+n_y-2))
    pval_s0 = 2*(1-stats.t.cdf(np.abs(tval_s0), n_x+n_y-2))

    return [fc, tval, pval, tval_s0, pval_s0]


def get_fdr_line(t_limit, s0, n_x, n_y, plot=False,
                 fc_s=np.arange(0, 6, 0.01), s_s=np.arange(0.005, 6, 0.005)):
    """
    Function to get the fdr line for a volcano plot as specified tval_s0
    limit, s0, n_x and n_y.
    """
    pvals = np.ones(len(fc_s))
    svals = np.zeros(len(fc_s))
    for i in np.arange(0, len(fc_s)):
        for j in np.arange(0, len(s_s)):
            res_s = perform_ttest_getMaxS(fc=fc_s[i], s=s_s[j], s0=s0, n_x=n_x, n_y=n_y)
            if res_s[3] >= t_limit:
                if svals[i] < s_s[j]:
                    svals[i] = s_s[j]
                    pvals[i] = res_s[2]

    s_df = pd.DataFrame(np.array([fc_s, svals, pvals]).T, columns=['fc_s','svals','pvals'])
    s_df = s_df[s_df.pvals != 1]

    s_df_neg = s_df.copy()
    s_df_neg.fc_s = -s_df_neg.fc_s

    s_df = s_df.append(s_df_neg)

    if (plot):
        fig = px.scatter(x=s_df.fc_s,
                         y=-np.log10(s_df.pvals),
                         template='simple_white')
        fig.show()

    return(s_df)



def perform_ttest_analysis(df, c1, c2, s0=1, n_perm=2, fdr=0.01, id_col='Genes', plot_fdr_line=False, parallelize=False):
    """
    Workflow function for the entire T-test analysis including FDR
    estimation and visualizing a volcanoe plot.
    """
    ttest_res = workflow_ttest(df, c1, c2, s0, parallelize=parallelize)
    ttest_perm_res = workflow_permutation_tvals(df, c1, c2, s0, n_perm, parallelize=parallelize)
    ttest_stats = get_fdr_stats_across_deltas(ttest_res, ttest_perm_res)
    ttest_res = annotate_fdr_significance(res_real=ttest_res, stats=ttest_stats, fdr=fdr)
    t_limit = get_tstat_limit(stats=ttest_stats, fdr=fdr)
    return(ttest_res, t_limit)


def perform_regression(y, X, s0):
    """
    Function to perform multiple linear regression including the s0 parameter.
    """
    # Use LinearRegression from sklearn.linear_model
    lm = LinearRegression()
    lm.fit(X, y)
    betas = np.append(lm.intercept_, lm.coef_)
    predictions = lm.predict(X)

    # Add intercept column to X dataframe
    newX = pd.DataFrame({"Constant": np.ones(len(X))}).join(pd.DataFrame(X))

    # Calculate mean squared error (MSE) of observed y vs. prediction
    # MSE = squared_error/degrees_of_freedom
    # degrees_of_freedom = number_of_samples - number_of_betas (including beta0 = intercept)
    MSE = (sum((y-predictions)**2))/(len(newX)-len(newX.columns))
    # Calculate t-values and p-values
    ts_b, ps_b = get_tstats(newX, betas, MSE)
    # Adjust the MSE by the minimum fold change that we trust (s0)
    MSE_s0 = MSE + s0
    # Calculate t-values and p-values
    ts_b_s0, ps_b_s0 = get_tstats(newX, betas, MSE_s0)
    betas_std = list()
    for i in np.arange(0, len(betas)):
        betas_std.append(betas[i]*(get_std(np.array(newX)[:, i])/get_std(y)))
    betas_std = np.array(betas_std)
    return betas, betas_std, predictions, MSE, ts_b, ps_b, MSE_s0, ts_b_s0, ps_b_s0


def get_tstats(newX, betas, MSE):
    # Calculate standard deviation for each beta
    # This scales the error as estimated by MSE to each beta
    # (error on the betas might be different for different betas depending on
    # the variable)
    var_b = MSE*(np.linalg.inv(np.dot(newX.T, newX)).diagonal())
    sd_b = np.sqrt(var_b)
    # Get t-values
    ts_b = betas/sd_b
    # Get p-values
    #ps_b =[2*(1-stats.t.cdf(np.abs(i),len(newX)-len(newX.columns))) for i in ts_b]
    #ps_b =[2*(1-nbs.t_cdf(np.abs(i),len(newX)-len(newX.columns),0,1)) for i in ts_b]
    ps_b = get_cdf(ts_b, len(newX)-len(newX.columns))
    return ts_b, ps_b


@nb.njit
def get_cdf(ts, df):
    pvals = [0.0][1:]
    for t in ts:
        pvals.append(2*(1-nbs.t_cdf(np.abs(t), df, 0, 1)))
    return (pvals)


def filter_nan(y, X):
    """
    Find nan values in quantitative data y and remove corresponding samples
    from both X and y.
    """
    nan_mask = np.invert(np.isnan(y))
    y_new = y[nan_mask]
    X_new = X[nan_mask]
    return y_new, X_new


def get_min_vars(X):
    """
    Get the minimum number of observations for each covariate.
    """
    min_vars = list()
    for d in np.arange(0, X.shape[1]):
        test_vals = X[:, d]
        n_vars = len(set(test_vals))
        if n_vars > 2:
            min_vars.append(n_vars)
        else:
            test_vals_0 = np.append(test_vals,[0,1]) # append 0 and 1 for counter
            var_count = Counter(test_vals_0)
            min_vars.append(min(var_count.values()))

    return min_vars


def regression_workflow(y, X, s0):
    y_new, X_new = filter_nan(y, X)
    # Removed min_var filtering after standard scaling of variable.
    # @ToDo: Include sanity check that sufficient categories are covered!
    # The following counts the minimum number of observations for each covariate
    #min_vars = get_min_vars(X_new)
    # A min(min_vars) value < 2 means essentially no values for at least one covariate were observed
    # A beta of zero and p-values of 1 are returned for these cases
    #if min(min_vars) < 2:
    #    betas, betas_std, tvals, pvals, tvals_s0, pvals_s0 = 0, 0, 0, 1, 0, 1
    #else:
    #    betas, betas_std, predictions, MSE, tvals, pvals, MSE_s0, tvals_s0, pvals_s0 = perform_regression(np.array(y_new), np.array(X_new), s0)
    betas, betas_std, predictions, MSE, tvals, pvals, MSE_s0, tvals_s0, pvals_s0 = perform_regression(np.array(y_new), np.array(X_new), s0)
    return betas, betas_std, tvals, pvals, tvals_s0, pvals_s0


def regression_workflow_permutation(y, X_rand, s0):
    res_rand = list()
    for X_r in X_rand:
        res_rand.append(regression_workflow(y, X_r, s0))
    # num_cores = multiprocessing.cpu_count()
    # res_rand = Parallel(n_jobs=num_cores-1)(delayed(regression_workflow)(y=y, X=X_r, s0=s0) for X_r in X_rand)
    return res_rand


def get_fdr_line_regression(t_limits, s0, X, plot = False,
                            fc_s = np.arange(0,6,0.01), s_s = np.arange(0.005,6,0.005)):
    """
    Function to get the fdr line for a volcano plot as specified tval_s0 limit, s0, n_x and n_y.
    """
    #pvals = [list(np.ones(len(fc_s)))] * X.shape[1]
    pvals = [list(np.ones(len(fc_s))) for i in range(0,X.shape[1])]
    #print(pvals)
    #svals = [list(np.zeros(len(fc_s)))] * X.shape[1]
    svals = [list(np.zeros(len(fc_s))) for i in range(0,X.shape[1])]
    #print(svals)
    for i in np.arange(0,len(fc_s)):
        for j in np.arange(0,len(s_s)):
            res_s = perform_ttest_getMaxS_regression(fc=fc_s[i], s=s_s[j], s0=s0, X=X)
            for k in np.arange(0,X.shape[1]):
                t_limit = t_limits[k]
                if (res_s[k][3] >= t_limit) and (svals[k][i] < s_s[j]):
                    svals[k][i] = s_s[j]
                    pvals[k][i] = res_s[k][2]
    s_df_list = list()
    for k in np.arange(0, X.shape[1]):
        s_df = pd.DataFrame(np.array([fc_s,svals[k],pvals[k]]).T, columns=['fc_s','svals','pvals'])
        s_df = s_df[s_df.pvals != 1]

        s_df_neg = s_df.copy()
        s_df_neg.fc_s = -s_df_neg.fc_s

        s_df = s_df.append(s_df_neg)

        if (plot):
            fig = px.scatter(x=s_df.fc_s,
                             y=-np.log10(s_df.pvals),
                             template='simple_white')
            fig.show()

        s_df_list.append(s_df)

    return(s_df_list)


def perform_ttest_getMaxS_regression(fc, s, s0, X):
    """
    Helper function to get ttest stats for specified standard errors s.
    Called from within get_fdr_line_regression.
    """
    newX = pd.DataFrame({"Constant":np.ones(len(X))}).join(pd.DataFrame(X))
    fc=np.repeat(fc,newX.shape[1])
    #print(fc)
    var_b = s*(np.linalg.inv(np.dot(newX.T,newX)).diagonal())
    sd_b = np.sqrt(var_b)
    ts_b = fc/sd_b
    #ps_b =[2*(1-stats.t.cdf(np.abs(i),len(newX)-len(newX.columns))) for i in ts_b]
    ps_b = get_cdf(ts_b, len(newX)-len(newX.columns))
    var_b_s0 = (s+s0)*(np.linalg.inv(np.dot(newX.T,newX)).diagonal())
    sd_b_s0 = np.sqrt(var_b_s0)
    ts_b_s0 = fc/sd_b_s0
    #ps_b_s0 =[2*(1-stats.t.cdf(np.abs(i),len(newX)-len(newX.columns))) for i in ts_b_s0]
    ps_b_s0 = get_cdf(ts_b_s0, len(newX)-len(newX.columns))
    res = [[fc[x], ts_b[x], ps_b[x], ts_b_s0[x], ps_b_s0[x]] for x in np.arange(1,newX.shape[1])]
    return res

def generate_perms(n, n_rand, seed=42):
    """
    Generate n_rand permutations of indeces ranging from 0 to n.
    """
    np.random.seed(seed)
    idx_v = np.arange(0, n)
    rand_v = list()
    n_rand_i = 0
    n_rand_max = math.factorial(n)-1
    if n_rand_max <= n_rand:
        print("{} random permutations cannot be created. The maximum of n_rand={} is used instead.".format(n_rand, n_rand_max))
        n_rand = n_rand_max
    while n_rand_i < n_rand:
        rand_i = list(np.random.permutation(idx_v))
        if np.all(rand_i == idx_v):
            next
        else:
            if rand_i in rand_v:
                next
            else:
                rand_v.append(rand_i)
                n_rand_i = len(rand_v)
    return rand_v


def permutate_multi_vars(X, rand_index, n_rand, seed=42):
    """
    Function to randomly permutate a multi-dimensional array X `n_rand` times
    at rand_index position.
    """
    x_perm_idx = generate_perms(X.shape[0], n_rand=n_rand, seed=seed)
    X_rand = list()
    for i in np.arange(0, len(x_perm_idx)):
        idx_r = np.array(x_perm_idx[i])
        idx_a = np.tile(np.arange(0, X.shape[0]), (X.shape[1], 1))
        idx_a[rand_index] = idx_r
        idx_a = np.transpose(idx_a)
        X_rand.append(np.take_along_axis(X, indices=idx_a, axis=0))

    return X_rand


def get_res_parallel(y_i, X, s0):
    betas, betas_std, tvals, pvals, tvals_s0, pvals_s0 = regression_workflow(y=y_i, X = X, s0 = s0)
    res = pd.DataFrame()
    res["fc"], res["fc_std"], res["tval"], res["pval"], res["tval_s0"], res["pval_s0"] = [betas, betas_std, tvals, pvals, tvals_s0, pvals_s0]
    return res


def get_perm_res_parallel(y_i, X_rand, s0):
    res_perm_list = regression_workflow_permutation(y_i, X_rand, s0=s0)
    return res_perm_list


def full_regression_analysis(quant_data, annotation, covariates, sample_column='sample_name', n_permutations=4, fdr=0.05, s0=0.05, seed=42):
    data_cols = annotation[sample_column].values
    quant_data = quant_data.dropna().reset_index(drop=True)
    y = quant_data[data_cols].to_numpy().astype('float')
    # @ToDo make sure that columns are sorted correctly!!!
    X = np.array(annotation[covariates])

    # standardize X:
    X_scaled = StandardScaler().fit_transform(X)

    num_cores = multiprocessing.cpu_count()
    res_list = Parallel(n_jobs=num_cores-1)(delayed(get_res_parallel)(y_i=y_i, X=X_scaled, s0=s0) for y_i in y)

    t_limit_dict = {}

    for test_index in np.arange(0, len(covariates)):
        test_covariate = covariates[test_index]
        print(test_covariate)
        X_rand = permutate_multi_vars(X_scaled, rand_index=test_index, n_rand=n_permutations, seed=seed)
        res_perm_list = Parallel(n_jobs=num_cores-1)(delayed(get_perm_res_parallel)(y_i=y_i, X_rand=X_rand, s0=s0) for y_i in y)

        i = test_index + 1
        res_i = [r.iloc[[i]] for r in res_list]
        res_i = pd.concat(res_i)

        # get tvals_s0 list of permutations for covariate i
        res_i_rand = [[0] * len(res_perm_list) for i in range(len(res_perm_list[0]))]
        for a in np.arange(0,len(res_perm_list)):
            for b in np.arange(0, len(res_perm_list[a])):
                #print(res_perm_list[a][b][3])
                res_i_rand[b][a] = res_perm_list[a][b][4][i]

        res_i_rand = [list(np.sort(np.abs(r))) for r in res_i_rand]
        fdr_d_i = get_fdr_stats_across_deltas(res_real=res_i, res_perm=res_i_rand)
        res_i = annotate_fdr_significance(res_real=res_i, stats=fdr_d_i, fdr=fdr)
        t_limit_i = get_tstat_limit(stats=fdr_d_i, fdr=fdr)
        t_limit_dict[test_covariate] = t_limit_i

        # @ToDo: This multiple testing should potentially be done across all covariates together?
        res_i['qval_BH'] = multipletests(res_i.pval, method='fdr_bh')[1]
        res_i['BH_FDR ' + str(int(fdr*100)) + '%'] = ["sig" if abs(x)<=fdr else "non_sig" for x in res_i["qval_BH"]]
        res_i['qval_BH_s0'] = multipletests(res_i.pval_s0, method='fdr_bh')[1]
        res_i['BH_FDR_s0 ' + str(int(fdr*100)) + '%'] = ["sig" if abs(x)<=fdr else "non_sig" for x in res_i["qval_BH_s0"]]

        res_i = res_i.add_prefix(covariates[test_index] + "_")

        quant_data = pd.concat([quant_data.reset_index(drop=True), res_i.reset_index(drop=True)], axis=1)

    return [quant_data, t_limit_dict]


def add_random_covariate(annotation, name='random', n_random=50, seed=42):
    annotation_test = annotation.copy()
    annotation_test.reset_index(drop=True, inplace=True)
    annotation_test[name] = 0
    random.seed(seed)
    rand_true = random.sample(range(0, annotation_test.shape[0]), n_random)
    annotation_test.loc[rand_true, name] = 1
    return annotation_test


def plot_evaluate_seed_and_perm(df, covariates):
    config = {'toImageButtonOptions': {'format': 'svg',
                                      'filename': 'permutation_test',
                                      'scale': 1
                                      }
             }
    for c in covariates:
        fig = px.line(df, x="permutations",
                      y=c,
                      line_dash="seed",
                      line_group="seed",
                      template="simple_white",
                      color_discrete_sequence=['lightgrey'])
        fig.update_traces(mode='lines+markers')
        fig.update_layout(width=620, height=350)
        fig.show(config=config)


def evaluate_seed_and_perm(quant_data,
                           annotation,
                           covariates,
                           perms,
                           seeds,
                           sample_column='sample_name',
                           fdr=0.05,
                           s0=0.05):
    resDF = pd.DataFrame({'seed': np.repeat(seeds, len(perms)),
                          'permutations': np.tile(perms, len(seeds))})
    resDF[covariates] = pd.DataFrame([np.repeat(0, len(covariates))],
                                     index=resDF.index)

    for i in np.arange(0,resDF.shape[0]):
        res, tlim = full_regression_analysis(quant_data=quant_data,
                                             annotation=annotation,
                                             covariates=covariates,
                                             n_permutations=resDF.permutations[i],
                                             sample_column=sample_column,
                                             fdr=fdr,
                                             s0=s0,
                                             seed=resDF.seed[i])

        for c in covariates:
            resDF.loc[i,c] = res[res[c + "_FDR " + str(int(fdr*100)) + "%"] == "sig"].shape[0]
    plot_evaluate_seed_and_perm(resDF, covariates=covariates)
    return resDF


def plot_evaluate_s0s(df, covariates):
    for c in covariates:
        fig = px.line(df, x="s0", y=c, title=c, template="simple_white", color_discrete_sequence=['lightgrey'])
        fig.update_traces(mode='lines+markers')
        fig.update_layout(width=800, height=500)
        fig.show()


def evaluate_s0s(quant_data,
                           annotation,
                           covariates,
                           s0s,
                           sample_column='sample_name',
                           n_permutations=5,
                           seed=42,
                           fdr=0.01):
    resDF = pd.DataFrame({'s0':s0s})
    resDF[covariates] = pd.DataFrame([np.repeat(0, len(covariates))], index=resDF.index)

    for i in np.arange(0,resDF.shape[0]):
        res, tlim = full_regression_analysis(quant_data=quant_data,
                                             annotation=annotation,
                                             covariates=covariates,
                                             sample_column=sample_column,
                                             n_permutations=n_permutations,
                                             fdr=fdr,
                                             s0=resDF.s0[i],
                                             seed=seed)

        for c in covariates:
            resDF.loc[i,c] = res[res[c + "_FDR " + str(int(fdr*100)) + "%"] == "sig"].shape[0]

    plot_evaluate_s0s(resDF, covariates=covariates)

    return resDF


def plot_pval_dist(df, covariates, mode='separate'):
    config = {'toImageButtonOptions': {'format': 'svg',
                                       'filename': 'pvalue_histogram',
                                       'scale': 1
                                      }
             }
    if (mode == 'separate'):
        for c in covariates:
            fig = px.histogram(df,
                               x=c+"_pval",
                               title=c,
                               nbins=50,
                               template='simple_white',
                               color_discrete_sequence=['lightgrey'])
            fig.update_layout(width=400, height=300)
            fig.show(config=config)
    elif (mode == 'joined'):
        df_cov = pd.DataFrame()
        for c in covariates:
            df_c = pd.DataFrame({'p-value': df[c+"_pval"], 'covariate': c})
            df_cov = pd.concat([df_cov, df_c])
        fig = px.histogram(df_cov, x='p-value',
                           color='covariate',
                           barmode='overlay',
                           opacity=0.6,
                           nbins=50,
                           template='simple_white')
        fig.update_layout(width=620, height=350)
        fig.show(config=config)
    else:
        raise ValueError("The mode parameter needs to be either 'separate' or 'joined'.")


def plot_beta_dist(df, covariates):
    for c in covariates:
        fig = px.histogram(df, x=c+"_fc", title=c, template='simple_white', color_discrete_sequence=['lightgrey'])
        fig.update_layout(width=800, height=500)
        fig.show()


def get_replacement_vals(df, threshold, mean_all, sd_all, ds_f):
    """
    Helper function for missing value imputation.
    """
    if df.percent_valid_vals == 100:
        # print(100)
        rep = []
    elif df.percent_valid_vals > threshold:
        # print(70)
        rep = np.random.normal(df.int_mean-(ds_f*df.int_sd), df.int_sd, df.invalid_vals)
    else:
        # print(0)
        rep = np.random.normal(mean_all-(ds_f*sd_all), sd_all, df.invalid_vals)
    return(rep)

def impute_missing_values(data, percent_impute=20, percent_self_impute=70, downshift_factor=1.8):
    """
    Function for missing value imputation. Proteins with less than
    'percent_impute' valid values are removed.
    For proteins with less than 'percent_self_impute' valid values, global mean
    and sd values are used for imputation.
    For proteins with more than 'percent_self_impute' valid values, mean and sd
    values for each specific protein are used.
    The 'downshift_factor' determines by how many standard deviations the mean
    of the imputed distribution is shifted.
    """

    data['int_mean'] = data.filter(regex=("Quantity")).mean(axis=1)
    data['int_sd'] = data.filter(regex=("Quantity")).std(axis=1)
    data['valid_vals'] = data.filter(regex=("Quantity")).count(axis=1)
    data['invalid_vals'] = data.filter(regex=("Quantity")).shape[1]-data.filter(regex=("Quantity")).count(axis=1)
    data['percent_valid_vals'] = 100/data.filter(regex=("Quantity")).shape[1]*data['valid_vals']

    data = data[data.percent_valid_vals >= percent_impute]

    overall_mean = np.mean(data.int_mean)
    overall_sd = np.mean(data.int_sd)

    nan_idx = np.where(data.isna())

    replacement = data.apply(get_replacement_vals, threshold=percent_self_impute, mean_all=overall_mean, sd_all=overall_sd, ds_f = downshift_factor, axis=1)
    replacement = [val for sublist in replacement for val in sublist]

    for i in np.arange(0, len(nan_idx[0])):
        data.iloc[nan_idx[0][i], nan_idx[1][i]] = replacement[i]

    return data
