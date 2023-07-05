
import scipy
import tqdm
import pandas as pd
import numpy as np
import plotly.express as px
from alphastats.statistics.StatisticUtils import StatisticUtils
import warnings

class MultiCovaAnalysis(StatisticUtils):
    def __init__(self, dataset, covariates: list, n_permutations: int=3, 
                 fdr: float=0.05, s0: float=0.05, subset: dict=None, plot:bool=False):

        self.dataset = dataset
        self.covariates = covariates
        self.n_permutations = n_permutations
        self.fdr = fdr
        self.s0 = s0
        self.subset = subset
        self.plot = plot

        self._check_covariat_input()
        self._subset_metadata()
        self._check_na_values()
        self._convert_string_to_binary()
        self._prepare_matrix()
    
    def _subset_metadata(self):
        columns_to_keep = self.covariates + [self.dataset.sample]
        if self.subset is not None:
             # dict structure {"column_name": ["group1", "group2"]}
            subset_column = list(self.subset.keys())[0]
            groups = self.subset.get(subset_column)
            self.metadata = self.dataset.metadata[self.dataset.metadata[subset_column].isin(groups)][columns_to_keep]

        else:
            self.metadata = self.dataset.metadata[columns_to_keep]
    
    def _check_covariat_input(self):
        # check whether covariates in metadata column
        misc_covariates = list(set(self.covariates) - set(self.dataset.metadata.columns.to_list()))
        if len(misc_covariates)> 0:
            warnings.warn(f"Covariates: {misc_covariates} are not found in Metadata.")
            self.covariates = [x for x in self.covariates if x not in misc_covariates]


    def _check_na_values(self):
        for covariate in self.covariates:
            if self.dataset.metadata[covariate].isna().any():
                self.covariates.remove(covariate)
                warnings.warn(f"Covariate: {covariate} contains missing values" +
                        f"in metadata and will not be used for analysis.")
                
    def _convert_string_to_binary(self):
        string_cols = self.metadata[self.covariates].select_dtypes(include=[object]).columns.to_list()
    
        if len(string_cols) > 0:
            for col in string_cols:
                col_values = list(set(self.metadata[col].to_list()))
                
                if len(col_values) == 2:
                    self.metadata[col] = np.where(self.metadata[col] == col_values[0], 0, 1)
                
                else:
                    if len(col_values) < 2: 
                        col_values.append("example")
                    
                    subset_prompt = "¨subset={" + col + ":[" + col_values[0] + ","+ col_values[1]+"]}"
                    warnings.warn(f"Covariate: {col} contains not exactly 2 binary values, instead {col_values}. "
                            f"Specify the values of the covariates you want to use for your analysis as: {subset_prompt} ")
                    self.covariates.remove(col)

    def _prepare_matrix(self):
        transposed = self.dataset.mat.transpose()
        transposed[self.dataset.index_column] = transposed.index
        transposed = transposed.reset_index(drop=True)
        self.transposed = transposed[self.metadata[self.dataset.sample].to_list()]

    def _plot_volcano_regression(self, res_real, variable):
        sig_col = res_real.filter(regex=variable+"_"+"FDR").columns[0]
        sig_level = sig_col.replace("_", " ")

        fig = px.scatter(x=res_real[variable+"_"+"fc"],
                        y=-np.log10(res_real[variable+"_"+"pval"]),
                        color=res_real[sig_col],
                        color_discrete_map={'sig': '#009599', 'non_sig': '#404040'},
                        hover_name=res_real[self.dataset.index_column],
                        title=variable,
                        labels=dict(x="beta value", y="-log10(p-value)", color=sig_level))
        fig.show()
        return fig

    
    def calculate(self):
        from alphastats.multicova import multicova
        
        if len(self.covariates) == 0:
            print("Covariates are invalid for analysis.")
            return
        
        res, tlim = multicova.full_regression_analysis(
            quant_data = self.transposed,
            annotation = self.metadata,
            covariates = self.covariates,
            sample_column = self.dataset.sample,
            n_permutations=self.n_permutations,
            fdr=self.fdr, 
            s0=self.s0
        )
        res[self.dataset.index_column] = self.dataset.mat.columns.to_list()
        plot_list = []
        
        if self.plot:
            for variable in self.covariates:
                plot = self._plot_volcano_regression(res_real=res, variable=variable)
                plot_list.append(plot)
        
        return res, plot_list
    




