from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px

import math

class SampleHistogram:
    """
    Plot denisty plot of each sample in a matrix
    """
    def __init__(self, dataset):
        self.dataset = dataset
        self._get_matrix()
    
    def _get_matrix(self):
        self.data = self.dataset.mat.transpose()
    
    def _get_position_in_matrix(self, count):
        count += 1
        row = math.ceil(count / 4 )
        col = count % 4
            
        if col == 0:
            col = 4
        
        return col, row
    
    def _initalize_plot(self):
        n_samples = len(self.data.columns.to_list())
        
        if n_samples < 4:
            n_rows = 1
            n_columns = n_samples
        
        else:
            n_rows = math.ceil(n_samples/4)
            n_columns = 4
        
        self.plot = make_subplots(
            rows=n_rows, 
            cols=n_columns,
            subplot_titles=self.data.columns.to_list()
        )
    
    def plot(self):
        self._initalize_plot()
        
        for count, x in enumerate(self.data.columns):
            col, row = self._get_position_in_matrix(count)

            data_column = self.data[self.data[x] != 0][x].dropna()
           
            self.plot.add_trace(go.Histogram(x=data_column),row=row,col=col)
        
        self.plot.update_layout(showlegend=False)

        return self.plot

