from alphastats.loader import BaseLoader

# Philosopher
class FragPipeLoader(BaseLoader):
    # https://github.com/Nesvilab/MSFragger/wiki/Interpreting-MSFragger-Output

    # SAMPLE
    # columm Spectrum

    # https://fragpipe.nesvilab.org/docs/tutorial_fragpipe_outputs.html#combined_proteintsv
    # Frag pipe
    # https://github.com/Nesvilab/philosopher/wiki/Combined-protein-reports
    def __init__(self):
        self.sample_value_column = "[experiment] MaxLFQ Intensity ", # normalized protein intensity using the unique+razor sequences after razor assignment calculated using the MaxLFQ method
       
        self.index_column = "Protein",
        #self.filter_column = "Indistinguishable Proteins"? "Protein Probability"
        self.software = "MSFragger_Philosopher"