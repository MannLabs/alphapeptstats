import numpy as no

class StatisticUtils:
    def __init__(self) -> None:
        pass
    
    def _add_metadata_column(self, group1_list: list, group2_list: list):

        # create new column in metadata with defined groups
        metadata = self.metadata

        sample_names = metadata[self.sample].to_list()
        misc_samples = list(set(group1_list + group2_list) - set(sample_names))
        if len(misc_samples) > 0:
            raise ValueError(
                f"Sample names: {misc_samples} are not described in Metadata."
            )

        column = "_comparison_column"
        conditons = [
            metadata[self.sample].isin(group1_list),
            metadata[self.sample].isin(group2_list),
        ]
        choices = ["group1", "group2"]
        metadata[column] = np.select(conditons, choices, default=np.nan)
        self.metadata = metadata

        return column, "group1", "group2"
