import plotly.express as px
import requests
import pandas as pd
from io import StringIO
import numpy as np
from alphastats.utils import check_internetconnection, check_if_df_empty


class enrichement_df(pd.DataFrame):
    # this is that added methods dont get lost when operatons on pd Dataframe get performed
    @property
    def _constructor(self):
        return enrichement_df

    def _modify_df(self):
        self["Description"] = self["term"] + " " + self["description"]
        self["-log10(P-value)"] = -np.log10(self["p_value"])
        self["over_under"] = np.where(
            self["over_under"] == "o", "over-represented", "under-represented"
        )

    @check_if_df_empty
    def plot_scatter(self):
        """
        Plot Scatterplot with -log10(p-value) on x-axis and effect size on y-axis.
        Datapoints will be colored by Gene Ontology-term category.

        Returns:
            plotly.graph_objects._figure.Figure: Scatterplot of GO terms
        """
        self._modify_df()
        plot = px.scatter(
            self,
            x="-log10(P-value)",
            y="effect_size",
            size=self["foreground_count"],
            color="category",
            hover_name=self["Description"],
        )
        return plot

    @check_if_df_empty
    def plot_bar(self):
        """
        Plot p-values as Barplot

        Returns:
            plotly.graph_objects._figure.Figure: Barplot
        """
        self._modify_df()
        plot = px.bar(
            self,
            x="-log10(P-value)",
            y="Description",
            orientation="h",
            color="over_under",
            color_discrete_map={
                "under-represented": "#009599",
                "over-represented": "#B65EAF",
            },
        )
        return plot


class Enrichment:
    @staticmethod
    def _extract_protein_ids(entry):
        try:
            proteins = entry.split(",")
            protein_id_list = []
            for protein in proteins:
                # 'sp|P0DMV9|HS71B_HUMAN,sp|P0DMV8|HS71A_HUMAN',
                if "|" in protein:
                    fasta_header_split = protein.split("|")
                else:
                    fasta_header_split = protein
                if isinstance(fasta_header_split, str):
                    #  'ENSEMBL:ENSBTAP00000007350',
                    if "ENSEMBL:" in fasta_header_split:
                        protein_id = fasta_header_split.replace("ENSEMBL:", "")
                    else:
                        protein_id = fasta_header_split
                else:
                    protein_id = fasta_header_split[1]
                protein_id_list.append(protein_id)
            protein_id_concentate = ";".join(protein_id_list)
            # ADD REV to the protein ID, else there will be duplicates in the ProteinGroup column
            if "REV_" in entry:
                protein_id_concentate = "REV_" + protein_id_concentate

        except AttributeError:
            protein_id_concentate = entry

        return protein_id_concentate

    def _get_ptm_proteins(self, sample=None):

        if self.evidence_df is None:
            raise ValueError(
                "No informations about PTMs."
                "Either load a list of ProteinIDs containing PTMs"
                "or DataSet.load_ptm_df()"
            )

        if "ProteinGroup" not in self.evidence_df.columns:
            self.evidence_df["ProteinGroup"] = self.evidence_df["Proteins"].map(
                self._extract_protein_ids
            )

        if isinstance(sample, str):
            protein_list = self.evidence_df[
                (self.evidence_df["Modifications"] != "Unmodified")
                & (self.evidence_df["Experiment"] == sample)
            ]["ProteinGroup"].to_list()

        elif isinstance(sample, list):
            protein_list = self.evidence_df[
                (self.evidence_df["Modifications"] != "Unmodified")
                & (self.evidence_df["Experiment"].isin(sample))
            ]["ProteinGroup"].to_list()

        else:
            protein_list = self.evidence_df[
                self.evidence_df["Modifications"] != "Unmodified"
            ]["ProteinGroup"].to_list()

        protein_list = [str(x) for x in protein_list]
        return protein_list

    def go_characterize_foreground(self, protein_list, tax_id=9606):
        """
        Display existing functional annotations for your protein(s) of interest.
        No statistical test for enrichment is performed.
        Using the API connection from a GO tool: https://agotool.org

        Args:
            tax_id (int, optional): NCBI taxon identifier used as background. Defaults to 9606 (=Homo sapiens).
            protein_list (list): list of enriched protein ids in the foreground sample.

        Returns:
            pandas.DataFrame: DataFrame
            * ``'rank'``: The rank is a combination of uncorrected p value and effect size (based on s value). It serves to highlight the most interesting results and tries to emphasize the importance of the effect size.
            * ``'term'``: A unique identifier for a specific functional category.
            * ``'description'``: A short description (or title) of a functional term.
            * ``'p value corrected'``: p value without multiple testing correction, stemming from either Fisher's exact test or Kolmorov Smirnov test (only for "Gene Ontology Cellular Component TEXTMINING", "Brenda Tissue Ontoloy", and "Disease Ontology" since these are based on a continuous score from text mining rather than a binary classification).
            * ``'effect size'``: Proportion of the Foregrounda nd the Background
            * ``'description'``: A short description (or title) of a functional term.
            * ``'year'``: Year of the scientific publication.
            * ``'over_under'``: Overrepresented (o) or underrepresented (u).
            * ``'s_value'``: The s value is a combination of (minus log) p value and effect size.
            * ``'ratio_in_foreground'``: The ratio in the ForeGround is calculated by dividing the number of positive associations for a given term by the number of input proteins (protein groups) for the Foreground.
            * ``'ratio_in_background'``: The ratio in the BackGround is analogous to the above ratio in the FG, using the associations for the background and Background input proteins instead.
            * ``'foreground_count'``: The ForeGround count consists of the number of all positive associations for the given term (i.e. how many proteins are associated with the given term).
            * ``'foreground_n'``: ForeGround n is comprised of the number of input proteins for the Foreground.
            * ``'background_count'``: The BackGround count is analogous to the "FG count" for the Background.
            * ``'background_n'``: BackGround n is analogous to "FG n".
            * ``'foreground_ids'``: ForeGround IDentifierS are semicolon separated protein identifers of the Forground that are associated with the given term.
            * ``'background_ids'``: BackGround IDentifierS are analogous to "FG IDs" for the Background.
            * ``'etype'``: Short for "Entity type", numeric internal identifer for different functional categories.

        """
        check_internetconnection()
        protein_list = [self._extract_protein_ids(protein) for protein in protein_list]
        protein_list = "%0d".join(protein_list)
        url = r"https://agotool.org/api_orig"

        result = requests.post(
            url,
            params={
                "output_format": "tsv",
                "enrichment_method": "characterize_foreground",
                "taxid": tax_id,
            },
            data={"foreground": protein_list},
        )

        result_df = enrichement_df(pd.read_csv(StringIO(result.text), sep="\t"))
        return result_df

    def go_abundance_correction(self, bg_sample, fg_sample=None, fg_protein_list=None):
        """
        Gene Ontology Enrichement Analysis with abundance correction.
        Using the API connection from a GO tool: https://agotool.org

        For the analysis modified proteins in the foreground sample are compared with proteins and
        their intensity of the background sample.
        In case there is no information about PTMs in the dataset a list of enriched proteins in the
        foreground can be loaded. This list of Protein IDs can be obtaint by performing a differential
        expression analysis or a ANOVA.


        Args:
            fg_sample (str): name of foreground sample
            bg_sample (str): name of background sample
            fg_protein_list (list, optional): list of enriched protein ids in the foreground sample. Defaults to None.

        Returns:
            pandas.DataFrame: DataFrame
            * ``'rank'``: The rank is a combination of uncorrected p value and effect size (based on s value). It serves to highlight the most interesting results and tries to emphasize the importance of the effect size.
            * ``'term'``: A unique identifier for a specific functional category.
            * ``'description'``: A short description (or title) of a functional term.
            * ``'p value corrected'``: p value without multiple testing correction, stemming from either Fisher's exact test or Kolmorov Smirnov test (only for "Gene Ontology Cellular Component TEXTMINING", "Brenda Tissue Ontoloy", and "Disease Ontology" since these are based on a continuous score from text mining rather than a binary classification).
            * ``'effect size'``: Proportion of the Foregrounda nd the Background
            * ``'description'``: A short description (or title) of a functional term.
            * ``'year'``: Year of the scientific publication.
            * ``'over_under'``: Overrepresented (o) or underrepresented (u).
            * ``'s_value'``: The s value is a combination of (minus log) p value and effect size.
            * ``'ratio_in_foreground'``: The ratio in the ForeGround is calculated by dividing the number of positive associations for a given term by the number of input proteins (protein groups) for the Foreground.
            * ``'ratio_in_background'``: The ratio in the BackGround is analogous to the above ratio in the FG, using the associations for the background and Background input proteins instead.
            * ``'foreground_count'``: The ForeGround count consists of the number of all positive associations for the given term (i.e. how many proteins are associated with the given term).
            * ``'foreground_n'``: ForeGround n is comprised of the number of input proteins for the Foreground.
            * ``'background_count'``: The BackGround count is analogous to the "FG count" for the Background.
            * ``'background_n'``: BackGround n is analogous to "FG n".
            * ``'foreground_ids'``: ForeGround IDentifierS are semicolon separated protein identifers of the Forground that are associated with the given term.
            * ``'background_ids'``: BackGround IDentifierS are analogous to "FG IDs" for the Background.
            * ``'etype'``: Short for "Entity type", numeric internal identifer for different functional categories.
        """

        check_internetconnection()
        # get PTMs for fg_sample
        if fg_protein_list is None:
            fg_protein_list = self._get_ptm_proteins(sample=fg_sample)

        fg_protein_list = [
            self._extract_protein_ids(protein) for protein in fg_protein_list
        ]
        fg_protein_list = "%0d".join(fg_protein_list)

        # get intensity for bg_sample
        bg_protein = "%0d".join(self.mat.loc[bg_sample].index.to_list())
        bg_intensity = "%0d".join(self.mat.loc[bg_sample].astype(str).values.tolist())

        url = r"https://agotool.org/api_orig"
        result = requests.post(
            url,
            params={
                "output_format": "tsv",
                "enrichment_method": "abundance_correction",
            },
            data={
                "foreground": fg_protein_list,
                "background": bg_protein,
                "background_intensity": bg_intensity,
            },
        )
        result_df = enrichement_df(pd.read_csv(StringIO(result.text), sep="\t"))
        return result_df

    def go_compare_samples(self, fg_sample, bg_sample):
        """
        Gene Ontology Enrichement Analysis without abundance correction.
        Using the API connection from a GO tool: https://agotool.org

        Args:
            fg_sample (str): name of the foreground sample
            bg_sample (str): name of the background sample

        Returns:
            pandas.DataFrame: DataFrame
            * ``'rank'``: The rank is a combination of uncorrected p value and effect size (based on s value). It serves to highlight the most interesting results and tries to emphasize the importance of the effect size.
            * ``'term'``: A unique identifier for a specific functional category.
            * ``'description'``: A short description (or title) of a functional term.
            * ``'p value corrected'``: p value without multiple testing correction, stemming from either Fisher's exact test or Kolmorov Smirnov test (only for "Gene Ontology Cellular Component TEXTMINING", "Brenda Tissue Ontoloy", and "Disease Ontology" since these are based on a continuous score from text mining rather than a binary classification).
            * ``'effect size'``: Proportion of the Foregrounda nd the Background
            * ``'description'``: A short description (or title) of a functional term.
            * ``'year'``: Year of the scientific publication.
            * ``'over_under'``: Overrepresented (o) or underrepresented (u).
            * ``'s_value'``: The s value is a combination of (minus log) p value and effect size.
            * ``'ratio_in_foreground'``: The ratio in the ForeGround is calculated by dividing the number of positive associations for a given term by the number of input proteins (protein groups) for the Foreground.
            * ``'ratio_in_background'``: The ratio in the BackGround is analogous to the above ratio in the FG, using the associations for the background and Background input proteins instead.
            * ``'foreground_count'``: The ForeGround count consists of the number of all positive associations for the given term (i.e. how many proteins are associated with the given term).
            * ``'foreground_n'``: ForeGround n is comprised of the number of input proteins for the Foreground.
            * ``'background_count'``: The BackGround count is analogous to the "FG count" for the Background.
            * ``'background_n'``: BackGround n is analogous to "FG n".
            * ``'foreground_ids'``: ForeGround IDentifierS are semicolon separated protein identifers of the Forground that are associated with the given term.
            * ``'background_ids'``: BackGround IDentifierS are analogous to "FG IDs" for the Background.
            * ``'etype'``: Short for "Entity type", numeric internal identifer for different functional categories.
        """

        check_internetconnection()
        # get protein ids for samples
        fg_proteins = "%0d".join(self._get_ptm_proteins(sample=fg_sample))
        bg_proteins = "%0d".join(self._get_ptm_proteins(sample=bg_sample))

        url = r"https://agotool.org/api_orig"
        result = requests.post(
            url,
            params={"output_format": "tsv", "enrichment_method": "compare_samples"},
            data={"foreground": fg_proteins, "background": bg_proteins},
        )
        result_df = enrichement_df(pd.read_csv(StringIO(result.text), sep="\t"))
        return result_df

    def go_genome(self, tax_id=9606, fg_sample=None, protein_list=None):
        """
        Gene Ontology Enrichement Analysis using a Background from UniProt Reference Proteomes.
        Using the API connection from a GO tool: https://agotool.org

        Args:
            tax_id (int, optional): NCBI taxon identifier used as background. Defaults to 9606 (=Homo sapiens).
            fg_sample (str, optional): name of sample used as foreground. Defaults to None.
            protein_list (list, optional): list of enriched protein ids in the foreground sample. Defaults to None.

        Returns:
            pandas.DataFrame: DataFrame
            * ``'rank'``: The rank is a combination of uncorrected p value and effect size (based on s value). It serves to highlight the most interesting results and tries to emphasize the importance of the effect size.
            * ``'term'``: A unique identifier for a specific functional category.
            * ``'description'``: A short description (or title) of a functional term.
            * ``'p value corrected'``: p value without multiple testing correction, stemming from either Fisher's exact test or Kolmorov Smirnov test (only for "Gene Ontology Cellular Component TEXTMINING", "Brenda Tissue Ontoloy", and "Disease Ontology" since these are based on a continuous score from text mining rather than a binary classification).
            * ``'effect size'``: Proportion of the Foregrounda nd the Background
            * ``'description'``: A short description (or title) of a functional term.
            * ``'year'``: Year of the scientific publication.
            * ``'over_under'``: Overrepresented (o) or underrepresented (u).
            * ``'s_value'``: The s value is a combination of (minus log) p value and effect size.
            * ``'ratio_in_foreground'``: The ratio in the ForeGround is calculated by dividing the number of positive associations for a given term by the number of input proteins (protein groups) for the Foreground.
            * ``'ratio_in_background'``: The ratio in the BackGround is analogous to the above ratio in the FG, using the associations for the background and Background input proteins instead.
            * ``'foreground_count'``: The ForeGround count consists of the number of all positive associations for the given term (i.e. how many proteins are associated with the given term).
            * ``'foreground_n'``: ForeGround n is comprised of the number of input proteins for the Foreground.
            * ``'background_count'``: The BackGround count is analogous to the "FG count" for the Background.
            * ``'background_n'``: BackGround n is analogous to "FG n".
            * ``'foreground_ids'``: ForeGround IDentifierS are semicolon separated protein identifers of the Forground that are associated with the given term.
            * ``'background_ids'``: BackGround IDentifierS are analogous to "FG IDs" for the Background.
            * ``'etype'``: Short for "Entity type", numeric internal identifer for different functional categories.
        """

        check_internetconnection()

        if protein_list is None:
            protein_list = self._get_ptm_proteins(sample=fg_sample)

        protein_list = [self._extract_protein_ids(protein) for protein in protein_list]
        protein_list = "%0d".join(protein_list)
        url = r"https://agotool.org/api_orig"

        result = requests.post(
            url,
            params={
                "output_format": "tsv",
                "enrichment_method": "genome",
                "taxid": tax_id,
            },
            data={"foreground": protein_list},
        )

        result_df = enrichement_df(pd.read_csv(StringIO(result.text), sep="\t"))
        return result_df
