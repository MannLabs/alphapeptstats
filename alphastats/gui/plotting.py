import streamlit as st


class Plotting:
    def load_plotting_options(self):
        st.write(self.metadata_columns)
        self.plotting_options = {
            "Sampledistribution": {
                "method": {"options": ["violin", "box"], "label": "Plot layout"},
                "color": {
                    "options": [None] + self.dataset.metadata.columns.to_list(),
                    "label": "Color according to",
                },
                "plotting_function": self.dataset.plot_sampledistribution,
            },
            "Intensity": {
                "id": {
                    "options": self.dataset.mat.columns.to_list(),
                    "label": "ProteinID/ProteinGroup",
                },
                "method": {
                    "options": ["violin", "box", "scatter"],
                    "label": "Plot layout",
                },
                "color": {
                    "options": [None] + self.dataset.metadata.columns.to_list(),
                    "label": "Color according to",
                },
                "plotting_function": self.dataset.plot_sampledistribution,
            },
            "t-SNE": {}
        }

    def display_plotly_figure(self, plot):
        st.plotly_chart(plot)

    def get_plotting_options_from_dict(self, plot):
        """ 
        extract plotting options from dict amd display as selectbox or checkbox
        give selceted options to plotting function
        """
        plot_dict = self.plotting_options.get(plot)
        parameter_dict = {}
        for parameter in plot_dict.keys():
            if "options" in parameter.keys():
                chosen_parameter = st.selectbox(
                    parameter.get("label"), options=parameter.get("options")
                )
            else:
                chosen_parameter = st.checkbox(parameter.get("label"))
            parameter_dict[parameter] = chosen_parameter
        return plot_dict["plotting_function"](**parameter_dict)

    def choose_plotoptions(self, plot):
        if self.plotting_options is None and self.dataset is not None:
            self.load_plotting_options()

        if plot in self.plotting_options.keys():
            self.get_plotting_options_from_dict(plot)

        elif plot == "Volcano":
            group = st.selectbox(
                "Grouping variable", options=[None] + self.metadata_columns
            )
            if group is not None:
                unique_values = self.get_unique_values_from_column(group)
                group1 = st.selectbox("Group 1", options=["<select>"] + unique_values)
                group2 = st.selectbox("Group 2", options=["<select>"] + unique_values)
                method = st.selectbox(
                    "Differential Analysis using:",
                    options=["<select>", "anova", "wald", "ttest"],
                )
                if (
                    group1 != "<select>"
                    and group2 != "<select>"
                    and method != "<select>"
                ):
                    self.dataset.plot_volcano(
                        column=group, group1=group1, group2=group2, method=method
                    )

        elif plot == "Clustermap":
            self.dataset.plot_clustermap()

        elif plot == "Dendrogram":
            self.dataset.plot_dendogram()

    def add_plot_widget(self):
        if "n_rows" not in st.session_state:
            st.session_state.n_rows = 1

        add = st.button(label="add")

        if add:
            st.session_state.n_rows += 1
            st.experimental_rerun()

        for i in range(st.session_state.n_rows):
            # add text inputs here
            plot = st.selectbox(
                "Plot",
                options=[
                    "PCA",
                    "t-SNE",
                    "Sampledistribution",
                    "Intensity",
                    "Volcano",
                    "Clustermap",
                    "Dendrogram",
                ],
            )  # Pass index as ke
            self.choose_plotoptions(plot)
    
    def plots(self):
        if self.dataset is None:
            st.write("Load data")
        else:
            self.add_plot_widget()
