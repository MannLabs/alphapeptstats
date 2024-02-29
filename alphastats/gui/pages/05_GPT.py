import os
import streamlit as st
from openai import OpenAI, OpenAIError, AuthenticationError

try:
    from alphastats.gui.utils.analysis_helper import (
        check_if_options_are_loaded,
        convert_df,
        display_df,
        display_figure,
        download_figure,
        download_preprocessing_info,
        get_analysis,
        load_options,
        save_plot_to_session_state,
        gui_volcano_plot_differential_expression_analysis,
        helper_compare_two_groups,
    )
    from alphastats.gui.utils.gpt_helper import (
        get_assistant_functions,
        display_proteins,
        get_gene_function,
        get_info,
        get_subgroups_for_each_group,
        turn_args_to_float,
        perform_dimensionality_reduction,
        wait_for_run_completion,
        send_message_save_thread,
        try_to_set_api_key,
    )
    from alphastats.gui.utils.ui_helper import sidebar_info

except ModuleNotFoundError:
    from utils.analysis_helper import (
        check_if_options_are_loaded,
        convert_df,
        display_df,
        display_figure,
        download_figure,
        download_preprocessing_info,
        get_analysis,
        load_options,
        save_plot_to_session_state,
        gui_volcano_plot_differential_expression_analysis,
        helper_compare_two_groups,
    )
    from utils.gpt_helper import (
        get_assistant_functions,
        display_proteins,
        get_gene_function,
        get_info,
        get_subgroups_for_each_group,
        turn_args_to_float,
        perform_dimensionality_reduction,
        wait_for_run_completion,
        send_message_save_thread,
        try_to_set_api_key,
    )
    from utils.ui_helper import sidebar_info


st.session_state.plot_dict = {}


@check_if_options_are_loaded
def select_analysis():
    """
    select box
    loads keys from option dicts
    """
    method = st.selectbox(
        "Analysis",
        options=["Volcano plot"],
        # options=list(st.session_state.interpretation_options.keys()),
    )
    return method


if "dataset" not in st.session_state:
    st.info("Import Data first")
    st.stop()


st.markdown("### GPT4 Analysis")

sidebar_info()


# set background to white so downloaded pngs dont have grey background
styl = f"""
    <style>
        .css-jc5rf5 {{
            position: absolute;
            background: rgb(255, 255, 255);
            color: rgb(48, 46, 48);
            inset: 0px;
            overflow: hidden;
        }}
    </style>
    """
st.markdown(styl, unsafe_allow_html=True)


if "plot_list" not in st.session_state:
    st.session_state["plot_list"] = []


if "openai_model" not in st.session_state:
    # st.session_state["openai_model"] = "gpt-3.5-turbo-16k"
    st.session_state["openai_model"] = "gpt-4-1106-preview"

if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "plot_submitted_clicked" not in st.session_state:
    st.session_state["plot_submitted_clicked"] = 0
    st.session_state["plot_submitted_counter"] = 0

if "lookup_submitted_clicked" not in st.session_state:
    st.session_state["lookup_submitted_clicked"] = 0
    st.session_state["lookup_submitted_counter"] = 0

if "gpt_submitted_clicked" not in st.session_state:
    st.session_state["gpt_submitted_clicked"] = 0
    st.session_state["gpt_submitted_counter"] = 0

c1, c2 = st.columns((1, 2))


with c1:
    method = select_analysis()
    chosen_parameter_dict = helper_compare_two_groups()
    api_key = st.text_input("API Key", type="password")

    try_to_set_api_key(api_key)

    try:
        client = OpenAI(api_key=st.secrets["openai_api_key"])
    except OpenAIError:
        pass
    method = st.selectbox(
        "Differential Analysis using:",
        options=["ttest", "anova", "wald", "sam", "paired-ttest", "welch-ttest"],
    )
    chosen_parameter_dict.update({"method": method})

    # TODO streamlit doesnt allow nested columns check for updates

    labels = st.checkbox("Add label", value=True)

    draw_line = st.checkbox("Draw line", value=True)

    alpha = st.number_input(
        label="alpha", min_value=0.001, max_value=0.050, value=0.050
    )

    organism = st.number_input(
        label="UniProt organism ID, for example human is 9606, R. norvegicus is 10116",
        value=9606,
    )
    st.session_state["organism"] = organism

    min_fc = st.select_slider("Foldchange cutoff", range(0, 3), value=1)

    plotting_parameter_dict = {
        "labels": labels,
        "draw_line": draw_line,
        "alpha": alpha,
        "min_fc": min_fc,
    }

    if method == "sam":
        perm = st.number_input(
            label="Number of Permutations", min_value=1, max_value=1000, value=10
        )
        fdr = st.number_input(
            label="FDR cut off", min_value=0.005, max_value=0.1, value=0.050
        )
        chosen_parameter_dict.update({"perm": perm, "fdr": fdr})

    plot_submitted = st.button("Plot")
    if plot_submitted:
        st.session_state["plot_submitted_clicked"] += 1


if (
    st.session_state["plot_submitted_counter"]
    < st.session_state["plot_submitted_clicked"]
):
    st.session_state["plot_submitted_counter"] = st.session_state[
        "plot_submitted_clicked"
    ]
    volcano_plot = gui_volcano_plot_differential_expression_analysis(
        chosen_parameter_dict, user_session_id=st.session_state.user_session_id
    )
    volcano_plot._update(plotting_parameter_dict)
    volcano_plot._annotate_result_df()
    volcano_plot._plot()
    genes_of_interest_colored = volcano_plot.get_colored_labels()
    genes_of_interest_colored_df = volcano_plot.get_colored_labels_df()
    print(genes_of_interest_colored_df)

    gene_names_colname = st.session_state["loader"].gene_names
    prot_ids_colname = st.session_state["loader"].index_column

    st.session_state["prot_id_to_gene"] = dict(
        zip(
            genes_of_interest_colored_df[prot_ids_colname].tolist(),
            genes_of_interest_colored_df[gene_names_colname].tolist(),
        )
    )
    st.session_state["gene_to_prot_id"] = dict(
        zip(
            genes_of_interest_colored_df[gene_names_colname].tolist(),
            genes_of_interest_colored_df[prot_ids_colname].tolist(),
        )
    )

    with c2:
        display_figure(volcano_plot.plot)

    if not genes_of_interest_colored:
        st.text("No proteins of interest found.")
        st.stop()
    print("genes_of_interest", genes_of_interest_colored)

    save_plot_to_session_state(volcano_plot, method)
    st.session_state["genes_of_interest_colored"] = genes_of_interest_colored
    # st.session_state["gene_functions"] = get_info(genes_of_interest_colored, organism)
    st.session_state["upregulated"] = [
        key
        for key in genes_of_interest_colored
        if genes_of_interest_colored[key] == "up"
    ]
    st.session_state["downregulated"] = [
        key
        for key in genes_of_interest_colored
        if genes_of_interest_colored[key] == "down"
    ]
    st.subheader("Genes of interest")
    c1, c2 = st.columns((1, 2), gap="medium")
    with c1:
        st.write("Upregulated genes")
        display_proteins(st.session_state["upregulated"], [])
    with c2:
        st.write("Downregulated genes")
        display_proteins([], st.session_state["downregulated"])

elif (
    st.session_state["plot_submitted_counter"] > 0
    and st.session_state["plot_submitted_counter"]
    == st.session_state["plot_submitted_clicked"]
    and len(st.session_state["plot_list"]) > 0
):
    with c2:
        display_figure(st.session_state["plot_list"][-1][1].plot)

    st.subheader("Genes of interest")
    c1, c2 = st.columns((1, 2), gap="medium")
    with c1:
        st.write("Upregulated genes")
        display_proteins(st.session_state["upregulated"], [])
    with c2:
        st.write("Downregulated genes")
        display_proteins([], st.session_state["downregulated"])

st.session_state["instructions"] = (
    f"You are an expert biologist and have extensive experience in molecular biology, medicine and biochemistry.{os.linesep}"
    "A user will present you with data regarding proteins upregulated in certain cells "
    "sourced from UniProt and abstracts from scientific publications. They seek your "
    "expertise in understanding the connections between these proteins and their potential role "
    f"in disease genesis. {os.linesep}Provide a detailed and insightful, yet concise response based on the given information. "
    f"The data you have has following groups and respective subgroups: {str(get_subgroups_for_each_group(st.session_state.dataset.metadata))}."
    "Plots are visualized using a graphical environment capable of rendering images, you don't need to worry about that."
)
if "column" in chosen_parameter_dict and "upregulated" in st.session_state:
    st.session_state["user_prompt"] = (
        f"We've recently identified several proteins that appear to be differently regulated in cells "
        f"when comparing {chosen_parameter_dict['group1']} and {chosen_parameter_dict['group2']} in the {chosen_parameter_dict['column']} group. "
        f"From our proteomics experiments, we know that the following ones are upregulated: {', '.join(st.session_state['upregulated'])}.{os.linesep}{os.linesep}"
        f"Here is the list of proteins that are downregulated: {', '.join(st.session_state['downregulated'])}.{os.linesep}{os.linesep}"
        f"Help us understand the potential connections between these proteins and how they might be contributing "
        f"to the differences. After that provide a high level summary"
    )

if "user_prompt" in st.session_state:
    st.subheader("Automatically generated prompt based on gene functions:")
    with st.expander("Adjust system prompt (see example below)", expanded=False):
        st.session_state["instructions"] = st.text_area(
            "", value=st.session_state["instructions"], height=150
        )

    with st.expander("Adjust user prompt", expanded=True):
        st.session_state["user_prompt"] = st.text_area(
            "", value=st.session_state["user_prompt"], height=200
        )

gpt_submitted = st.button("Run GPT analysis")

if gpt_submitted and "user_prompt" not in st.session_state:
    st.warning("Please enter a user prompt first")
    st.stop()

if gpt_submitted:
    st.session_state["gpt_submitted_clicked"] += 1

# creating new assistant only once TODO: add a button to create new assistant
if (
    st.session_state["gpt_submitted_clicked"]
    > st.session_state["gpt_submitted_counter"]
):
    try_to_set_api_key()
    
    client = OpenAI(api_key=st.secrets["openai_api_key"])
    
    try:
        st.session_state["assistant"] = client.beta.assistants.create(
        instructions=st.session_state["instructions"],
        name="Proteomics interpreter",
        model=st.session_state["openai_model"],
        tools=get_assistant_functions(
            gene_to_prot_id_dict=st.session_state["gene_to_prot_id"],
            metadata=st.session_state["dataset"].metadata,
            subgroups_for_each_group=get_subgroups_for_each_group(
                st.session_state["dataset"].metadata
            ),
        ),
    )
    except AuthenticationError:
        st.warning("Incorrect API key provided. Please enter a valid API key, it should look like this: sk-XXXXX")
        st.stop()

if "plot_enum_dict" not in st.session_state:
    st.session_state["plot_enum_dict"] = {}

if (
    st.session_state["gpt_submitted_counter"]
    < st.session_state["gpt_submitted_clicked"]
):
    st.session_state["gpt_submitted_counter"] = st.session_state[
        "gpt_submitted_clicked"
    ]
    st.session_state["plot_enum_dict"] = {}
    thread = client.beta.threads.create()
    st.session_state["thread_id"] = thread.id
    plots = send_message_save_thread(client, st.session_state["user_prompt"])
    if plots:
        st.session_state["plot_enum_dict"][len(st.session_state.messages) - 1] = plots

if st.session_state["gpt_submitted_clicked"] > 0:
    if prompt := st.chat_input("Say something"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        plots = send_message_save_thread(client, prompt)
        if plots:
            st.session_state["plot_enum_dict"][
                len(st.session_state.messages) - 1
            ] = plots
    for num, role_content_dict in enumerate(st.session_state.messages):
        with st.chat_message(role_content_dict["role"]):
            st.markdown(role_content_dict["content"])
            if num in st.session_state["plot_enum_dict"]:
                for plot in st.session_state["plot_enum_dict"][num]:
                    st.plotly_chart(plot)
    print(st.session_state["plot_enum_dict"])
