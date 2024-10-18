import os

import pandas as pd
import streamlit as st
from openai import AuthenticationError

from alphastats.gui.utils.analysis_helper import (
    display_figure,
    gui_volcano_plot_differential_expression_analysis,
    helper_compare_two_groups,
    save_plot_to_session_state,
)
from alphastats.gui.utils.gpt_helper import (
    display_proteins,
    get_assistant_functions,
    get_general_assistant_functions,
    get_subgroups_for_each_group,
)
from alphastats.gui.utils.ollama_utils import LLMIntegration
from alphastats.gui.utils.openai_utils import (
    set_api_key,
)
from alphastats.gui.utils.options import interpretation_options
from alphastats.gui.utils.ui_helper import StateKeys, init_session_state, sidebar_info

init_session_state()
sidebar_info()


def select_analysis():
    """
    select box
    loads keys from option dicts
    """
    method = st.selectbox(
        "Analysis",
        # options=["Volcano plot"],
        options=list(interpretation_options(st.session_state).keys()),
    )
    return method


if StateKeys.DATASET not in st.session_state:
    st.info("Import Data first")
    st.stop()


st.markdown("### LLM Analysis")

sidebar_info()
init_session_state()


# set background to white so downloaded pngs dont have grey background
styl = """
    <style>
        .css-jc5rf5 {
            position: absolute;
            background: rgb(255, 255, 255);
            color: rgb(48, 46, 48);
            inset: 0px;
            overflow: hidden;
        }
    </style>
    """
st.markdown(styl, unsafe_allow_html=True)

# Initialize session state variables
if StateKeys.LLM_INTEGRATION not in st.session_state:
    st.session_state[StateKeys.LLM_INTEGRATION] = None
if StateKeys.API_TYPE not in st.session_state:
    st.session_state[StateKeys.API_TYPE] = "gpt"

if StateKeys.PLOT_LIST not in st.session_state:
    st.session_state[StateKeys.PLOT_LIST] = []

if StateKeys.MESSAGES not in st.session_state:
    st.session_state[StateKeys.MESSAGES] = []

if StateKeys.PLOT_SUBMITTED_CLICKED not in st.session_state:
    st.session_state[StateKeys.PLOT_SUBMITTED_CLICKED] = 0
    st.session_state[StateKeys.PLOT_SUBMITTED_COUNTER] = 0

if StateKeys.LOOKUP_SUBMITTED_CLICKED not in st.session_state:
    st.session_state[StateKeys.LOOKUP_SUBMITTED_CLICKED] = 0
    st.session_state[StateKeys.LOOKUP_SUBMITTED_COUNTER] = 0

if StateKeys.GPT_SUBMITTED_CLICKED not in st.session_state:
    st.session_state[StateKeys.GPT_SUBMITTED_CLICKED] = 0
    st.session_state[StateKeys.GPT_SUBMITTED_COUNTER] = 0


st.markdown("#### Configure LLM")

c1, _ = st.columns((1, 2))
with c1:
    st.session_state[StateKeys.API_TYPE] = st.selectbox(
        "Select LLM",
        ["gpt4o", "llama3.1 70b"],
        index=0 if st.session_state[StateKeys.API_TYPE] == "gpt4o" else 1,
    )

    if st.session_state[StateKeys.API_TYPE] == "gpt4o":
        api_key = st.text_input("Enter OpenAI API Key", type="password")
        set_api_key(api_key)

st.markdown("#### Analysis")
c1, c2 = st.columns((1, 2))

with c1:
    method = select_analysis()
    chosen_parameter_dict = helper_compare_two_groups()

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
    st.session_state[StateKeys.ORGANISM] = organism

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
        st.session_state[StateKeys.PLOT_SUBMITTED_CLICKED] += 1


if (
    st.session_state[StateKeys.PLOT_SUBMITTED_COUNTER]
    < st.session_state[StateKeys.PLOT_SUBMITTED_CLICKED]
):
    st.session_state[StateKeys.PLOT_SUBMITTED_COUNTER] = st.session_state[
        StateKeys.PLOT_SUBMITTED_CLICKED
    ]
    volcano_plot = gui_volcano_plot_differential_expression_analysis(
        chosen_parameter_dict
    )
    volcano_plot._update(plotting_parameter_dict)
    volcano_plot._annotate_result_df()
    volcano_plot._plot()
    genes_of_interest_colored = volcano_plot.get_colored_labels()
    genes_of_interest_colored_df = volcano_plot.get_colored_labels_df()
    print(genes_of_interest_colored_df)

    gene_names_colname = st.session_state[StateKeys.LOADER].gene_names
    prot_ids_colname = st.session_state[StateKeys.LOADER].index_column

    st.session_state[StateKeys.PROT_ID_TO_GENE] = dict(
        zip(
            genes_of_interest_colored_df[prot_ids_colname].tolist(),
            genes_of_interest_colored_df[gene_names_colname].tolist(),
        )
    )
    st.session_state[StateKeys.GENE_TO_PROT_ID] = dict(
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
    st.session_state[StateKeys.GENES_OF_INTEREST_COLORED] = genes_of_interest_colored
    # st.session_state["gene_functions"] = get_info(genes_of_interest_colored, organism)
    st.session_state[StateKeys.UPREGULATED] = [
        key
        for key in genes_of_interest_colored
        if genes_of_interest_colored[key] == "up"
    ]
    st.session_state[StateKeys.DOWNREGULATED] = [
        key
        for key in genes_of_interest_colored
        if genes_of_interest_colored[key] == "down"
    ]
    st.subheader("Genes of interest")
    c1, c2 = st.columns((1, 2), gap="medium")
    with c1:
        st.write("Upregulated genes")
        display_proteins(st.session_state[StateKeys.UPREGULATED], [])
    with c2:
        st.write("Downregulated genes")
        display_proteins([], st.session_state[StateKeys.DOWNREGULATED])

elif (
    st.session_state[StateKeys.PLOT_SUBMITTED_COUNTER] > 0
    and st.session_state[StateKeys.PLOT_SUBMITTED_COUNTER]
    == st.session_state[StateKeys.PLOT_SUBMITTED_CLICKED]
    and len(st.session_state[StateKeys.PLOT_LIST]) > 0
):
    with c2:
        display_figure(st.session_state[StateKeys.PLOT_LIST][-1][1].plot)

    st.subheader("Genes of interest")
    c1, c2 = st.columns((1, 2), gap="medium")
    with c1:
        st.write("Upregulated genes")
        display_proteins(st.session_state[StateKeys.UPREGULATED], [])
    with c2:
        st.write("Downregulated genes")
        display_proteins([], st.session_state[StateKeys.DOWNREGULATED])

st.session_state[StateKeys.INSTRUCTIONS] = (
    f"You are an expert biologist and have extensive experience in molecular biology, medicine and biochemistry.{os.linesep}"
    "A user will present you with data regarding proteins upregulated in certain cells "
    "sourced from UniProt and abstracts from scientific publications. They seek your "
    "expertise in understanding the connections between these proteins and their potential role "
    f"in disease genesis. {os.linesep}Provide a detailed and insightful, yet concise response based on the given information. Use formatting to make your response more human readable."
    f"The data you have has following groups and respective subgroups: {str(get_subgroups_for_each_group(st.session_state[StateKeys.DATASET].metadata))}."
    "Plots are visualized using a graphical environment capable of rendering images, you don't need to worry about that. If the data coming to"
    " you from a function has references to the literature (for example, PubMed), always quote the references in your response."
)
if "column" in chosen_parameter_dict and StateKeys.UPREGULATED in st.session_state:
    st.session_state[StateKeys.USER_PROMPT] = (
        f"We've recently identified several proteins that appear to be differently regulated in cells "
        f"when comparing {chosen_parameter_dict['group1']} and {chosen_parameter_dict['group2']} in the {chosen_parameter_dict['column']} group. "
        f"From our proteomics experiments, we know that the following ones are upregulated: {', '.join(st.session_state[StateKeys.UPREGULATED])}.{os.linesep}{os.linesep}"
        f"Here is the list of proteins that are downregulated: {', '.join(st.session_state[StateKeys.DOWNREGULATED])}.{os.linesep}{os.linesep}"
        f"Help us understand the potential connections between these proteins and how they might be contributing "
        f"to the differences. After that provide a high level summary"
    )

if StateKeys.USER_PROMPT in st.session_state:
    st.subheader("Automatically generated prompt based on gene functions:")
    with st.expander("Adjust system prompt (see example below)", expanded=False):
        st.session_state[StateKeys.INSTRUCTIONS] = st.text_area(
            "", value=st.session_state[StateKeys.INSTRUCTIONS], height=150
        )

    with st.expander("Adjust user prompt", expanded=True):
        st.session_state[StateKeys.USER_PROMPT] = st.text_area(
            "", value=st.session_state[StateKeys.USER_PROMPT], height=200
        )

gpt_submitted = st.button("Run LLM analysis")

if gpt_submitted and StateKeys.USER_PROMPT not in st.session_state:
    st.warning("Please enter a user prompt first")
    st.stop()

if gpt_submitted:
    st.session_state[StateKeys.GPT_SUBMITTED_CLICKED] += 1

# creating new assistant only once TODO: add a button to create new assistant
if (
    st.session_state[StateKeys.GPT_SUBMITTED_CLICKED]
    > st.session_state[StateKeys.GPT_SUBMITTED_COUNTER]
):
    if st.session_state[StateKeys.API_TYPE] == "gpt4o":
        set_api_key()

    try:
        if st.session_state[StateKeys.API_TYPE] == "gpt4o":
            st.session_state[StateKeys.LLM_INTEGRATION] = LLMIntegration(
                api_type="gpt",
                api_key=st.session_state[StateKeys.OPENAI_API_KEY],
                dataset=st.session_state[StateKeys.DATASET],
                metadata=st.session_state[StateKeys.DATASET].metadata,
            )
        else:
            st.session_state[StateKeys.LLM_INTEGRATION] = LLMIntegration(
                api_type="ollama",
                base_url=os.getenv("OLLAMA_BASE_URL", None),
                dataset=st.session_state[StateKeys.DATASET],
                metadata=st.session_state[StateKeys.DATASET].metadata,
            )
        st.success(
            f"{st.session_state[StateKeys.API_TYPE].upper()} integration initialized successfully!"
        )
    except AuthenticationError:
        st.warning(
            "Incorrect API key provided. Please enter a valid API key, it should look like this: sk-XXXXX"
        )
        st.stop()

if (
    StateKeys.LLM_INTEGRATION not in st.session_state
    or not st.session_state[StateKeys.LLM_INTEGRATION]
):
    st.warning("Please initialize the model first")
    st.stop()

llm = st.session_state[StateKeys.LLM_INTEGRATION]

# Set instructions and update tools
llm.tools = [
    *get_general_assistant_functions(),
    *get_assistant_functions(
        gene_to_prot_id_dict=st.session_state[StateKeys.GENE_TO_PROT_ID],
        metadata=st.session_state[StateKeys.DATASET].metadata,
        subgroups_for_each_group=get_subgroups_for_each_group(
            st.session_state[StateKeys.DATASET].metadata
        ),
    ),
]

if StateKeys.ARTIFACTS not in st.session_state:
    st.session_state[StateKeys.ARTIFACTS] = {}

if (
    st.session_state[StateKeys.GPT_SUBMITTED_COUNTER]
    < st.session_state[StateKeys.GPT_SUBMITTED_CLICKED]
):
    st.session_state[StateKeys.GPT_SUBMITTED_COUNTER] = st.session_state[
        StateKeys.GPT_SUBMITTED_CLICKED
    ]
    st.session_state[StateKeys.ARTIFACTS] = {}
    llm.messages = [
        {"role": "system", "content": st.session_state[StateKeys.INSTRUCTIONS]}
    ]
    response = llm.chat_completion(st.session_state[StateKeys.USER_PROMPT])

if st.session_state[StateKeys.GPT_SUBMITTED_CLICKED] > 0:
    if prompt := st.chat_input("Say something"):
        response = llm.chat_completion(prompt)
    for num, role_content_dict in enumerate(st.session_state[StateKeys.MESSAGES]):
        if role_content_dict["role"] == "tool" or role_content_dict["role"] == "system":
            continue
        if "tool_calls" in role_content_dict:
            continue
        with st.chat_message(role_content_dict["role"]):
            st.markdown(role_content_dict["content"])
            if num in st.session_state[StateKeys.ARTIFACTS]:
                for artefact in st.session_state[StateKeys.ARTIFACTS][num]:
                    if isinstance(artefact, pd.DataFrame):
                        st.dataframe(artefact)
                    elif "plotly" in str(type(artefact)):
                        st.plotly_chart(artefact)
