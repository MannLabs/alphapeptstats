import os
import streamlit as st
import pandas as pd
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
        get_general_assistant_functions,
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
        get_general_assistant_functions,
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


st.markdown("### Chat with GPT")

sidebar_info()


# set background to white so downloaded pngs dont have grey background
style = f"""
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
st.markdown(style, unsafe_allow_html=True)


if "plot_list_chat" not in st.session_state:
    st.session_state["plot_list_chat"] = []

if "plotting_options" not in st.session_state:
    st.session_state["plotting_options"] = {}

if "openai_model" not in st.session_state:
    # st.session_state["openai_model"] = "gpt-3.5-turbo-16k"
    st.session_state["openai_model"] = "gpt-4-0125-preview"  # "gpt-4-1106-preview"

if "messages_chat" not in st.session_state:
    st.session_state["messages_chat"] = []

if "prompt_submitted_clicked_chat" not in st.session_state:
    st.session_state["prompt_submitted_clicked_chat"] = 0
    st.session_state["prompt_submitted_counter_chat"] = 0

if "gpt_submitted_clicked_chat" not in st.session_state:
    st.session_state["gpt_submitted_clicked_chat"] = 0
    st.session_state["gpt_submitted_counter_chat"] = 0

c1, c2 = st.columns((1, 2))
st.subheader("Necessary context:")
st.session_state["upregulated_chat"] = st.text_area(
    "List of upregulated proteins / genes:", height=75
)

st.session_state["downregulated_chat"] = st.text_area(
    "List of downregulated proteins / genes:", height=75
)


import re


def custom_string_list_parser(input_str):
    """
    Parses a string that represents a list of strings, handling misformatted double quotes.

    Args:
    - input_str (str): The string representation of the list to be parsed.

    Returns:
    - list: A Python list of strings that have been extracted and corrected.
    """

    # Trim leading and trailing whitespaces and brackets if present
    trimmed_str = re.sub(r"^[\[\{\(]", "", input_str.strip())
    trimmed_str = re.sub(r"[\]\}\)]$", "", trimmed_str)

    # Split by separators, considering quotes but ignoring misformatted ones
    items = re.split(r',(?=(?:[^"]*"[^"]*")*[^"]*$)', trimmed_str)

    clean_items = []
    for item in items:
        # Remove leading and trailing whitespaces
        item = item.strip()
        # Remove leading and trailing quotes
        item = re.sub(r'^"|"$', "", item)
        # Handle escaping properly. Replace two double quotes with one.
        # If quotes were meant to be part of the string, they should be doubled.
        item = item.replace('""', '"')

        clean_items.append(item)

    return clean_items


# Example usage
input_str = '["VCL;HEL114", "P4HB", "An erroneous ""string"" here", "PSME2"]'
parsed_list = custom_string_list_parser(input_str)
print(parsed_list)

st.session_state["upregulated_chat"] = custom_string_list_parser(
    st.session_state["upregulated_chat"]
)
st.session_state["downregulated_chat"] = custom_string_list_parser(
    st.session_state["downregulated_chat"]
)

with c1:

    api_key = st.text_input("API Key", type="password")

with c2:
    organism = st.number_input(
        label="UniProt organism ID, e.g. human is 9606, R. norvegicus is 10116",
        value=9606,
    )
    st.session_state["organism"] = organism

try_to_set_api_key(api_key)

try:
    client = OpenAI(api_key=st.secrets["openai_api_key"])
except OpenAIError:
    pass

    # TODO streamlit doesnt allow nested columns check for updates

    # st.session_state["prot_id_to_gene"] = dict(
    #     zip(
    #         genes_of_interest_colored_df[prot_ids_colname].tolist(),
    #         genes_of_interest_colored_df[gene_names_colname].tolist(),
    #     )
    # )
    # st.session_state["gene_to_prot_id"] = dict(
    #     zip(
    #         genes_of_interest_colored_df[gene_names_colname].tolist(),
    #         genes_of_interest_colored_df[prot_ids_colname].tolist(),
    #     )
    # )

    # st.session_state["genes_of_interest_colored"] = genes_of_interest_colored
    # # st.session_state["gene_functions"] = get_info(genes_of_interest_colored, organism)
    # st.session_state["upregulated"] = [
    #     key
    #     for key in genes_of_interest_colored
    #     if genes_of_interest_colored[key] == "up"
    # ]
    # st.session_state["downregulated"] = [
    #     key
    #     for key in genes_of_interest_colored
    #     if genes_of_interest_colored[key] == "down"
    # ]

c1, c2 = st.columns((1, 1))

with c1:
    group1 = st.text_input("Comparison group 1. e.g. healthy")
    st.session_state["group1_chat"] = group1

with c2:
    group2 = st.text_input("Comparison group 2. e.g. diseased")
    st.session_state["group2_chat"] = group2

prompt_submitted = st.button("Create user prompt")
st.session_state["prompt_submitted_counter"] = 0


if prompt_submitted:
    st.session_state["prompt_submitted_clicked_chat"] += 1

if st.session_state["prompt_submitted_clicked_chat"] == 0:
    st.stop()

if (
    not st.session_state["upregulated_chat"]
    or not st.session_state["downregulated_chat"]
):
    st.warning("Please enter upregulated and downregulated proteins")
    st.stop()

if not st.session_state["group1_chat"] or not st.session_state["group2_chat"]:
    st.warning("Please enter group names")
    st.stop()

if (
    st.session_state["prompt_submitted_clicked_chat"]
    > st.session_state["prompt_submitted_counter_chat"]
):
    st.session_state["prompt_submitted_counter_chat"] = st.session_state[
        "prompt_submitted_clicked_chat"
    ]
    st.session_state["instructions"] = (
        f"You are an expert biologist and have extensive experience in molecular biology, medicine and biochemistry.{os.linesep}"
        "A user will present you with data regarding proteins upregulated in certain cells "
        "sourced from UniProt and abstracts from scientific publications. They seek your "
        "expertise in understanding the connections between these proteins and their potential role "
        f"in disease genesis. {os.linesep}Provide a detailed and insightful, yet concise response based on the given information. "
        "Plots are visualized using a graphical environment capable of rendering images, you don't need to worry about that."
    )
    st.session_state["user_prompt_chat"] = (
        f"We've recently identified several proteins that appear to be differently regulated in cells "
        f"when comparing {st.session_state['group1_chat']} and {st.session_state['group2_chat']}. "
        f"From our proteomics experiments, we know that the following ones are upregulated: {', '.join(st.session_state['upregulated_chat'])}.{os.linesep}{os.linesep}"
        f"Here is the list of proteins that are downregulated: {', '.join(st.session_state['downregulated_chat'])}.{os.linesep}{os.linesep}"
        f"Help us understand the potential connections between these proteins and how they might be contributing "
        f"to the differences. After that provide a high level summary"
    )


if "user_prompt_chat" in st.session_state:
    st.subheader("Automatically generated prompt:")
    with st.expander("Adjust system prompt (see example below)", expanded=False):
        st.session_state["instructions"] = st.text_area(
            "", value=st.session_state["instructions"], height=150
        )

    with st.expander("Adjust user prompt", expanded=True):
        st.session_state["user_prompt_chat"] = st.text_area(
            "", value=st.session_state["user_prompt_chat"], height=200
        )

gpt_submitted = st.button("Run GPT analysis")

if gpt_submitted and "user_prompt_chat" not in st.session_state:
    st.warning("Please enter a user prompt first")
    st.stop()

if gpt_submitted:
    st.session_state["gpt_submitted_clicked_chat"] += 1

if (
    st.session_state["gpt_submitted_clicked_chat"]
    > st.session_state["gpt_submitted_counter_chat"]
):
    try_to_set_api_key()

    client = OpenAI(api_key=st.secrets["openai_api_key"])

    try:
        st.session_state["assistant_chat"] = client.beta.assistants.create(
            instructions=st.session_state["instructions"],
            name="Proteomics interpreter",
            model=st.session_state["openai_model"],
            tools=get_general_assistant_functions(),
        )
        print(
            st.session_state["assistant_chat"], type(st.session_state["assistant_chat"])
        )
    except AuthenticationError:
        st.warning(
            "Incorrect API key provided. Please enter a valid API key, it should look like this: sk-XXXXX"
        )
        st.stop()

if "artefact_enum_dict_chat" not in st.session_state:
    st.session_state["artefact_enum_dict_chat"] = {}

if (
    st.session_state["gpt_submitted_counter_chat"]
    < st.session_state["gpt_submitted_clicked_chat"]
):
    st.session_state["gpt_submitted_counter_chat"] = st.session_state[
        "gpt_submitted_clicked_chat"
    ]
    st.session_state["artefact_enum_dict_chat"] = {}
    thread = client.beta.threads.create()
    thread_id = thread.id
    st.session_state["thread_id_chat"] = thread_id
    artefacts = send_message_save_thread(
        client,
        st.session_state["user_prompt_chat"],
        st.session_state["assistant_chat"].id,
        st.session_state["thread_id_chat"],
        "messages_chat",
    )
    if artefacts:
        st.session_state["artefact_enum_dict_chat"][
            len(st.session_state.messages_chat) - 1
        ] = artefacts

if st.session_state["gpt_submitted_clicked_chat"] > 0:
    if prompt := st.chat_input("Say something"):
        st.session_state.messages_chat.append({"role": "user", "content": prompt})
        artefacts = send_message_save_thread(
            client,
            prompt,
            st.session_state["assistant_chat"].id,
            st.session_state["thread_id_chat"],
            "messages_chat",
        )
        if artefacts:
            st.session_state["artefact_enum_dict_chat"][
                len(st.session_state.messages_chat) - 1
            ] = artefacts
    for num, role_content_dict in enumerate(st.session_state.messages_chat):
        with st.chat_message(role_content_dict["role"]):
            st.markdown(role_content_dict["content"])
            if num in st.session_state["artefact_enum_dict_chat"]:
                for artefact in st.session_state["artefact_enum_dict_chat"][num]:
                    if isinstance(artefact, pd.DataFrame):
                        st.dataframe(artefact)
                    else:
                        st.plotly_chart(artefact)
    print(st.session_state["artefact_enum_dict_chat"])
