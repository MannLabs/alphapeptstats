from typing import Optional, List
from pathlib import Path

import time
import json


import openai
import streamlit as st

try:
    from alphastats.gui.utils.gpt_helper import (
        turn_args_to_float,
    )
    from alphastats.gui.utils.uniprot_utils import (
        get_gene_function,
    )

except ModuleNotFoundError:
    from utils.gpt_helper import (
        turn_args_to_float,
    )
    from utils.uniprot_utils import (
        get_gene_function,
    )
    from utils.openai_utils import (
        wait_for_run_completion,
    )


def wait_for_run_completion(
    client: openai.OpenAI, thread_id: int, run_id: int, check_interval: int = 2
) -> Optional[List]:
    """
    Wait for a run and function calls to complete and return the plots, if they were created by function calling.

    Args:
        client (openai.OpenAI): The OpenAI client.
        thread_id (int): The thread ID.
        run_id (int): The run ID.
        check_interval (int, optional): The interval to check for run completion. Defaults to 2 seconds.

    Returns:
        Optional[list]: A list of plots, if any.
    """
    artefacts = []
    while True:
        run_status = client.beta.threads.runs.retrieve(
            thread_id=thread_id, run_id=run_id
        )
        print(run_status.status, run_id, run_status.required_action)
        assistant_functions = {
            "create_intensity_plot",
            "perform_dimensionality_reduction",
            "create_sample_histogram",
            "st.session_state.dataset.plot_volcano",
            "st.session_state.dataset.plot_sampledistribution",
            "st.session_state.dataset.plot_intensity",
            "st.session_state.dataset.plot_pca",
            "st.session_state.dataset.plot_umap",
            "st.session_state.dataset.plot_tsne",
            "get_enrichment_data",
        }
        if run_status.status == "completed":
            print("Run is completed!")
            if artefacts:
                print("Returning artefacts")
                return artefacts
            break
        elif run_status.status == "requires_action":
            print("requires_action", run_status)
            print(
                [
                    st.session_state[StateKeys.PLOTTING_OPTIONS][i]["function"].__name__
                    for i in st.session_state[StateKeys.PLOTTING_OPTIONS]
                ]
            )
            tool_calls = run_status.required_action.submit_tool_outputs.tool_calls
            tool_outputs = []
            for tool_call in tool_calls:
                print("### calling:", tool_call.function.name)
                if tool_call.function.name == "get_gene_function":
                    print(
                        type(tool_call.function.arguments), tool_call.function.arguments
                    )
                    prompt = json.loads(tool_call.function.arguments)["gene_name"]
                    gene_function = get_gene_function(prompt)
                    tool_outputs.append(
                        {
                            "tool_call_id": tool_call.id,
                            "output": gene_function,
                        },
                    )
                elif (
                    tool_call.function.name
                    in [
                        st.session_state[StateKeys.PLOTTING_OPTIONS][i][
                            "function"
                        ].__name__
                        for i in st.session_state[StateKeys.PLOTTING_OPTIONS]
                    ]
                    or tool_call.function.name in assistant_functions
                ):
                    args = tool_call.function.arguments
                    args = turn_args_to_float(args)
                    print(f"{tool_call.function.name}(**{args})")
                    artefact = eval(f"{tool_call.function.name}(**{args})")
                    artefact_json = artefact.to_json()

                    tool_outputs.append(
                        {"tool_call_id": tool_call.id, "output": artefact_json},
                    )
                    artefacts.append(artefact)

            if tool_outputs:
                _run = client.beta.threads.runs.submit_tool_outputs(
                    thread_id=thread_id, run_id=run_id, tool_outputs=tool_outputs
                )
                print("submitted")
        else:
            print("Run is not yet completed. Waiting...", run_status.status, run_id)
            time.sleep(check_interval)


def send_message_save_thread(
    client: openai.OpenAI,
    message: str,
    assistant_id: str,
    thread_id: str,
    storing_variable: str = "messages",
) -> Optional[List]:
    """
    Send a message to the OpenAI ChatGPT model and save the thread in the session state, return plots if GPT called a function to create them.

    Args:
        client (openai.OpenAI): The OpenAI client.
        message (str): The message to send to the ChatGPT model.

    Returns:
        Optional[list]: A list of plots, if any.
    """
    message = client.beta.threads.messages.create(
        thread_id=thread_id, role="user", content=message
    )

    run = client.beta.threads.runs.create(
        thread_id=thread_id,
        assistant_id=assistant_id,
    )
    try:
        plots = wait_for_run_completion(client, thread_id, run.id)
    except KeyError as e:
        print(e)
        plots = None
    messages = client.beta.threads.messages.list(thread_id=thread_id)
    st.session_state[storing_variable] = []
    for num, message in enumerate(messages.data[::-1]):
        role = message.role
        if message.content:
            content = message.content[0].text.value
        else:
            content = "Sorry, I was unable to process this message. Try again or change your request."
        st.session_state[storing_variable].append({"role": role, "content": content})
    if not plots:
        return
    return plots


def try_to_set_api_key(api_key: str = None) -> None:
    """
    Checks if the OpenAI API key is available in the environment / system variables.
    If the API key is not available, saves the key to secrets.toml in the repository root directory.

    Args:
        api_key (str, optional): The OpenAI API key. Defaults to None.

    Returns:
        None
    """
    if api_key and "api_key" not in st.session_state:
        st.session_state["openai_api_key"] = api_key
        secret_path = Path(st.secrets._file_paths[-1])
        secret_path.parent.mkdir(parents=True, exist_ok=True)
        with open(secret_path, "w") as f:
            f.write(f'openai_api_key = "{api_key}"')
        openai.OpenAI.api_key = api_key
        return
    try:
        openai.OpenAI.api_key = st.secrets["openai_api_key"]
    except KeyError:
        st.write(
            "OpenAI API key not found in environment variables. Please enter your API key to continue."
        )
