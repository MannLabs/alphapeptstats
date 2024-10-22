"""Module to integrate different LLM APIs and handle chat interactions."""

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import plotly.io as pio
from IPython.display import HTML, Markdown, display
from openai import OpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessageToolCall

from alphastats.DataSet import DataSet
from alphastats.llm.enrichment_analysis import get_enrichment_data
from alphastats.llm.llm_functions import (
    get_assistant_functions,
    get_general_assistant_functions,
    perform_dimensionality_reduction,
)
from alphastats.llm.llm_utils import (
    get_protein_id_for_gene_name,
    get_subgroups_for_each_group,
)
from alphastats.llm.prompts import get_tool_call_message
from alphastats.llm.uniprot_utils import get_gene_function

logger = logging.getLogger(__name__)


class Models:
    GPT = "gpt-4o"
    OLLAMA = "llama3.1:70b"


class LLMIntegration:
    """A class to integrate different LLM APIs and handle chat interactions.

    This class provides methods to interact with GPT and Ollama APIs, manage conversation
    history, handle function calls, and manage artifacts.

    Parameters
    ----------
    api_type : str
        The type of API to use, will be forwarded to the client.
    system_message : str
        The system message that should be given to the model.
    base_url : str, optional
        The base URL for the API, by default None
    api_key : str, optional
        The API key for authentication, by default None
    dataset : Any, optional
        The dataset to be used in the conversation, by default None
    gene_to_prot_id_map: optional
        Mapping of gene names to protein IDs

    Attributes
    ----------
    _client : OpenAI
        The OpenAI client instance
    _model : str
        The name of the language model being used
    _messages : List[Dict[str, Any]]
        The conversation history used for the LLM. Could be truncate at some point.
    _all_messages : List[Dict[str, Any]]
        The whole conversation history, used for display.
    _dataset : Any
        The dataset being used
    _metadata : Any
        Metadata associated with the dataset
    _tools : List[Dict[str, Any]]
        List of available tools or functions
    _artifacts : Dict[str, Any]
        Dictionary to store conversation artifacts
    """

    def __init__(
        self,
        api_type: str,
        system_message: str,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        dataset: Optional[DataSet] = None,
        gene_to_prot_id_map: Optional[Dict[str, str]] = None,
    ):
        self._model = api_type

        if api_type == Models.OLLAMA:
            url = f"{base_url or 'http://localhost:11434'}/v1"
            self._client = OpenAI(base_url=url, api_key="ollama")
        elif api_type == Models.GPT:
            self._client = OpenAI(api_key=api_key)
        else:
            raise ValueError(f"Invalid API type: {api_type}")

        self._dataset = dataset
        self._metadata = None if dataset is None else dataset.metadata
        self._gene_to_prot_id_map = gene_to_prot_id_map

        self._tools = self._get_tools()
        self._messages = []
        self._all_messages = []
        self._append_message("system", system_message)
        self._artifacts = {}

    def _get_tools(self) -> List[Dict[str, Any]]:
        """
        Get the list of available tools or functions.

        Returns
        -------
        List[Dict[str, Any]]
            A list of dictionaries describing the available tools
        """

        tools = [
            *get_general_assistant_functions(),
        ]
        if self._metadata is not None and self._gene_to_prot_id_map is not None:
            tools += (
                *get_assistant_functions(
                    gene_to_prot_id_map=self._gene_to_prot_id_map,
                    metadata=self._metadata,
                    subgroups_for_each_group=get_subgroups_for_each_group(
                        self._metadata
                    ),
                ),
            )

        return tools

    def _append_message(
        self,
        role: str,
        content: str,
        *,
        tool_calls: Optional[List[ChatCompletionMessageToolCall]] = None,
        tool_call_id: Optional[str] = None,
    ) -> None:
        """Construct a message and append it to the conversation history."""
        message = {
            "role": role,
            "content": content,
        }

        if tool_calls is not None:
            message["tool_calls"] = tool_calls

        if tool_call_id is not None:
            message["tool_call_id"] = tool_call_id

        self._messages.append(message)
        self._all_messages.append(message)

        self._truncate_conversation_history()

    def _truncate_conversation_history(self, max_tokens: int = 100000):
        """
        Truncate the conversation history to stay within token limits.

        Parameters
        ----------
        max_tokens : int, optional
            The maximum number of tokens to keep in history, by default 100000
        """
        total_tokens = sum(
            len(message["content"].split()) for message in self._messages
        )
        while total_tokens > max_tokens and len(self._messages) > 1:
            removed_message = self._messages.pop(0)
            total_tokens -= len(removed_message["content"].split())

    def _parse_model_response(
        self, response: ChatCompletion
    ) -> Tuple[str, List[ChatCompletionMessageToolCall]]:
        """Parse the response from the language model.

        Parameters
        ----------
        response : ChatCompletion
            The raw response from the language model

        Returns
        -------
        Message content and list of tool calls
        """
        message = response.choices[0].message
        return message.content, message.tool_calls

    def _execute_function(
        self, function_name: str, function_args: Dict[str, Any]
    ) -> Any:
        """
        Execute a function based on its name and arguments.

        Parameters
        ----------
        function_name : str
            The name of the function to execute
        function_args : Dict[str, Any]
            The arguments to pass to the function

        Returns
        -------
        Any
            The result of the function execution

        Raises
        ------
        ValueError
            If the function is not implemented or the dataset is not available
        """
        try:
            # first try to find the function in the non-Dataset functions
            if (
                function := {
                    "get_gene_function": get_gene_function,
                    "get_enrichment_data": get_enrichment_data,
                }.get(function_name)
            ) is not None:
                return function(**function_args)

            # special treatment for these functions
            elif function_name == "perform_dimensionality_reduction":
                # TODO add API in dataset
                perform_dimensionality_reduction(self._dataset, **function_args)

            elif function_name == "plot_intensity":
                # TODO move this logic to dataset
                gene_name = function_args.pop("gene_name")
                protein_id = get_protein_id_for_gene_name(
                    gene_name, self._gene_to_prot_id_map
                )
                function_args["protein_id"] = protein_id

                return self._dataset.plot_intensity(**function_args)

            # fallback: try to find the function in the Dataset functions
            else:
                plot_function = getattr(
                    self._dataset,
                    function_name.split(".")[-1],
                    None,  # TODO why split?
                )
                if plot_function:
                    return plot_function(**function_args)
            raise ValueError(
                f"Function {function_name} not implemented or dataset not available"
            )

        except Exception as e:
            return f"Error executing {function_name}: {str(e)}"

    def _handle_function_calls(
        self,
        tool_calls: List[ChatCompletionMessageToolCall],
    ) -> Tuple[str, List[ChatCompletionMessageToolCall]]:
        """
        Handle function calls from the language model and manage resulting artifacts.

        Parameters
        ----------
        tool_calls : List[Any]
            List of tool calls from the language model

        Returns
        -------
        Dict[str, Any]
            The parsed response after handling function calls, including any new artifacts

        """
        # TODO avoid infinite loops
        new_artifacts = {}

        tool_call_message = get_tool_call_message(tool_calls)
        self._append_message("assistant", tool_call_message, tool_calls=tool_calls)

        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)

            function_result = self._execute_function(function_name, function_args)
            artifact_id = f"{function_name}_{tool_call.id}"

            new_artifacts[artifact_id] = function_result

            content = json.dumps(
                {"result": str(function_result), "artifact_id": artifact_id}
            )
            self._append_message("tool", content, tool_call_id=tool_call.id)

        post_artifact_message_idx = len(self._messages)
        self._artifacts[post_artifact_message_idx] = new_artifacts.values()

        logger.info(
            f"Calling 'chat.completions.create' {self._messages=} {self._tools=} .."
        )
        response = self._client.chat.completions.create(
            model=self._model,
            messages=self._messages,
            tools=self._tools,
        )
        logger.info(".. done")

        return self._parse_model_response(response)

    def get_print_view(self, show_all=False) -> List[Dict[str, Any]]:
        """Get a structured view of the conversation history for display purposes."""

        print_view = []
        for num, role_content_dict in enumerate(self._all_messages):
            if not show_all and (role_content_dict["role"] in ["tool", "system"]):
                continue
            if not show_all and "tool_calls" in role_content_dict:
                continue

            print_view.append(
                {
                    "role": role_content_dict["role"],
                    "content": role_content_dict["content"],
                    "artifacts": self._artifacts.get(num, []),
                }
            )
        return print_view

    def chat_completion(
        self, prompt: str, role: str = "user"
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Generate a chat completion based on the given prompt and manage any resulting artifacts.

        Parameters
        ----------
        prompt : str
            The user's input prompt
        role : str, optional
            The role of the message sender, by default "user"

        Returns
        -------
        Tuple[str, Dict[str, Any]]
            A tuple containing the generated response and a dictionary of new artifacts
        """
        self._append_message(role, prompt)

        try:
            logger.info(
                f"Calling 'chat.completions.create' {self._messages=} {self._tools=} .."
            )
            response = self._client.chat.completions.create(
                model=self._model,
                messages=self._messages,
                tools=self._tools,
            )
            logger.info(".. done")

            content, tool_calls = self._parse_model_response(response)

            if tool_calls:
                if content:
                    raise ValueError(
                        f"Unexpected content {content} with tool calls {tool_calls}"
                    )

                content, _ = self._handle_function_calls(tool_calls)

            self._append_message("assistant", content)

        except ArithmeticError as e:
            error_message = f"Error in chat completion: {str(e)}"
            self._append_message("system", error_message)
            return error_message, {}

    # TODO this seems to be for notebooks?
    # we need some "export mode" where everything is shown
    def display_chat_history(self):
        """
        Display the chat history, including messages, function calls, and associated artifacts.

        This method renders the chat history in a structured format, aligning artifacts
        with their corresponding messages and the model's interpretation.

        Returns
        -------
        None
        """
        for message in self._messages:
            role = message["role"].capitalize()
            content = message["content"]

            if role == "Assistant" and "tool_calls" in message:
                display(Markdown(f"**{role}**: {content}"))
                for tool_call in message["tool_calls"]:
                    function_name = tool_call.function.name
                    function_args = tool_call.function.arguments
                    display(Markdown(f"*Function Call*: `{function_name}`"))
                    display(Markdown(f"*Arguments*: ```json\n{function_args}\n```"))

            elif role == "Tool":
                tool_result = json.loads(content)
                artifact_id = tool_result.get("artifact_id")
                if artifact_id and artifact_id in self._artifacts:
                    artifact = self._artifacts[artifact_id]
                    display(
                        Markdown(f"**Function Result** (Artifact ID: {artifact_id}):")
                    )
                    self._display_artifact(artifact)
                else:
                    display(Markdown(f"**Function Result**: {content}"))

            else:
                display(Markdown(f"**{role}**: {content}"))

    def _display_artifact(self, artifact):
        """
        Display an artifact based on its type.

        Parameters
        ----------
        artifact : Any
            The artifact to display

        Returns
        -------
        None
        """
        if isinstance(artifact, pd.DataFrame):
            display(artifact)
        elif str(type(artifact)) == "<class 'plotly.graph_objs._figure.Figure'>":
            display(HTML(pio.to_html(artifact, full_html=False)))
        else:
            display(Markdown(f"```\n{str(artifact)}\n```"))
