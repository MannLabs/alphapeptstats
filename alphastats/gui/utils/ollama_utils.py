import json
import logging
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import plotly.io as pio
from IPython.display import HTML, Markdown, display
from openai import OpenAI

from alphastats.gui.utils.enrichment_analysis import get_enrichment_data
from alphastats.gui.utils.gpt_helper import (
    get_assistant_functions,
    get_gene_to_prot_id_mapping,
    get_general_assistant_functions,
    get_subgroups_for_each_group,
    perform_dimensionality_reduction,
)

# from alphastats.gui.utils.artefacts import ArtifactManager
from alphastats.gui.utils.uniprot_utils import get_gene_function

logger = logging.getLogger(__name__)


class Models:
    GPT = "gpt-4o"
    OLLAMA = "llama3.1:70b"


class LLMIntegration:
    """
    A class to integrate different Language Model APIs and handle chat interactions.

    This class provides methods to interact with GPT and Ollama APIs, manage conversation
    history, handle function calls, and manage artifacts.

    Parameters
    ----------
    api_type : str, optional
        The type of API to use ('gpt' or 'ollama'), by default 'gpt'
    base_url : str, optional
        The base URL for the API, by default None
    api_key : str, optional
        The API key for authentication, by default None
    dataset : Any, optional
        The dataset to be used in the conversation, by default None
    metadata : Any, optional
        Metadata associated with the dataset, by default None

    Attributes
    ----------
    client : OpenAI
        The OpenAI client instance
    model : str
        The name of the language model being used
    messages : List[Dict[str, Any]]
        The conversation history
    dataset : Any
        The dataset being used
    metadata : Any
        Metadata associated with the dataset
    tools : List[Dict[str, Any]]
        List of available tools or functions
    artifacts : Dict[str, Any]
        Dictionary to store conversation artifacts
    """

    def __init__(
        self,
        api_type: str,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        dataset=None,
        gene_to_prot_id_map=None,
    ):
        self.model = api_type

        if api_type == Models.OLLAMA:
            url = f"{base_url or 'http://localhost:11434'}/v1"
            self.client = OpenAI(base_url=url, api_key="ollama")
        elif api_type == Models.GPT:
            self.client = OpenAI(api_key=api_key)
        else:
            raise ValueError(f"Invalid API type: {api_type}")

        self.messages = []
        self.dataset = dataset
        self.metadata = None if dataset is None else dataset.metadata
        self._gene_to_prot_id_map = gene_to_prot_id_map

        self.tools = self._get_tools()
        self.artifacts = {}
        # self.artifact_manager = ArtifactManager()
        self.message_artifact_map = {}

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
        if self.metadata is not None and self._gene_to_prot_id_map is not None:
            tools += (
                *get_assistant_functions(
                    gene_to_prot_id_dict=self._gene_to_prot_id_map,
                    metadata=self.metadata,
                    subgroups_for_each_group=get_subgroups_for_each_group(
                        self.metadata
                    ),
                ),
            )

        return tools

    def truncate_conversation_history(self, max_tokens: int = 100000):
        """
        Truncate the conversation history to stay within token limits.

        Parameters
        ----------
        max_tokens : int, optional
            The maximum number of tokens to keep in history, by default 100000

        Returns
        -------
        None
        """
        total_tokens = sum(len(m["content"].split()) for m in self.messages)
        while total_tokens > max_tokens and len(self.messages) > 1:
            # TODO messages should still be displayed!
            removed_message = self.messages.pop(0)
            total_tokens -= len(removed_message["content"].split())

    def parse_model_response(self, response: Any) -> Dict[str, Any]:
        """
        Parse the response from the language model.

        Parameters
        ----------
        response : Any
            The raw response from the language model

        Returns
        -------
        Dict[str, Any]
            A dictionary containing the parsed content and tool calls
        """
        return {
            "content": response.choices[0].message.content,
            "tool_calls": response.choices[0].message.tool_calls,
        }

    def execute_function(
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
                    "perform_dimensionality_reduction": perform_dimensionality_reduction,
                }.get(function_name)
            ) is not None:
                return function(**function_args)

            # special treatment for this one
            elif function_name == "plot_intensity":
                gene_name = function_args.pop("gene_name")
                protein_id = get_gene_to_prot_id_mapping(
                    gene_name, self._gene_to_prot_id_map
                )
                function_args["protein_id"] = protein_id

                return self.dataset.plot_intensity(**function_args)

            # fallback: try to find the function in the Dataset functions
            else:
                plot_function = getattr(
                    self.dataset,
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

    def handle_function_calls(
        self,
        tool_calls: List[Any],
    ) -> Dict[str, Any]:
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
        new_artifacts = {}

        funcs_and_args = "\n".join(
            [
                f"Calling function: {tool_call.function.name} with arguments: {tool_call.function.arguments}"
                for tool_call in tool_calls
            ]
        )
        self.messages.append(
            {"role": "assistant", "content": funcs_and_args, "tool_calls": tool_calls}
        )

        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)

            function_result = self.execute_function(function_name, function_args)
            artifact_id = f"{function_name}_{tool_call.id}"

            new_artifacts[artifact_id] = function_result

            self.messages.append(
                {
                    "role": "tool",
                    "content": json.dumps(
                        {"result": str(function_result), "artifact_id": artifact_id}
                    ),
                    "tool_call_id": tool_call.id,
                }
            )

        post_artefact_message_idx = len(self.messages)
        self.artifacts[post_artefact_message_idx] = new_artifacts.values()

        logger.info(
            f"Calling 'chat.completions.create' {self.messages=} {self.tools=} .."
        )
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            tools=self.tools,
        )
        logger.info(".. done")

        parsed_response = self.parse_model_response(response)
        parsed_response["new_artifacts"] = new_artifacts

        return parsed_response

    def get_print_view(self, show_all=False) -> List[Dict[str, Any]]:
        """Get a structured view of the conversation history for display purposes."""

        print_view = []
        for num, role_content_dict in enumerate(self.messages):
            if not show_all and (role_content_dict["role"] in ["tool", "system"]):
                continue
            if not show_all and "tool_calls" in role_content_dict:
                continue

            print_view.append(
                {
                    "role": role_content_dict["role"],
                    "content": role_content_dict["content"],
                    "artifacts": self.artifacts.get(num, []),
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

        Raises
        ------
        ArithmeticError
            If there's an error in chat completion
        """
        self.messages.append({"role": role, "content": prompt})
        self.truncate_conversation_history()

        try:
            logger.info(
                f"Calling 'chat.completions.create' {self.messages=} {self.tools=} .."
            )
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                tools=self.tools,
            )
            logger.info(".. done")

            parsed_response = self.parse_model_response(response)
            new_artifacts = {}

            if parsed_response["tool_calls"]:
                parsed_response = self.handle_function_calls(
                    parsed_response["tool_calls"]
                )
                new_artifacts = parsed_response.pop("new_artifacts", {})

            self.messages.append(
                {"role": "assistant", "content": parsed_response["content"]}
            )
            return parsed_response[
                "content"
            ], new_artifacts  # TODO response is not used

        except ArithmeticError as e:
            error_message = f"Error in chat completion: {str(e)}"
            self.messages.append({"role": "system", "content": error_message})
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
        for message in self.messages:
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
                if artifact_id and artifact_id in self.artifacts:
                    artifact = self.artifacts[artifact_id]
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
