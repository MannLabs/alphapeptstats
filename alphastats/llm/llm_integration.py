"""Module to integrate different LLM APIs and handle chat interactions."""

import json
import logging
import warnings
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import plotly.io as pio
import tiktoken
from IPython.display import HTML, Markdown, display
from openai import OpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessageToolCall

from alphastats.dataset.dataset import DataSet
from alphastats.llm.llm_functions import (
    GENERAL_FUNCTION_MAPPING,
    get_assistant_functions,
    get_general_assistant_functions,
)
from alphastats.llm.llm_utils import (
    get_subgroups_for_each_group,
)
from alphastats.llm.prompts import get_tool_call_message

logger = logging.getLogger(__name__)


class Models:
    """Names of the available models.

    Note that this will be directly passed to the OpenAI client.
    """

    GPT4O = "gpt-4o"
    OLLAMA_31_70B = "llama3.1:70b"
    OLLAMA_31_8B = "llama3.1:8b"  # for testing only


class LLMIntegration:
    """A class to integrate different LLM APIs and handle chat interactions.

    This class provides methods to interact with GPT and Ollama APIs, manage conversation
    history, handle function calls, and manage artifacts.

    Parameters
    ----------
    model_name : str
        The type of API to use, will be forwarded to the client.
    system_message : str
        The system message that should be given to the model.
    base_url : str, optional
        The base URL for the API, by default None
    api_key : str, optional
        The API key for authentication, by default None
    dataset : Any, optional
        The dataset to be used in the conversation, by default None
    genes_of_interest: optional
        List of regulated genes
    """

    def __init__(
        self,
        model_name: str,
        *,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        system_message: str = None,
        load_tools: bool = True,
        dataset: Optional[DataSet] = None,
        genes_of_interest: Optional[List[str]] = None,
        max_tokens=100000,
    ):
        self._model = model_name

        if model_name in [Models.OLLAMA_31_70B, Models.OLLAMA_31_8B]:
            url = f"{base_url}/v1"  # TODO: enable to configure this per model
            self._client = OpenAI(base_url=url, api_key="ollama")
        elif model_name in [Models.GPT4O]:
            self._client = OpenAI(api_key=api_key)
        else:
            raise ValueError(f"Invalid model name: {model_name}")

        self._dataset = dataset
        self._metadata = None if dataset is None else dataset.metadata
        self._genes_of_interest = genes_of_interest
        self._max_tokens = max_tokens

        self._tools = self._get_tools() if load_tools else None

        self._artifacts = {}
        self._messages = []  # the conversation history used for the LLM, could be truncated at some point.
        self._all_messages = []  # full conversation history for display
        if system_message is not None:
            self._append_message("system", system_message)

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
        if self._metadata is not None and self._genes_of_interest is not None:
            tools += (
                *get_assistant_functions(
                    genes_of_interest=self._genes_of_interest,
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

    def estimate_tokens(
        self, messages: List[Dict[str, str]], average_chars_per_token: float = 3.6
    ) -> int:
        """
        Estimate the number of tokens in a list of messages.

        Parameters
        ----------
        messages : List[Dict[str, str]]
            A list of messages to estimate the number of tokens for
        average_chars_per_token : float, optional
            The average number of characters per token, by default 3.6

        Returns
        -------
        int
            The estimated number of tokens
        """
        try:
            enc = tiktoken.encoding_for_model(self._model)
            total_tokens = sum(
                [len(enc.encode(message["content"])) for message in messages if message]
            )
        except KeyError:
            total_tokens = sum(
                [
                    len(message["content"]) / average_chars_per_token
                    for message in messages
                    if message
                ]
            )
        return total_tokens

    def _truncate_conversation_history(
        self, average_chars_per_token: float = 3.6
    ) -> None:
        """
        Truncate the conversation history to stay within token limits.

        Parameters
        ----------
        max_tokens : int, optional
            The maximum number of tokens to keep in history, by default 100000
        average_chars_per_token : float, optional
            The average number of characters per token, by default 3.6. Normal english language has 4 per token. Every ID included in the text is 1 token per character. Parsed uniprot entries are between 3.6 and 3.9 judging from experience with https://platform.openai.com/tokenizer.
        """
        # TODO: avoid important messages being removed (e.g. facts about genes)
        # TODO: find out how messages can be None type and handle them earlier
        total_tokens = self.estimate_tokens(self._messages, average_chars_per_token)
        while total_tokens > self._max_tokens and len(self._messages) > 1:
            warnings.warn(
                f"Truncating conversation history to stay within token limits.\nRemoved message:\n{self._messages[0]['role']}: {self._messages[0]['content'][0:min(30, len(self._messages[0]['content']))]}..."
            )
            removed_message = self._messages.pop(0)
            while (
                removed_message["role"] == "assistant"
                and self._messages[0]["role"] == "tool"
            ):
                # This is required as the chat completion fails if there are tool outputs without corresponding tool calls in the message history.
                warnings.warn(
                    f"Removing corresponsing tool output as well.\nRemoved message:\n{self._messages[0]['role']}: {self._messages[0]['content'][0:min(30, len(self._messages[0]['content']))]}..."
                )
                self._messages.pop(0)
                if len(self._messages) == 0:
                    raise ValueError(
                        "Truncating conversation history failed, as the tool replies exceeded the token limit. Please increase the token limit and reset the LLM analysis."
                    )
            total_tokens = self.estimate_tokens(self._messages, average_chars_per_token)

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
        """
        # first try to find the function in the non-Dataset functions
        if (function := GENERAL_FUNCTION_MAPPING.get(function_name)) is not None:
            return function(**function_args)
        # look up the function in the DataSet class
        else:
            function = getattr(
                self._dataset,
                function_name.split(".")[-1],
                None,  # TODO why split?
            )
            if function:
                return function(**function_args)

        raise ValueError(
            f"Function {function_name} not implemented or dataset not available"
        )

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

            try:
                function_result = self._execute_function(function_name, function_args)
            except Exception as e:
                function_result = f"Error executing {function_name}: {str(e)}"

            artifact_id = f"{function_name}_{tool_call.id}"
            new_artifacts[artifact_id] = function_result

            content = json.dumps(
                {"result": str(function_result), "artifact_id": artifact_id}
            )
            self._append_message("tool", content, tool_call_id=tool_call.id)

        post_artifact_message_idx = len(self._all_messages)
        self._artifacts[post_artifact_message_idx] = new_artifacts.values()

        response = self._chat_completion_create()

        return self._parse_model_response(response)

    def _chat_completion_create(self) -> ChatCompletion:
        """Create a chat completion based on the current conversation history."""
        logger.info(f"Calling 'chat.completions.create' {self._messages[-1]} ..")
        result = self._client.chat.completions.create(
            model=self._model,
            messages=self._messages,
            tools=self._tools,
        )
        logger.info(".. done")
        return result

    def get_print_view(self, show_all=False) -> List[Dict[str, Any]]:
        """Get a structured view of the conversation history for display purposes."""

        print_view = []
        for message_idx, role_content_dict in enumerate(self._all_messages):
            if not show_all and (role_content_dict["role"] in ["tool", "system"]):
                continue
            if not show_all and "tool_calls" in role_content_dict:
                continue
            in_context = role_content_dict in self._messages

            print_view.append(
                {
                    "role": role_content_dict["role"],
                    "content": role_content_dict["content"],
                    "artifacts": self._artifacts.get(message_idx, []),
                    "in_context": in_context,
                }
            )
        return print_view

    def get_chat_log_txt(self) -> str:
        """Get a chat log in text format for saving. It excludes tool replies, as they are usually also represented in the artifacts."""
        messages = self.get_print_view(show_all=True)
        chatlog = ""
        for message in messages:
            if message["role"] == "tool":
                continue
            chatlog += f"{message['role'].capitalize()}: {message['content']}\n"
            if len(message["artifacts"]) > 0:
                chatlog += "-----\n"
            for artifact in message["artifacts"]:
                chatlog += f"Artifact: {artifact}\n"
            chatlog += "----------\n"
        return chatlog

    def chat_completion(self, prompt: str, role: str = "user") -> None:
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
            response = self._chat_completion_create()

            content, tool_calls = self._parse_model_response(response)

            if tool_calls:
                if content:
                    raise ValueError(
                        f"Unexpected content {content} with tool calls {tool_calls}."
                    )

                content, _ = self._handle_function_calls(tool_calls)

            self._append_message("assistant", content)

        except ArithmeticError as e:
            error_message = f"Error in chat completion: {str(e)}"
            self._append_message("system", error_message)

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
            tokens = self.estimate_tokens([message])

            if role == "Assistant" and "tool_calls" in message:
                display(Markdown(f"**{role}**: {content} *({str(tokens)} tokens)*"))
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
                        Markdown(
                            f"**Function Result** (Artifact ID: {artifact_id}, *{str(tokens)} tokens*):"
                        )
                    )
                    self._display_artifact(artifact)
                else:
                    display(
                        Markdown(
                            f"**Function Result** *({str(tokens)} tokens)*: {content}"
                        )
                    )

            else:
                display(Markdown(f"**{role}**: {content} *({str(tokens)} tokens)*"))

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
