"""Module to integrate different LLM APIs and handle chat interactions."""

import base64
import json
import logging
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import plotly.io as pio
import pytz
import tiktoken
from IPython.display import HTML, Markdown, display
from litellm import completion
from openai.types.chat import ChatCompletion, ChatCompletionMessageToolCall

from alphastats.dataset.dataset import DataSet
from alphastats.dataset.keys import ConstantsClass
from alphastats.llm.llm_functions import (
    GENERAL_FUNCTION_MAPPING,
    get_assistant_functions,
    get_general_assistant_functions,
)
from alphastats.llm.llm_utils import (
    get_subgroups_for_each_group,
)
from alphastats.llm.prompts import (
    IMAGE_ANALYSIS_PROMPT,
    IMAGE_REPRESENTATION_PROMPT,
    ITERABLE_ARTIFACT_REPRESENTATION_PROMPT,
    NO_REPRESENTATION_PROMPT,
    SINGLE_ARTIFACT_REPRESENTATION_PROMPT,
    get_tool_call_message,
)
from alphastats.plots.plot_utils import PlotlyObject

logger = logging.getLogger(__name__)


class Model:
    class ModelProperties(metaclass=ConstantsClass):
        """Properties of the models used in the LLM integration."""

        REQUIRES_API_KEY = "requires_api_key"  # pragma: allowlist secret
        SUPPORTS_BASE_URL = "supports_base_url"
        MULTIMODAL = "multimodal"

    MODELS = {
        "openai/gpt-4o": {
            ModelProperties.REQUIRES_API_KEY: True,
            ModelProperties.MULTIMODAL: True,
        },
        "openai/gpt-o3": {
            ModelProperties.REQUIRES_API_KEY: True,
            ModelProperties.MULTIMODAL: True,
        },
        "anthropic/claude-sonnet-4-20250514": {
            ModelProperties.REQUIRES_API_KEY: True,
        },
        "ollama/llama3.1:8b": {
            ModelProperties.SUPPORTS_BASE_URL: True,
        },
        "GWDG/llama-3.3-70b-instruct": {
            ModelProperties.SUPPORTS_BASE_URL: True,
            ModelProperties.REQUIRES_API_KEY: True,
        },
        "GWDG/mistral-large-instruct": {
            ModelProperties.SUPPORTS_BASE_URL: True,
            ModelProperties.REQUIRES_API_KEY: True,
        },
        "GWDG/qwen2.5-72b-instruct": {
            ModelProperties.SUPPORTS_BASE_URL: True,
            ModelProperties.REQUIRES_API_KEY: True,
            ModelProperties.MULTIMODAL: True,
        },
        "GWDG/qwen2.5-vl-72b-instruct": {
            ModelProperties.SUPPORTS_BASE_URL: True,
            ModelProperties.REQUIRES_API_KEY: True,
            ModelProperties.MULTIMODAL: True,
        },
    }

    def __init__(self, model_name: str):
        """Initialize the Models class."""
        if model_name not in self.MODELS:
            raise ValueError(
                f"Invalid model name: {model_name}. Available models: {self.get_available_models()}"
            )

        self._model_properties = self.MODELS[model_name]

    def requires_api_key(self) -> bool:
        """Check if the model requires API key authentication."""
        return self._model_properties.get(self.ModelProperties.REQUIRES_API_KEY, False)

    def supports_base_url(self) -> bool:
        """Check if the model supports a custom base URL."""
        return self._model_properties.get(self.ModelProperties.SUPPORTS_BASE_URL, False)

    def is_multimodal(self) -> bool:
        """Check if the model supports multimodal inputs."""
        return self._model_properties.get(self.ModelProperties.MULTIMODAL, False)

    @staticmethod
    def get_available_models() -> List[str]:
        """Get a list of available model names."""
        return list(Model.MODELS.keys())


# we do not want to pass our internal keys to the LLM
DO_NOT_PASS_PREFIX = "___"


class MessageKeys(metaclass=ConstantsClass):
    """Keys for the message dictionary."""

    ROLE = "role"
    CONTENT = "content"
    TOOL_CALLS = "tool_calls"
    TOOL_CALL_ID = "tool_call_id"
    RESULT = "result"
    ARTIFACT_ID = "artifact_id"
    IN_CONTEXT = f"{DO_NOT_PASS_PREFIX}in_context"
    ARTIFACTS = "artifacts"
    PINNED = f"{DO_NOT_PASS_PREFIX}pinned"
    TIMESTAMP = f"{DO_NOT_PASS_PREFIX}timestamp"
    IMAGE_URL = "image_url"
    EXCHANGE_ID = f"{DO_NOT_PASS_PREFIX}exchange_id"


class Roles(metaclass=ConstantsClass):
    """Names of the available roles."""

    USER = "user"
    ASSISTANT = "assistant"  # might show as tool-call in error messages
    TOOL = "tool"
    SYSTEM = "system"


class LLMClientWrapper:
    """A class to provide the OpenAI client."""

    def __init__(
        self,
        model_name: str,
        *,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        """Initialize the client provider.

        model_name : str
            The type of API to use, will be forwarded to the client.
        base_url : str, optional
            The base URL for the API, by default None
        api_key : str, optional
            The API key for authentication, by default None
        """

        model = Model(model_name)
        if model_name.startswith("GWDG"):
            # The GWDG supplies an openai like API
            model_name = model_name.replace("GWDG/", "openai/")
        self.model_name = model_name

        if model.requires_api_key() and not api_key:
            raise ValueError("API key is required for this model.")

        self.api_key = api_key

        self.base_url = base_url
        self.is_multimodal = model.is_multimodal()

    def chat_completion_create(
        self, *, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]
    ) -> ChatCompletion:
        """Create a chat completion based on the current conversation history."""

        messages_ = self._strip_off_internal_keys(messages)

        last_message = messages_[-1]
        logger.info(f"Calling 'chat.completions.create' {last_message} ..")

        response = completion(
            model=self.model_name,
            messages=messages_,
            tools=tools,
            api_key=self.api_key if self.api_key else None,
            api_base=self.base_url if self.base_url else None,
        )
        return response

    @staticmethod
    def _strip_off_internal_keys(
        messages: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Strip off internal keys from the messages."""
        return [
            {k: v for k, v in message.items() if not k.startswith(DO_NOT_PASS_PREFIX)}
            for message in messages
        ]


class LLMIntegration:
    """A class to integrate different LLM APIs and handle chat interactions.

    This class provides methods to interact with GPT and Ollama APIs, manage conversation
    history, handle function calls, and manage artifacts.

    Parameters
    ----------
    client_wrapper : LLMClientWrapper
        The client provider to be used.
    system_message : str
        The system message that should be given to the model.
    load_tools : bool
        Whether to load the tools or not, by default True
    dataset : Any, optional
        The dataset to be used in the conversation, by default None
    genes_of_interest: optional
        List of regulated genes
    max_tokens : int
        The maximum number of tokens for the conversation history, by default 100000
    """

    def __init__(
        self,
        client_wrapper: LLMClientWrapper,
        *,
        system_message: str = None,
        load_tools: bool = True,
        dataset: Optional[DataSet] = None,
        genes_of_interest: Optional[List[str]] = None,
        max_tokens=100000,
    ):
        self.client_wrapper = client_wrapper

        self._dataset = dataset
        self._metadata = None if dataset is None else dataset.metadata
        self._genes_of_interest = genes_of_interest
        self._max_tokens = max_tokens

        self._tools = self._get_tools() if load_tools else None

        self._artifacts = {}
        self._messages = []  # the conversation history used for the LLM, could be truncated at some point.

        # an "exchange" is a message from the user and a response from the LLM (incl. tool calls and responses)
        # => at least 2, but maybe more messages
        self._exchange_count = 0

        if system_message is not None:
            self._append_message("system", system_message, pin_message=True)

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

    @staticmethod
    def _plotly_to_base64(figure) -> str:
        """Convert Plotly figure to base64-encoded PNG image."""
        img_bytes = pio.to_image(figure, format="png")
        return base64.b64encode(img_bytes).decode("utf-8")

    def _append_message(
        self,
        role: str,
        content: Union[str, List[Dict[str, Any]]],
        *,
        tool_calls: Optional[List[ChatCompletionMessageToolCall]] = None,
        tool_call_id: Optional[str] = None,
        pin_message: bool = False,
        keep_list: bool = False,
    ) -> None:
        """Construct a message and append it to the conversation history."""
        message = {
            MessageKeys.EXCHANGE_ID: self._exchange_count,
            MessageKeys.TIMESTAMP: datetime.now(tz=pytz.utc).strftime(
                "%Y-%m-%dT%H:%M:%S"
            ),
            MessageKeys.PINNED: pin_message,
            MessageKeys.ROLE: role,
        }

        if keep_list:
            message[MessageKeys.CONTENT] = content
        else:
            message[MessageKeys.CONTENT] = str(content)

        if tool_calls is not None:
            message[MessageKeys.TOOL_CALLS] = tool_calls

        if tool_call_id is not None:
            message[MessageKeys.TOOL_CALL_ID] = tool_call_id

        self._messages.append(message)

    @staticmethod
    def estimate_tokens(
        messages: List[Dict[str, str]],
        model: str = "model",
        average_chars_per_token: float = 3.6,
    ) -> float:
        """
        Estimate the number of tokens in a list of messages.

        Parameters
        ----------
        messages : List[Dict[str, str]]
            A list of messages to estimate the number of tokens for
        model : str, optional
            The model to use for tokenization, by default "model", could be for example "gpt-4o"
        average_chars_per_token : float, optional
            The average number of characters per token, by default 3.6

        Returns
        -------
        float
            The estimated number of tokens
        """
        # TODO we could store this with each message to avoid re-calculating it

        total_tokens = 0
        try:
            tiktoken_model_name = model.split("/")[-1]  # TODO this is a hack!
            enc = tiktoken.encoding_for_model(tiktoken_model_name)
            for message in messages:
                if message and MessageKeys.CONTENT in message:
                    content = message[MessageKeys.CONTENT]
                    if isinstance(content, str):
                        total_tokens += len(enc.encode(content))
                    elif isinstance(content, list):
                        for part in content:
                            if (
                                isinstance(part, dict)
                                and part.get("type") == "text"
                                and isinstance(part.get("text"), str)
                            ):
                                total_tokens += len(enc.encode(part["text"]))
                            elif (
                                isinstance(part, dict)
                                and part.get("type") == "image_url"
                            ):
                                total_tokens += (
                                    250  # Placeholder token count for an image
                                )
        except KeyError:
            for message in messages:
                if message and MessageKeys.CONTENT in message:
                    content = message[MessageKeys.CONTENT]
                    if isinstance(content, str):
                        total_tokens += len(content) / average_chars_per_token
                    elif isinstance(content, list):
                        for part in content:
                            if (
                                isinstance(part, dict)
                                and part.get("type") == "text"
                                and isinstance(part.get("text"), str)
                            ):
                                total_tokens += (
                                    len(part["text"]) / average_chars_per_token
                                )
                            elif (
                                isinstance(part, dict)
                                and part.get("type") == "image_url"
                            ):
                                total_tokens += 250  # Placeholder
        return int(total_tokens)

    def _truncate(
        self, messages: List[Dict[str, Any]], average_chars_per_token: float = 3.6
    ) -> List[Dict[str, Any]]:
        """
        Truncate messages to stay within token limits.

        Assumes that messages are ordered "oldest first".
        In the process pinned messages and their corresponding exchanges are preserved.
        The most recent exchanges are kept until the token limit is reached.

        Parameters
        ----------
        average_chars_per_token : float, optional
            The average number of characters per token, by default 3.6. Normal english language has 4 per token. Every ID included in the text is 1 token per character. Parsed uniprot entries are between 3.6 and 3.9 judging from experience with https://platform.openai.com/tokenizer.
        """
        # TODO: avoid important messages being removed (e.g. facts about genes)
        # TODO: find out how messages can be None type and handle them earlier

        pinned_exchange_ids = set()
        not_pinned_exchanges_token_count = defaultdict(lambda: 0)
        pinned_tokens = 0

        # first, get pinned exchanges and token counts
        for message in messages:
            exchange_id = message[MessageKeys.EXCHANGE_ID]

            message_token_count = self.estimate_tokens(
                [message], self.client_wrapper.model_name, average_chars_per_token
            )

            if message[MessageKeys.PINNED]:
                pinned_exchange_ids.add(exchange_id)
            if exchange_id in pinned_exchange_ids:
                pinned_tokens += message_token_count
                continue

            not_pinned_exchanges_token_count[exchange_id] += message_token_count

        if pinned_tokens > self._max_tokens:
            raise ValueError(
                # TODO enable increasing the token limit.
                "Pinned messages exceed the maximum token limit. Please increase the token limit."  #  or unpin some messages.
            )

        # find out which messages we can keep
        total_tokens_added = pinned_tokens
        exchange_ids_to_keep = set()
        for exchange_id, exchange_token_count in reversed(  # youngest first
            not_pinned_exchanges_token_count.items()
        ):
            total_tokens_added += exchange_token_count
            if total_tokens_added <= self._max_tokens:
                exchange_ids_to_keep.add(exchange_id)
                continue
            break

        # finally, filter messages based on the kept exchange IDs
        kept_messages = []
        for message in messages:
            if message[MessageKeys.EXCHANGE_ID] in pinned_exchange_ids.union(
                exchange_ids_to_keep
            ):
                kept_messages.append(message)

        return kept_messages

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
        pending_image_analysis_user_messages: List[List[Dict[str, Any]]] = []

        tool_call_message = get_tool_call_message(tool_calls)
        self._append_message(Roles.ASSISTANT, tool_call_message, tool_calls=tool_calls)

        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)

            try:
                function_result = self._execute_function(function_name, function_args)
            except Exception as e:
                function_result = f"Error executing {function_name}: {str(e)}"

            artifact_id = f"{function_name}_{tool_call.id}"
            new_artifacts[artifact_id] = function_result

            result_representation = self._create_string_representation(
                function_result,
                function_name,
                function_args,
                self.client_wrapper.is_multimodal,
            )

            message_content_for_tool_response = json.dumps(
                {
                    MessageKeys.RESULT: result_representation,
                    MessageKeys.ARTIFACT_ID: artifact_id,
                }
            )

            self._append_message(
                Roles.TOOL, message_content_for_tool_response, tool_call_id=tool_call.id
            )

            if isinstance(function_result, PlotlyObject) and (
                image_analysis_content_parts := self._get_image_analysis_message(
                    function_result
                )
            ):
                pending_image_analysis_user_messages.append(
                    image_analysis_content_parts
                )

        for user_message_content_parts in pending_image_analysis_user_messages:
            self._append_message(
                Roles.USER,
                user_message_content_parts,
                pin_message=False,
                keep_list=True,
            )

        post_artifact_message_idx = len(self._messages)

        self._artifacts[post_artifact_message_idx] = list(new_artifacts.values())

        response = self.client_wrapper.chat_completion_create(
            messages=self._truncate(self._messages), tools=self._tools
        )

        return self._parse_model_response(response)

    def _get_image_analysis_message(self, function_result: Any) -> List[Dict[str, str]]:
        """Get prompt to handle image generation and analysis."""

        image_analysis_message = []

        if not self.client_wrapper.is_multimodal:
            return image_analysis_message

        try:
            image_data = self._plotly_to_base64(function_result)
        except Exception as e:
            logger.warning(f"Failed to convert Plotly figure to image: {str(e)}")
        else:
            image_analysis_message = [
                {
                    "type": "text",
                    "text": (IMAGE_ANALYSIS_PROMPT),
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_data}",
                        "detail": "high",
                    },
                },
            ]

        return image_analysis_message

    @staticmethod
    def _is_image_analysis_message(message: Dict[str, Any]) -> bool:
        """Check if a user message is an image analysis prompt."""
        if message[MessageKeys.ROLE] == Roles.USER and isinstance(
            message[MessageKeys.CONTENT], list
        ):
            is_image_analysis_text = False
            has_image_url = False
            for part in message[MessageKeys.CONTENT]:
                if isinstance(part, dict):
                    if part.get("type") == "text" and part.get("text", "").startswith(
                        IMAGE_REPRESENTATION_PROMPT[:20]
                    ):
                        is_image_analysis_text = True
                    elif part.get("type") == "image_url":
                        has_image_url = True
            return is_image_analysis_text or has_image_url
        return False

    @staticmethod
    def _create_string_representation(
        function_result: Any,
        function_name: str,
        function_args: Dict,
        is_multimodal: bool,
    ) -> str:
        """Create a string representation of the function result.

        Parameters
        ----------
        function_result : Any
            The result of the function call
        function_name : str
            The name of the function
        function_args : Dict
            The arguments passed to the function
        is_multimodal : bool
            Whether the current model supports multimodal inputs

        Returns
        -------
        str
            A string representation of the function result
        """
        result_type = type(function_result).__name__
        primitive_types = (int, float, str, bool)
        simple_iterable_types = (list, tuple, set)

        if isinstance(function_result, PlotlyObject):
            return (
                IMAGE_REPRESENTATION_PROMPT
                if is_multimodal
                else NO_REPRESENTATION_PROMPT
            )

        if isinstance(function_result, primitive_types):
            return str(function_result)

        if isinstance(function_result, pd.DataFrame):
            return function_result.to_json()

        items_to_check = None
        is_collection_type = False

        if isinstance(function_result, dict):
            items_to_check = function_result.values()
            is_collection_type = True
        elif isinstance(function_result, simple_iterable_types):
            items_to_check = function_result
            is_collection_type = True

        if is_collection_type:
            if all(isinstance(item, primitive_types) for item in items_to_check):
                return str(function_result)
            else:
                return (
                    ITERABLE_ARTIFACT_REPRESENTATION_PROMPT.format(
                        function_name,
                        json.dumps(function_args),
                        result_type,
                        len(function_result),
                    )
                    + NO_REPRESENTATION_PROMPT
                )

        return (
            SINGLE_ARTIFACT_REPRESENTATION_PROMPT.format(
                function_name, json.dumps(function_args), result_type
            )
            + NO_REPRESENTATION_PROMPT
        )

    def get_print_view(
        self, show_all=False
    ) -> Tuple[List[Dict[str, Any]], float, float]:
        """Get a structured view of the conversation history for display purposes."""

        print_view = []
        total_tokens = 0
        pinned_tokens = 0
        truncated_messages = self._truncate(self._messages)
        for message_idx, message in enumerate(self._messages):
            tokens = self.estimate_tokens([message], self.client_wrapper.model_name)
            in_context = message in truncated_messages
            if in_context:
                total_tokens += tokens
            if message[MessageKeys.PINNED]:
                pinned_tokens += tokens
            if not show_all and (
                message[MessageKeys.ROLE] in [Roles.TOOL, Roles.SYSTEM]
            ):
                continue
            if not show_all and MessageKeys.TOOL_CALLS in message:
                continue

            if self._is_image_analysis_message(message):
                continue

            print_view.append(
                {
                    MessageKeys.ROLE: message[MessageKeys.ROLE],
                    MessageKeys.CONTENT: message[MessageKeys.CONTENT],
                    MessageKeys.ARTIFACTS: self._artifacts.get(message_idx, []),
                    MessageKeys.IN_CONTEXT: in_context,
                    MessageKeys.PINNED: message[MessageKeys.PINNED],
                    MessageKeys.TIMESTAMP: message[MessageKeys.TIMESTAMP],
                }
            )

        return print_view, total_tokens, pinned_tokens

    def get_chat_log_txt(self) -> str:
        """Get a chat log in text format for saving."""
        messages, _, _ = self.get_print_view(show_all=True)
        chatlog = ""
        for message in messages:
            chatlog += f"[{message[MessageKeys.TIMESTAMP]}] {message[MessageKeys.ROLE].capitalize()}: {message[MessageKeys.CONTENT]}\n"
            if len(message[MessageKeys.ARTIFACTS]) > 0:
                chatlog += "-----\n"
            for artifact in message[MessageKeys.ARTIFACTS]:
                chatlog += f"Artifact: {artifact}\n"
            chatlog += "----------\n"
        return chatlog

    def chat_completion(
        self,
        prompt: str,
        role: str = Roles.USER,
        *,
        pin_message: bool = False,
        pass_tools: bool = True,
    ) -> None:
        """
        Generate a chat completion based on the given prompt and manage any resulting artifacts.

        Parameters
        ----------
        prompt : str
            The user's input prompt
        role : str, optional
            The role of the message sender, by default "user"
        pin_message : bool, optional
            Whether the prompt and assistant reply should be pinned, by default False
        pass_tools : bool, optional
            Whether to pass the tools to the model, by default True

        Returns
        -------
        Tuple[str, Dict[str, Any]]
            A tuple containing the generated response and a dictionary of new artifacts
        """
        self._exchange_count += 1

        self._append_message(role, prompt, pin_message=pin_message)

        try:
            response = self.client_wrapper.chat_completion_create(
                messages=self._truncate(self._messages),
                tools=self._tools if pass_tools else None,
            )

            content, tool_calls = self._parse_model_response(response)

            if tool_calls:
                if content:
                    self._append_message(
                        Roles.ASSISTANT, content, pin_message=pin_message
                    )

                content, _ = self._handle_function_calls(tool_calls)

            self._append_message(Roles.ASSISTANT, content, pin_message=pin_message)

        except ArithmeticError as e:
            error_message = f"Error in chat completion: {str(e)}"
            self._append_message(Roles.SYSTEM, error_message)

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
