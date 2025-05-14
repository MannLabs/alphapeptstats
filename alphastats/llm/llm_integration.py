"""Module to integrate different LLM APIs and handle chat interactions."""

import base64
import json
import logging
import warnings
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import plotly.io as pio
import pytz
import tiktoken
from IPython.display import HTML, Markdown, display
from openai import OpenAI
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
from alphastats.llm.prompts import get_tool_call_message

logger = logging.getLogger(__name__)


class Models(metaclass=ConstantsClass):
    """Names of the available models.

    Note that this will be directly passed to the OpenAI client.
    """

    GPT4O = "gpt-4o"
    OLLAMA_31_70B = "llama3.1:70b"
    OLLAMA_31_8B = "llama3.1:8b"  # for testing only
    OLLAMA_33_70B_INSTRUCT = "llama-3.3-70b-instruct"
    MISTRAL_LARGE_INSTRUCT = "mistral-large-instruct"
    QWEN_25_72B_INSTRUCT = "qwen2.5-72b-instruct"


class ModelFlags(metaclass=ConstantsClass):
    """Requirements for the different models."""

    REQUIRES_API_KEY = [Models.GPT4O]
    REQUIRES_BASE_URL = [
        Models.OLLAMA_31_70B,
        Models.OLLAMA_31_8B,
        Models.OLLAMA_33_70B_INSTRUCT,
        Models.MISTRAL_LARGE_INSTRUCT,
        Models.QWEN_25_72B_INSTRUCT,
    ]
    MULTIMODAL = [Models.GPT4O]


class MessageKeys(metaclass=ConstantsClass):
    """Keys for the message dictionary."""

    ROLE = "role"
    CONTENT = "content"
    TOOL_CALLS = "tool_calls"
    TOOL_CALL_ID = "tool_call_id"
    RESULT = "result"
    ARTIFACT_ID = "artifact_id"
    IN_CONTEXT = "in_context"
    ARTIFACTS = "artifacts"
    PINNED = "pinned"
    TIMESTAMP = "timestamp"
    IMAGE_URL = "image_url"


class Roles(metaclass=ConstantsClass):
    """Names of the available roles."""

    USER = "user"
    ASSISTANT = "assistant"  # might show as tool-call in error messages
    TOOL = "tool"
    SYSTEM = "system"


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

        if model_name in ModelFlags.REQUIRES_BASE_URL:
            url = f"{base_url}/v1"  # TODO: enable to configure this per model
            self._client = OpenAI(base_url=url, api_key=api_key)
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
    ) -> None:
        """Construct a message and append it to the conversation history."""
        message = {
            MessageKeys.ROLE: role,
            MessageKeys.PINNED: pin_message,
        }

        if (
            role == Roles.USER
            and isinstance(content, list)
            and any(
                isinstance(part, dict) and part.get("type") == "image_url"
                for part in content
            )
        ):
            message[MessageKeys.CONTENT] = content
        elif not isinstance(content, str):
            message[MessageKeys.CONTENT] = str(content)
        else:
            message[MessageKeys.CONTENT] = content

        if tool_calls is not None:
            message[MessageKeys.TOOL_CALLS] = tool_calls

        if tool_call_id is not None:
            message[MessageKeys.TOOL_CALL_ID] = tool_call_id

        message[MessageKeys.TIMESTAMP] = datetime.now(tz=pytz.utc).strftime(
            "%Y-%m-%dT%H:%M:%S"
        )

        self._messages.append(message)
        self._all_messages.append(message)

        # self._truncate_conversation_history()

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
        total_tokens = 0
        try:
            enc = tiktoken.encoding_for_model(model)
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
        return total_tokens

    def _truncate_conversation_history(
        self, average_chars_per_token: float = 3.6
    ) -> None:
        """
        Truncate the conversation history to stay within token limits.

        In the process pinned messages are preserved. If a tool call is removed, corresponding tool outputs are also removed, as this would otherwise raise an error in the chat completion.

        Parameters
        ----------
        average_chars_per_token : float, optional
            The average number of characters per token, by default 3.6. Normal english language has 4 per token. Every ID included in the text is 1 token per character. Parsed uniprot entries are between 3.6 and 3.9 judging from experience with https://platform.openai.com/tokenizer.
        """
        # TODO: avoid important messages being removed (e.g. facts about genes)
        # TODO: find out how messages can be None type and handle them earlier
        while (
            self.estimate_tokens(self._messages, self._model, average_chars_per_token)
            > self._max_tokens
        ):
            if len(self._messages) == 1:
                raise ValueError(
                    "Truncating conversation history failed, as the only remaining message exceeds the token limit. Please increase the token limit and reset the LLM interpretation."
                )
            oldest_not_pinned = -1
            for message_idx, message in enumerate(self._messages):
                if not message[MessageKeys.PINNED]:
                    oldest_not_pinned = message_idx
                    break
            if oldest_not_pinned == -1:
                raise ValueError(
                    "Truncating conversation history failed, as all remaining messages are pinned. Please increase the token limit and reset the LLM interpretation, or unpin messages."
                )
            removed_message = self._messages.pop(oldest_not_pinned)
            warnings.warn(
                f"Truncating conversation history to stay within token limits.\nRemoved message:{removed_message[MessageKeys.ROLE]}: {removed_message[MessageKeys.CONTENT][0:30]}..."
            )
            while (
                removed_message[MessageKeys.ROLE] == Roles.ASSISTANT
                and self._messages[oldest_not_pinned][MessageKeys.ROLE] == Roles.TOOL
            ):
                # This is required as the chat completion fails if there are tool outputs without corresponding tool calls in the message history.
                removed_toolmessage = self._messages.pop(oldest_not_pinned)
                warnings.warn(
                    f"Removing corresponsing tool output as well.\nRemoved message:{removed_toolmessage[MessageKeys.ROLE]}: {removed_toolmessage[MessageKeys.CONTENT][0:30]}..."
                )
                if len(self._messages) == oldest_not_pinned:
                    raise ValueError(
                        "Truncating conversation history failed, as the artifact from the last call exceeds the token limit. Please increase the token limit and reset the LLM interpretation."
                    )

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

            image_data = None
            if "PlotlyObject" in str(type(function_result)):
                try:
                    image_data = self._plotly_to_base64(function_result)
                except Exception as e:
                    logger.warning(
                        f"Failed to convert Plotly figure to image: {str(e)}"
                    )

            result_representation = self._create_string_representation(
                function_result, function_name, function_args
            )

            content_dict = {
                MessageKeys.RESULT: result_representation,
                MessageKeys.ARTIFACT_ID: artifact_id,
            }

            content_json_string = json.dumps(content_dict)
            self._append_message(
                Roles.TOOL, content_json_string, tool_call_id=tool_call.id
            )

            if image_data and self._model in ModelFlags.MULTIMODAL:
                user_image_analysis_prompt_content = [
                    {
                        "type": "text",
                        "text": (
                            "The previous tool call generated the following image. "
                            "Please analyze it in the context of our current discussion and your previous actions."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_data}",
                            "detail": "high",
                        },
                    },
                ]
                self._append_message(
                    Roles.USER, user_image_analysis_prompt_content, pin_message=False
                )

        post_artifact_message_idx = len(self._all_messages)
        self._artifacts[post_artifact_message_idx] = list(
            new_artifacts.values()
        )  # Ensure it's a list

        response = self._chat_completion_create()

        return self._parse_model_response(response)

    @staticmethod
    def _is_image_analysis_prompt(message: Dict[str, Any]) -> bool:
        """Check if a user message is an image analysis prompt."""
        if message[MessageKeys.ROLE] == Roles.USER and isinstance(
            message[MessageKeys.CONTENT], list
        ):
            is_image_analysis_text = False
            has_image_url = False
            for part in message[MessageKeys.CONTENT]:
                if isinstance(part, dict):
                    if part.get("type") == "text" and part.get("text", "").startswith(
                        "The previous tool call generated the following image."
                    ):
                        is_image_analysis_text = True
                    elif part.get("type") == "image_url":
                        has_image_url = True
            return is_image_analysis_text and has_image_url
        return False

    @staticmethod
    def _create_string_representation(
        function_result: Any, function_name: str, function_args: Dict
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

        Returns
        -------
        str
            A string representation of the function result
        """
        result_type = type(function_result).__name__
        primitive_types = (int, float, str, bool)
        simple_iterable_types = (list, tuple, set)
        iterable_artifact_description = "Function {} with arguments {} returned a {}, containing {} elements, some of which are non-trivial to represent as text."
        single_artifact_description = "Function {} with arguments {} returned a {}."
        LLM_instructions = " There is currently no text representation for this artifact that can be interpreted meaningfully. If the user asks for guidance how to interpret the artifact please rely on the desription of the tool function and the arguments it was called with."
        if isinstance(function_result, primitive_types):
            return str(function_result)
        elif isinstance(function_result, pd.DataFrame):
            return function_result.to_json()
        elif isinstance(function_result, dict):
            if all(
                isinstance(element, primitive_types)
                for element in function_result.values()
            ):
                return str(function_result)
            else:
                return (
                    iterable_artifact_description.format(
                        function_name,
                        json.dumps(function_args),
                        result_type,
                        len(function_result),
                    )
                    + LLM_instructions
                )
        elif isinstance(function_result, simple_iterable_types):
            if all(isinstance(element, primitive_types) for element in function_result):
                return str(function_result)
            else:
                return (
                    iterable_artifact_description.format(
                        function_name,
                        json.dumps(function_args),
                        result_type,
                        len(function_result),
                    )
                    + LLM_instructions
                )
        elif "PlotlyObject" in str(type(function_result)):
            return "This is a visualization result that will be shown visually. Please analyze what you see in this plot."
        else:
            return (
                single_artifact_description.format(
                    function_name, json.dumps(function_args), result_type
                )
                + LLM_instructions
            )

    def _chat_completion_create(self) -> ChatCompletion:
        """Create a chat completion based on the current conversation history."""
        logger.info(f"Calling 'chat.completions.create' {self._messages[-1]} ..")
        for message in self._messages:
            print(message)
        result = self._client.chat.completions.create(
            model=self._model,
            messages=self._messages,
            tools=self._tools,
        )
        logger.info(".. done")
        return result

    def get_print_view(
        self, show_all=False
    ) -> Tuple[List[Dict[str, Any]], float, float]:
        """Get a structured view of the conversation history for display purposes."""

        print_view = []
        total_tokens = 0
        pinned_tokens = 0
        for message_idx, message in enumerate(self._all_messages):
            tokens = self.estimate_tokens([message], self._model)
            in_context = message in self._messages
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

            if self._is_image_analysis_prompt(message):
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
        self, prompt: str, role: str = Roles.USER, *, pin_message=False
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

        Returns
        -------
        Tuple[str, Dict[str, Any]]
            A tuple containing the generated response and a dictionary of new artifacts
        """
        self._append_message(role, prompt, pin_message=pin_message)

        try:
            response = self._chat_completion_create()

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
            role = message[MessageKeys.ROLE]
            content = message[MessageKeys.CONTENT]
            tokens = self.estimate_tokens([message], self._model)

            if role == Roles.ASSISTANT and MessageKeys.TOOL_CALLS in message:
                display(
                    Markdown(
                        f"**{role.capitalize()}**: {content} *({str(tokens)} tokens)*"
                    )
                )
                for tool_call in message[MessageKeys.TOOL_CALLS]:
                    function_name = tool_call.function.name
                    function_args = tool_call.function.arguments
                    display(Markdown(f"*Function Call*: `{function_name}`"))
                    display(Markdown(f"*Arguments*: ```json\n{function_args}\n```"))

            elif role == Roles.TOOL:
                tool_result = json.loads(content)
                artifact_id = tool_result.get(MessageKeys.ARTIFACT_ID)
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
                display(
                    Markdown(
                        f"**{role.capitalize()}**: {content} *({str(tokens)} tokens)*"
                    )
                )

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
