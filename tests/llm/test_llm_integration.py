from datetime import datetime
from unittest import mock, skip
from unittest.mock import MagicMock, Mock, patch

import pandas as pd
import plotly.graph_objects as go
import pytest
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessage,
    ChatCompletionMessageToolCall,
)
from openai.types.chat.chat_completion_message_tool_call import Function

import alphastats.llm.llm_integration
from alphastats.llm.llm_integration import (
    LLMClientWrapper,
    LLMIntegration,
    ModelFlags,
    Models,
)
from alphastats.plots.plot_utils import PlotlyObject


@pytest.fixture
def mock_openai_client():
    with patch("alphastats.llm.llm_integration.OpenAI") as mock_client:
        yield mock_client


@pytest.fixture
def llm_integration(mock_openai_client) -> LLMIntegration:
    """Fixture providing a basic LLM instance with test configuration"""
    dataset = Mock()
    dataset.plot_intensity = Mock(return_value="Plot created")
    dataset.custom_function = Mock(return_value="Dataset function called")
    dataset.metadata = pd.DataFrame({"group1": ["A", "B"], "group2": ["C", "D"]})
    client_wrapper = MagicMock()
    return LLMIntegration(
        client_wrapper,
        system_message="Test system message",
        dataset=dataset,
        genes_of_interest={"GENE1": "PROT1", "GENE2": "PROT2"},
    )


@pytest.fixture
def llm_with_conversation(llm_integration: LLMIntegration) -> LLMIntegration:
    """Setup LLM with a sample conversation history"""
    # Add various message types to conversation history
    llm_integration._all_messages = [
        {
            "role": "system",
            "content": "System message",
            "pinned": True,
            "timestamp": "2022-01-01T00:00:00",
        },
        {
            "role": "user",
            "content": "User message 1",
            "pinned": False,
            "timestamp": "2022-01-01T00:00:00",
        },
        {
            "role": "assistant",
            "content": "Assistant message 1",
            "pinned": False,
            "timestamp": "2022-01-01T00:00:00",
        },
        {
            "role": "assistant",
            "content": "Assistant with tool calls",
            "tool_calls": [
                {"id": "123", "type": "function", "function": {"name": "test"}}
            ],
            "pinned": False,
            "timestamp": "2022-01-01T00:00:00",
        },
        {
            "role": "tool",
            "content": "Tool response",
            "pinned": False,
            "timestamp": "2022-01-01T00:00:00",
        },
        {
            "role": "user",
            "content": "User message 2",
            "pinned": False,
            "timestamp": "2022-01-01T00:00:00",
        },
        {
            "role": "assistant",
            "content": "Assistant message 2",
            "pinned": False,
            "timestamp": "2022-01-01T00:00:00",
        },
    ]

    llm_integration._messages = llm_integration._all_messages[0:3].copy()

    # Add some artifacts
    llm_integration._artifacts = {
        2: ["Artifact for message 2"],
        4: ["Tool artifact 1", "Tool artifact 2"],
        6: ["Artifact for message 6"],
    }

    return llm_integration


@pytest.fixture
def mock_chat_completion():
    """Fixture providing a mock successful chat completion"""
    mock_response = Mock(spec=ChatCompletion)
    mock_response.choices = [
        Mock(
            message=ChatCompletionMessage(
                role="assistant", content="Test response", tool_calls=None
            )
        )
    ]
    return mock_response


@pytest.fixture
def mock_tool_call_completion():
    """Fixture providing a mock completion with tool calls"""
    mock_response = Mock(spec=ChatCompletion)
    mock_response.choices = [
        Mock(
            message=ChatCompletionMessage(
                role="assistant",
                content="",
                tool_calls=[
                    ChatCompletionMessageToolCall(
                        id="test-id",
                        type="function",
                        function={"name": "test_function", "arguments": "{}"},
                    )
                ],
            )
        )
    ]
    return mock_response


@pytest.fixture
def mock_general_function_mapping():
    def mock_general_function(param1, param2):
        """An example for a function."""
        return f"General function called with {param1} and {param2}"

    return {"test_general_function": mock_general_function}


def test_initialization_gpt4(mock_openai_client):
    """Test initialization with GPT-4 configuration"""
    LLMClientWrapper(
        model_name=Models.GPT4O,
        api_key="test-key",  # pragma: allowlist secret
    )

    mock_openai_client.assert_called_once_with(
        api_key="test-key"  # pragma: allowlist secret
    )


def test_initialization_ollama(mock_openai_client):
    """Test initialization with Ollama configuration"""
    LLMClientWrapper(
        model_name=Models.OLLAMA_31_8B,
        base_url="http://localhost:11434",
        api_key="some_api_key",  # pragma: allowlist secret
    )

    mock_openai_client.assert_called_once_with(
        base_url="http://localhost:11434/v1",
        api_key="some_api_key",  # pragma: allowlist secret
    )


def test_initialization_invalid_model():
    """Test initialization with invalid model type"""
    with pytest.raises(ValueError, match="Invalid model name"):
        LLMClientWrapper(model_name="invalid-model")


@patch(f"{alphastats.llm.llm_integration.__name__}.datetime")
def test_append_message(mock_datetime, llm_integration: LLMIntegration):
    """Test message appending functionality"""

    mock_datetime.now.return_value = datetime(2022, 1, 1, 0, 0, 0)

    llm_integration._append_message("user", "Test message")

    assert len(llm_integration._messages) == 2  # Including system message
    assert len(llm_integration._all_messages) == 2
    assert llm_integration._messages[-1] == {
        "role": "user",
        "content": "Test message",
        "pinned": False,
        "timestamp": "2022-01-01T00:00:00",
    }


def test_append_message_with_tool_calls(llm_integration: LLMIntegration):
    """Test message appending with tool calls"""
    tool_calls = [
        ChatCompletionMessageToolCall(
            id="test-id",
            type="function",
            function={"name": "test_function", "arguments": "{}"},
        )
    ]

    llm_integration._append_message("assistant", "Test message", tool_calls=tool_calls)

    assert llm_integration._messages[-1]["tool_calls"] == tool_calls


@pytest.mark.parametrize(
    "num_messages,message_length,max_tokens,expected_messages",
    [
        (5, 100, 200, 2),  # Should truncate to 2 messages
        (3, 50, 1000, 3),  # Should keep all messages
        (10, 20, 100, 5),  # Should truncate to 5 messages
    ],
)
def test_truncate_conversation_history_success(
    llm_integration, num_messages, message_length, max_tokens, expected_messages
):
    """Test conversation history truncation with different scenarios"""
    # Add multiple messages
    message_content = "Test " * message_length
    llm_integration._max_tokens = max_tokens
    for _ in range(num_messages):
        llm_integration._append_message("user", message_content.strip())

    llm_integration._truncate_conversation_history()

    # Adding 1 to account for the initial system message
    assert len(llm_integration._messages) <= expected_messages + 1
    assert llm_integration._messages[0]["role"] == "system"


def test_truncate_conversation_history_pinned_too_large(llm_integration):
    """Test conversation history truncation with pinned messages that exceed the token limit"""
    # Add multiple messages
    message_content = "Test " * 100
    llm_integration._max_tokens = 200
    llm_integration._append_message("user", message_content.strip(), pin_message=True)
    llm_integration._append_message("user", message_content.strip(), pin_message=False)
    with pytest.raises(ValueError, match=r".*all remaining messages are pinned.*"):
        llm_integration._append_message(
            "assistant", message_content.strip(), pin_message=True
        )


def test_truncate_conversation_history_tool_output_popped(llm_integration):
    message_content = "Test " * 50
    llm_integration._max_tokens = 120
    # removal of assistant would suffice for total tokens, but tool output should be dropped as well
    llm_integration._append_message("assistant", message_content.strip())
    llm_integration._append_message("tool", message_content.strip())
    with (
        pytest.warns(UserWarning, match=r".*Truncating conversation history.*"),
        pytest.warns(UserWarning, match=r".*Removing corresponsing tool.*"),
    ):
        llm_integration._append_message("user", message_content.strip())

    assert len(llm_integration._messages) == 2
    assert llm_integration._messages[0]["role"] == "system"
    assert llm_integration._messages[1]["role"] == "user"


def test_truncate_conversation_history_last_tool_output_error(llm_integration):
    message_content = "Test " * 50
    llm_integration._max_tokens = 100
    # removal of assistant would suffice for total tokens, but tool output should be dropped as well
    llm_integration._append_message("assistant", message_content.strip())
    with pytest.raises(ValueError, match=r".*last call exceeds the token limit.*"):
        llm_integration._append_message("tool", message_content.strip())


def test_truncate_conversation_history_single_large_message(llm_integration):
    llm_integration._max_tokens = 1
    with pytest.raises(
        ValueError, match=r".*only remaining message exceeds the token limit*"
    ):
        llm_integration._truncate_conversation_history()


def test_estimate_tokens_gpt():
    """Test token estimation for a given message"""
    message_content = "Test message"
    tokens = LLMIntegration.estimate_tokens(
        [{"content": message_content}], model=Models.GPT4O
    )

    assert tokens == 2


def test_estimate_tokens_ollama():
    """Test token estimation for a given message with Ollama model, falls back on average chars per token"""
    message_content = "Test message"
    tokens = LLMIntegration.estimate_tokens(
        [{"content": message_content}],
        model=Models.OLLAMA_31_8B,
        average_chars_per_token=3.6,
    )

    assert tokens == 12 / 3.6


def test_estimate_tokens_default():
    """Test token estimation for a given message with Ollama model, falls back on average chars per token"""
    message_content = "Test message"
    tokens = LLMIntegration.estimate_tokens([{"content": message_content}])

    assert tokens == 12 / 3.6


def test_chat_completion_success(llm_integration: LLMIntegration, mock_chat_completion):
    """Test successful chat completion"""
    llm_integration.client_wrapper.chat_completion_create.return_value = (
        mock_chat_completion
    )

    llm_integration.chat_completion("Test prompt")

    assert llm_integration._messages == [
        {
            "content": "Test system message",
            "role": "system",
            "pinned": True,
            "timestamp": mock.ANY,
        },
        {
            "content": "Test prompt",
            "role": "user",
            "pinned": False,
            "timestamp": mock.ANY,
        },
        {
            "content": "Test response",
            "role": "assistant",
            "pinned": False,
            "timestamp": mock.ANY,
        },
    ]


def test_chat_completion_with_error(llm_integration: LLMIntegration):
    """Test chat completion with error handling"""
    llm_integration.client_wrapper.chat_completion_create.side_effect = ArithmeticError(
        "Test error"
    )

    llm_integration.chat_completion("Test prompt")

    assert (
        "Error in chat completion: Test error"
        in llm_integration._messages[-1]["content"]
    )


def test_parse_model_response(
    llm_integration: LLMIntegration, mock_tool_call_completion
):
    """Test parsing model response with tool calls"""
    content, tool_calls = llm_integration._parse_model_response(
        mock_tool_call_completion
    )

    assert content == ""
    assert len(tool_calls) == 1
    assert tool_calls[0].id == "test-id"
    assert tool_calls[0].type == "function"


@pytest.mark.parametrize(
    "function_name,function_args,expected_result",
    [
        (
            "test_general_function",
            {"param1": "value1", "param2": "value2"},
            "General function called with value1 and value2",
        ),
    ],
)
def test_execute_general_function(
    llm_integration: LLMIntegration,
    mock_general_function_mapping,
    function_name,
    function_args,
    expected_result,
):
    """Test execution of functions from GENERAL_FUNCTION_MAPPING"""
    with patch(
        "alphastats.llm.llm_integration.GENERAL_FUNCTION_MAPPING",
        mock_general_function_mapping,
    ):
        result = llm_integration._execute_function(function_name, function_args)
        assert result == expected_result


def test_execute_dataset_function(llm_integration: LLMIntegration):
    """Test execution of a function from the dataset"""
    result = llm_integration._execute_function("custom_function", {"param1": "value1"})

    assert result == "Dataset function called"
    llm_integration._dataset.custom_function.assert_called_once_with(param1="value1")


def test_execute_dataset_function_with_dots(llm_integration: LLMIntegration):
    """Test execution of a dataset function when name contains dots"""
    result = llm_integration._execute_function(
        "dataset.custom_function", {"param1": "value1"}
    )

    assert result == "Dataset function called"
    llm_integration._dataset.custom_function.assert_called_once_with(param1="value1")


@skip  # TODO fix this test
def test_execute_nonexistent_function(llm_integration: LLMIntegration):
    """Test execution of a non-existent function"""

    result = llm_integration._execute_function(
        "nonexistent_function", {"param1": "value1"}
    )

    assert "Error executing nonexistent_function" in result
    assert "not implemented or dataset not available" in result


def test_execute_function_with_error(
    llm_integration: LLMIntegration, mock_general_function_mapping
):
    """Test handling of function execution errors"""

    def failing_function(**kwargs):
        raise ValueError("Test error")

    with (
        patch(
            "alphastats.llm.llm_integration.GENERAL_FUNCTION_MAPPING",
            {"failing_function": failing_function},
        ),
        pytest.raises(ValueError, match="Test error"),
    ):
        llm_integration._execute_function("failing_function", {"param1": "value1"})


def test_execute_function_without_dataset(mock_openai_client):
    """Test function execution when dataset is not available"""
    llm = LLMIntegration(LLMClientWrapper(model_name=Models.GPT4O, api_key="test-key"))

    with pytest.raises(
        ValueError,
        match="Function dataset_function not implemented or dataset not available",
    ):
        llm._execute_function("dataset_function", {"param1": "value1"})


@patch("alphastats.llm.llm_integration.LLMIntegration._execute_function")
def test_handle_function_calls(
    mock_execute_function, mock_openai_client, mock_chat_completion
):
    """Test handling of function calls in the chat completion response."""
    mock_execute_function.return_value = "some_function_result"

    llm_integration = LLMIntegration(
        LLMClientWrapper(
            model_name=Models.GPT4O,
            api_key="test-key",  # pragma: allowlist secret
        ),
        system_message="Test system message",
    )

    tool_calls = [
        ChatCompletionMessageToolCall(
            id="test-id",
            type="function",
            function={"name": "test_function", "arguments": '{"arg1": "value1"}'},
        )
    ]

    mock_openai_client.return_value.chat.completions.create.return_value = (
        mock_chat_completion
    )

    # when
    result = llm_integration._handle_function_calls(tool_calls)

    assert result == ("Test response", None)

    mock_execute_function.assert_called_once_with("test_function", {"arg1": "value1"})

    expected_messages = [
        {
            "role": "system",
            "content": "Test system message",
            "pinned": True,
            "timestamp": mock.ANY,
        },
        {
            "role": "assistant",
            "content": 'Calling function: test_function with arguments: {"arg1": "value1"}',
            "tool_calls": [
                ChatCompletionMessageToolCall(
                    id="test-id",
                    function=Function(
                        arguments='{"arg1": "value1"}', name="test_function"
                    ),
                    type="function",
                )
            ],
            "pinned": False,
            "timestamp": mock.ANY,
        },
        {
            "role": "tool",
            "content": '{"result": "some_function_result", "artifact_id": "test_function_test-id"}',
            "tool_call_id": "test-id",
            "pinned": False,
            "timestamp": mock.ANY,
        },
    ]
    mock_openai_client.return_value.chat.completions.create.assert_called_once_with(
        model="gpt-4o", messages=expected_messages, tools=llm_integration._tools
    )

    assert list(llm_integration._artifacts[3]) == ["some_function_result"]

    assert llm_integration._messages == expected_messages


@patch("alphastats.llm.llm_integration.LLMIntegration._get_image_analysis_message")
@patch("alphastats.llm.llm_integration.LLMIntegration._execute_function")
def test_handle_function_calls_with_images(
    mock_execute_function,
    mock_get_image_analysis_message,
    mock_openai_client,
    mock_chat_completion,
):
    """Test handling of function calls that return images in the chat completion response."""
    mock_execute_function.return_value = PlotlyObject()

    llm_integration = LLMIntegration(
        LLMClientWrapper(
            model_name=Models.GPT4O,
            api_key="test-key",  # pragma: allowlist secret
        ),
        system_message="Test system message",
    )

    tool_calls = [
        ChatCompletionMessageToolCall(
            id="test-id",
            type="function",
            function={"name": "test_function", "arguments": '{"arg1": "value1"}'},
        )
    ]

    mock_openai_client.return_value.chat.completions.create.return_value = (
        mock_chat_completion
    )
    mock_get_image_analysis_message.return_value = [
        {"image_analysis_message": "something"}
    ]

    # when
    _ = llm_integration._handle_function_calls(tool_calls)

    assert {
        "role": "user",
        "pinned": False,
        "content": [{"image_analysis_message": "something"}],
        "timestamp": mock.ANY,
    } in mock_openai_client.return_value.chat.completions.create.call_args_list[
        0
    ].kwargs["messages"]


def test_get_image_analysis_message_returns_empty_prompt_if_model_not_multimodal():
    """Test that the _get_image_analysis_message method returns an empty prompt if the model is not multimodal."""
    llm_integration = LLMIntegration(
        LLMClientWrapper(
            model_name=Models.OLLAMA_31_70B,
            api_key="test-key",  # pragma: allowlist secret
        ),
        system_message="Test system message",
    )

    # when
    result = llm_integration._get_image_analysis_message(MagicMock())

    assert result == []


@patch("alphastats.llm.llm_integration.LLMIntegration._plotly_to_base64")
def test_get_image_analysis_message_returns_prompt_with_image_data_for_multimodal_model(
    mock_plotly_to_base64,
):
    """Test that the _get_image_analysis_message method returns a prompt with image data for a multimodal model."""

    llm_integration = LLMIntegration(
        LLMClientWrapper(
            model_name=ModelFlags.MULTIMODAL[0],
            api_key="test-key",  # pragma: allowlist secret
        ),
        system_message="Test system message",
    )

    function_result = MagicMock()
    mock_plotly_to_base64.return_value = "mock_base64_image_data"

    # when
    result = llm_integration._get_image_analysis_message(function_result)

    assert result == [
        {
            "type": "text",
            "text": "The previous tool call generated the following image. Please analyze it in the context of our current discussion and your previous actions.",
        },
        {
            "type": "image_url",
            "image_url": {
                "url": "data:image/png;base64,mock_base64_image_data",
                "detail": "high",
            },
        },
    ]


@patch("alphastats.llm.llm_integration.LLMIntegration._plotly_to_base64")
def test_get_image_analysis_message_handles_plotly_conversion_failure_gracefully(
    mock_plotly_to_base64,
):
    """Test that the _get_image_analysis_message method handles plotly conversion failure gracefully."""
    llm_integration = LLMIntegration(
        LLMClientWrapper(
            model_name=ModelFlags.MULTIMODAL[0],
            api_key="test-key",  # pragma: allowlist secret
        ),
        system_message="Test system message",
    )

    function_result = MagicMock()
    mock_plotly_to_base64.side_effect = Exception("Conversion failed")

    # when
    result = llm_integration._get_image_analysis_message(function_result)

    assert result == []


def test_get_print_view_default(llm_with_conversation: LLMIntegration):
    """Test get_print_view with default settings (show_all=False)"""
    print_view, _, _ = llm_with_conversation.get_print_view()

    # Should only include user and assistant messages without tool_calls
    assert print_view == [
        {
            "artifacts": [],
            "content": "User message 1",
            "role": "user",
            "in_context": True,
            "pinned": False,
            "timestamp": mock.ANY,
        },
        {
            "artifacts": ["Artifact for message 2"],
            "content": "Assistant message 1",
            "role": "assistant",
            "in_context": True,
            "pinned": False,
            "timestamp": mock.ANY,
        },
        {
            "artifacts": [],
            "content": "User message 2",
            "role": "user",
            "in_context": False,
            "pinned": False,
            "timestamp": mock.ANY,
        },
        {
            "artifacts": ["Artifact for message 6"],
            "content": "Assistant message 2",
            "role": "assistant",
            "in_context": False,
            "pinned": False,
            "timestamp": mock.ANY,
        },
    ]


def test_get_print_view_show_all(llm_with_conversation: LLMIntegration):
    """Test get_print_view with default settings (show_all=True)"""
    print_view, _, _ = llm_with_conversation.get_print_view(show_all=True)

    # Should only include user and assistant messages without tool_calls
    assert print_view == [
        {
            "artifacts": [],
            "content": "System message",
            "role": "system",
            "in_context": True,
            "pinned": True,
            "timestamp": mock.ANY,
        },
        {
            "artifacts": [],
            "content": "User message 1",
            "role": "user",
            "in_context": True,
            "pinned": False,
            "timestamp": mock.ANY,
        },
        {
            "artifacts": ["Artifact for message 2"],
            "content": "Assistant message 1",
            "role": "assistant",
            "in_context": True,
            "pinned": False,
            "timestamp": mock.ANY,
        },
        {
            "artifacts": [],
            "content": "Assistant with tool calls",
            "role": "assistant",
            "in_context": False,
            "pinned": False,
            "timestamp": mock.ANY,
        },
        {
            "artifacts": ["Tool artifact 1", "Tool artifact 2"],
            "content": "Tool response",
            "role": "tool",
            "in_context": False,
            "pinned": False,
            "timestamp": mock.ANY,
        },
        {
            "artifacts": [],
            "content": "User message 2",
            "role": "user",
            "in_context": False,
            "pinned": False,
            "timestamp": mock.ANY,
        },
        {
            "artifacts": ["Artifact for message 6"],
            "content": "Assistant message 2",
            "role": "assistant",
            "in_context": False,
            "pinned": False,
            "timestamp": mock.ANY,
        },
    ]


@pytest.mark.parametrize(
    "function_result, function_name, function_args, output, is_multimodal",
    [
        (
            "primitive_result",
            "some_function_name",
            {"arg1": "value1"},
            "primitive_result",
            False,
        ),
        (
            [1, 2, 3],
            "some_function_name",
            {"returns": "primitive list"},
            "[1, 2, 3]",
            False,
        ),
        (
            ("arg1", 1),
            "some_function_name",
            {"returns": "a tuple with primitive values"},
            "('arg1', 1)",
            False,
        ),
        (
            {"arg1": "value1"},
            "some_function_name",
            {"returns": "a dictionary with primitive values"},
            "{'arg1': 'value1'}",
            False,
        ),
        (
            ("DataFrame", pd.DataFrame([[1, 2, 3]], columns=["a", "b", "c"])),
            "some_function_name",
            {"returns": "a tuple with non-primitive elements"},
            'Function some_function_name with arguments {"returns": "a tuple with non-primitive elements"} returned a tuple, containing 2 elements, some of which are non-trivial to represent as text. There is currently no text representation for this artifact that can be interpreted meaningfully. If the user asks for guidance how to interpret the artifact please rely on the description of the tool function and the arguments it was called with.',
            False,
        ),
        (
            {"DataFrame": pd.DataFrame([[1, 2, 3]], columns=["a", "b", "c"])},
            "some_function_name",
            {"returns": "a dictionary with non-primitive values"},
            'Function some_function_name with arguments {"returns": "a dictionary with non-primitive values"} returned a dict, containing 1 elements, some of which are non-trivial to represent as text. There is currently no text representation for this artifact that can be interpreted meaningfully. If the user asks for guidance how to interpret the artifact please rely on the description of the tool function and the arguments it was called with.',
            False,
        ),
        (
            pd.DataFrame([[1, 2, 3]], columns=["a", "b", "c"]),
            "some_function_name",
            {"arg1": "value1"},
            r'{"a":{"0":1},"b":{"0":2},"c":{"0":3}}',
            False,
        ),
        (
            go.Figure(),
            "some_function_name",
            {"arg1": "value1"},
            'Function some_function_name with arguments {"arg1": "value1"} returned a Figure. There is currently no text representation for this artifact that can be interpreted meaningfully. If the user asks for guidance how to interpret the artifact please rely on the description of the tool function and the arguments it was called with.',
            False,
        ),
        (
            PlotlyObject(),
            "some_function_name",
            {"arg1": "value1"},
            " There is currently no text representation for this artifact that can be interpreted meaningfully. If the user asks for guidance how to interpret the artifact please rely on the description of the tool function and the arguments it was called with.",
            False,
        ),
        (
            PlotlyObject(),
            "some_function_name",
            {"arg1": "value1"},
            "This is a visualization result that will be provided as an image.",
            True,
        ),
    ],
)
def test_str_repr(function_result, function_name, function_args, output, is_multimodal):
    assert (
        LLMIntegration._create_string_representation(
            function_result, function_name, function_args, is_multimodal=is_multimodal
        )
        == output
    )
