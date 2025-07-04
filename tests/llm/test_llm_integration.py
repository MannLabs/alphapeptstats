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
from alphastats.llm.llm_integration import LLMClientWrapper, LLMIntegration, MessageKeys
from alphastats.plots.plot_utils import PlotlyObject

GPT_MODEL_NAME = "openai/gpt-4o"
OLLAMA_MODEL_NAME = "ollama/llama3.1:8b"


@pytest.fixture
def mock_completion():
    with patch("alphastats.llm.llm_integration.completion") as mock_client:
        yield mock_client


@pytest.fixture
def llm_integration(mock_completion) -> LLMIntegration:
    """Fixture providing a basic LLM instance with test configuration"""
    dataset = Mock()
    dataset.plot_intensity = Mock(return_value="Plot created")
    dataset.custom_function = Mock(return_value="Dataset function called")
    dataset.metadata = pd.DataFrame(
        {"sample_": ["S1", "S2"], "group1": ["A", "B"], "group2": ["C", "D"]}
    )
    client_wrapper = MagicMock()
    return LLMIntegration(
        client_wrapper,
        system_message="Test system message",
        dataset=dataset,
    )


@pytest.fixture
def llm_with_conversation(llm_integration: LLMIntegration) -> LLMIntegration:
    """Setup LLM with a sample conversation history"""
    # Add various message types to conversation history
    llm_integration._messages = [
        {
            "role": "system",
            "content": "System message",
            "___exchange_id": 1,
            "___pinned": True,
            "___timestamp": datetime(2022, 1, 1, 0, 0, 0),
        },
        {
            "role": "user",
            "content": "User message 1",
            "___exchange_id": 2,
            "___pinned": False,
            "___timestamp": datetime(2022, 1, 1, 0, 0, 0),
        },
        {
            "role": "assistant",
            "content": "Assistant message 1",
            "___exchange_id": 2,
            "___pinned": False,
            "___timestamp": datetime(2022, 1, 1, 0, 0, 0),
        },
        {
            "role": "assistant",
            "content": "Assistant with tool calls",
            "tool_calls": [
                {"id": "123", "type": "function", "function": {"name": "test"}}
            ],
            "___exchange_id": 2,
            "___pinned": False,
            "___timestamp": datetime(2022, 1, 1, 0, 0, 0),
        },
        {
            "role": "tool",
            "content": "Tool response",
            "___exchange_id": 2,
            "___pinned": False,
            "___timestamp": datetime(2022, 1, 1, 0, 0, 0),
        },
        {
            "role": "user",
            "content": "User message 2",
            "___exchange_id": 3,
            "___pinned": False,
            "___timestamp": datetime(2022, 1, 1, 0, 0, 0),
        },
        {
            "role": "assistant",
            "content": "Assistant message 2",
            "___exchange_id": 3,
            "___pinned": False,
            "___timestamp": datetime(2022, 1, 1, 0, 0, 0),
        },
    ]

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


@patch(f"{alphastats.llm.llm_integration.__name__}.datetime")
def test_append_message(mock_datetime, llm_integration: LLMIntegration):
    """Test message appending functionality"""

    mock_datetime.now.return_value = datetime(2022, 1, 1, 0, 0, 0)

    llm_integration._append_message("user", "Test message")

    assert len(llm_integration._messages) == 2  # Including system message
    assert llm_integration._messages[-1] == {
        "___exchange_id": 0,
        "___timestamp": datetime(2022, 1, 1, 0, 0, 0),
        "___pinned": False,
        "role": "user",
        "content": "Test message",
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


def test_estimate_tokens_gpt():
    """Test token estimation for a given message"""
    message_content = "Test message"
    tokens = LLMIntegration.estimate_tokens(
        [{"content": message_content}], model=GPT_MODEL_NAME
    )

    assert tokens == 2


def test_estimate_tokens_ollama():
    """Test token estimation for a given message with Ollama model, falls back on average chars per token"""
    message_content = "Test message"
    tokens = LLMIntegration.estimate_tokens(
        [{"content": message_content}],
        model=OLLAMA_MODEL_NAME,
        average_chars_per_token=3.9,
    )

    assert tokens == int(12 / 3.9)


def test_estimate_tokens_default():
    """Test token estimation for a given message with Ollama model, falls back on average chars per token"""
    message_content = "Test message"
    tokens = LLMIntegration.estimate_tokens([{"content": message_content}])

    assert tokens == int(12 / 3.6)


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
            "___exchange_id": 0,
            "___pinned": True,
            "___timestamp": mock.ANY,
        },
        {
            "content": "Test prompt",
            "role": "user",
            "___exchange_id": 1,
            "___pinned": False,
            "___timestamp": mock.ANY,
        },
        {
            "content": "Test response",
            "role": "assistant",
            "___exchange_id": 1,
            "___pinned": False,
            "___timestamp": mock.ANY,
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


def test_execute_function_without_dataset(mock_completion):
    """Test function execution when dataset is not available"""
    llm = LLMIntegration(
        LLMClientWrapper(
            model_name=GPT_MODEL_NAME,
            api_key="test-key",  # pragma: allowlist secret
        )
    )

    with pytest.raises(
        ValueError,
        match="Function dataset_function not implemented or dataset not available",
    ):
        llm._execute_function("dataset_function", {"param1": "value1"})


@patch("alphastats.llm.llm_integration.LLMIntegration._execute_function")
def test_handle_function_calls(
    mock_execute_function, mock_completion, mock_chat_completion
):
    """Test handling of function calls in the chat completion response."""
    mock_execute_function.return_value = "some_function_result"

    llm_integration = LLMIntegration(
        LLMClientWrapper(
            model_name=GPT_MODEL_NAME,
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

    mock_completion.return_value = mock_chat_completion

    # when
    result = llm_integration._handle_function_calls(tool_calls)

    assert result == ("Test response", None)

    mock_execute_function.assert_called_once_with("test_function", {"arg1": "value1"})

    expected_messages = [
        {
            "role": "system",
            "content": "Test system message",
            "___exchange_id": 0,
            "___pinned": True,
            "___timestamp": mock.ANY,
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
            "___exchange_id": 0,
            "___pinned": False,
            "___timestamp": mock.ANY,
        },
        {
            "role": "tool",
            "content": '{"result": "some_function_result", "artifact_id": "test_function_test-id"}',
            "tool_call_id": "test-id",
            "___exchange_id": 0,
            "___pinned": False,
            "___timestamp": mock.ANY,
        },
    ]
    mock_completion.assert_called_once_with(
        model=GPT_MODEL_NAME,
        api_key="test-key",  # pragma: allowlist secret
        base_url=None,
        messages=[
            {k: v for k, v in message.items() if not k.startswith("___")}
            for message in expected_messages
        ],
        tools=llm_integration._tools,
    )

    assert list(llm_integration._artifacts[3]) == ["some_function_result"]

    assert llm_integration._messages == expected_messages


@patch("alphastats.llm.llm_integration.LLMIntegration._get_image_analysis_message")
@patch("alphastats.llm.llm_integration.LLMIntegration._execute_function")
def test_handle_function_calls_with_images(
    mock_execute_function,
    mock_get_image_analysis_message,
    mock_completion,
    mock_chat_completion,
):
    """Test handling of function calls that return images in the chat completion response."""
    mock_execute_function.return_value = PlotlyObject()

    llm_integration = LLMIntegration(
        LLMClientWrapper(
            model_name=GPT_MODEL_NAME,
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

    mock_completion.return_value = mock_chat_completion
    mock_get_image_analysis_message.return_value = [
        {"image_analysis_message": "something"}
    ]

    # when
    _ = llm_integration._handle_function_calls(tool_calls)

    assert {
        "role": "user",
        "content": [{"image_analysis_message": "something"}],
    } in mock_completion.call_args_list[0].kwargs["messages"]


def test_get_image_analysis_message_returns_empty_prompt_if_model_not_multimodal():
    """Test that the _get_image_analysis_message method returns an empty prompt if the model is not multimodal."""
    llm_integration = LLMIntegration(
        LLMClientWrapper(
            model_name=OLLAMA_MODEL_NAME,
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
            model_name=GPT_MODEL_NAME,
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
            model_name=GPT_MODEL_NAME,
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
            "___in_context": True,
            "___pinned": False,
            "___timestamp": mock.ANY,
        },
        {
            "artifacts": ["Artifact for message 2"],
            "content": "Assistant message 1",
            "role": "assistant",
            "___in_context": True,
            "___pinned": False,
            "___timestamp": mock.ANY,
        },
        {
            "artifacts": [],
            "content": "User message 2",
            "role": "user",
            "___in_context": True,
            "___pinned": False,
            "___timestamp": mock.ANY,
        },
        {
            "artifacts": ["Artifact for message 6"],
            "content": "Assistant message 2",
            "role": "assistant",
            "___in_context": True,
            "___pinned": False,
            "___timestamp": mock.ANY,
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
            "___in_context": True,
            "___pinned": True,
            "___timestamp": mock.ANY,
        },
        {
            "artifacts": [],
            "content": "User message 1",
            "role": "user",
            "___in_context": True,
            "___pinned": False,
            "___timestamp": mock.ANY,
        },
        {
            "artifacts": ["Artifact for message 2"],
            "content": "Assistant message 1",
            "role": "assistant",
            "___in_context": True,
            "___pinned": False,
            "___timestamp": mock.ANY,
        },
        {
            "artifacts": [],
            "content": "Assistant with tool calls",
            "role": "assistant",
            "___in_context": True,
            "___pinned": False,
            "___timestamp": mock.ANY,
        },
        {
            "artifacts": ["Tool artifact 1", "Tool artifact 2"],
            "content": "Tool response",
            "role": "tool",
            "___in_context": True,
            "___pinned": False,
            "___timestamp": mock.ANY,
        },
        {
            "artifacts": [],
            "content": "User message 2",
            "role": "user",
            "___in_context": True,
            "___pinned": False,
            "___timestamp": mock.ANY,
        },
        {
            "artifacts": ["Artifact for message 6"],
            "content": "Assistant message 2",
            "role": "assistant",
            "___in_context": True,
            "___pinned": False,
            "___timestamp": mock.ANY,
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


@pytest.fixture
def llm_integration_ut():
    """Create a mock instance with required attributes and methods"""
    instance = LLMIntegration(MagicMock())
    instance._max_tokens = 1000
    instance.estimate_tokens = MagicMock(return_value=100)  # Default token estimate
    return instance


@pytest.fixture
def sample_messages():
    """Create sample messages for truncation testing"""
    return [
        {
            MessageKeys.CONTENT: "m1",
            MessageKeys.EXCHANGE_ID: 1,
            MessageKeys.PINNED: True,
        },
        {
            MessageKeys.CONTENT: "m2",
            MessageKeys.EXCHANGE_ID: 1,
            MessageKeys.PINNED: False,
        },
        {
            MessageKeys.CONTENT: "m3",
            MessageKeys.EXCHANGE_ID: 2,
            MessageKeys.PINNED: False,
        },
        {
            MessageKeys.CONTENT: "m4",
            MessageKeys.EXCHANGE_ID: 2,
            MessageKeys.PINNED: False,
        },
        {
            MessageKeys.CONTENT: "m5",
            MessageKeys.EXCHANGE_ID: 3,
            MessageKeys.PINNED: True,
        },
        {
            MessageKeys.CONTENT: "m6",
            MessageKeys.EXCHANGE_ID: 4,
            MessageKeys.PINNED: False,
        },
        {
            MessageKeys.CONTENT: "m7",
            MessageKeys.EXCHANGE_ID: 5,
            MessageKeys.PINNED: True,
        },
        {
            MessageKeys.CONTENT: "m8",
            MessageKeys.EXCHANGE_ID: 6,
            MessageKeys.PINNED: False,
        },
        # 4 messages pinned => 400 tokens
    ]


def test_empty_messages_list(llm_integration_ut):
    """Test with empty messages list"""
    result = llm_integration_ut._truncate([])
    assert result == []


def test_all_messages_fit_within_limit(llm_integration_ut, sample_messages):
    """Test when all messages fit within token limit"""
    result = llm_integration_ut._truncate(sample_messages)
    assert result == sample_messages


def test_pinned_messages_exceed_limit_raises_error(llm_integration_ut, sample_messages):
    """Test that pinned messages exceeding limit raises ValueError"""
    llm_integration_ut._max_tokens = 100

    with pytest.raises(ValueError) as exc_info:
        llm_integration_ut._truncate(sample_messages)

    assert "Pinned messages exceed the maximum token limit" in str(exc_info.value)


@pytest.mark.parametrize(
    "max_tokens,expected_messages",
    [
        (400, ["m1", "m2", "m5", "m7"]),
        (401, ["m1", "m2", "m5", "m7"]),
        (499, ["m1", "m2", "m5", "m7"]),
        (500, ["m1", "m2", "m5", "m7", "m8"]),
        (600, ["m1", "m2", "m5", "m6", "m7", "m8"]),
        (799, ["m1", "m2", "m5", "m6", "m7", "m8"]),  # "m3" + "m4" do not fit
        (800, ["m1", "m2", "m3", "m4", "m5", "m6", "m7", "m8"]),
    ],
)
def test_correct_truncation(
    llm_integration_ut, sample_messages, max_tokens, expected_messages
):
    """Test that messages are truncated correctly based on max_tokens and exchanges."""
    llm_integration_ut._max_tokens = max_tokens

    result = llm_integration_ut._truncate(sample_messages)

    assert [msg["content"] for msg in result] == expected_messages
