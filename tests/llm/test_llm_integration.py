from unittest import skip
from unittest.mock import Mock, patch

import pandas as pd
import pytest
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessage,
    ChatCompletionMessageToolCall,
)
from openai.types.chat.chat_completion_message_tool_call import Function

from alphastats.llm.llm_integration import LLMIntegration, Models


@pytest.fixture
def mock_openai_client():
    with patch("alphastats.llm.llm_integration.OpenAI") as mock_client:
        yield mock_client


@pytest.fixture
def llm_integration(mock_openai_client):
    """Fixture providing a basic LLM instance with test configuration"""
    dataset = Mock()
    dataset.plot_intensity = Mock(return_value="Plot created")
    dataset.custom_function = Mock(return_value="Dataset function called")
    dataset.metadata = pd.DataFrame({"group1": ["A", "B"], "group2": ["C", "D"]})
    return LLMIntegration(
        model_name=Models.GPT4O,
        api_key="test-key",  # pragma: allowlist secret
        system_message="Test system message",
        dataset=dataset,
        genes_of_interest={"GENE1": "PROT1", "GENE2": "PROT2"},
    )


@pytest.fixture
def llm_with_conversation(llm_integration):
    """Setup LLM with a sample conversation history"""
    # Add various message types to conversation history
    llm_integration._all_messages = [
        {"role": "system", "content": "System message", "pinned": True},
        {"role": "user", "content": "User message 1", "pinned": False},
        {"role": "assistant", "content": "Assistant message 1", "pinned": False},
        {
            "role": "assistant",
            "content": "Assistant with tool calls",
            "tool_calls": [
                {"id": "123", "type": "function", "function": {"name": "test"}}
            ],
            "pinned": False,
        },
        {"role": "tool", "content": "Tool response", "pinned": False},
        {"role": "user", "content": "User message 2", "pinned": False},
        {"role": "assistant", "content": "Assistant message 2", "pinned": False},
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
    LLMIntegration(
        model_name=Models.GPT4O,
        api_key="test-key",  # pragma: allowlist secret
    )

    mock_openai_client.assert_called_once_with(
        api_key="test-key"  # pragma: allowlist secret
    )


def test_initialization_ollama(mock_openai_client):
    """Test initialization with Ollama configuration"""
    LLMIntegration(
        model_name=Models.OLLAMA_31_8B,
        base_url="http://localhost:11434",
    )

    mock_openai_client.assert_called_once_with(
        base_url="http://localhost:11434/v1",
        api_key="ollama",  # pragma: allowlist secret
    )


def test_initialization_invalid_model():
    """Test initialization with invalid model type"""
    with pytest.raises(ValueError, match="Invalid model name"):
        LLMIntegration(model_name="invalid-model")


def test_append_message(llm_integration):
    """Test message appending functionality"""
    llm_integration._append_message("user", "Test message")

    assert len(llm_integration._messages) == 2  # Including system message
    assert len(llm_integration._all_messages) == 2
    assert llm_integration._messages[-1] == {
        "role": "user",
        "content": "Test message",
        "pinned": False,
    }


def test_append_message_with_tool_calls(llm_integration):
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
def test_truncate_conversation_history(
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


def test_estimate_tokens_gpt(llm_integration):
    """Test token estimation for a given message"""
    message_content = "Test message"
    tokens = llm_integration.estimate_tokens([{"content": message_content}])

    assert tokens == 2


def test_estimate_tokens_ollama(llm_integration):
    """Test token estimation for a given message with Ollama model, falls back on average chars per token"""
    llm_integration._model = Models.OLLAMA_31_8B
    message_content = "Test message"
    tokens = llm_integration.estimate_tokens(
        [{"content": message_content}], average_chars_per_token=3.6
    )

    assert tokens == 12 / 3.6


def test_chat_completion_success(llm_integration, mock_chat_completion):
    """Test successful chat completion"""
    llm_integration._client.chat.completions.create.return_value = mock_chat_completion

    llm_integration.chat_completion("Test prompt")

    assert llm_integration._messages == [
        {
            "content": "Test system message",
            "role": "system",
            "pinned": True,
        },
        {
            "content": "Test prompt",
            "role": "user",
            "pinned": False,
        },
        {
            "content": "Test response",
            "role": "assistant",
            "pinned": False,
        },
    ]


def test_chat_completion_with_error(llm_integration):
    """Test chat completion with error handling"""
    llm_integration._client.chat.completions.create.side_effect = ArithmeticError(
        "Test error"
    )

    llm_integration.chat_completion("Test prompt")

    assert (
        "Error in chat completion: Test error"
        in llm_integration._messages[-1]["content"]
    )


def test_parse_model_response(llm_integration, mock_tool_call_completion):
    """Test parsing model response with tool calls"""
    content, tool_calls = llm_integration._parse_model_response(
        mock_tool_call_completion
    )

    assert content == ""
    assert len(tool_calls) == 1
    assert tool_calls[0].id == "test-id"
    assert tool_calls[0].type == "function"


def test_chat_completion_with_content_and_tool_calls(llm_integration):
    """Test that chat completion raises error when receiving both content and tool calls"""
    mock_response = Mock(spec=ChatCompletion)
    mock_response.choices = [
        Mock(
            message=ChatCompletionMessage(
                role="assistant",
                content="Some content",
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
    llm_integration._client.chat.completions.create.return_value = mock_response

    with pytest.raises(ValueError, match="Unexpected content.*with tool calls"):
        llm_integration.chat_completion("Test prompt")


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
    llm_integration,
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


def test_execute_dataset_function(llm_integration):
    """Test execution of a function from the dataset"""
    result = llm_integration._execute_function("custom_function", {"param1": "value1"})

    assert result == "Dataset function called"
    llm_integration._dataset.custom_function.assert_called_once_with(param1="value1")


def test_execute_dataset_function_with_dots(llm_integration):
    """Test execution of a dataset function when name contains dots"""
    result = llm_integration._execute_function(
        "dataset.custom_function", {"param1": "value1"}
    )

    assert result == "Dataset function called"
    llm_integration._dataset.custom_function.assert_called_once_with(param1="value1")


@skip  # TODO fix this test
def test_execute_nonexistent_function(llm_integration):
    """Test execution of a non-existent function"""

    result = llm_integration._execute_function(
        "nonexistent_function", {"param1": "value1"}
    )

    assert "Error executing nonexistent_function" in result
    assert "not implemented or dataset not available" in result


def test_execute_function_with_error(llm_integration, mock_general_function_mapping):
    """Test handling of function execution errors"""

    def failing_function(**kwargs):
        raise ValueError("Test error")

    with patch(
        "alphastats.llm.llm_integration.GENERAL_FUNCTION_MAPPING",
        {"failing_function": failing_function},
    ), pytest.raises(ValueError, match="Test error"):
        llm_integration._execute_function("failing_function", {"param1": "value1"})


def test_execute_function_without_dataset(mock_openai_client):
    """Test function execution when dataset is not available"""
    llm = LLMIntegration(model_name=Models.GPT4O, api_key="test-key")

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
        model_name=Models.GPT4O,
        api_key="test-key",  # pragma: allowlist secret
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
    result = llm_integration._handle_function_calls(tool_calls)

    assert result == ("Test response", None)

    mock_execute_function.assert_called_once_with("test_function", {"arg1": "value1"})

    expected_messages = [
        {"role": "system", "content": "Test system message", "pinned": True},
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
        },
        {
            "role": "tool",
            "content": '{"result": "some_function_result", "artifact_id": "test_function_test-id"}',
            "tool_call_id": "test-id",
            "pinned": False,
        },
    ]
    mock_openai_client.return_value.chat.completions.create.assert_called_once_with(
        model="gpt-4o", messages=expected_messages, tools=llm_integration._tools
    )

    assert list(llm_integration._artifacts[3]) == ["some_function_result"]

    assert llm_integration._messages == expected_messages


def test_get_print_view_default(llm_with_conversation):
    """Test get_print_view with default settings (show_all=False)"""
    print_view = llm_with_conversation.get_print_view()

    # Should only include user and assistant messages without tool_calls
    assert print_view == [
        {
            "artifacts": [],
            "content": "User message 1",
            "role": "user",
            "in_context": True,
            "pinned": False,
        },
        {
            "artifacts": ["Artifact for message 2"],
            "content": "Assistant message 1",
            "role": "assistant",
            "in_context": True,
            "pinned": False,
        },
        {
            "artifacts": [],
            "content": "User message 2",
            "role": "user",
            "in_context": False,
            "pinned": False,
        },
        {
            "artifacts": ["Artifact for message 6"],
            "content": "Assistant message 2",
            "role": "assistant",
            "in_context": False,
            "pinned": False,
        },
    ]


def test_get_print_view_show_all(llm_with_conversation):
    """Test get_print_view with default settings (show_all=True)"""
    print_view = llm_with_conversation.get_print_view(show_all=True)

    # Should only include user and assistant messages without tool_calls
    assert print_view == [
        {
            "artifacts": [],
            "content": "System message",
            "role": "system",
            "in_context": True,
            "pinned": True,
        },
        {
            "artifacts": [],
            "content": "User message 1",
            "role": "user",
            "in_context": True,
            "pinned": False,
        },
        {
            "artifacts": ["Artifact for message 2"],
            "content": "Assistant message 1",
            "role": "assistant",
            "in_context": True,
            "pinned": False,
        },
        {
            "artifacts": [],
            "content": "Assistant with tool calls",
            "role": "assistant",
            "in_context": False,
            "pinned": False,
        },
        {
            "artifacts": ["Tool artifact 1", "Tool artifact 2"],
            "content": "Tool response",
            "role": "tool",
            "in_context": False,
            "pinned": False,
        },
        {
            "artifacts": [],
            "content": "User message 2",
            "role": "user",
            "in_context": False,
            "pinned": False,
        },
        {
            "artifacts": ["Artifact for message 6"],
            "content": "Assistant message 2",
            "role": "assistant",
            "in_context": False,
            "pinned": False,
        },
    ]
