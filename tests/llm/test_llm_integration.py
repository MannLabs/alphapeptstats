from unittest import skip
from unittest.mock import Mock, patch

import pandas as pd
import pytest
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessage,
    ChatCompletionMessageToolCall,
)

from alphastats.llm.llm_integration import LLMIntegration, Models


@pytest.fixture
def mock_openai_client():
    with patch("alphastats.llm.llm_integration.OpenAI") as mock_client:
        yield mock_client


@pytest.fixture
def llm_integration(mock_openai_client):
    """Fixture providing a basic LLM instance with test configuration"""
    return LLMIntegration(
        api_type=Models.GPT4O,
        api_key="test-key",  # pragma: allowlist secret
        system_message="Test system message",
    )


def test_initialization_gpt4(mock_openai_client):
    """Test initialization with GPT-4 configuration"""
    LLMIntegration(
        api_type=Models.GPT4O,
        api_key="test-key",  # pragma: allowlist secret
    )

    mock_openai_client.assert_called_once_with(
        api_key="test-key"  # pragma: allowlist secret
    )


def test_initialization_ollama(mock_openai_client):
    """Test initialization with Ollama configuration"""
    LLMIntegration(
        api_type=Models.OLLAMA_31_8B,
        base_url="http://localhost:11434",
    )

    mock_openai_client.assert_called_once_with(
        base_url="http://localhost:11434/v1",
        api_key="ollama",  # pragma: allowlist secret
    )


def test_initialization_invalid_model():
    """Test initialization with invalid model type"""
    with pytest.raises(ValueError, match="Invalid API type"):
        LLMIntegration(api_type="invalid-model")


def test_append_message(llm_integration):
    """Test message appending functionality"""
    llm_integration._append_message("user", "Test message")

    assert len(llm_integration._messages) == 2  # Including system message
    assert len(llm_integration._all_messages) == 2
    assert llm_integration._messages[-1] == {"role": "user", "content": "Test message"}


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
    for _ in range(num_messages):
        llm_integration._append_message("user", message_content)

    llm_integration._truncate_conversation_history(max_tokens=max_tokens)

    # Adding 1 to account for the initial system message
    assert len(llm_integration._messages) <= expected_messages + 1


@pytest.fixture
def mock_successful_completion():
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


def test_chat_completion_success(llm_integration, mock_successful_completion):
    """Test successful chat completion"""
    llm_integration._client.chat.completions.create.return_value = (
        mock_successful_completion
    )

    llm_integration.chat_completion("Test prompt")

    assert llm_integration._messages == [
        {
            "content": "Test system message",
            "role": "system",
        },
        {
            "content": "Test prompt",
            "role": "user",
        },
        {
            "content": "Test response",
            "role": "assistant",
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


# Mock functions for testing
def mock_general_function(param1, param2):
    return f"General function called with {param1} and {param2}"


@pytest.fixture
def mock_dataset():
    dataset = Mock()
    dataset.plot_intensity = Mock(return_value="Plot created")
    dataset.custom_function = Mock(return_value="Dataset function called")
    dataset.metadata = pd.DataFrame({"group1": ["A", "B"], "group2": ["C", "D"]})
    return dataset


@pytest.fixture
def mock_gene_map():
    return {"GENE1": "PROT1", "GENE2": "PROT2"}


@pytest.fixture
def llm_with_dataset(mock_openai_client, mock_dataset, mock_gene_map):
    return LLMIntegration(
        api_type=Models.GPT4O,
        api_key="test-key",  # pragma: allowlist secret
        dataset=mock_dataset,
        gene_to_prot_id_map=mock_gene_map,
    )


@pytest.fixture
def mock_general_function_mapping():
    return {"test_general_function": mock_general_function}


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
    llm_with_dataset,
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
        result = llm_with_dataset._execute_function(function_name, function_args)
        assert result == expected_result


@pytest.mark.parametrize(
    "gene_name,plot_args,expected_protein_id",
    [
        ("GENE1", {"param1": "value1"}, "PROT1"),
        ("GENE2", {"param2": "value2"}, "PROT2"),
    ],
)
def test_execute_plot_intensity(
    llm_with_dataset, gene_name, plot_args, expected_protein_id
):
    """Test execution of plot_intensity with gene name translation"""
    function_args = {"protein_id": gene_name, **plot_args}

    result = llm_with_dataset._execute_function("plot_intensity", function_args)

    # Verify the dataset's plot_intensity was called with correct protein ID
    llm_with_dataset._dataset.plot_intensity.assert_called_once()
    call_args = llm_with_dataset._dataset.plot_intensity.call_args[1]
    assert call_args["protein_id"] == expected_protein_id
    assert result == "Plot created"


def test_execute_dataset_function(llm_with_dataset):
    """Test execution of a function from the dataset"""
    result = llm_with_dataset._execute_function("custom_function", {"param1": "value1"})

    assert result == "Dataset function called"
    llm_with_dataset._dataset.custom_function.assert_called_once_with(param1="value1")


def test_execute_dataset_function_with_dots(llm_with_dataset):
    """Test execution of a dataset function when name contains dots"""
    result = llm_with_dataset._execute_function(
        "dataset.custom_function", {"param1": "value1"}
    )

    assert result == "Dataset function called"
    llm_with_dataset._dataset.custom_function.assert_called_once_with(param1="value1")


@skip  # TODO fix this test
def test_execute_nonexistent_function(llm_with_dataset):
    """Test execution of a non-existent function"""

    result = llm_with_dataset._execute_function(
        "nonexistent_function", {"param1": "value1"}
    )

    assert "Error executing nonexistent_function" in result
    assert "not implemented or dataset not available" in result


def test_execute_function_with_error(llm_with_dataset, mock_general_function_mapping):
    """Test handling of function execution errors"""

    def failing_function(**kwargs):
        raise ValueError("Test error")

    with patch(
        "alphastats.llm.llm_integration.GENERAL_FUNCTION_MAPPING",
        {"failing_function": failing_function},
    ), pytest.raises(ValueError, match="Test error"):
        llm_with_dataset._execute_function("failing_function", {"param1": "value1"})


def test_execute_function_without_dataset(mock_openai_client):
    """Test function execution when dataset is not available"""
    llm = LLMIntegration(api_type=Models.GPT4O, api_key="test-key")

    with pytest.raises(
        ValueError,
        match="Function dataset_function not implemented or dataset not available",
    ):
        llm._execute_function("dataset_function", {"param1": "value1"})
