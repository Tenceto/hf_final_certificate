import pytest
import io
import sys
from unittest.mock import patch, MagicMock, AsyncMock, ANY
from gaia_agent.agent import GAIAAgent
from gaia_agent import tools
from gaia_agent import prompts

# Mark all tests in this file as async
pytestmark = pytest.mark.asyncio


@pytest.fixture
def mock_env_gemini(monkeypatch):
    """Sets a dummy GEMINI_API_KEY in environment variables."""
    monkeypatch.setenv("GEMINI_API_KEY", "dummy_gemini_key")


@pytest.fixture
def mock_env_openai(monkeypatch):
    """Sets a dummy OPENAI_API_KEY in environment variables."""
    monkeypatch.setenv("OPENAI_API_KEY", "dummy_openai_key")


@pytest.fixture
def gemini_config():
    """Provides a sample config for Gemini."""
    return {
        "provider": "gemini",
        "model_name": "gemini-pro",
        "temperature": 0.5,
    }


@pytest.fixture
def openai_config():
    """Provides a sample config for OpenAI."""
    return {
        "provider": "openai",
        "model_name": "gpt-4",
        "temperature": 0.7,
    }


@pytest.fixture
def invalid_config():
    """Provides an invalid config."""
    return {
        "provider": "unknown_provider",
        "model_name": "some-model",
        "temperature": 0.5,
    }


@patch("gaia_agent.agent.ReActAgent", autospec=True)
@patch("gaia_agent.agent.GoogleGenAI", autospec=True)
async def test_init_gemini_provider(
    mock_google_genai, mock_react_agent, gemini_config, mock_env_gemini
):
    """Tests successful initialization with the Gemini provider."""
    mock_llm_instance = MagicMock()
    mock_google_genai.return_value = mock_llm_instance

    agent = GAIAAgent(config=gemini_config)

    # Assert GoogleGenAI was called correctly
    mock_google_genai.assert_called_once_with(
        model=gemini_config["model_name"],
        api_key="dummy_gemini_key",
        temperature=gemini_config["temperature"],
    )

    # Assert ReActAgent was called correctly
    mock_react_agent.assert_called_once_with(
        tools=tools.tool_list,  # Use the actual tool_list object
        llm=mock_llm_instance,
        verbose=True,
        memory=None,
    )
    assert agent.agent == mock_react_agent.return_value


@patch("gaia_agent.agent.ReActAgent", autospec=True)
@patch("gaia_agent.agent.OpenAI", autospec=True)
async def test_init_openai_provider(
    mock_openai, mock_react_agent, openai_config, mock_env_openai
):
    """Tests successful initialization with the OpenAI provider."""
    mock_llm_instance = MagicMock()
    mock_openai.return_value = mock_llm_instance

    agent = GAIAAgent(config=openai_config)

    # Assert OpenAI was called correctly
    mock_openai.assert_called_once_with(
        model=openai_config["model_name"],
        api_key="dummy_openai_key",
        temperature=openai_config["temperature"],
    )

    # Assert ReActAgent was called correctly
    mock_react_agent.assert_called_once_with(
        tools=tools.tool_list, llm=mock_llm_instance, verbose=True, memory=None
    )
    assert agent.agent == mock_react_agent.return_value


async def test_init_invalid_provider(invalid_config):
    """Tests initialization failure with an unsupported provider."""
    with pytest.raises(ValueError, match="Unsupported provider: unknown_provider"):
        GAIAAgent(config=invalid_config)


@patch("gaia_agent.agent.ReActAgent", autospec=True)
@patch(
    "gaia_agent.agent.GoogleGenAI", autospec=True
)  # Patch one LLM type is enough if we mock ReActAgent properly
async def test_run_without_reasoning(
    mock_google_genai, mock_react_agent_class, gemini_config, mock_env_gemini
):
    """Tests the run method without capturing reasoning."""
    mock_llm_instance = MagicMock()
    mock_google_genai.return_value = mock_llm_instance

    # Mock the ReActAgent instance and its async 'achat' method
    mock_react_agent_instance = MagicMock()
    mock_response = MagicMock()
    mock_response.response = "Paris."
    mock_react_agent_instance.achat = AsyncMock(return_value=mock_response)
    mock_react_agent_class.return_value = mock_react_agent_instance

    agent = GAIAAgent(config=gemini_config)
    question = "What is the capital of France?"
    expected_prompt = prompts.general_instructions.format(question=question)
    response, reasoning = await agent.run(question, return_reasoning_process=False)

    # Check if ReActAgent's achat was called correctly
    mock_react_agent_instance.achat.assert_awaited_once_with(expected_prompt)
    # Check the returned values
    assert response == "Paris."
    assert reasoning == ""


@patch("gaia_agent.agent.ReActAgent", autospec=True)
@patch("gaia_agent.agent.GoogleGenAI", autospec=True)
@patch("sys.stdout", new_callable=io.StringIO)  # Mock stdout
async def test_run_with_reasoning(
    mock_stdout,
    mock_google_genai,
    mock_react_agent_class,
    gemini_config,
    mock_env_gemini,
):
    """Tests the run method WITH capturing reasoning."""
    mock_llm_instance = MagicMock()
    mock_google_genai.return_value = mock_llm_instance

    mock_react_agent_instance = MagicMock()
    mock_response = MagicMock()
    mock_response.response = "Paris"

    # Simulate verbose output: make the mocked achat write to the *actual* (but captured) stdout
    async def mock_achat_with_output(*args, **kwargs):
        print("Thought: I need to answer the question.", file=sys.stdout)
        print("Action: Search", file=sys.stdout)
        print("Observation: Found Paris.", file=sys.stdout)
        sys.stdout.flush()  # Ensure it's written before returning
        return mock_response

    mock_react_agent_instance.achat = AsyncMock(side_effect=mock_achat_with_output)
    mock_react_agent_class.return_value = mock_react_agent_instance

    agent = GAIAAgent(config=gemini_config)
    question = "Capital of France?"
    expected_prompt = prompts.general_instructions.format(question=question)
    response, reasoning = await agent.run(question, return_reasoning_process=True)

    mock_react_agent_instance.achat.assert_awaited_once_with(expected_prompt)
    # Check the returned values
    assert response == "Paris"
    # Check the captured stdout content
    expected_reasoning = (
        "Thought: I need to answer the question.\n"
        "Action: Search\n"
        "Observation: Found Paris.\n"
    )
    assert reasoning == expected_reasoning
