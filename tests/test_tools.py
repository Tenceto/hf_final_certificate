import pytest
import pandas as pd
from unittest.mock import patch, MagicMock, mock_open, ANY
import requests
import speech_recognition as sr
from pydub import AudioSegment
from gaia_agent.tools import (
    web_search_and_scrape,
    string_reverse,
    read_excel_file,
    read_python_script,
    parse_audio_file,
    tool_list,
)
from llama_index.core.tools import FunctionTool


@pytest.fixture
def mock_requests_response_success():
    """Fixture for a successful requests.get response."""
    mock_response = MagicMock(spec=requests.Response)
    mock_response.status_code = 200
    mock_response.text = """
        <html><head><title>Test Page</title><style>body{color:red}</style></head>
        <body><p>This is the main content.</p><script>alert('hi');</script> Extra spaces   here.</body>
        </html>
    """
    mock_response.raise_for_status.return_value = None  # Simulate no HTTP error
    return mock_response


@patch("gaia_agent.tools.requests.get")
@patch("gaia_agent.tools.DDGS")
def test_web_search_scrape_success(
    mock_ddgs_context, mock_requests_get, mock_requests_response_success
):
    """Tests successful web search and scrape."""
    # Mock DDGS search results
    mock_ddgs_instance = mock_ddgs_context.return_value.__enter__.return_value
    mock_ddgs_instance.text.return_value = iter(
        [{"href": "http://example.com/success"}]
    )
    mock_requests_get.return_value = mock_requests_response_success

    result = web_search_and_scrape("test query", num_results=1)

    mock_ddgs_instance.text.assert_called_once_with("test query", max_results=1)
    mock_requests_get.assert_called_once_with("http://example.com/success", timeout=10)
    mock_requests_response_success.raise_for_status.assert_called_once()
    assert result.startswith("Scraped content from http://example.com/success:\n\n")
    assert "Test Page This is the main content. Extra spaces here." in result
    assert "<script>" not in result  # Check script tags are removed
    assert "<style>" not in result  # Check style tags are removed
    assert "  " not in result  # Check extra spaces are collapsed


@patch("gaia_agent.tools.DDGS")
def test_web_search_no_results(mock_ddgs_context):
    """Tests web search returning no results."""
    mock_ddgs_instance = mock_ddgs_context.return_value.__enter__.return_value
    mock_ddgs_instance.text.return_value = iter([])  # Simulate empty results

    result = web_search_and_scrape("test query")

    assert result == "Error: Web search did not return any results."
    mock_ddgs_instance.text.assert_called_once_with("test query", max_results=1)


@patch("gaia_agent.tools.requests.get")
@patch("gaia_agent.tools.DDGS")
def test_web_search_no_url_in_result(mock_ddgs_context, mock_requests_get):
    """Tests web search result missing the 'href' key."""
    mock_ddgs_instance = mock_ddgs_context.return_value.__enter__.return_value
    mock_ddgs_instance.text.return_value = iter(
        [{"title": "No URL here"}]  # Missing 'href'
    )

    result = web_search_and_scrape("test query")

    assert result == "Error: Search result did not contain a valid URL."
    mock_requests_get.assert_not_called()  # requests.get should not be called


@patch("gaia_agent.tools.requests.get")
@patch("gaia_agent.tools.DDGS")
def test_web_search_request_exception(mock_ddgs_context, mock_requests_get):
    """Tests handling of requests.exceptions.RequestException."""
    mock_ddgs_instance = mock_ddgs_context.return_value.__enter__.return_value
    mock_ddgs_instance.text.return_value = iter([{"href": "http://example.com/fail"}])
    mock_requests_get.side_effect = requests.exceptions.RequestException(
        "Connection timed out"
    )

    result = web_search_and_scrape("test query")

    assert result.startswith("Error: Failed to fetch URL http://example.com/fail.")
    assert "Connection timed out" in result


@patch("gaia_agent.tools.requests.get")
@patch("gaia_agent.tools.DDGS")
def test_web_search_unexpected_exception(mock_ddgs_context, mock_requests_get):
    """Tests handling of unexpected exceptions during scraping."""
    mock_ddgs_instance = mock_ddgs_context.return_value.__enter__.return_value
    mock_ddgs_instance.text.return_value = iter([{"href": "http://example.com/weird"}])
    # Simulate an error AFTER getting the response
    mock_requests_get.return_value = MagicMock()
    with patch(
        "gaia_agent.tools.BeautifulSoup", side_effect=Exception("Parsing failed")
    ):
        result = web_search_and_scrape("test query")

    assert "Error: An unexpected error occurred during scraping." in result
    assert "URL: http://example.com/weird" in result  # Check URL is included in error
    assert "Parsing failed" in result


def test_string_reverse_simple():
    assert string_reverse("hello") == "olleh"


def test_string_reverse_empty():
    assert string_reverse("") == ""


@patch("gaia_agent.tools.pd.read_excel")
def test_read_excel_file_success(mock_read_excel):
    """Tests successful reading and conversion of Excel file."""
    # Mock pandas DataFrame and its to_markdown method
    mock_df = MagicMock(spec=pd.DataFrame)
    mock_df.to_markdown.return_value = "| col1 | col2 |\n|---|---|\n| val1 | val2 |"
    mock_read_excel.return_value = mock_df

    # Filename argument is ignored due to hardcoding in the function
    result = read_excel_file("any_filename_here_ignored.xlsx")

    # Verify read_excel was called with the hardcoded path
    mock_read_excel.assert_called_once_with("files/menu_items.xlsx")
    mock_df.to_markdown.assert_called_once_with(index=False)
    assert result == "| col1 | col2 |\n|---|---|\n| val1 | val2 |"


@patch(
    "gaia_agent.tools.pd.read_excel", side_effect=FileNotFoundError("File not found")
)
def test_read_excel_file_not_found(mock_read_excel):
    """Tests handling of FileNotFoundError when reading Excel."""
    result = read_excel_file("ignored.xlsx")

    mock_read_excel.assert_called_once_with("files/menu_items.xlsx")
    assert result.startswith("Error: Failed to read the Excel file.")
    assert "File not found" in result


@patch("builtins.open", new_callable=mock_open, read_data="print('hello world')")
def test_read_python_script_success(mock_file):
    """Tests successful reading of a Python script."""
    result = read_python_script("ignored_filename.py")

    # Assert open was called with the hardcoded path within the function
    mock_file.assert_called_once_with("files/python_code.py", "r")
    assert result == "print('hello world')"


@patch("builtins.open", side_effect=FileNotFoundError("Script not found"))
def test_read_python_script_not_found(mock_file):
    """Tests handling FileNotFoundError when reading Python script."""
    result = read_python_script("ignored.py")

    mock_file.assert_called_once_with("files/python_code.py", "r")
    assert result.startswith("Error: Failed to read the Python script.")
    assert "Script not found" in result


@patch("gaia_agent.tools.os.path.exists")
@patch("gaia_agent.tools.AudioSegment.from_mp3")
@patch("gaia_agent.tools.sr.Recognizer")
@patch("gaia_agent.tools.sr.AudioFile")
@patch("gaia_agent.tools.io.BytesIO", MagicMock())
def test_parse_audio_file_success(
    mock_audio_file_ctx, mock_recognizer_cls, mock_from_mp3, mock_exists
):
    """Tests successful audio parsing."""
    # Arrange Mocks
    mock_exists.return_value = True
    mock_audio_segment = MagicMock(spec=AudioSegment)
    mock_from_mp3.return_value = mock_audio_segment
    mock_recognizer_instance = MagicMock()
    mock_recognizer_cls.return_value = mock_recognizer_instance
    mock_recognizer_instance.recognize_google.return_value = (
        "This is the transcribed text."
    )

    # This represents the 'audio_data' variable in the original code
    mock_recorded_audio_data = MagicMock()
    mock_recognizer_instance.record.return_value = mock_recorded_audio_data

    # This is the object returned when AudioFile(...) is called
    mock_audio_file_call_result = mock_audio_file_ctx.return_value
    # This is the object returned by __enter__(), which is assigned to 'source'
    mock_source_object_from_with = mock_audio_file_call_result.__enter__.return_value

    result = parse_audio_file("test.mp3")

    mock_exists.assert_called_once_with("files/test.mp3")
    mock_from_mp3.assert_called_once_with("files/test.mp3")
    mock_audio_segment.export.assert_called_once_with(
        ANY, format="wav"
    )  # ANY matches the BytesIO obj

    # Check sr.AudioFile was used with the BytesIO object
    mock_audio_file_ctx.assert_called_once_with(ANY)
    # Check recognizer.record was called with the object yielded by the 'with' statement
    mock_recognizer_instance.record.assert_called_once_with(
        mock_source_object_from_with
    )
    # Check recognizer.recognize_google was called with the result of record()
    mock_recognizer_instance.recognize_google.assert_called_once_with(
        mock_recorded_audio_data
    )

    assert result == "This is the transcribed text."


@patch("gaia_agent.tools.os.path.exists")
def test_parse_audio_file_not_found(mock_exists):
    """Tests audio parsing when file does not exist."""
    mock_exists.return_value = False

    result = parse_audio_file("nonexistent.mp3")

    mock_exists.assert_called_once_with("files/nonexistent.mp3")
    assert result == "Error: File not found at files/nonexistent.mp3"


@patch("gaia_agent.tools.os.path.exists", return_value=True)
@patch("gaia_agent.tools.AudioSegment.from_mp3")
@patch("gaia_agent.tools.sr.Recognizer")
@patch("gaia_agent.tools.sr.AudioFile")
def test_parse_audio_google_api_error(
    mock_audio_file_ctx, mock_recognizer_cls, mock_from_mp3, mock_exists
):
    """Tests handling sr.RequestError from Google Speech API."""
    mock_audio_segment = MagicMock(spec=AudioSegment)
    mock_from_mp3.return_value = mock_audio_segment
    mock_recognizer_instance = MagicMock()
    mock_recognizer_cls.return_value = mock_recognizer_instance
    # Simulate Google API failure
    mock_recognizer_instance.recognize_google.side_effect = sr.RequestError(
        "API unavailable"
    )

    mock_audio_source = MagicMock()
    mock_audio_file_instance = mock_audio_file_ctx.return_value.__enter__.return_value
    mock_audio_file_instance = mock_audio_source

    result = parse_audio_file("test.mp3")

    assert result.startswith(
        "Error: Could not request results from Google Web Speech API;"
    )
    assert "API unavailable" in result


def test_tool_list_structure():
    """Tests that the tool_list is created correctly."""
    assert isinstance(tool_list, list)
    assert len(tool_list) == 5
    for tool in tool_list:
        assert isinstance(tool, FunctionTool)

    expected_names = [
        "web_search_and_scrape",
        "string_reverse",
        "read_excel_file",
        "read_python_script",
        "parse_audio_file",
    ]
    actual_names = [tool.metadata.name for tool in tool_list]
    assert sorted(actual_names) == sorted(expected_names)
