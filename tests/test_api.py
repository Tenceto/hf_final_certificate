from unittest.mock import patch, MagicMock
import requests
from gaia_agent.api import (
    fetch_questions,
    submit_answers,
    get_file,
    QUESTIONS_URL,
    SUBMIT_URL,
    FILES_URL,
)


@patch("gaia_agent.api.requests.get")
def test_fetch_questions_success(mock_get):
    """Test successful fetching of questions."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = [
        {"task_id": "1", "question": "Q1"},
        {"task_id": "2", "question": "Q2"},
    ]
    mock_response.raise_for_status.return_value = None
    mock_get.return_value = mock_response

    questions = fetch_questions()

    assert questions == [
        {"task_id": "1", "question": "Q1"},
        {"task_id": "2", "question": "Q2"},
    ]
    mock_get.assert_called_once_with(QUESTIONS_URL, timeout=15)


@patch("gaia_agent.api.requests.get")
def test_fetch_questions_empty(mock_get):
    """Test fetching an empty list of questions."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = []
    mock_response.raise_for_status.return_value = None
    mock_get.return_value = mock_response

    questions = fetch_questions()

    assert questions is None
    mock_get.assert_called_once_with(QUESTIONS_URL, timeout=15)


@patch("gaia_agent.api.requests.get")
def test_fetch_questions_request_error(mock_get):
    """Test handling requests.exceptions.RequestException."""
    mock_get.side_effect = requests.exceptions.RequestException("Network error")

    questions = fetch_questions()

    assert questions is None
    mock_get.assert_called_once_with(QUESTIONS_URL, timeout=15)


@patch("gaia_agent.api.requests.get")
def test_fetch_questions_json_decode_error(mock_get):
    """Test handling JSONDecodeError."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.raise_for_status.return_value = None
    mock_response.json.side_effect = requests.exceptions.JSONDecodeError(
        "Invalid JSON", "", 0
    )
    mock_response.text = "Invalid JSON response"
    mock_get.return_value = mock_response

    questions = fetch_questions()

    assert questions is None
    mock_get.assert_called_once_with(QUESTIONS_URL, timeout=15)


@patch("gaia_agent.api.requests.post")
def test_submit_answers_success(mock_post):
    """Test successful submission of answers."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"score": 10.0, "message": "Success"}
    mock_response.raise_for_status.return_value = None
    mock_post.return_value = mock_response

    answers_data = [{"task_id": "1", "submitted_answer": "A1"}]
    result = submit_answers("user", "http://code.url", answers_data)

    assert result == {"score": 10.0, "message": "Success"}
    expected_payload = {
        "username": "user",
        "agent_code": "http://code.url",
        "answers": answers_data,
    }
    mock_post.assert_called_once_with(SUBMIT_URL, json=expected_payload)


@patch("gaia_agent.api.requests.post")
def test_submit_answers_request_error(mock_post):
    """Test handling requests.exceptions.RequestException during submission."""
    mock_post.side_effect = requests.exceptions.RequestException("Submit error")

    answers_data = [{"task_id": "1", "submitted_answer": "A1"}]
    result = submit_answers("user", "http://code.url", answers_data)

    assert result is None
    expected_payload = {
        "username": "user",
        "agent_code": "http://code.url",
        "answers": answers_data,
    }
    mock_post.assert_called_once_with(SUBMIT_URL, json=expected_payload)


@patch("gaia_agent.api.requests.post")
def test_submit_answers_json_decode_error(mock_post):
    """Test handling JSONDecodeError during submission."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.raise_for_status.return_value = None
    mock_response.json.side_effect = requests.exceptions.JSONDecodeError(
        "Bad JSON", "", 0
    )
    mock_response.text = "Bad JSON response"
    mock_post.return_value = mock_response

    answers_data = [{"task_id": "1", "submitted_answer": "A1"}]
    result = submit_answers("user", "http://code.url", answers_data)

    assert result is None
    expected_payload = {
        "username": "user",
        "agent_code": "http://code.url",
        "answers": answers_data,
    }
    mock_post.assert_called_once_with(SUBMIT_URL, json=expected_payload)


@patch("gaia_agent.api.requests.get")
def test_get_file_success(mock_get):
    """Test successful file download."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.content = b"file content"
    mock_response.raise_for_status.return_value = None
    mock_get.return_value = mock_response

    task_id = "task123"
    file_content = get_file(task_id)

    assert file_content == b"file content"
    expected_url = f"{FILES_URL}/{task_id}"
    mock_get.assert_called_once_with(expected_url)


@patch("gaia_agent.api.requests.get")
def test_get_file_request_error(mock_get):
    """Test handling requests.exceptions.RequestException during file download."""
    mock_get.side_effect = requests.exceptions.RequestException("File download error")

    task_id = "task123"
    file_content = get_file(task_id)

    assert file_content is None
    expected_url = f"{FILES_URL}/{task_id}"
    mock_get.assert_called_once_with(expected_url)


@patch("gaia_agent.api.requests.get")
def test_get_file_http_error(mock_get):
    """Test handling HTTP errors (like 404) during file download."""
    mock_response = MagicMock()
    mock_response.status_code = 404
    mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
        "Not Found"
    )
    mock_get.return_value = mock_response

    task_id = "task_not_found"
    file_content = get_file(task_id)

    assert file_content is None
    expected_url = f"{FILES_URL}/{task_id}"
    mock_get.assert_called_once_with(expected_url)
