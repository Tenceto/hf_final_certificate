import requests
from typing import Any

API_URL = "https://agents-course-unit4-scoring.hf.space"
QUESTIONS_URL = f"{API_URL}/questions"
SUBMIT_URL = f"{API_URL}/submit"
FILES_URL = f"{API_URL}/files"


def fetch_questions(timeout=15):
    """Fetches questions from the API."""
    print("Fetching questions...")

    # Make the GET request to fetch questions
    try:
        response = requests.get(QUESTIONS_URL, timeout=timeout)
        response.raise_for_status()
        questions_data = response.json()
        if not questions_data:
            print("Fetched questions list is empty.")
            return None
        print(f"Fetched {len(questions_data)} questions.")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching questions: {e}")
        return f"Error fetching questions: {e}", None
    except requests.exceptions.JSONDecodeError as e:
        print(f"Error decoding JSON response from questions endpoint: {e}")
        print(f"Response text: {response.text[:500]}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred fetching questions: {e}")
        return None

    return questions_data


def submit_answers(
    username: str, agent_code: str, answers: list[dict[str, Any]]
) -> dict[str, Any]:
    """Submit agent answers and get score."""
    data = {"username": username, "agent_code": agent_code, "answers": answers}
    try:
        response = requests.post(SUBMIT_URL, json=data)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error submitting answers: {e}")
        return None
    except requests.exceptions.JSONDecodeError as e:
        print(f"Error decoding JSON response from submit endpoint: {e}")
        print(f"Response text: {response.text[:500]}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred submitting answers: {e}")
        return None


def get_file(task_id: str) -> bytes:
    """Download a file for a specific task"""
    try:
        response = requests.get(f"{FILES_URL}/{task_id}")
        response.raise_for_status()
        return response.content
    except requests.RequestException as e:
        # No file for this task
        return None
