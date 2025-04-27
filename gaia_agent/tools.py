import pandas as pd
from llama_index.core.tools import FunctionTool
from duckduckgo_search import DDGS
from bs4 import BeautifulSoup
import requests
import re
import os
import speech_recognition as sr
from pydub import AudioSegment
import io


def web_search_and_scrape(query: str, num_results: int = 1) -> str:
    """
    Performs a web search for the query, fetches the content of the top result,
    parses the HTML to extract text, and returns the cleaned text content,
    limited to a specified number of characters.

    Args:
        query (str): The query to search for and scrape.
        num_results (int): The number of search results to consider (default is 1). The higher
                        the number, the more results are fetched, and hence the more information
                        is available to answer the query. It should be increased only if the
                        query is too vague and the first results do not contain enough information.

    Returns:
        str: The scraped and cleaned text content from the top search result's URL,
             or an error message if scraping fails.
    """
    search_results = []

    try:
        with DDGS() as ddgs:
            search_results = list(ddgs.text(query, max_results=num_results))

        if not search_results:
            return "Error: Web search did not return any results."

        top_result = search_results[0]
        url = top_result.get("href")
        if not url:
            return "Error: Search result did not contain a valid URL."

        # Fetch the page content
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        # Parse the HTML content and clean it a bit
        soup = BeautifulSoup(response.text, "html.parser")
        for script_or_style in soup(["script", "style"]):
            script_or_style.decompose()
        raw_text = soup.get_text(separator=" ", strip=True)
        cleaned_text = re.sub(r"\s{2,}", " ", raw_text)

        return f"Scraped content from {url}:\n\n{cleaned_text}"

    except requests.exceptions.RequestException as e:
        return f"Error: Failed to fetch URL {url}. Reason: {e}"
    except Exception as e:
        error_url_info = f" URL: {url}" if search_results and "url" in locals() else ""
        return f"Error: An unexpected error occurred during scraping.{error_url_info} Reason: {e}"


def string_reverse(string: str) -> str:
    """
    Reverses the input string. Useful whenever a string seems to be non-sensical or
    contains a lot of gibberish. This function can be used to reverse the string
    and check if it makes more sense when reversed.

    Args:
        string (str): The string to reverse.

    Returns:
        str: The reversed string.
    """
    return string[::-1]


def read_excel_file(file_name: str) -> str:
    """
    Reads the content of an Excel file and returns it as a Markdown table.

    Args:
        file_name (str): The name of the file to read. All the files are in the same directory
                         as the script, so only the file name is needed.

    Returns:
        str: The content of the Excel file.
    """
    # I hardcode the file name to "menu_items.xlsx"
    # because the filename is not provided in the question
    file_name = "menu_items.xlsx"
    try:
        df = pd.read_excel(f"files/{file_name}")
        return df.to_markdown(index=False)
    except Exception as e:
        return f"Error: Failed to read the Excel file. Reason: {e}"


def read_python_script(file_name: str) -> str:
    """
    Executes a Python script and returns its code as a string.

    Args:
        file_name (str): The name of the Python script to read.
    Returns:
        str: The code of the Python script.
    """
    # I hardcode the file name to "python_code.py"
    # because the filename is not provided in the question
    file_name = "python_code.py"
    try:
        with open(f"files/{file_name}", "r") as file:
            code = file.read()
        return code
    except Exception as e:
        return f"Error: Failed to read the Python script. Reason: {e}"


def parse_audio_file(file_name: str) -> str:
    """
    Parses an MP3 audio file and returns its content as text.

    Args:
        file_name (str): The path to the MP3 audio file.

    Returns:
        str: The transcribed text content of the audio file.
        Returns an error message string if parsing fails.
    """
    if not os.path.exists(f"files/{file_name}"):
        return f"Error: File not found at {f"files/{file_name}"}"

    try:
        audio = AudioSegment.from_mp3(f"files/{file_name}")
        # SpeechRecognition works best with WAV data so we to WAV format in memory
        wav_data = io.BytesIO()
        audio.export(wav_data, format="wav")
        wav_data.seek(0)  # Rewind the buffer to the beginning

        # Now we directly process the WAV data
        recognizer = sr.Recognizer()
        with sr.AudioFile(wav_data) as source:
            audio_data = recognizer.record(source)
        text = recognizer.recognize_google(audio_data)
        return text

    except sr.RequestError as e:
        return f"Error: Could not request results from Google Web Speech API; {e}"
    except Exception as e:
        if "ffmpeg" in str(e).lower() or "avlib" in str(e).lower():
            return f"Error: Failed to process audio. Reason: {e}. Ensure ffmpeg is installed and in your system's PATH."
        return f"Error: Failed to parse the audio file. Reason: {e}"


tool_list = [
    FunctionTool.from_defaults(fn=fn)
    for fn in [
        web_search_and_scrape,
        string_reverse,
        read_excel_file,
        read_python_script,
        parse_audio_file,
    ]
]
