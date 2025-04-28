# Hugging Face Final Project: GAIA Agent :robot:

Welcome to my resolution of the final project for the [Hugging Face Agents Course](https://huggingface.co/learn/agents-course/unit4/hands-on)!

There is much room for improvement, since this is just the first version. I am currently working on adding additional **tools for image and video processing**.

However, the current code already helped me pass the final exam, so feel free to try it out or use it as a baseline for your implementation.

## Getting started :chart_with_upwards_trend:

First of all, you must set the necessary environment variables.
Create a file `.env` inside the root directory following `.env.example`.
Namely, the content must be:

```
GEMINI_API_KEY="your-gemini-api-key"
OPENAI_API_KEY="your-openai-api-key"
```

You only have to set one of the two if you want to use Gemini or OpenAI.
For now, these are the only two providers supported by my implementation.

To install all Python dependencies, run:
`uv sync`

One last thing: since one of the tools for the LLM transcribes a voice note into text, you need FFMPEG on your computer. In macOS, you can install it by running:
`brew install ffmpeg`

## Running the script :runner:

The main script automatically fetches all questions, downloads all required files, and submits the agent's answers.
Before running it, remember to change the `config.yaml` accordingly.
In particular, set `username` to your Hugging Face username, and `agent_code` to your repository URL.

To execute the main script, just run the command:
`uv run main.py`
