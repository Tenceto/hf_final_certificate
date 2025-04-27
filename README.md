# Hugging Face Final Project: GAIA Agent :robot:

Welcome to my resolution of the final project for the Hugging Face Agents Course!

There is several room for improvement, since this is just a first version. I am currently working on:

- Adding additional tools for image and video processing
- Adding unit tests

However, the current code already helped me pass the final exam, so feel free to try it out or use it as a baseline for your own implementation.

## Getting Started

First of all, the necessary environment variables must be set.
Create a file `.env` inside the root directory following `.env.example`.
Namely, the content must be:

```
GEMINI_API_KEY="your-gemini-api-key"
OPENAI_API_KEY="your-openai-api-key"
```

You only have to set one of the two if you want to just use Gemini or just use OpenAI.
For now, these are the only two providers that are supported by my implementation.

To install all Python dependencies, simply run:
`uv sync`

One last thing: since one of the tools for the LLM transcripts a voicenote into text, you need FFMPEG in your computer. In MacOS, you can install it by running:
`brew install ffmpeg`

### Running the script :books:

The main script automatically fetches all questions, downloads all required files, and submits the agent's answers.
Before running it, remember to change the `config.yaml` accordingly.
In particular, set `username` to your Hugging Face username, and `agent_code` to the URL to your own repository.

To execute the main script just run the command:
`uv run main.py`
