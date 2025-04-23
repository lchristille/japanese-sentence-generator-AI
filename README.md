# Japanese Phrases Generator Project

A multi-agent system for generating Japanese phrases across different verb tenses.

## System Overview
This project implements an intelligent system that leverages multiple AI agents to generate grammatically correct Japanese sentences. The system focuses on verb conjugation and tense variations.

### Key Components

#### Agents
- **Chatbot Agent**: Manages user interactions and maintains system state.
- **Sentence Expert Agent**: Specialized in Japanese sentence construction and grammar rules.

#### Features
- Dynamic phrase generation based on user input.
- Support for multiple Japanese verb tenses.
- State management to track selected verbs and tenses.
- Modular and extensible architecture.

## Setup

### 1. Install Dependencies

Create a venv/conda/uv for your project before installing and running.
Linux and MacOS will refuse to install packages with pip without an python environment.

Install the required Python packages:
```bash
pip install -r requirements.txt
```

### 2. Configure Environment
Duplicate the .env.template file, rename it .env and put your Google API Key.

If you don't already have an API key, you can grab one from [AI Studio](https://aistudio.google.com/app/apikey). You can find [detailed instructions in the docs](https://ai.google.dev/gemini-api/docs/api-key).

This project uses `gemini-2.0-flash` model, that has currently a free-tier. You can find more information [here](https://ai.google.dev/gemini-api/docs/rate-limits).
You will be the sole responsible for economic charges on your Google account.

### 3. Run
```shell
python run.py
```

Example conversation (only human parts)
- What tenses do you support?
- Can you generate a sentence with the verb 食べる at the imperative tense?
- Can you add verbs 食べる、見る and 写す?
- Can you add verbs 走る and 今日?
- What there is my notebook?
- add Request, volitional, past negative polite and conditional
- add non past tense
- generate sentences
