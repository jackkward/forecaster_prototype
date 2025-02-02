# forecaster_prototype
 Global events forecaster that uses OSINT/news + commercially available LLMs to predict the future - geopolitically speaking :)


# Forecast.py Usage Guide

`forecast.py` is a Python script that leverages Anthropic and OpenAI’s language models to answer scenario-based questions using curated OSINT content files. It performs the following tasks:

- **Loads a system prompt** to instruct the model.
- **Scans a content directory** for articles with tagged summaries and full text.
- **Filters relevant articles** using a human-like evaluation based on a provided question.
- **Builds a prompt** that combines context and your question.
- **Queries the language model** for answers.
- **Saves the results** to a timestamped file.

---

## Prerequisites

- **Python Version:** Python 3.8 or later is recommended.
- **Required API Keys:**
  - `ANTHROPIC_API_KEY`: Your API key for Anthropic.
  - `OPENAI_API_KEY`: Your API key for OpenAI.
- **Dependencies:**  
  Install the required packages via pip:
  ```bash
  pip install anthropic openai pydantic

---
```
forecaster_prototype/
├── forecast.py
├── system_prompts/
│   └── forecast_sysprompt.txt         # System prompt file for model instructions.
├── content/                           # Directory containing text files with articles.
│   ├── article1.txt                   # Each file should include <SUMMARY> and <CONTENT> tags.
│   ├── article2.txt
│   └── ...
└── scenario_eval/
    └── syria/                         # Example scenario directory.
        ├── questions.txt              # File with one question per line.
        └── results/                   # Directory where answer files are saved.
    └── ...
```
---

## Content File Format

Each text file in the content/ directory should include:

A summary enclosed in <SUMMARY> and </SUMMARY> tags.
The full content enclosed in <CONTENT> and </CONTENT> tags.

Example file (`article1.txt`):
```
<SUMMARY>
This article overviews the recent policy changes and their economic impact.
</SUMMARY>

<CONTENT>
Here is a detailed analysis including statistical data, expert opinions, and potential future implications...
</CONTENT>
```

---

## Usage

Prepare Your Content:

- Populate the content/ directory with your article files (formatted as shown above).

Prepare Your Questions:

- Create a file at `scenario_eval/syria/questions.txt` (or any other scenario you'd like to evaluate – you might need to add more content) containing one question per line.
  
Run the Script:
- From the project root, execute: `python forecast.py`
  
The script will:
- Load the system prompt.
- Scan and parse content files.
- Optionally filter articles for relevance (comparing summary relevance to qeustion).
- Query the Anthropic model for each question.
- Save the results to a timestamped file inside scenario_eval/syria/results/
        
