# forecaster_prototype
 Global events forecaster that uses OSINT/news + commercially available LLMs to predict the future - geopolitically speaking :)

# Example Output:

Q1: What year will China invade Taiwan? What is making them more or less likely to do so?

A1:
Let me analyze this question carefully based on the available intelligence:

<thinking>
1. First, I must note that predicting the exact timing of a potential invasion is highly speculative and would involve significant uncertainty. Instead, I should focus on analyzing risk factors and trends.

2. Key factors increasing risk of invasion:
- Growing military coordination between China and other adversaries (Russia, North Korea, Iran) suggests increased confidence in managing multiple conflicts
- China's economic challenges under Xi may create incentives for external distraction
- Xi's prioritization of political control over economic prosperity indicates higher risk tolerance
- US military assessments suggest America is stretched thin and unprepared for multiple conflicts
- Xi's personal ideology emphasizes reunification and nationalism

3. Key factors decreasing risk:
- China's economic difficulties may make them more cautious about destabilizing actions
- US and allies are actively preparing for this scenario and strengthening deterrence
- The economic costs of invasion would be severe for China's already troubled economy
- Military challenges remain significant given Taiwan's defenses and likely US intervention
- Failure would be catastrophic for CCP legitimacy

4. Timeline considerations:
- Multiple sources indicate heightened concern about the 2025-2027 timeframe
- This aligns with PLA modernization goals and US military assessments
- However, China's domestic economic challenges may force delay
- US election cycles and political transitions create windows of opportunity/vulnerability
</thinking>

Based on the available intelligence, I cannot make a definitive prediction about the exact year China will invade Taiwan, as such precision would be misleading. However, the intelligence suggests several important trends:

1. The period between 2025-2027 appears to be a timeframe of elevated risk, based on multiple indicators including PLA modernization goals and US military assessments.

2. The likelihood of invasion appears to be INCREASING due to:
- Growing military coordination between China and other US adversaries that could complicate US response
- Xi Jinping's emphasis on political control over economic prosperity
- US military assessments indicating America is stretched thin for multiple conflicts
- China's domestic challenges potentially increasing incentives for external distraction

3. However, significant DETERRING factors remain:
- China's serious economic difficulties and property sector crisis
- Improved US-allied preparation and coordination
- The enormous economic costs China would face
- The catastrophic consequences of failure for CCP legitimacy

The most prudent assessment is that while risk is elevated in the 2025-2027 timeframe, the timing would likely depend on a complex mix of factors including: US political transitions, Taiwan's elections, China's economic conditions, and the international security environment. Rather than predicting a specific year, focus should be on monitoring key indicators and strengthening deterrence.

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

A summary enclosed in `<SUMMARY>` and `</SUMMARY>` tags.
The full content enclosed in `<CONTENT>` and `</CONTENT>` tags.

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
        
