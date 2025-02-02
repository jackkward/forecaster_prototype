import os
import json
import re
import logging
from pathlib import Path
from datetime import datetime
from time import sleep
from typing import List

import anthropic
import openai
from pydantic import BaseModel

# Configure logging for traceability.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# Suppress HTTP request logs from API clients
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("anthropic").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)  # Suppress underlying HTTP client logs

class BatchDecisions(BaseModel):
    decisions: List[str]

class ClaudeQuestioner:
    def __init__(self, api_key: str, model: str = "claude-3-sonnet-20240229"):
        """
        Initialize the questioner with an Anthropic API key.
        """
        self.anthropic_client = anthropic.Anthropic(api_key=api_key)
        self.openai_client = openai.OpenAI(api_key=api_key)
        self.model = model
        self.system_prompt = ""
        self.context = ""
        self.conversation_history = []

    def load_system_prompt(self, file_path: str) -> None:
        """
        Load and store the system prompt from the given file.
        """
        logging.info("Loading system prompt from %s", file_path)
        prompt_file = Path(file_path)
        if not prompt_file.exists():
            raise FileNotFoundError(f"System prompt file not found: {file_path}")
        self.system_prompt = prompt_file.read_text(encoding="utf-8").strip()
        logging.info("Loaded system prompt (%d characters)", len(self.system_prompt))

    def _load_content_cache(self, content_dir: Path) -> List[dict]:
        """
        Scan the content directory for text files, extract summaries from <SUMMARY> tags,
        and return a list of dictionaries with file paths and summaries.
        """
        logging.info("Scanning content directory: %s", content_dir)
        if not content_dir.exists():
            raise FileNotFoundError(f"Content directory not found: {content_dir}")
        
        cache = []
        for file_path in content_dir.rglob("*.txt"):
            try:
                text = file_path.read_text(encoding="utf-8")
                summary_match = re.search(r"<SUMMARY>(.*?)</SUMMARY>", text, re.DOTALL)
                if summary_match:
                    summary = summary_match.group(1).strip()
                    cache.append({"file_path": str(file_path), "summary": summary})
                else:
                    logging.warning("No <SUMMARY> tags found in %s", file_path.name)
            except Exception as e:
                logging.error("Error reading %s: %s", file_path, e)
        return cache

    def _load_full_content(self, file_path: str) -> str:
        """
        Load and return full content enclosed in <CONTENT> tags from the specified file.
        """
        try:
            text = Path(file_path).read_text(encoding="utf-8")
            content_match = re.search(r"<CONTENT>(.*?)</CONTENT>", text, re.DOTALL)
            if content_match:
                return content_match.group(1).strip()
            else:
                logging.warning("No <CONTENT> tags found in %s", file_path)
        except Exception as e:
            logging.error("Error reading full content from %s: %s", file_path, e)
        return ""

    def _filter_relevant_articles_human_like(self, question: str, cache: List[dict], batch_size: int = 5) -> List[str]:
        """
        Process the cache of article summaries in batches.
        For each batch, ask the LLM (via OpenAI's structured outputs API) to decide whether each summary is relevant.
        The expected output is a JSON object with a single key 'decisions', whose value is a list of "yes"/"no" strings.
        """
        relevant_files = []
        for i in range(0, len(cache), batch_size):
            batch = cache[i : i + batch_size]
            logging.info(f"Processing batch {(i//batch_size)+1} of {(len(cache)-1)//batch_size + 1}")
            
            # Build the prompt with instructions for a structured JSON response.
            prompt_lines = [
                f"You are reviewing article summaries to decide if the full article might help answer the question:",
                f"\"{question}\"",
                "",
                "For each summary below, respond with only 'yes' or 'no' in a JSON object.",
                "Your output must be valid JSON, following this schema:",
                '{"decisions": [ "yes", "no", ... ]}',
                "",
                "Summaries:",
            ]
            for idx, item in enumerate(batch, start=1):
                prompt_lines.append(f"{idx}. {item['summary']}")
            prompt = "\n".join(prompt_lines)
            
            try:
                # Use structured outputs API to get consistent JSON output.
                response = openai.beta.chat.completions.parse(
                    model="gpt-4o-2024-08-06",
                    messages=[{"role": "user", "content": prompt}],
                    response_format=BatchDecisions,
                    max_tokens=50,
                    temperature=0.0
                )
                parsed_response = response.choices[0].message.parsed
                decisions = [decision.strip().lower() for decision in parsed_response.decisions]
                
                if len(decisions) != len(batch):
                    logging.warning("Mismatch in decision count and batch size; skipping this batch.")
                    continue
                
                for decision, item in zip(decisions, batch):
                    logging.debug(f"Summary: {item['summary'][:100]}... -> Decision: {decision}")
                    if decision == "yes":
                        logging.info(f"Selected article: {Path(item['file_path']).name} (Decision: {decision})")
                        relevant_files.append(item["file_path"])
            except Exception as e:
                logging.error("Error during batch relevance evaluation: %s", e)
        logging.info("Filtered down to %d relevant articles using human-like evaluation.", len(relevant_files))
        return relevant_files

    def load_context(self, scenario_name: str, question: str = "") -> None:
        """
        Load context from a directory of articles. Optionally filter for relevant articles using a question.
        """
        logging.info("Loading context for scenario: %s", scenario_name)
        content_dir = Path("content")
        cache = self._load_content_cache(content_dir)
        if not cache:
            raise ValueError("No content files found in the content directory.")
        
        logging.info(f"Loaded {len(cache)} articles from {content_dir}")
        
        # If a question is provided, filter the articles using the human-like binary decision process.
        if question:
            logging.info("Filtering articles based on the question.")
            relevant_files = self._filter_relevant_articles_human_like(question, cache)
        else:
            logging.info("No question provided; using all available articles.")
            relevant_files = [item["file_path"] for item in cache]

        if not relevant_files:
            raise ValueError("No relevant articles found for the given question.")

        context_parts = []
        for file in relevant_files:
            content = self._load_full_content(file)
            if content:
                file_stem = Path(file).stem
                context_parts.append(f"=== {file_stem} ===\n{content}")
                logging.info("Loaded content from %s (%d characters)", file, len(content))
            else:
                logging.warning("Skipping %s due to empty content.", file)

        self.context = "\n\n".join(context_parts)
        logging.info("Final context loaded from %d files (total length %d characters)",
                     len(context_parts), len(self.context))

    def get_token_count(self, messages: list) -> int:
        """
        Use Anthropic's token counter to count tokens for the current system prompt and conversation.
        """
        try:
            token_response = self.anthropic_client.messages.count_tokens(
                model=self.model,
                system=self.system_prompt,
                messages=messages
            )
            token_count = token_response.input_tokens
            logging.info("Current token count: %d", token_count)
            return token_count
        except Exception as e:
            logging.warning("Could not count tokens: %s", e)
            return 0

    def ask_question(self, question: str) -> str:
        """
        Build a full prompt with context and question, ensure the token count is within limits,
        then query the model and return its response.
        """
        full_prompt = (
            f"Here is the relevant context:\n\n{self.context}\n\n"
            f"Based on this context, please answer the following question:\n\n{question}"
        )
        self.conversation_history.append({"role": "user", "content": full_prompt})
        token_count = self.get_token_count(self.conversation_history)

        # Truncate context if token count is too high.
        while token_count > 190000 and self.context:
            logging.warning("Token count (%d) exceeds limit. Truncating context.", token_count)
            self.context = self.context[: len(self.context) // 2]
            full_prompt = (
                f"Here is the relevant raw intelligence: <RAW_INTELLIGENCE>\n\n{self.context}\n\n"
                f"</RAW_INTELLIGENCE>\n\nPlease answer the following question:\n\n{question}"
            )
            self.conversation_history[-1]["content"] = full_prompt
            token_count = self.get_token_count(self.conversation_history)

        if token_count > 190000:
            error_msg = f"Prompt too long ({token_count} tokens). Please reduce context or question size."
            logging.error(error_msg)
            return f"Error: {error_msg}"

        try:
            response = self.anthropic_client.messages.create(
                model=self.model,
                max_tokens=4096,
                temperature=0,
                system=self.system_prompt,
                messages=self.conversation_history
            )
            answer = response.content[0].text
            logging.info("Received answer (first 100 chars): %s", answer[:100])
            return answer
        except Exception as e:
            logging.error("Error asking question: %s", e)
            return f"Error: {e}"

    def ask_multiple_questions(self, questions: List[str]) -> List[tuple]:
        """
        Ask multiple questions sequentially. Resets the conversation history for each question.
        Returns a list of (question, answer) tuples.
        """
        qa_pairs = []
        for question in questions:
            logging.info("Processing question: %s", question)
            self.conversation_history = []  # Reset history for each question.
            answer = self.ask_question(question)
            qa_pairs.append((question, answer))
            # Uncomment the following sleep if needed to avoid rate limiting.
            # sleep(120)
        return qa_pairs

def save_results(scenario_name: str, results: str) -> None:
    """
    Save the results string to a timestamped file within the scenario's results directory.
    """
    results_dir = Path(f"scenario_eval/{scenario_name}/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = results_dir / f"{scenario_name}_{timestamp}.txt"
    try:
        filename.write_text(results, encoding="utf-8")
        logging.info("Results saved to: %s", filename)
    except Exception as e:
        logging.error("Error saving results: %s", e)

def main():
    # Retrieve API key.
    api_key = os.getenv("ANTHROPIC_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Please set the ANTHROPIC_API_KEY environment variable.")
    if not openai_api_key:
        raise ValueError("Please set the OPENAI_API_KEY environment variable.")

    # Define scenario and questions file.
    scenario_name = "syria_general"
    questions_file = Path(f"scenario_eval/{scenario_name}/questions.txt")
    if not questions_file.exists():
        raise FileNotFoundError(f"Questions file not found: {questions_file}")

    questions = [
        line.strip() for line in questions_file.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if not questions:
        raise ValueError("No questions found in the questions file.")

    # Initialize the questioner.
    questioner = ClaudeQuestioner(api_key)
    system_prompt_path = "system_prompts/forecast_sysprompt.txt"
    questioner.load_system_prompt(system_prompt_path)

    # Load context (using the first question as a hint for relevance if desired).
    questioner.load_context(scenario_name, question=questions[0])

    # Ask the questions.
    qa_pairs = questioner.ask_multiple_questions(questions)

    # Format and save the results.
    results_text = "\n".join(
        f"Q{i+1}: {q}\n\nA{i+1}:\n{a}\n{'='*80}"
        for i, (q, a) in enumerate(qa_pairs)
    )
    save_results(scenario_name, results_text)

if __name__ == "__main__":
    main()
