import os
import re
from typing import List, Dict, Tuple
import anthropic
from pypdf import PdfReader
import json
from time import sleep

class PDFChapterSummarizer:
    def __init__(self, api_key: str):
        """
        Initialize the summarizer with your Anthropic API key.
        
        Args:
            api_key (str): Your Anthropic API key
        """
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = "claude-3-opus-20240229"  # Using Opus for highest quality summaries
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text content from a PDF file.
        
        Args:
            pdf_path (str): Path to the PDF file
            
        Returns:
            str: Extracted text content
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found at: {pdf_path}")
        
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text

    def detect_chapters(self, text: str) -> List[Tuple[str, str]]:
        """
        Detect chapters in the text and split content accordingly.
        
        Args:
            text (str): Full text content from PDF
            
        Returns:
            List[Tuple[str, str]]: List of (chapter_title, chapter_content) tuples
        """
        # Common chapter patterns
        chapter_patterns = [
            r'Chapter\s+\d+[\s:]+([^\n]+)',
            r'CHAPTER\s+\d+[\s:]+([^\n]+)',
        ]
        
        chapters = []
        lines = text.split('\n')
        current_chapter = "Introduction"  # Default title for content before first chapter
        current_content = []
        chapter_starts = []  # Store indices where chapters start
        
        # First pass: identify chapter starting points
        for i, line in enumerate(lines):
            for pattern in chapter_patterns:
                if re.match(pattern, line.strip()):
                    if current_content and current_chapter:
                        # Store the current chapter before starting a new one
                        chapters.append((current_chapter, '\n'.join(current_content)))
                    current_chapter = line.strip()
                    current_content = []
                    chapter_starts.append(i)
                    break
            if i not in chapter_starts:
                current_content.append(line)
        
        # Add the final chapter if there's content
        if current_content:
            chapters.append((current_chapter, '\n'.join(current_content)))
        
        # If no chapters were detected, return all content as a single chapter
        if not chapters:
            return [("Main Content", text)]
        
        return chapters

    def generate_summary(self, chapter_title: str, chapter_content: str) -> str:
        """
        Generate a summary for a single chapter using the Anthropic API.
        
        Args:
            chapter_title (str): Title of the chapter
            chapter_content (str): Content of the chapter
            
        Returns:
            str: Generated summary
        """
        prompt = f"""Please provide a detailed, informative summary of the following chapter from a book as it relates to performing rigorous analysis.
        Focus on key concepts, main arguments, and important details. In particular, focus on compiling the insights as they relate to performing good analysis.
        
        Chapter Title: {chapter_title}
        
        Chapter Content:
        {chapter_content}
        
        Focus on compiling the content's insights as they relate to performing good analysis.

        Do not answer with any preamble - do not say something like "Here's a summary of the chapter", just output the summary as described above.
        """
        
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                temperature=0,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return response.content[0].text
        except Exception as e:
            print(f"Error generating summary for chapter '{chapter_title}': {str(e)}")
            return f"Error generating summary: {str(e)}"

    def summarize_pdf(self, pdf_path: str, output_path: str = None) -> Dict[str, str]:
        """
        Process a PDF file and generate summaries for all chapters.
        
        Args:
            pdf_path (str): Path to the PDF file
            output_path (str, optional): Path to save the JSON output
            
        Returns:
            Dict[str, str]: Dictionary mapping chapter titles to their summaries
        """
        print(f"Processing PDF: {pdf_path}")
        
        # Extract text from PDF
        text = self.extract_text_from_pdf(pdf_path)
        
        # Detect chapters
        chapters = self.detect_chapters(text)
        print(f"Detected {len(chapters)} chapters")
        
        # Generate summaries
        summaries = {}
        for i, (title, content) in enumerate(chapters, 1):
            print(f"Generating summary for chapter {i}/{len(chapters)}: {title}")
            summary = self.generate_summary(title, content)
            summaries[title] = summary
            sleep(100)  # Rate limiting
        
        # Save to file if output path is provided
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(summaries, f, indent=2, ensure_ascii=False)
            print(f"Summaries saved to: {output_path}")
        
        return summaries

def main():
    # Get API key from environment variable
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("Please set the ANTHROPIC_API_KEY environment variable")
    
    # Initialize summarizer
    summarizer = PDFChapterSummarizer(api_key)
    
    # Example usage
    pdf_path = "Pyschology-of-Intelligence-Analysis.pdf"
    output_path = "chapter_summaries.json"
    
    try:
        summaries = summarizer.summarize_pdf(pdf_path, output_path)
        
        # Print summaries to console
        for chapter, summary in summaries.items():
            print(f"\n{'='*80}\n")
            print(f"Summary for {chapter}")
            print(f"\n{summary}")
            print(f"\n{'='*80}\n")
            
    except Exception as e:
        print(f"Error processing PDF: {str(e)}")

if __name__ == "__main__":
    main()