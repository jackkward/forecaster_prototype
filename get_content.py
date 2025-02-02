import requests
from datetime import datetime
import os
import re

API_KEY = os.getenv("EXA_API_KEY")

results = []

# for domain in domains:
payload = {
    "query": """
    News coverage of geopolitical events, conflicts, elections, political affiliations, military innovations, and trade in the Middle East, Iran, Taiwan, China, Japan, Ukraine, Russia, Pakistan, India, and/or the United States from highly respected news outlets.
    """,
    "useAutoprompt": True,
    "type": "auto",
    "category": "research paper",
    "numResults": 100,
    "startPublishedDate": "2024-01-01T05:00:00.000Z",
    "endPublishedDate": "2025-01-27T04:59:59.999Z",
    "startCrawlDate": "2025-01-01T05:00:01.000Z",
    "contents": {
        "text": {
            "maxCharacters": 20000
        },
        "highlights": True,
        "summary": {
            "query": "Generate a 3-10 sentence summary of the article contents, calling out the key facts from the article."
        }
    }
}

headers = {
    "x-api-key": API_KEY,
    "Content-Type": "application/json"
}

response = requests.post("https://api.exa.ai/search", json=payload, headers=headers)
results.append(response.json())

def sanitize_filename(filename):
    # Remove invalid filename characters and limit length
    return re.sub(r'[<>:"/\\|?*]', '', filename)[:150]

def get_domain_folder(url):
    # Extract domain from URL and clean it
    domain = url.split('/')[2]  # Get domain from URL
    # Remove common prefixes and .com/.org etc
    domain = domain.replace('www.', '')
    domain = domain.split('.')[0]  # Get first part of domain (ft, bloomberg, etc)
    return domain

# Create base articles directory if it doesn't exist
articles_dir = "content"
os.makedirs(articles_dir, exist_ok=True)

for batch in results:
    for article in batch["results"]:
        # Format date and create filename
        formatted_date = datetime.fromisoformat(article["publishedDate"].replace('Z', '+00:00'))
        date_str = formatted_date.strftime('%Y-%m-%d')
        
        # Get domain folder and create if it doesn't exist
        domain_folder = get_domain_folder(article["url"])
        domain_path = os.path.join(articles_dir, domain_folder)
        os.makedirs(domain_path, exist_ok=True)
        
        # Create sanitized filename
        safe_title = sanitize_filename(article["title"])
        filename = f"{date_str}_{safe_title}.txt"
        filepath = os.path.join(domain_path, filename)
        
        # Write content to file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"<TITLE>{article['title']}</TITLE>\n")
            f.write(f"<PUBLISHED_DATE>{date_str}</PUBLISHED_DATE>\n")
            f.write(f"<URL>{article['url']}</URL>\n")
            f.write(f"<AUTHOR>{article['author']}</AUTHOR>\n\n")
            
            f.write("<SUMMARY>\n")
            # Format summary with newlines and dashes after periods
            summary = article['summary'].replace(". ", ".\n- ")
            if summary.startswith("- "):
                summary = summary[2:]  # Remove leading dash if present
            f.write(summary)
            f.write("\n\n</SUMMARY>\n")
            
            f.write("<HIGHLIGHTS>\n")
            # Format highlights as bullet points with newlines
            highlights = article['highlights']
            if isinstance(highlights, list):
                for highlight in highlights:
                    # Format each highlight with newlines and dashes after periods
                    formatted_highlight = highlight.replace(". ", ".\n- ")
                    if formatted_highlight.startswith("- "):
                        formatted_highlight = formatted_highlight[2:]  # Remove leading dash if present
                    f.write(f"- {formatted_highlight}\n\n")
            else:
                formatted_highlights = str(highlights).replace(". ", ".\n- ")
                if formatted_highlights.startswith("- "):
                    formatted_highlights = formatted_highlights[2:]
                f.write(formatted_highlights)
            f.write("</HIGHLIGHTS>\n")
            
            f.write("<CONTENT>\n")
            # Add newlines after periods for better paragraph separation
            text = article['text']
            text = text.replace(". ", ".\n\n")
            f.write(text)
            f.write("\n\n</CONTENT>\n")
        
        # Optional: Print confirmation of file creation
        print(f"Saved article to: {domain_folder}/{filename}")
