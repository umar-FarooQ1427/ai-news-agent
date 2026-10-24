
import json
import requests
from pathlib import Path
from datetime import datetime

# ==========================================
# 1. CONFIGURATION
# ==========================================

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2:3b"

PROJECT_FOLDER = Path(__file__).parent
INPUT_FILE = PROJECT_FOLDER / "news.json"
OUTPUT_FILE = PROJECT_FOLDER / "ai_briefing.md"

# Analyze a limited number of articles to start with.
MAX_ARTICLES = 10


# ==========================================
# 2. ASK THE AI TO ANALYZE AN ARTICLE
# ==========================================

def analyze_article(article):
    """Ask the local AI model to analyze a news article."""

    title = article.get("title", "Untitled")
    category = article.get("category", "Technology")
    source = article.get("source", "Unknown source")
    link = article.get("link", "")

    # Use a feed description if your collector provides one.
    description = article.get("description", "")

    if description:
        article_text = description[:2500]
        evidence_note = (
            "Use the supplied article description as context."
        )
    else:
        article_text = title
        evidence_note = (
            "Only the title and metadata are available. "
            "Do not invent facts or claim to know the full article."
        )

    prompt = f"""
You are a careful technology news analyst.

Analyze this news item for a daily briefing.

Category: {category}
Source: {source}
Title: {title}
Available article text: {article_text}

Instructions:
- Write a concise summary in simple English.
- Explain why the news may matter.
- Assign an importance score from 1 to 5.
- Score 1 means low importance; 5 means very high importance.
- Base your score only on the available information.
- Do not invent facts, dates, incidents, or technical details.
- If information is insufficient, say so clearly.
- {evidence_note}

Return your answer using these exact headings:

Summary:
Why it matters:
Importance: [number]/5
Confidence: [Low, Medium, or High]
"""

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.2
        }
    }

    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=240
    )

    response.raise_for_status()

    result = response.json()

    return result.get(
        "response",
        "The model returned no analysis."
    ).strip()


# ==========================================
# 3. LOAD NEWS FROM JSON
# ==========================================

def load_news():
    """Load the articles collected by news_collector.py."""

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"{INPUT_FILE} not found. "
            "Run news_collector.py first."
        )

    with INPUT_FILE.open("r", encoding="utf-8") as file:
        articles = json.load(file)

    if not isinstance(articles, list):
        raise ValueError(
            "news.json must contain a list of articles."
        )

    return articles


# ==========================================
# 4. GENERATE THE AI BRIEFING
# ==========================================

def generate_briefing(articles):
    """Analyze articles and save the results as Markdown."""

    selected_articles = articles[:MAX_ARTICLES]

    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        today = datetime.now().strftime("%Y-%m-%d")

        file.write(f"# AI Technology News Briefing — {today}\n\n")
        file.write(
            f"AI analyzed {len(selected_articles)} "
            f"of {len(articles)} collected articles.\n\n"
        )

        for index, article in enumerate(
            selected_articles, start=1
        ):
            title = article.get("title", "Untitled")
            link = article.get("link", "")
            category = article.get("category", "Technology")
            source = article.get("source", "Unknown source")

            print(
                f"\nAnalyzing article {index}/"
                f"{len(selected_articles)}: {title}"
            )

            file.write(f"## {index}. {title}\n\n")
            file.write(f"- **Category:** {category}\n")
            file.write(f"- **Source:** {source}\n")

            try:
                analysis = analyze_article(article)
                file.write(f"\n{analysis}\n\n")

            except requests.exceptions.ConnectionError:
                message = (
                    "Could not connect to Ollama. "
                    "Check that Ollama is running."
                )
                print(message)
                file.write(f"\n**Analysis error:** {message}\n\n")

            except requests.exceptions.RequestException as error:
                message = f"AI request failed: {error}"
                print(message)
                file.write(f"\n**Analysis error:** {message}\n\n")

            if link:
                file.write(f"[Read original article]({link})\n\n")

            file.write("---\n\n")

    print(f"\nAI briefing saved to: {OUTPUT_FILE}")


# ==========================================
# 5. MAIN PROGRAM
# ==========================================

def main():
    print("=" * 55)
    print("       AI NEWS AGENT - ARTICLE ANALYZER")
    print("=" * 55)

    try:
        articles = load_news()

        if not articles:
            print("No articles found in news.json.")
            print("Run news_collector.py to collect news first.")
            return

        print(f"\nLoaded {len(articles)} articles.")
        print(f"AI will analyze up to {MAX_ARTICLES} articles.")

        generate_briefing(articles)

        print("\nFinished! Open ai_briefing.md to read your report.")

    except FileNotFoundError as error:
        print(f"\nFile error: {error}")

    except json.JSONDecodeError:
        print("\nnews.json is not valid JSON.")
        print("Run news_collector.py again.")

    except requests.exceptions.ConnectionError:
        print("\nCannot connect to Ollama.")
        print("Make sure Ollama is installed and running.")

    except Exception as error:
        print(f"\nUnexpected error: {error}")


if __name__ == "__main__":
    main()
