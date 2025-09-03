import base64
import requests
from fake_useragent import UserAgent
import random
import re
from io import BytesIO
from typing import List, Optional
from together import Together
import PyPDF2
from requests_html import HTMLSession

# Together.ai configuration
TOGETHER_API_KEY = "2d5341da842ba4f133068a76fba1bc069199737b6178293b22765df2df33465f"
MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
MAX_CHUNK_SIZE = 5000  # Characters per chunk

ua = UserAgent()
user_agents = [
    # Chrome variants
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
    # Firefox variants
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:122.0) Gecko/20100101 Firefox/122.0',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:122.0) Gecko/20100101 Firefox/122.0',
    'Mozilla/5.0 (X11; Linux x86_64; rv:122.0) Gecko/20100101 Firefox/122.0',
    # Safari variants
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2.1 Safari/605.1.15',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15',
    # Edge variants
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36 Edg/122.0.0.0',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36 Edg/121.0.0.0',
    # Mobile variants
    'Mozilla/5.0 (iPhone; CPU iPhone OS 17_2_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Mobile/15E148 Safari/604.1',
    'Mozilla/5.0 (Linux; Android 14; SM-G998B) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Mobile Safari/537.36',
    # Additional random agents
    ua.chrome,
    ua.firefox,
    ua.safari,
    ua.edge,
]

def download_pdf(url: str) -> Optional[bytes]:
    session = HTMLSession()
    
    try:
        # Set browser-like headers
        headers = {
            'User-Agent': random.choice(user_agents),
            'Accept': 'application/pdf,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Referer': 'https://www.esmoopen.com/',
            'Connection': 'keep-alive',
            'DNT': '1',
            'Upgrade-Insecure-Requests': '1'
        }
        response = session.get(url, headers=headers, timeout=20)
        response.raise_for_status()

        # Content-type must be PDF
        if "application/pdf" in response.headers.get("Content-Type", ""):
            return response.content
        else:
            print("Server responded, but not with a PDF.")
            return None

    except Exception as e:
        print(f"Error downloading PDF: {e}")
        return None

def sentence_safe_chunks(text: str, chunk_size: int = MAX_CHUNK_SIZE, overlap: int = 500) -> List[str]:
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= chunk_size:
            current_chunk += sentence + " "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
    if current_chunk:
        chunks.append(current_chunk.strip())

    # Add overlap to maintain context
    overlapped_chunks = []
    for i in range(len(chunks)):
        prev = chunks[i-1][-overlap:] if i > 0 else ""
        current = chunks[i]
        overlapped_chunks.append(f"[CONTEXT]\n{prev}\n\n[CURRENT]\n{current}")
    return overlapped_chunks

def process_pdf_chunk_with_llm(chunk: str, previous_summary: str = "") -> str:
    client = Together(api_key=TOGETHER_API_KEY)

    prompt = f"""You are analyzing a section of a medical research paper. Summarize the content using **only well-written paragraphs**. Avoid bullet points and repetition.

Sections to cover if present: Introduction, Methods, Results, Discussion, Conclusion.

Here's the current section:
{chunk}

Previously summarized sections (avoid repeating them):
{previous_summary if previous_summary else "None yet."}

Write in formal academic tone using paragraph formatting only.
"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are building a comprehensive medical document analysis."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=2000,
            temperature=0.1
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error processing chunk: {e}")
        return ""

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    try:
        pdf_reader = PyPDF2.PdfReader(BytesIO(pdf_bytes))
        return "\n".join(page.extract_text() or "" for page in pdf_reader.pages)
    except Exception as e:
        print(f"Error extracting text: {e}")
        return ""

def process_complete_pdf(pdf_url: str) -> str:
    print(f"Processing PDF from: {pdf_url}")
    pdf_bytes = download_pdf(pdf_url)
    if not pdf_bytes:
        return "Failed to download PDF"
    
    full_text = extract_text_from_pdf(pdf_bytes)
    chunks = sentence_safe_chunks(full_text)
    if not chunks:
        return "Failed to extract text from PDF"

    print(f"Processing {len(chunks)} chunks...")
    chunk_summaries = []
    accumulated_summary = ""

    for i, chunk in enumerate(chunks, 1):
        print(f"Processing chunk {i}/{len(chunks)}...")
        summary = process_pdf_chunk_with_llm(chunk, accumulated_summary)
        chunk_summaries.append(summary)
        accumulated_summary += " " + summary  # Build-up context without repeating whole text

    print("Consolidating final results...")
    client = Together(api_key=TOGETHER_API_KEY)
    try:
        combined_prompt = """You are finalizing a comprehensive medical research summary. The following are summaries of different sections from a research paper. Merge them into a single coherent, non-repetitive, paragraph-formatted document. Maintain logical flow and avoid repeating information.

Summaries:
""" + "\n\n".join(chunk_summaries)

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are finalizing a scientific paper summary."},
                {"role": "user", "content": combined_prompt}
            ],
            max_tokens=2500,
            temperature=0.1
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error consolidating results: {e}")
        return "\n\n".join(chunk_summaries)

if __name__ == "__main__":
    pdf_url = "https://www.esmoopen.com/article/S2059-7029(24)01764-2/pdf"
    result = process_complete_pdf(pdf_url)
    print("\n=== COMPREHENSIVE ANALYSIS ===\n")
    print(result)
