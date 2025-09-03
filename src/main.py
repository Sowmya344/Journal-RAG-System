import os
import re
import json
import psycopg2
import logging
import time
import hashlib
import random
from typing import Optional, Dict, Any, List, Iterator
from datetime import datetime

#---- Langfuse imports -----
from langfuse import Langfuse
from langfuse.decorators import observe, langfuse_context
import asyncio
from functools import wraps
import traceback
# --- Core FastAPI and Pydantic Imports ---
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# --- ML and Utility Imports ---
from sentence_transformers import SentenceTransformer
from supabase import create_client, Client
# --- CHANGED: Switched from OpenAI to Google Gemini ---
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai # Added for Gemini specific configuration and token counting
from psycopg2.extras import RealDictCursor
from langchain.text_splitter import RecursiveCharacterTextSplitter


# --- Set up logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Enhanced Medical Research RAG API",
    version="3.0.0",
    description="An API combining a direct ReACT agent for common queries and an advanced multi-step agent for complex analysis. Includes database browsing and filtering capabilities."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_api_cache = {}
_cache_ttl = 300  # Cache LLM responses for 5 minutes

# --- Manually Inserted Configuration ---
# Database and Supabase configuration
SUPABASE_URL = "https://nobnxgtmpspfvhhxtlvw.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im5vYm54Z3RtcHNwZnZoaHh0bHZ3Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDk0NDIxMjgsImV4cCI6MjA2NTAxODEyOH0.BqdlpaONkK0UkF1BY-i0dqu1W9czgj3LCasEttCneDo" # Corrected an invalid character in key if any

DB_URL = "postgresql://postgres.nobnxgtmpspfvhhxtlvw:IgqCrDmMvSEorSkO@aws-0-ap-south-1.pooler.supabase.com:6543/postgres"

# LLM Configuration (Modified for Google Gemini 2.0 Flash)
LLM_MODEL_NAME = "gemini-1.5-flash-latest" # Use Gemini 2.0 Flash
GEMINI_API_KEY = "" 
LLM_API_KEY = GEMINI_API_KEY 
#Langfuse
LANGFUSE_SECRET_KEY = ""
LANGFUSE_PUBLIC_KEY = ""
LANGFUSE_HOST = ""
LANGFUSE_ENABLED = "true"

LANGFUSE_EVAL_SCOPE = os.getenv("LANGFUSE_EVAL_SCOPE", "analyze_research_endpoint,advanced_agent_search").split(',')
LANGFUSE_EVAL_SCOPE = [s.strip() for s in LANGFUSE_EVAL_SCOPE if s.strip()]
# Initialize Langfuse client
if LANGFUSE_ENABLED:
    try:
        langfuse = Langfuse(
            secret_key=LANGFUSE_SECRET_KEY,
            public_key=LANGFUSE_PUBLIC_KEY,
            host=LANGFUSE_HOST
        )
        logger.info("Langfuse initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Langfuse: {e}")
        langfuse = None
        LANGFUSE_ENABLED = False
else:
    langfuse = None

# Initialize clients
model: SentenceTransformer = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def safe_langfuse_trace(func):
    """Decorator that safely wraps functions with Langfuse tracing"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not LANGFUSE_ENABLED or not langfuse:
            return func(*args, **kwargs)
        
        try:
            # --- MODIFIED: Removed langfuse=langfuse from observe() call ---
            return observe()(func)(*args, **kwargs) 
        except Exception as e:
            logger.warning(f"Langfuse tracing error in {func.__name__}: {e}")
            return func(*args, **kwargs)
    
    return wrapper

# Global variable for the tokenizer (now the Gemini model for counting)
gemini_model_for_counting: Any = None 

def initialize_llm_and_agent():
    """Initialize the LLM and agent instances."""
    global llm, agent, gemini_model_for_counting # Modified to use new tokenizer variable
    
    try:
        # --- NEW: Configure Google Generative AI with API key ---
        genai.configure(api_key=LLM_API_KEY)
        
        # Initialize model for token counting
        gemini_model_for_counting = get_tokenizer()
        
        # Initialize LLM
        llm = LiteLLM(
            model_name=LLM_MODEL_NAME,
            api_key=LLM_API_KEY,
            # max_tokens will be passed dynamically to generate
        )
        
        # Initialize ReACT Agent
        agent = ReACTAgent(llm)
        
        logger.info("LLM and ReACT Agent initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize LLM and Agent: {e}")
        raise e

# --- Schema, Prompts, and Pydantic Models ---
SCHEMA_DESCRIPTION = """
-- Main table for articles
articles (id BIGINT, title TEXT, article_url TEXT, pdf_url TEXT, journal_id INTEGER, article_type_id INTEGER, published_date DATE)
-- Table for authors
authors (id SERIAL, name TEXT)
-- Junction table for articles and authors
article_authors (article_id INTEGER, author_id INTEGER)
-- Table for academic journals
journals (id SERIAL, name TEXT)
-- Table for types of articles
article_types (id SERIAL, name TEXT)
-- Table with text chunks for semantic search
article_chunks (id BIGINT, article_id BIGINT, chunk_no INTEGER, chunk_text TEXT, chunk_embedding VECTOR)
"""

MULTI_STEP_AGENT_PROMPT_TEMPLATE = """
You are a highly intelligent research assistant...
(Content is the same as in the original 1st file, keeping it for the advanced agent)
"""
TEXT_TO_SQL_PROMPT_TEMPLATE = """
You are a PostgreSQL expert...
(Content is the same as in the original 1st file, keeping it for the text-to-sql agent)
"""

class PydanticModels:
    class QueryRequest(BaseModel):
        question: str
        user_context: Optional[Dict[str, Any]] = None # Kept for compatibility with App.js

    class TextToSqlRequest(BaseModel):
        query: str

# --- Helper Functions (from original advanced script) ---
def create_vector_index():
    """Creates an HNSW index on the chunk_embedding column if it doesn't exist."""
    try:
        with psycopg2.connect(DB_URL) as conn, conn.cursor() as cur:
            cur.execute("SELECT indexname FROM pg_indexes WHERE schemaname = 'public' AND tablename = 'article_chunks' AND indexname = 'article_chunks_embedding_hnsw_idx'")
            if cur.fetchone() is None:
                logger.info("--- Creating HNSW index on article_chunks.chunk_embedding ---")
                cur.execute("CREATE INDEX CONCURRENTLY article_chunks_embedding_hnsw_idx ON article_chunks USING hnsw (chunk_embedding vector_cosine_ops)")
                logger.info("--- HNSW index created successfully ---")
            else:
                logger.info("--- HNSW index already exists ---")
    except psycopg2.Error as e:
        logger.error(f"Error creating vector index: {e}")

def get_tokenizer():
    """Returns a GenerativeModel instance for token counting."""
    try:
        # The genai.configure must be called before this function is truly useful
        return genai.GenerativeModel(LLM_MODEL_NAME)
    except Exception as e:
        logger.error(f"Failed to get Gemini model for token counting: {e}")
        return None # Indicate failure

def split_large_text_into_sub_chunks(text: str, model_for_counting: Any, chunk_size: int = 1500, chunk_overlap: int = 100) -> List[str]:
    """
    Splits text into sub-chunks.
    For Gemini, we use a generic RecursiveCharacterTextSplitter as tiktoken is not applicable.
    Chunk sizes will be approximate in terms of Gemini tokens.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_text(text)

def count_tokens(text: str, model_for_counting: Any) -> int:
    """
    Counts tokens using the Google Gemini model.
    Provides a fallback if the model is not available.
    """
    if model_for_counting:
        try:
            return model_for_counting.count_tokens(text).total_tokens
        except Exception as e:
            logger.warning(f"Error counting tokens with Gemini model: {e}. Falling back to character count / 4.")
            # Fallback approximation
            return len(text) // 4
    else:
        logger.warning("Gemini model for token counting not initialized. Falling back to character count / 4.")
        return len(text) // 4

def prune_messages(messages: List[Dict], max_tokens: int, model_for_counting: Any) -> List[Dict]:
    # (Implementation from original 1st file)
    system_message = messages[0]
    conversational_messages = messages[1:]
    # Pass the Gemini model for counting to count_tokens
    token_counts = {id(msg): count_tokens(json.dumps(msg), model_for_counting) for msg in messages}
    current_tokens = sum(token_counts.values())
    
    while current_tokens > max_tokens and conversational_messages:
        oldest_msg = conversational_messages.pop(0)
        current_tokens -= token_counts.get(id(oldest_msg), 0)
        
    return [system_message] + conversational_messages

# --- Core LLM Class using LangChain and Google Gemini ---
class LiteLLM:
    def __init__(self, model_name: str, api_key: str, max_tokens: int = 800):
        """
        Initializes the LLM wrapper using ChatGoogleGenerativeAI to connect to Google Gemini.
        Note: temperature and max_tokens are handled dynamically in generate method.
        """
        # --- MODIFIED: Initialize ChatGoogleGenerativeAI without temperature/max_tokens here,
        #              as they will be passed dynamically per call via 'config'. ---
        self.client = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key, # Use google_api_key parameter
            convert_system_message_to_human=True # Recommended for Gemini if using system messages
        )
        self.model_name = model_name

    def generate(self, prompt: str, temperature: float = 0.7, max_tokens: Optional[int] = None, response_format: Optional[Dict] = None, **kwargs) -> str:
        """
        Generates a response from the LLM, with caching and Langfuse tracing.
        """
        cache_key = hashlib.md5((prompt + str(temperature) + str(max_tokens) + str(response_format)).encode()).hexdigest()
        current_time = time.time()
        
        # Check cache first
        if cache_key in _api_cache and (current_time - _api_cache[cache_key][1] < _cache_ttl):
            logger.info(f"Returning cached LLM response (max_tokens: {max_tokens or 'default'}).")
            
            # Log cache hit to Langfuse
            if LANGFUSE_ENABLED and langfuse:
                try:
                    langfuse_context.update_current_observation(
                        metadata={
                            "cache_hit": True,
                            "model": self.model_name,
                            "temperature": temperature,
                            "max_tokens": max_tokens or "default"
                        }
                    )
                except Exception as e:
                    logger.debug(f"Langfuse cache hit logging error: {e}")
            
            return _api_cache[cache_key][0]

        # Make actual LLM call with Langfuse tracing
        try:
            # --- MODIFIED: Build generation_config for dynamic parameters ---
            generation_config = {"temperature": temperature}
            if max_tokens is not None:
                generation_config['max_output_tokens'] = max_tokens
            
            # --- Removed response_format as it's not a direct Gemini Chat API parameter ---
            # For structured output, one would use LangChain's with_structured_output or tool_calling features.

            # Create Langfuse generation if enabled
            if LANGFUSE_ENABLED and langfuse:
                try:
                    # Start generation trace
                    generation = langfuse.generation(
                        name="llm_generation",
                        model=self.model_name,
                        input=prompt,
                        metadata={
                            "temperature": temperature,
                            "max_tokens": max_tokens,
                            "cache_hit": False,
                            "provider": "google_gemini" # --- CHANGED: Provider name ---
                        }
                    )
                except Exception as e:
                    logger.debug(f"Langfuse generation start error: {e}")
                    generation = None
            else:
                generation = None

            # Make the actual LLM call
            # --- MODIFIED: Pass generation_config via 'config' parameter to invoke ---
            response = self.client.invoke(prompt, config={"generation_config": generation_config})
            response_text = response.content
            
            # Update Langfuse generation with response
            if generation:
                try:
                    # Count tokens (approximate)
                    # Pass the global Gemini model for counting
                    input_tokens = count_tokens(prompt, gemini_model_for_counting)
                    output_tokens = count_tokens(response_text, gemini_model_for_counting)
                    
                    generation.end(
                        output=response_text,
                        usage={
                            "input": input_tokens,
                            "output": output_tokens,
                            "total": input_tokens + output_tokens
                        }
                    )
                except Exception as e:
                    logger.debug(f"Langfuse generation end error: {e}")
            
            # Cache the response
            _api_cache[cache_key] = (response_text, current_time)
            return response_text
            
        except Exception as e:
            # Log error to Langfuse
            if generation:
                try:
                    generation.end(
                        level="ERROR",
                        status_message=str(e)
                    )
                except Exception as trace_error:
                    logger.debug(f"Langfuse error logging error: {trace_error}")
            
            error_msg = str(e)
            logger.warning(f"LLM generation error: {error_msg}")
            if "rate_limit" in error_msg.lower() or "429" in error_msg:
                raise HTTPException(status_code=429, detail="LLM rate limit exceeded after multiple retries.")
            else:
                raise HTTPException(status_code=500, detail=f"LLM generation failed: {e}")


def get_embedding(text: str) -> List[float]:
    """Global embedding function."""
    return model.encode([f"search_query: {text}"])[0].tolist()

def stream_relevant_sentences(question: str, text_to_search: str) -> Iterator[str]:
    """Enhanced sentence streaming function for better context refinement with focus on quantitative data."""
    logger.info(f"Streaming relevant sentences from text (length: {len(text_to_search)})...")
    # Pass the global Gemini model for counting to split_large_text_into_sub_chunks
    sub_chunks = split_large_text_into_sub_chunks(text_to_search, gemini_model_for_counting)
    seen_sentences = set()
    
    # ENHANCED PROMPT FOR BETTER SENTENCE EXTRACTION
    prompt_template = """
    Analyze the 'Source Text' and provide a comprehensive analysis for the 'User's Question'. Follow these steps in order:

    ## STEP 1: EXTRACT RELEVANT SENTENCES
    First, identify and copy EXACT, VERBATIM sentences from the source text that are relevant to the question. Prioritize sentences that contain:
    - Numerical data, percentages, statistical measures
    - Specific research findings and results
    - Comparative data between groups or conditions
    - Key mechanisms, processes, or qualitative findings
    - Important methodological details

    Copy each relevant sentence word-for-word, each on a new line. Focus on concrete, specific information rather than general statements.

    If NO relevant sentences are found, write: NO_RELEVANT_SENTENCES_FOUND and skip to Step 2.

    ## STEP 2: SYNTHESIZE ANALYSIS
    After completing Step 1, analyze the extracted data and provide a synthesis appropriate to the question type:

    **For questions with numerical data:**
    - **Quantitative Summary**: Calculate or identify typical ranges, averages, or representative outcomes
    - **Key Patterns**: What the data collectively shows across studies
    - **Practical Interpretation**: What these numbers mean in real-world terms
    - **Confidence Level**: How reliable is this conclusion given the available data

    **For questions without numerical data:**
    - **Key Findings**: Main conclusions or insights from the research
    - **Mechanisms/Processes**: How things work or what happens (if applicable)
    - **Practical Implications**: What this means for treatment/decisions
    - **Confidence Level**: How well-established these findings are

    User's Question: "{question}"
    Source Text:
    ---
    {sub_chunk}
    ---

    ## STEP 1 - EXTRACTED SENTENCES:
    [List verbatim sentences with relevant information here]

    ## STEP 2 - SYNTHESIZED ANALYSIS:
    [Provide appropriate synthesis based on whether the question/data is numerical or qualitative]
    """
    
    for i, sub_chunk in enumerate(sub_chunks[:min(7, len(sub_chunks))]):  # Increased from 5 to 7 for more comprehensive extraction
        try:
            prompt = prompt_template.format(question=question, sub_chunk=sub_chunk)
            response_text = llm.generate(prompt, temperature=0.0)  # Use temperature 0 for consistent extraction
            if "NO_RELEVANT_SENTENCES_FOUND" in response_text: 
                continue
            
            for sentence in response_text.splitlines():
                s = sentence.strip()
                if s and s not in seen_sentences and len(s) > 20:  # Increased minimum length for more substantial sentences
                    seen_sentences.add(s)
                    yield s
        except Exception as e:
            logger.error(f"Error during sentence streaming from chunk {i+1}: {e}")
            continue

# --- ReACT Agent (from 2nd script, integrated and enhanced) ---
class ReACTAgent:
    def __init__(self, llm_instance: LiteLLM):
        self.llm = llm_instance
        self.max_iterations = 5

    @safe_langfuse_trace
    def semantic_search_tool(self, query: str, limit: int = 10, similarity_threshold: float = 0.4) -> Dict[str, Any]:
        """Perform semantic search with Langfuse tracing."""
        if LANGFUSE_ENABLED.lower() == "true":
            langfuse_context.update_current_observation(
                name="semantic_search",
                input={"query": query, "limit": limit, "similarity_threshold": similarity_threshold}
            )
        
        try:
            embedding = get_embedding(query)
            result = supabase.rpc('optimized_semantic_search', {
                'query_embedding': embedding,
                'similarity_threshold': similarity_threshold,
                'match_count': limit
            }).execute()
            
            if not result.data:
                search_result = {"observation": "No relevant articles found after optimization and filtering.", "articles": []}
            else:
                articles_data = {}
                for chunk in result.data:
                    article_id = chunk['article_id']
                    if article_id not in articles_data:
                        articles_data[article_id] = {
                            'article_id': article_id, 
                            'title': chunk.get('title', 'Unknown'),
                            'chunks': [], 
                            'similarity': chunk.get('similarity', 0),
                            'article_url': chunk.get('article_url', ''), 
                            'published_date': chunk.get('published_date', '') # Keep published_date here for potential use
                        }
                    articles_data[article_id]['chunks'].append({'text': chunk['chunk_text']})
                    if chunk.get('similarity', 0) > articles_data[article_id]['similarity']:
                        articles_data[article_id]['similarity'] = chunk.get('similarity', 0)

                sorted_articles = sorted(articles_data.values(), key=lambda x: x['similarity'], reverse=True)
                search_result = {"observation": f"Found %s relevant articles." % len(sorted_articles), "articles": sorted_articles}
            
            if LANGFUSE_ENABLED.lower() == "true":
                langfuse_context.update_current_observation(
                    output=search_result,
                    metadata={
                        "articles_found": len(search_result.get("articles", [])),
                        "embedding_model": "nomic-ai/nomic-embed-text-v1.5"
                    }
                )
            
            return search_result
            
        except Exception as e:
            error_message = str(e)
            logger.error(f"ReACT Agent Semantic search error: %s" % error_message)
            
            if LANGFUSE_ENABLED.lower() == "true":
                langfuse_context.update_current_observation(
                    level="ERROR",
                    status_message=error_message
                )
            
            return {"observation": f"Error in semantic search: %s" % error_message, "articles": []}

    @safe_langfuse_trace
    def search_by_author_tool(self, author_name: str) -> Dict[str, Any]:
        """Search for articles by author with Langfuse tracing."""
        if LANGFUSE_ENABLED.lower() == "true":
            langfuse_context.update_current_observation(
                name="search_by_author",
                input={"author_name": author_name}
            )
        
        try:
            result = supabase.rpc('get_articles_by_author', {'author_name': author_name}).execute()
            if not result.data:
                search_result = {"observation": f"No articles found for author: %s" % author_name, "article_ids": []}
            else:
                article_ids = [row['article_id'] for row in result.data]
                search_result = {"observation": f"Found %s articles by author '%s'" % (len(article_ids), author_name), "article_ids": article_ids}
            
            if LANGFUSE_ENABLED.lower() == "true":
                langfuse_context.update_current_observation(
                    output=search_result,
                    metadata={"articles_found": len(search_result.get("article_ids", []))}
                )
            
            return search_result
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Author search error: %s" % e)
            
            if LANGFUSE_ENABLED.lower() == "true":
                langfuse_context.update_current_observation(
                    level="ERROR",
                    status_message=error_msg
                )
            
            return {"observation": f"Error searching by author: %s" % error_msg, "article_ids": []}

    def get_article_details_tool(self, article_ids: List[int]) -> Dict[str, Any]:
        """Get detailed information about specific articles including URLs and PDF URLs."""
        if not article_ids:
            return {"observation": "No article IDs provided", "articles": []}
        
        try:
            # Ensure article_url and pdf_url are selected
            placeholders = ','.join(['%s'] * len(article_ids[:20])) # Limit to 20 to prevent overly long queries
            sql_query = f"""
            SELECT a.id, a.title, a.article_url, a.pdf_url, a.published_date, j.name as journal, 
                   array_agg(DISTINCT au.name) FILTER (WHERE au.name IS NOT NULL) as authors
            FROM articles a
            LEFT JOIN journals j ON a.journal_id = j.id
            LEFT JOIN article_authors aa ON a.id = aa.article_id
            LEFT JOIN authors au ON aa.author_id = au.id
            WHERE a.id IN (%s)
            GROUP BY a.id, j.name ORDER BY a.published_date DESC
            """ % placeholders
            with psycopg2.connect(DB_URL) as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(sql_query, article_ids[:20])
                results = [dict(row) for row in cur.fetchall()]
            return {"observation": f"Retrieved details for %s articles" % len(results), "articles": results}
        except Exception as e:
            logger.error(f"Get article details error: %s" % e)
            return {"observation": f"Error getting article details: %s" % str(e), "articles": []}

    @safe_langfuse_trace
    def classify_query(self, question: str) -> str:
        """Classify query type with Langfuse tracing."""
        if LANGFUSE_ENABLED.lower() == "true":
            langfuse_context.update_current_observation(
                name="query_classification",
                input={"question": question}
            )
        
        classification_prompt = f"""
        Analyze this medical research question and classify it into one category:
        1. "factual": Asks for specific facts, counts, or lists (e.g., "How many articles?", "List papers...").
        2. "conceptual": Asks about concepts, treatments, findings (e.g., "What is diabetes?", "How does aspirin work?").
        3. "author_search": Specifically asks to find papers by an author.
        Question: "%s"
        Respond with ONLY the category name: factual, conceptual, or author_search
        """ % question
        
        try:
            response = self.llm.generate(classification_prompt, temperature=0.0).strip().lower()
            classification = response if response in ["factual", "conceptual", "author_search"] else "conceptual"
            
            if LANGFUSE_ENABLED.lower() == "true":
                langfuse_context.update_current_observation(
                    output={"classification": classification},
                    metadata={"raw_response": response}
                )
            
            return classification
            
        except Exception as e:
            logger.error(f"Query classification error: %s" % e)
            
            if LANGFUSE_ENABLED.lower() == "true":
                langfuse_context.update_current_observation(
                    level="ERROR",
                    status_message=str(e)
                )
            
            return "conceptual"

    def _format_article_sources(self, detailed_articles: List[Dict], original_semantic_results: Optional[List[Dict]] = None) -> List[Dict]:
        """
        Formats raw article details (from get_article_details_tool) for frontend consumption,
        ensuring 'url' (mapped from 'article_url') and 'pdf_url' are present.
        Optionally merges similarity scores if original_semantic_results (from semantic_search_tool) are provided.
        """
        formatted_sources = []
        similarity_map = {}
        if original_semantic_results:
            # Create a map for quick lookup of similarity from the semantic search results
            # The key is article_id, value is its similarity score
            similarity_map = {art_sem['article_id']: art_sem.get('similarity', 0) for art_sem in original_semantic_results}

        for art in detailed_articles:
            source_item = {
                "id": art.get('id'),
                "title": art.get('title', 'Unknown'),
                "url": art.get('article_url', ''),  # Map article_url to 'url' for frontend
                "pdf_url": art.get('pdf_url', ''),  # Explicitly include pdf_url
                "published_date": art.get('published_date').isoformat() if art.get('published_date') else None,
                "journal": art.get('journal', None),
                "authors": art.get('authors', [])
            }
            # Add similarity if available and matched by ID
            if original_semantic_results and art.get('id') in similarity_map:
                source_item['similarity'] = similarity_map[art.get('id')]
            formatted_sources.append(source_item)
        
        # Sort by similarity if it was included, otherwise by published date (desc) as a fallback
        if original_semantic_results:
            formatted_sources.sort(key=lambda x: x.get('similarity', 0), reverse=True)
        else:
            formatted_sources.sort(key=lambda x: x.get('published_date', '0000-01-01'), reverse=True)

        return formatted_sources

    def _handle_author_search(self, question: str) -> Dict[str, Any]:
        """Handle author search queries."""
        author_extraction_prompt = f"""
        Extract the author's name from this question. Return only the name, nothing else.
        Question: "%s"
        Author name:
        """ % question
        
        try:
            author_name = self.llm.generate(author_extraction_prompt, temperature=0.0).strip()
            if not author_name:
                return {
                    "status": "error",
                    "answer": "Could not extract author name from the question. Please try phrasing like 'papers by John Doe'.",
                    "sources": [],
                    "query_type": "author_search"
                }
            
            search_result = self.search_by_author_tool(author_name)
            
            if not search_result.get("article_ids"):
                return {
                    "status": "no_results",
                    "answer": f"No articles found for author: %s" % author_name,
                    "sources": [],
                    "query_type": "author_search"
                }
            
            article_ids = search_result["article_ids"][:10]
            
            # Fetch full details including article_url and pdf_url
            details_result = self.get_article_details_tool(article_ids)
            detailed_articles = details_result.get("articles", [])
            
            # Construct a human-readable answer string
            if detailed_articles:
                answer_parts = [f"Found %s articles by %s:" % (len(detailed_articles), author_name)]
                for article in detailed_articles:
                    article_info = f"Title: %s" % article['title']
                    if article['journal']:
                        article_info += f"\nJournal: %s" % article['journal']
                    if article['published_date']:
                        article_info += f"\nPublished: %s" % article['published_date'].isoformat()
                    if article['authors']:
                        article_info += f"\nAuthors: %s" % ', '.join(article['authors'])
                    answer_parts.append(article_info)
                answer = "\n\n---\n\n".join(answer_parts)
            else:
                answer = f"Found %s article IDs by %s, but could not retrieve detailed information." % (len(article_ids), author_name)

            # Format the sources list for the frontend, ensuring URLs are present
            sources = self._format_article_sources(detailed_articles)

            return {
                "status": "success",
                "answer": answer,
                "sources": sources,
                "query_type": "author_search"
            }
                
        except Exception as e:
            logger.error(f"Error in author search: %s" % e, exc_info=True)
            return {
                "status": "error",
                "answer": f"Error processing author search: %s" % str(e),
                "sources": [],
                "query_type": "author_search"
            }

    def _handle_factual_query(self, question: str) -> Dict[str, Any]:
        """Handle factual queries that need specific data."""
        # Use semantic search to find relevant information
        search_result = self.semantic_search_tool(question, limit=15)
        
        if not search_result.get("articles"):
            return {
                "status": "no_results",
                "answer": "No relevant information found in the database.",
                "sources": [],
                "query_type": "factual"
            }
        
        # Get detailed article info for full URLs, including PDF URLs
        article_ids = [art["article_id"] for art in search_result["articles"]]
        details_result = self.get_article_details_tool(article_ids)
        detailed_articles = details_result.get("articles", [])

        # Format sources for the frontend, ensuring URL and PDF URL are present, and similarity is included
        sources = self._format_article_sources(detailed_articles, search_result["articles"])
        
        # Generate structured answer using the improved prompt template
        answer = self._generate_structured_answer(question, search_result["articles"], query_type="factual")
        
        return {
            "status": "success",
            "answer": answer,
            "sources": sources,
            "query_type": "factual"
        }

    def _handle_conceptual_query(self, question: str) -> Dict[str, Any]:
        """Handle conceptual queries that need explanations."""
        # Use semantic search to find relevant information
        search_result = self.semantic_search_tool(question, limit=10)
        
        if not search_result.get("articles"):
            return {
                "status": "no_results",
                "answer": "No relevant information found in the database.",
                "sources": [],
                "query_type": "conceptual"
            }
        
        # Get detailed article info for full URLs, including PDF URLs
        article_ids = [art["article_id"] for art in search_result["articles"]]
        details_result = self.get_article_details_tool(article_ids)
        detailed_articles = details_result.get("articles", [])

        # Format sources for the frontend, ensuring URL and PDF URL are present, and similarity is included
        sources = self._format_article_sources(detailed_articles, search_result["articles"])
        
        # Generate structured answer using the improved prompt template
        answer = self._generate_structured_answer(question, search_result["articles"], query_type="conceptual")
        
        return {
            "status": "success",
            "answer": answer,
            "sources": sources,
            "query_type": "conceptual"
        }
    
    @safe_langfuse_trace
    def _generate_structured_answer(self, question: str, articles: List[Dict], query_type: str) -> str:
        """Generate structured answer with user-friendly format."""
        if LANGFUSE_ENABLED.lower() == "true":
            langfuse_context.update_current_observation(
                name="generate_structured_answer",
                input={
                    "question": question,
                    "articles_count": len(articles),
                    "query_type": query_type
                }
            )
        
        # Build context from articles
        context_parts = []
        article_limit = 7 if query_type == "factual" else 5
        
        for art in articles[:article_limit]:
            chunks_text = ' '.join([c['text'] for c in art.get('chunks', [])])
            context_parts.append(f"STUDY: %s\nCONTENT: %s" % (art.get('title', 'N/A'), chunks_text))
        
        context_text = "\n\n---\n\n".join(context_parts)
        
        # Updated prompt template for user-friendly output
        prompt_template = """
    You are a medical research assistant. Analyze the provided research studies and answer the user's question in a clear, accessible way.

    User's Question: "{question}"

    Research Studies:
    ---
    {context_text}
    ---

    Please provide your response in this exact format:

    **Summary:**
    [Provide a brief, easy-to-understand overview of the key findings in 2-3 sentences. Include the most important percentage or improvement rate when available.]

    **Detailed Analysis:**
    1. [First key finding or insight, explained clearly]
    2. [Second key finding or insight, explained clearly]
    3. [Third key finding or insight, explained clearly]
    [Continue with additional numbered points as needed, but keep to 5-7 points maximum]

    Guidelines:
    - Include key percentages, survival rates, or improvement statistics where relevant (e.g., "33% vs 22% survival rate")
    - Focus on the most meaningful numbers - recovery rates, response rates, survival percentages
    - Avoid overwhelming technical statistics (p-values, confidence intervals, hazard ratios) unless crucial
    - Provide context for numbers (e.g., "survival improved from 22% to 33%")
    - Use accessible language that a non-expert can understand
    - Highlight the most important and actionable information
    - Keep each numbered point concise but informative
    """
        
        formatted_prompt = prompt_template.format(
            question=question,
            context_text=context_text
        )
        
        try:
            # Adjust temperature and max_tokens based on query type
            temperature = 0.0 if query_type == "factual" else 0.1
            max_tokens = 1500  # Reduced for more concise responses
            
            response = self.llm.generate(formatted_prompt, temperature=temperature, max_tokens=max_tokens)
            
            if LANGFUSE_ENABLED.lower() == "true":
                langfuse_context.update_current_observation(
                    output={"answer": response},
                    metadata={
                        "context_length": len(context_text),
                        "articles_used": len(articles),
                        "query_type": query_type,
                        "temperature": temperature,
                        "max_tokens": max_tokens
                    }
                )
            
            return response
            
        except Exception as e:
            logger.error(f"Structured answer generation error: %s" % e)
            
            if LANGFUSE_ENABLED.lower() == "true":
                langfuse_context.update_current_observation(
                    level="ERROR",
                    status_message=str(e)
                )
            
            raise e

    def run(self, question: str) -> Dict[str, Any]:
        """Main ReACT loop with Langfuse tracing."""
        if not self.llm:
            raise HTTPException(status_code=503, detail="LLM is not configured.")
        
        if LANGFUSE_ENABLED.lower() == "true" and langfuse:
            try:
                trace = langfuse.trace(
                    name="react_agent_run",
                    input={"question": question},
                    metadata={"agent_type": "ReACT", "max_iterations": self.max_iterations}
                )
                langfuse_context.update_current_trace(trace)
            except Exception as e:
                logger.debug(f"Langfuse trace creation error: %s" % e)
        else:
            trace = None
        
        try:
            query_type = self.classify_query(question)
            logger.info(f"Query classified as: %s" % query_type)
            
            if query_type == "author_search":
                result = self._handle_author_search(question)
            elif query_type == "factual":
                result = self._handle_factual_query(question)
            else:
                result = self._handle_conceptual_query(question)
            
            if trace:
                try:
                    trace.update(
                        output=result,
                        metadata={
                            "query_type": query_type,
                            "status": result.get("status", "unknown"),
                            "sources_count": len(result.get("sources", []))
                        }
                    )
                except Exception as e:
                    logger.debug(f"Langfuse trace update error: %s" % e)
            
            return result
            
        except Exception as e:
            logger.error(f"ReACT agent run error: %s" % e)
            
            if trace:
                try:
                    trace.update(
                        level="ERROR",
                        status_message=str(e)
                    )
                except Exception as trace_error:
                    logger.debug(f"Langfuse error trace update error: %s" % trace_error)
            
            raise e
# --- ADVANCED AGENT TOOLS AND LOGIC (from original 1st script) ---
# Initialize LLM and Agent
llm: Optional[LiteLLM] = None
agent: Optional[ReACTAgent] = None
# This variable is assigned in initialize_llm_and_agent
gemini_model_for_counting: Any = None # Initialize it globally too for advanced functions

# --- Modified Advanced Functions ---
@safe_langfuse_trace
def advanced_semantic_search_tool(query: str, article_ids_filter: Optional[List[int]] = None) -> Dict[str, Any]:
    """Enhanced semantic search with Langfuse tracing."""
    if LANGFUSE_ENABLED:
        langfuse_context.update_current_observation(
            name="advanced_semantic_search",
            input={"query": query, "article_ids_filter": article_ids_filter}
        )
    
    logger.info(f"Executing ADVANCED semantic_search_tool with query='%s'..." % query)
    embedding = get_embedding(query)
    
    rpc_params = {'query_embedding': embedding, 'similarity_threshold': 0.4, 'match_count': 17}
    rpc_function = "match_article_chunks_by_ids" if article_ids_filter else "match_article_chunks_react"
    if article_ids_filter: 
        rpc_params['id_filter'] = article_ids_filter

    try:
        response = supabase.rpc(rpc_function, rpc_params).execute()
        chunks_from_db = response.data
        
        if not chunks_from_db:
            result = {"observation": "No relevant article chunks found.", "context": None, "references": []}
        else:
            all_chunk_text = "\n".join([c.get("chunk_text", "") for c in chunks_from_db])
            # Pass the global Gemini model for counting
            final_evidence_sentences = list(stream_relevant_sentences(question=query, text_to_search=all_chunk_text))
            
            if not final_evidence_sentences:
                final_evidence = "\n\n".join([f"FROM STUDY '%s': %s" % (c.get('title', 'Unknown'), c.get('chunk_text', '')) for c in chunks_from_db[:5]])
            else:
                final_evidence = "\n".join(final_evidence_sentences)
            
            if not final_evidence:
                result = {"observation": "Found chunks, but no relevant content after refinement.", "context": None, "references": []}
            else:
                unique_articles = {c['article_id']: {"title": c.get('title'), "url": c.get('article_url')} for c in chunks_from_db}
                result = {
                    "observation": f"Successfully extracted %s relevant data points from %s studies." % (len(final_evidence_sentences), len(unique_articles)), 
                    "context": final_evidence, 
                    "references": list(unique_articles.values())
                }
        
        if LANGFUSE_ENABLED:
            langfuse_context.update_current_observation(
                output=result,
                metadata={
                    "chunks_found": len(chunks_from_db),
                    "evidence_sentences": len(final_evidence_sentences) if 'final_evidence_sentences' in locals() else 0,
                    "unique_articles": len(result.get("references", []))
                }
            )
        
        return result
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error in advanced_semantic_search_tool: %s" % e, exc_info=True)
        
        if LANGFUSE_ENABLED:
            langfuse_context.update_current_observation(
                level="ERROR",
                status_message=error_msg
            )
        
        return {"observation": f"Error: %s" % error_msg, "context": None, "references": []}

def execute_sql_tool(sql_query: str) -> Dict[str, Any]:
    """Executes a SQL query for the advanced agent."""
    logger.info(f"Executing execute_sql_tool with query='%s'" % sql_query)
    if not sql_query.strip().upper().startswith("SELECT"):
        return {"observation": "Error: Only SELECT statements are allowed."}

    try:
        with psycopg2.connect(DB_URL) as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql_query)
            results = [dict(row) for row in cur.fetchall()]
        
        if results and all('id' in row for row in results):
            return {"observation": f"Retrieved %s article IDs." % [row['id'] for row in results], "article_ids": [row['id'] for row in results]}
        
        return {"observation": f"Query returned %s rows." % len(results), "sql_data": results}
    except psycopg2.Error as e:
        logger.error(f"Database error on query '%s': %s" % (sql_query, e))
        return {"observation": f"Database error: %s" % e}

def finish_tool(answer: str) -> Dict[str, Any]:
    return {"final_answer": answer}

TOOLS = {
    "execute_sql": {"function": execute_sql_tool, "description": "Runs a SQL query...", "parameters": {}},
    "semantic_search": {"function": advanced_semantic_search_tool, "description": "Searches for conceptual info...", "parameters": {}},
    "finish": {"function": finish_tool, "description": "Provides the final answer...", "parameters": {}}
}

def run_agentic_rag(user_question: str, max_steps: int = 4) -> Dict[str, Any]:
    """The original, multi-step agentic workflow - simplified implementation."""
    start_time = time.time()
    logger.info(f"--- Starting Advanced Agentic RAG for question: '%s' ---" % user_question)
    
    try:
        # For now, use the same logic as the ReACT agent but return in expected format
        if not agent:
            return {
                "answer": "Advanced agent is not available. LLM not initialized.",
                "sources": [],
                "processing_time_seconds": round(time.time() - start_time, 2)
            }
        
        result = agent.run(user_question)
        
        return {
            "answer": result.get("answer", "No answer generated"),
            "sources": result.get("sources", []),
            "processing_time_seconds": round(time.time() - start_time, 2),
            "status": result.get("status", "unknown"),
            "query_type": result.get("query_type", "unknown")
        }
        
    except Exception as e:
        logger.error(f"Error in advanced agentic RAG: %s" % e)
        return {
            "answer": f"Error in advanced processing: %s" % str(e),
            "sources": [],
            "processing_time_seconds": round(time.time() - start_time, 2)
        }


@app.post("/analyze-research", summary="Main RAG Analysis", tags=["RAG"])
async def analyze_research(request: PydanticModels.QueryRequest):
    """Main endpoint with Langfuse tracing."""
    trace = None # Initialize trace to None
    # Conditional trace creation for evaluation
    if LANGFUSE_ENABLED and langfuse and "analyze_research_endpoint" in LANGFUSE_EVAL_SCOPE:
        try:
            trace = langfuse.trace(
                name="analyze_research_endpoint", # This name must match one in LANGFUSE_EVAL_SCOPE
                input={"question": request.question, "user_context": request.user_context},
                metadata={"endpoint": "/analyze-research", "method": "POST"}
            )
            # IMPORTANT: Update the current context so nested @observe calls belong to this trace
            langfuse_context.update_current_trace(trace)
        except Exception as e:
            logger.debug(f"Langfuse endpoint trace creation error: {e}")
            trace = None # Ensure trace is None if creation fails
    
    try:
        if not request.question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        
        result = agent.run(request.question)
        
        # Update trace with success result
        if trace: # Only update if a trace was actually created
            try:
                trace.update(
                    output=result,
                    metadata={
                        "status": "success",
                        "query_type": result.get("query_type", "unknown"),
                        "sources_count": len(result.get("sources", [])),
                        "processing_time": time.time() - trace.timestamp.timestamp() if hasattr(trace, 'timestamp') else None
                    }
                )
            except Exception as e:
                logger.debug(f"Langfuse trace update error: {e}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error in /analyze-research: {e}", exc_info=True)
        
        # Update trace with error
        if trace: # Only update if a trace was actually created
            try:
                trace.update(
                    level="ERROR",
                    status_message=str(e),
                    metadata={"error_type": type(e).__name__}
                )
            except Exception as trace_error:
                logger.debug(f"Langfuse error trace update error: {trace_error}")
        
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

@app.post("/agent-search", summary="Advanced Multi-Step Agentic RAG", tags=["RAG"])
async def handle_agent_search(request: PydanticModels.QueryRequest):
    """Advanced agent endpoint with Langfuse tracing."""
    trace = None # Initialize trace to None
    # Conditional trace creation for evaluation
    if LANGFUSE_ENABLED and langfuse and "advanced_agent_search" in LANGFUSE_EVAL_SCOPE:
        try:
            trace = langfuse.trace(
                name="advanced_agent_search", # This name must match one in LANGFUSE_EVAL_SCOPE
                input={"question": request.question},
                metadata={"endpoint": "/agent-search", "agent_type": "advanced"}
            )
            # IMPORTANT: Update the current context so nested @observe calls belong to this trace
            langfuse_context.update_current_trace(trace)
        except Exception as e:
            logger.debug(f"Langfuse advanced agent trace error: {e}")
            trace = None # Ensure trace is None if creation fails
    
    try:
        result = run_agentic_rag(request.question)
        
        # Update trace with success result
        if trace: # Only update if a trace was actually created
            try:
                # Assuming run_agentic_rag already updates the trace within its own logic,
                # you might only need a final update here, or omit if run_agentic_rag completes it.
                # For simplicity, let's add a generic update here too.
                trace.update(
                    output=result,
                    metadata={
                        "status": result.get("status", "unknown"),
                        "query_type": result.get("query_type", "unknown"),
                        "processing_time": result.get("processing_time_seconds")
                    }
                )
            except Exception as e:
                logger.debug(f"Langfuse trace update error: {e}")
        
        return result
    except Exception as e:
        logger.error(f"Error in /agent-search: {e}")
        
        # Update trace with error
        if trace: # Only update if a trace was actually created
            try:
                trace.update(
                    level="ERROR",
                    status_message=str(e),
                    metadata={"error_type": type(e).__name__}
                )
            except Exception as trace_error:
                logger.debug(f"Langfuse error trace update error: {trace_error}")
        
        raise HTTPException(status_code=500, detail=str(e))


# --- NEW ENDPOINTS FOR DATABASE BROWSING AND DEBUGGING ---
@app.get("/get-articles", summary="Get Articles for Frontend", tags=["Database"])
async def get_articles(
    page: int = 1,
    limit: int = 50,
    search: Optional[str] = None,
    author: Optional[str] = None,
    journal: Optional[str] = None,
    article_type: Optional[str] = None,
    year_from: Optional[int] = None,
    year_to: Optional[int] = None
):
    """
    This is the endpoint your frontend is actually calling.
    It returns articles with pagination and filtering.
    """
    try:
        offset = (page - 1) * limit
        
        # Build the base query
        base_query = """
        SELECT DISTINCT
            a.id,
            a.title,
            a.article_url,
            a.pdf_url,
            a.published_date,
            j.name as journal,
            at.name as article_type,
            array_agg(DISTINCT au.name) FILTER (WHERE au.name IS NOT NULL) as authors
        FROM articles a
        LEFT JOIN journals j ON a.journal_id = j.id
        LEFT JOIN article_types at ON a.article_type_id = at.id
        LEFT JOIN article_authors aa ON a.id = aa.article_id
        LEFT JOIN authors au ON aa.author_id = au.id
        """
        
        # Build WHERE conditions
        where_conditions = []
        params = []
        
        if search:
            where_conditions.append("a.title ILIKE %s")
            params.append(f"%{search}%")
            
        if author:
            where_conditions.append("""
                EXISTS (
                    SELECT 1 FROM article_authors aa2 
                    JOIN authors au2 ON aa2.author_id = au2.id 
                    WHERE aa2.article_id = a.id AND au2.name ILIKE %s
                )
            """)
            params.append(f"%{author}%")
            
        if journal and journal != "All Journals":
            where_conditions.append("j.name = %s")
            params.append(journal)
            
        if article_type and article_type != "All Types":
            where_conditions.append("at.name = %s")
            params.append(article_type)
            
        if year_from:
            where_conditions.append("EXTRACT(YEAR FROM a.published_date) >= %s")
            params.append(year_from)
            
        if year_to:
            where_conditions.append("EXTRACT(YEAR FROM a.published_date) <= %s")
            params.append(year_to)
        
        # Complete the query
        if where_conditions:
            query = base_query + " WHERE " + " AND ".join(where_conditions)
        else:
            query = base_query
            
        query += " GROUP BY a.id, j.name, at.name ORDER BY a.published_date DESC LIMIT %s OFFSET %s"
        params.extend([limit, offset])
        
        # Get total count for pagination
        count_base = "SELECT COUNT(DISTINCT a.id) FROM articles a"
        if where_conditions:
            count_query = count_base + " LEFT JOIN journals j ON a.journal_id = j.id LEFT JOIN article_types at ON a.article_type_id = at.id LEFT JOIN article_authors aa ON a.id = aa.article_id LEFT JOIN authors au ON aa.author_id = au.id WHERE " + " AND ".join(where_conditions)
            count_params = params[:-2]  # Exclude limit and offset
        else:
            count_query = count_base
            count_params = []
        
        with psycopg2.connect(DB_URL) as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Execute count query
            cur.execute(count_query, count_params)
            total_count = cur.fetchone()['count']
            
            # Execute main query
            cur.execute(query, params)
            articles = [dict(row) for row in cur.fetchall()]
            
            # Convert dates to strings for JSON serialization
            for article in articles:
                if article['published_date']:
                    article['published_date'] = article['published_date'].isoformat()
            
        return {
            "articles": articles,
            "total": total_count,
            "page": page,
            "limit": limit,
            "totalPages": (total_count + limit - 1) // limit,
            "hasMore": (page * limit) < total_count
        }
        
    except Exception as e:
        logger.error(f"Error in get_articles: %s" % e, exc_info=True)
        # Return empty result instead of error to prevent frontend crashes
        return {
            "articles": [],
            "total": 0,
            "page": page,
            "limit": limit,
            "totalPages": 0,
            "hasMore": False,
            "error": str(e)
        }

# Also add the other endpoints your frontend might be calling:

@app.get("/get-journals", summary="Get All Journals", tags=["Database"])
async def get_journals():
    """Get list of all journals for dropdown filter."""
    try:
        with psycopg2.connect(DB_URL) as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT DISTINCT name FROM journals WHERE name IS NOT NULL ORDER BY name")
            journals = [{"value": row['name'], "label": row['name']} for row in cur.fetchall()]
            return {"journals": [{"value": "All Journals", "label": "All Journals"}] + journals}
    except Exception as e:
        logger.error(f"Error getting journals: %s" % e)
        return {"journals": [{"value": "All Journals", "label": "All Journals"}]}

@app.get("/get-article-types", summary="Get All Article Types", tags=["Database"])
async def get_article_types():
    """Get list of all article types for dropdown filter."""
    try:
        with psycopg2.connect(DB_URL) as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT DISTINCT name FROM article_types WHERE name IS NOT NULL ORDER BY name")
            types = [{"value": row['name'], "label": row['name']} for row in cur.fetchall()]
            return {"types": [{"value": "All Types", "label": "All Types"}] + types}
    except Exception as e:
        logger.error(f"Error getting article types: %s" % e)
        return {"types": [{"value": "All Types", "label": "All Types"}]}

@app.get("/database-info", summary="Get Database Info", tags=["Database"])
async def get_database_info():
    """Get basic database information for the frontend."""
    try:
        with psycopg2.connect(DB_URL) as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Get total article count
            cur.execute("SELECT COUNT(*) as count FROM articles")
            total_articles = cur.fetchone()['count']
            
            # Get date range
            cur.execute("SELECT MIN(published_date) as min_date, MAX(published_date) as max_date FROM articles WHERE published_date IS NOT NULL")
            date_range = cur.fetchone()
            
            return {
                "totalArticles": total_articles,
                "dateRange": {
                    "minDate": date_range['min_date'].isoformat() if date_range['min_date'] else None,
                    "maxDate": date_range['max_date'].isoformat() if date_range['max_date'] else None
                }
            }
    except Exception as e:
        logger.error(f"Error getting database info: %s" % e)
        return {
            "totalArticles": 0,
            "dateRange": {"minDate": None, "maxDate": None},
            "error": str(e)
        }

# --- Server Startup ---
@app.on_event("startup")
async def on_startup():
    logger.info("Server is starting up...")
    
    # Initialize LLM and agent
    try:
        initialize_llm_and_agent()
    except Exception as e:
        logger.error(f"Failed to initialize LLM and agent: %s" % e)
        # Don't raise error to allow server to start for other endpoints
    
    # Langfuse health check
    if LANGFUSE_ENABLED and langfuse:
        try:
            # Test Langfuse connection
            test_trace = langfuse.trace(name="startup_health_check")
            test_trace.update(output={"status": "healthy"})
            logger.info("Langfuse connection verified")
        except Exception as e:
            logger.warning(f"Langfuse health check failed: %s" % e)
    
    if not DB_URL or "[YOUR-PASSWORD]" in DB_URL:
        logger.critical("FATAL: DATABASE_URL is not configured properly")
    else:
        create_vector_index()

if __name__ == "__main__":
    import uvicorn
    # PORT environment variable is still useful for deployment flexibility
    port = int(os.getenv("PORT", 8000))

    uvicorn.run(app, host="0.0.0.0", port=port)
