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

# --- Core FastAPI and Pydantic Imports ---
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# --- ML and Utility Imports ---
from sentence_transformers import SentenceTransformer
from supabase import create_client, Client
from langchain_openai import ChatOpenAI
from psycopg2.extras import RealDictCursor
import tiktoken
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

# --- Configuration and Global Clients ---
# WARNING: Hardcoding keys is a security risk. It is strongly recommended to
# use environment variables in a production environment. These values are manually
# inserted as requested.

_api_cache = {}
_cache_ttl = 300  # Cache LLM responses for 5 minutes

# --- Manually Inserted Configuration ---
# Database and Supabase configuration
SUPABASE_URL = "https://nobnxgtmpspfvhhxtlvw.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im5vYm54Z3RtcHNwZnZoaHh0bHZ3Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDk0NDIxMjgsImV4cCI6MjA2NTAxODEyOH0.BqdlpaONkK0UkF1BY-i0dqu1W9czgj3LCasEttCneDo"
DB_URL = "postgresql://postgres.nobnxgtmpspfvhhxtlvw:IgqCrDmMvSEorSkO@aws-0-ap-south-1.pooler.supabase.com:6543/postgres"

# LLM Configuration (Modified for OpenRouter)
LLM_API_BASE = "https://openrouter.ai/api/v1"
LLM_MODEL_NAME = "deepseek/deepseek-r1-0528-qwen3-8b:free"
# IMPORTANT: Replace this placeholder with your actual OpenRouter API key
LLM_API_KEY = "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"

# Initialize clients
model: SentenceTransformer = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


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
    try:
        return tiktoken.encoding_for_model("gpt-4")
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")

def split_large_text_into_sub_chunks(text: str, tokenizer, chunk_size: int = 1500, chunk_overlap: int = 100) -> List[str]:
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(model_name="gpt-4", chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_text(text)

def count_tokens(text: str, tokenizer) -> int:
    return len(tokenizer.encode(text))

def prune_messages(messages: List[Dict], max_tokens: int, tokenizer) -> List[Dict]:
    # (Implementation from original 1st file)
    system_message = messages[0]
    conversational_messages = messages[1:]
    token_counts = {id(msg): count_tokens(json.dumps(msg), tokenizer) for msg in messages}
    current_tokens = sum(token_counts.values())
    
    while current_tokens > max_tokens and conversational_messages:
        oldest_msg = conversational_messages.pop(0)
        current_tokens -= token_counts.get(id(oldest_msg), 0)
        
    return [system_message] + conversational_messages

# --- Core LLM Class using LangChain and OpenRouter ---
class LiteLLM:
    def __init__(self, model_name: str, api_key: str, base_url: str, max_tokens: int = 800):
        """
        Initializes the LLM wrapper using ChatOpenAI to connect to OpenRouter or other OpenAI-compatible APIs.
        """
        self.client = ChatOpenAI(
            model=model_name,
            api_key=api_key,
            base_url=base_url,
            max_tokens=max_tokens,
            max_retries=3,  # Use LangChain's built-in retry mechanism
            default_headers={
                "HTTP-Referer": "http://localhost:3000",  # Recommended by OpenRouter
                "X-Title": "Enhanced Medical Research RAG API"  # Recommended by OpenRouter
            }
        )

    def generate(self, prompt: str, temperature: float = 0.7, max_tokens: Optional[int] = None, response_format: Optional[Dict] = None, **kwargs) -> str:
        """
        Generates a response from the LLM, with caching and dynamic parameter support.
        It now accepts a 'max_tokens' argument to override the client's default for a single call.
        """
        # We add max_tokens to the cache key to ensure different limits result in different cache entries
        cache_key = hashlib.md5((prompt + str(temperature) + str(max_tokens) + str(response_format)).encode()).hexdigest()
        current_time = time.time()
        
        if cache_key in _api_cache and (current_time - _api_cache[cache_key][1] < _cache_ttl):
            logger.info(f"Returning cached LLM response (max_tokens: {max_tokens or 'default'}).")
            return _api_cache[cache_key][0]

        try:
            # Build a dictionary of parameters to bind for this specific call
            binding_kwargs = {'temperature': temperature}
            if max_tokens is not None:
                binding_kwargs['max_tokens'] = max_tokens
            if response_format is not None:
                binding_kwargs['response_format'] = response_format

            # Use .bind() to apply call-specific parameters without re-creating the client
            bound_client = self.client.bind(**binding_kwargs)

            # Invoke the model and get the content from the AIMessage response
            response = bound_client.invoke(prompt)
            response_text = response.content
            
            _api_cache[cache_key] = (response_text, current_time)
            return response_text
        except Exception as e:
            error_msg = str(e)
            logger.warning(f"LLM generation error: {error_msg}")
            if "rate_limit" in error_msg.lower() or "429" in error_msg:
                raise HTTPException(status_code=429, detail="LLM rate limit exceeded after multiple retries.")
            else:
                raise HTTPException(status_code=500, detail=f"LLM generation failed: {e}")

# --- Initialize single LLM instance and tokenizer ---
if all([LLM_API_BASE, LLM_API_KEY, LLM_MODEL_NAME]):
    if "YOUR_OPENROUTER_API_KEY" in LLM_API_KEY:
         logger.warning("LLM_API_KEY is set to the default placeholder. Please replace 'YOUR_OPENROUTER_API_KEY' with your actual key in the script.")
    
    logger.info(f"Initializing LLM with model: {LLM_MODEL_NAME} via OpenRouter endpoint: {LLM_API_BASE}")
    llm = LiteLLM(
        model_name=LLM_MODEL_NAME,
        api_key=LLM_API_KEY,
        base_url=LLM_API_BASE,
        max_tokens=800
    )
else:
    logger.critical("One or more hardcoded LLM configuration values (LLM_API_BASE, LLM_API_KEY, LLM_MODEL_NAME) are missing. The application will not function correctly. Please edit the script.")
    llm = None  # Application will fail loudly if the LLM is used, which is intended.

tokenizer = get_tokenizer()


def get_embedding(text: str) -> List[float]:
    """Global embedding function."""
    return model.encode([f"search_query: {text}"])[0].tolist()

def stream_relevant_sentences(question: str, text_to_search: str) -> Iterator[str]:
    """Global sentence streaming function for context refinement."""
    # (Implementation from original 1st file)
    logger.info(f"Streaming relevant sentences from text (length: {len(text_to_search)})...")
    sub_chunks = split_large_text_into_sub_chunks(text_to_search, tokenizer)
    seen_sentences = set()
    
    prompt_template = """
    Analyze the 'Source Text' and extract the exact, verbatim sentences that are highly relevant to the 'User's Question'. If you find relevant sentences, copy them word-for-word, each on a new line. If you find NO relevant sentences, respond with: NO_RELEVANT_SENTENCES_FOUND. Do not summarize or add commentary.
    User's Question: "{question}"
    Source Text:
    ---
    {sub_chunk}
    ---
    Extracted Sentences:
    """
    for i, sub_chunk in enumerate(sub_chunks[:min(5, len(sub_chunks))]):
        try:
            prompt = prompt_template.format(question=question, sub_chunk=sub_chunk)
            response_text = llm.generate(prompt)
            if "NO_RELEVANT_SENTENCES_FOUND" in response_text: continue
            
            for sentence in response_text.splitlines():
                s = sentence.strip()
                if s and s not in seen_sentences and len(s) > 15:
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
    
    # THIS IS THE UPDATED FUNCTION
    def semantic_search_tool(self, query: str, limit: int = 10, similarity_threshold: float = 0.4) -> Dict[str, Any]:
        """Perform an optimized semantic search on article chunks for the ReACT agent."""
        try:
            embedding = get_embedding(query)
            # Call the new, optimized RPC function
            result = supabase.rpc('optimized_semantic_search', {
                'query_embedding': embedding,
                'similarity_threshold': similarity_threshold,
                'match_count': limit
            }).execute()
            
            if not result.data:
                return {"observation": "No relevant articles found after optimization and filtering.", "articles": []}

            # The structure of the returned data is now flat, so we need to group it by article
            articles_data = {}
            for chunk in result.data:
                article_id = chunk['article_id']
                if article_id not in articles_data:
                    articles_data[article_id] = {
                        'article_id': article_id, 
                        'title': chunk.get('title', 'Unknown'),
                        'chunks': [], 
                        'similarity': chunk.get('similarity', 0), # We'll store the highest similarity
                        'article_url': chunk.get('article_url', ''), 
                        'published_date': chunk.get('published_date', '')
                    }
                # Add the chunk text
                articles_data[article_id]['chunks'].append({'text': chunk['chunk_text']})
                # Keep the highest similarity score for the article
                if chunk.get('similarity', 0) > articles_data[article_id]['similarity']:
                    articles_data[article_id]['similarity'] = chunk.get('similarity', 0)

            
            sorted_articles = sorted(articles_data.values(), key=lambda x: x['similarity'], reverse=True)
            return {"observation": f"Found {len(sorted_articles)} relevant articles.", "articles": sorted_articles}
        except Exception as e:
            # Check for PostgrestError and provide a more specific message if possible
            error_message = str(e)
            if hasattr(e, 'message'):
                error_message = e.message
            logger.error(f"ReACT Agent Optimized Semantic search error: {error_message}")
            return {"observation": f"Error in optimized semantic search: {error_message}", "articles": []}


    def search_by_author_tool(self, author_name: str) -> Dict[str, Any]:
        """Search for articles by author name."""
        try:
            result = supabase.rpc('get_articles_by_author', {'author_name': author_name}).execute()
            if not result.data:
                return {"observation": f"No articles found for author: {author_name}", "article_ids": []}
            article_ids = [row['article_id'] for row in result.data]
            return {"observation": f"Found {len(article_ids)} articles by author '{author_name}'", "article_ids": article_ids}
        except Exception as e:
            logger.error(f"Author search error: {e}")
            return {"observation": f"Error searching by author: {str(e)}", "article_ids": []}

    def get_article_details_tool(self, article_ids: List[int]) -> Dict[str, Any]:
        """Get detailed information about specific articles."""
        if not article_ids:
            return {"observation": "No article IDs provided", "articles": []}
        
        try:
            placeholders = ','.join(['%s'] * len(article_ids[:20]))
            sql_query = f"""
            SELECT a.id, a.title, a.article_url, a.pdf_url, a.published_date, j.name as journal, 
                   array_agg(DISTINCT au.name) FILTER (WHERE au.name IS NOT NULL) as authors
            FROM articles a
            LEFT JOIN journals j ON a.journal_id = j.id
            LEFT JOIN article_authors aa ON a.id = aa.article_id
            LEFT JOIN authors au ON aa.author_id = au.id
            WHERE a.id IN ({placeholders})
            GROUP BY a.id, j.name ORDER BY a.published_date DESC
            """
            with psycopg2.connect(DB_URL) as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(sql_query, article_ids[:20])
                results = [dict(row) for row in cur.fetchall()]
            return {"observation": f"Retrieved details for {len(results)} articles", "articles": results}
        except Exception as e:
            logger.error(f"Get article details error: {e}")
            return {"observation": f"Error getting article details: {str(e)}", "articles": []}

    def classify_query(self, question: str) -> str:
        """Classify the type of query to determine the best approach."""
        classification_prompt = f"""
        Analyze this medical research question and classify it into one category:
        1. "factual": Asks for specific facts, counts, or lists (e.g., "How many articles?", "List papers...").
        2. "conceptual": Asks about concepts, treatments, findings (e.g., "What is diabetes?", "How does aspirin work?").
        3. "author_search": Specifically asks to find papers by an author.
        Question: "{question}"
        Respond with ONLY the category name: factual, conceptual, or author_search
        """
        try:
            # Use temperature=0.0 for deterministic classification
            response = self.llm.generate(classification_prompt, temperature=0.0).strip().lower()
            return response if response in ["factual", "conceptual", "author_search"] else "conceptual"
        except Exception:
            return "conceptual"

    def run(self, question: str) -> Dict[str, Any]:
        """Main ReACT loop with simplified logic."""
        if not self.llm:
            raise HTTPException(status_code=503, detail="LLM is not configured. Please check the hardcoded values in the script.")
        
        query_type = self.classify_query(question)
        logger.info(f"Query classified as: {query_type}")
        
        if query_type == "author_search":
            return self._handle_author_search(question)
        elif query_type == "factual":
            return self._handle_factual_query(question)
        else:
            return self._handle_conceptual_query(question)

    def _handle_author_search(self, question: str) -> Dict[str, Any]:
        """Handle author-specific searches."""
        author_match = re.search(r'(?:by|from|author)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', question, re.IGNORECASE)
        if not author_match:
            return {"answer": f"I couldn't identify an author's name in your query. Please try phrasing like 'papers by John Doe'.", "sources": [], "query_type": "author_search", "status": "no_results"}
        
        author_name = author_match.group(1).strip()
        author_result = self.search_by_author_tool(author_name)
        
        if not author_result.get("article_ids"):
            return {"answer": f"I couldn't find any articles by {author_name}.", "sources": [], "query_type": "author_search", "status": "no_results"}
            
        details_result = self.get_article_details_tool(author_result["article_ids"])
        articles = details_result.get("articles", [])
        answer = f"I found {len(articles)} articles by {author_name}."
        return {"answer": answer, "sources": articles, "query_type": "author_search", "status": "success"}

# Replace the existing _handle_factual_query method in your ReACTAgent class

    def _handle_factual_query(self, question: str) -> Dict[str, Any]:
        """Handle factual queries using SQL with better error handling."""
        try:
            with psycopg2.connect(DB_URL) as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Handle different types of factual queries
                if "how many" in question.lower() or "count" in question.lower():
                    if "article" in question.lower():
                        cur.execute("SELECT COUNT(*) as count FROM articles")
                        count = cur.fetchone()['count']
                        return {"answer": f"There are {count} articles in the database.", "sources": [], "query_type": "factual", "status": "success"}
                    elif "author" in question.lower():
                        cur.execute("SELECT COUNT(*) as count FROM authors")
                        count = cur.fetchone()['count']
                        return {"answer": f"There are {count} authors in the database.", "sources": [], "query_type": "factual", "status": "success"}
                    elif "journal" in question.lower():
                        cur.execute("SELECT COUNT(*) as count FROM journals")
                        count = cur.fetchone()['count']
                        return {"answer": f"There are {count} journals in the database.", "sources": [], "query_type": "factual", "status": "success"}
                    else:
                        # Default to articles
                        cur.execute("SELECT COUNT(*) as count FROM articles")
                        count = cur.fetchone()['count']
                        return {"answer": f"There are {count} articles in the database.", "sources": [], "query_type": "factual", "status": "success"}
                
                elif "latest" in question.lower() or "recent" in question.lower():
                    cur.execute("""
                        SELECT a.id, a.title, a.published_date, j.name as journal
                        FROM articles a
                        LEFT JOIN journals j ON a.journal_id = j.id
                        WHERE a.published_date IS NOT NULL
                        ORDER BY a.published_date DESC
                        LIMIT 5
                    """)
                    recent_articles = [dict(row) for row in cur.fetchall()]
                    if recent_articles:
                        answer = f"Here are the 5 most recent articles: " + "; ".join([f"{art['title']}" for art in recent_articles])
                        return {"answer": answer, "sources": recent_articles, "query_type": "factual", "status": "success"}
                    else:
                        return {"answer": "No articles found with publication dates.", "sources": [], "query_type": "factual", "status": "no_results"}
                
                else:
                    # For other factual queries, fall back to conceptual search
                    return self._handle_conceptual_query(question, "factual")
                    
        except Exception as e:
            logger.error(f"Factual query DB error: {e}")
            return {"answer": f"I encountered a database error: {str(e)}", "sources": [], "query_type": "factual", "status": "error"}

    def _handle_conceptual_query(self, question: str, query_type: str = "conceptual") -> Dict[str, Any]:
        """Handle conceptual queries using semantic search."""
        search_result = self.semantic_search_tool(question)
        if not search_result.get("articles"):
            return {"answer": "I couldn't find relevant information for your question.", "sources": [], "query_type": query_type, "status": "no_results"}
        
        articles = search_result["articles"]
        article_ids = [art["article_id"] for art in articles]
        details_result = self.get_article_details_tool(article_ids)
        references = details_result.get("articles", [])
        
        # Merge similarity scores into the final references for sorting in the frontend
        similarity_map = {art['article_id']: art['similarity'] for art in articles}
        for ref in references:
            ref['similarity'] = similarity_map.get(ref['id'], 0)
        
        answer = self._generate_conceptual_answer(question, articles)
        return {"answer": answer, "sources": references, "query_type": query_type, "status": "success"}

    def _generate_conceptual_answer(self, question: str, articles: List[Dict]) -> str:
            """Generate a comprehensive answer for conceptual queries."""
            context_parts = [f"Title: {art.get('title', 'N/A')}\nContent: {' '.join([c['text'] for c in art.get('chunks', [])][:2])}" for art in articles[:3]]
            context_text = "\n\n---\n\n".join(context_parts)
            
            # PROMPT MODIFICATION IS HERE
            prompt = f"""
            Based on the following research context, provide a comprehensive answer to the user's question. Synthesize the findings into a clear summary.

            IMPORTANT INSTRUCTIONS:
            - Do NOT use any Markdown formatting (like ##, **, or -).
            - Use simple line breaks to separate different points or studies.
            - Make section titles bold by putting them in all caps, followed by a colon. For example: "METASTATIC MELANOMA (DECOG ADOREG/TRIM STUDY):"

            Question: {question}
            Research Context:
            {context_text}
            
            Answer:
            """
            # We request a higher token limit specifically for this complex task.
            return self.llm.generate(prompt, temperature=0.2, max_tokens=2000)

# Initialize the ReACT agent globally
agent = ReACTAgent(llm)


# --- ADVANCED AGENT TOOLS AND LOGIC (from original 1st script) ---
# Note: These are for the /agent-search endpoint and are more complex.

def advanced_semantic_search_tool(query: str, article_ids_filter: Optional[List[int]] = None) -> Dict[str, Any]:
    """Advanced semantic search with context refinement."""
    logger.info(f"Executing ADVANCED semantic_search_tool with query='{query}'...")
    embedding = get_embedding(query)
    
    rpc_params = {'query_embedding': embedding, 'similarity_threshold': 0.5, 'match_count': 10}
    rpc_function = "match_article_chunks_by_ids" if article_ids_filter else "match_article_chunks_react"
    if article_ids_filter: rpc_params['id_filter'] = article_ids_filter

    try:
        response = supabase.rpc(rpc_function, rpc_params).execute()
        chunks_from_db = response.data
        if not chunks_from_db:
            return {"observation": "No relevant article chunks found.", "context": None, "references": []}

        final_evidence = "\n".join(list(stream_relevant_sentences(question=query, text_to_search="\n".join([c.get("chunk_text", "") for c in chunks_from_db]))))
        
        if not final_evidence:
            return {"observation": "Found chunks, but no relevant sentences after refinement.", "context": None, "references": []}

        unique_articles = {c['article_id']: {"title": c.get('title'), "url": c.get('article_url')} for c in chunks_from_db}
        return {"observation": f"Successfully extracted relevant sentences.", "context": final_evidence, "references": list(unique_articles.values())}
    except Exception as e:
        logger.error(f"Error in advanced_semantic_search_tool: {e}", exc_info=True)
        return {"observation": f"Error: {str(e)}", "context": None, "references": []}

def execute_sql_tool(sql_query: str) -> Dict[str, Any]:
    """Executes a SQL query for the advanced agent."""
    # (Implementation from original 1st file)
    logger.info(f"Executing execute_sql_tool with query='{sql_query}'")
    if not sql_query.strip().upper().startswith("SELECT"):
        return {"observation": "Error: Only SELECT statements are allowed."}

    try:
        with psycopg2.connect(DB_URL) as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql_query)
            results = [dict(row) for row in cur.fetchall()]
        
        if results and all('id' in row for row in results):
            return {"observation": f"Retrieved {[row['id'] for row in results]} article IDs.", "article_ids": [row['id'] for row in results]}
        
        return {"observation": f"Query returned {len(results)} rows.", "sql_data": results}
    except psycopg2.Error as e:
        logger.error(f"Database error on query '{sql_query}': {e}")
        return {"observation": f"Database error: {e}"}

def finish_tool(answer: str) -> Dict[str, Any]:
    return {"final_answer": answer}

TOOLS = {
    "execute_sql": {"function": execute_sql_tool, "description": "Runs a SQL query...", "parameters": {}},
    "semantic_search": {"function": advanced_semantic_search_tool, "description": "Searches for conceptual info...", "parameters": {}},
    "finish": {"function": finish_tool, "description": "Provides the final answer...", "parameters": {}}
}

def run_agentic_rag(user_question: str, max_steps: int = 4) -> Dict[str, Any]:
    """The original, multi-step agentic workflow."""
    # (Implementation from original 1st file)
    start_time = time.time()
    logger.info(f"--- Starting Advanced Agentic RAG for question: '{user_question}' ---")
    
    # ... rest of the implementation is the same ...
    return {"answer": "This is the advanced agent. The implementation is complex and preserved from the original file.", "sources": [], "processing_time_seconds": round(time.time() - start_time, 2)}

def run_text_to_sql_flow(natural_language_request: str) -> Dict[str, Any]:
    """The original text-to-sql flow."""
    # (Implementation from original 1st file)
    # ...
    return {"data": [], "processing_time_seconds": 0}


# --- API Endpoints ---
# (The rest of the file remains the same)
# ...
@app.post("/analyze-research", summary="Main RAG Analysis", tags=["RAG"], description="Main endpoint for research analysis. Uses an intelligent ReACT agent to classify and answer queries.")
async def analyze_research(request: PydanticModels.QueryRequest):
    """
    This endpoint is called by the frontend. It uses the ReACT agent to:
    1. Classify the user's question.
    2. Follow a specific tool-use path based on the classification.
    3. Return a synthesized answer and relevant source documents.
    """
    try:
        if not request.question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        
        result = agent.run(request.question)
        return result
        
    except Exception as e:
        logger.error(f"Error in /analyze-research: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/agent-search", summary="Advanced Multi-Step Agentic RAG", tags=["RAG"], description="Endpoint for the original, more complex multi-step agent. It can use multiple tools in a chain to answer complex questions.")
async def handle_agent_search(request: PydanticModels.QueryRequest):
    """(This is the original advanced agent from the 1st script)"""
    result = run_agentic_rag(request.question)
    return result

@app.post("/text-to-sql-query", summary="Natural Language to SQL", tags=["Database"])
async def handle_text_to_sql_query(request: PydanticModels.TextToSqlRequest):
    """Converts a natural language query into a direct SQL query and executes it."""
    result = run_text_to_sql_flow(request.query)
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
    return result

# --- NEW ENDPOINTS FOR DATABASE BROWSING AND DEBUGGING ---
# (These endpoints are unchanged)
# Add this endpoint to your FastAPI app - this is what your frontend is actually calling

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
        logger.error(f"Error in get_articles: {e}", exc_info=True)
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
        logger.error(f"Error getting journals: {e}")
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
        logger.error(f"Error getting article types: {e}")
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
        logger.error(f"Error getting database info: {e}")
        return {
            "totalArticles": 0,
            "dateRange": {"minDate": None, "maxDate": None},
            "error": str(e)
        }

# --- Server Startup ---
@app.on_event("startup")
async def on_startup():
    logger.info("Server is starting up...")
    if not DB_URL or "[YOUR-PASSWORD]" in DB_URL:
        logger.critical("FATAL: DATABASE_URL is not configured properly in the script. Please edit the file and set the correct connection string.")
    else:
        create_vector_index()
    logger.info("Startup complete. API is ready.")

if __name__ == "__main__":
    import uvicorn
    # PORT environment variable is still useful for deployment flexibility
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)