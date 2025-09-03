import os
import re
import json
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from supabase import create_client
from together import Together
from datetime import datetime

app = FastAPI()

# Add CORS middleware (unchanged)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load embedding model (unchanged)
model = SentenceTransformer("nomic-ai/nomic-embed-text-v1", trust_remote_code=True)

# Supabase setup (unchanged)
supabase = create_client(
    'https://nobnxgtmpspfvhhxtlvw.supabase.co', 
    'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im5vYm54Z3RtcHNwZnZoaHh0bHZ3Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDk0NDIxMjgsImV4cCI6MjA2NTAxODEyOH0.BqdlpaONkK0UkF1BY-i0dqu1W9czgj3LCasEttCneDo'
)

class QueryRequest(BaseModel):
    question: str
    user_context: Optional[Dict[str, Any]] = None

class TogetherLLM:
    def __init__(self, model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"):
        self.client = Together(api_key="2d5341da842ba4f133068a76fba1bc069199737b6178293b22765df2df33465f")
        self.model = model

    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a medical research expert assistant with SQL knowledge. Provide accurate, detailed answers."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1024,
            temperature=0.3,
            top_k=40,
            top_p=0.9
        )
        return response.choices[0].message.content.strip()

together_llm = TogetherLLM()

def chunk_text(text: str, max_length: int = 512) -> List[str]:
    """
    Split text into chunks of a specified maximum length.
    """
    words = text.split()
    chunks = []
    current_chunk = []
    for word in words:
        current_chunk.append(word)
        if len(' '.join(current_chunk)) > max_length:
            chunks.append(' '.join(current_chunk[:-1]))
            current_chunk = [word]
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    return chunks

def detect_query_type(question: str) -> str:
    """
    Enhanced query type detection that handles hybrid queries
    """
    prompt = f"""
    Analyze this research database query and determine if it's asking for:
    - METADATA: Information about articles themselves (author, publication date, journal, counts, lists, recent/latest articles) can be found by an sql query
    - SEMANTIC: Content-based questions about research findings, topics, or medical concepts
    - HYBRID: Questions that combine content topics with metadata filters (e.g., "papers about cancer from 2020-2022")

    Question: "{question}"

    Respond with exactly one word: METADATA, SEMANTIC, or HYBRID
    """
    
    response = together_llm.generate(prompt).strip().upper()
    
    # Extract the classification from response
    if 'HYBRID' in response:
        query_type = 'hybrid'
    elif 'METADATA' in response:
        query_type = 'metadata'
    elif 'SEMANTIC' in response:
        query_type = 'semantic'
    else:
        # Fallback logic for better detection
        question_lower = question.lower()
        has_topic = any(topic in question_lower for topic in ['about', 'on', 'regarding', 'related to'])
        has_date = any(year in question_lower for year in ['2020', '2021', '2022', '2023', '2024']) or \
                   any(word in question_lower for word in ['from', 'between', 'during', 'in'])
        has_author = 'by' in question_lower or 'author' in question_lower
        
        if has_topic and (has_date or has_author):
            query_type = 'hybrid'
        elif has_author or has_date or 'count' in question_lower:
            query_type = 'metadata'
        else:
            query_type = 'semantic'
    
    print(f"Detected query type: {query_type} for question: {question}")
    return query_type

def extract_topic_and_filters(question: str) -> dict:
    """
    Extract semantic topic and metadata filters from hybrid queries
    """
    filters = {
        'topic': None,
        'start_year': None,
        'end_year': None,
        'author': None
    }
    
    question_lower = question.lower()
    
    # Extract topic (what comes after "about", "on", "regarding", etc.)
    topic_patterns = [
        r'papers about ([^f]+?)(?:\s+(?:from|released|published))',
        r'articles about ([^f]+?)(?:\s+(?:from|released|published))',
        r'papers on ([^f]+?)(?:\s+(?:from|released|published))',
        r'articles on ([^f]+?)(?:\s+(?:from|released|published))',
        r'regarding ([^f]+?)(?:\s+(?:from|released|published))',
        r'related to ([^f]+?)(?:\s+(?:from|released|published))',
    ]
    
    for pattern in topic_patterns:
        match = re.search(pattern, question_lower)
        if match:
            filters['topic'] = match.group(1).strip()
            break
    
    # If no specific pattern found, try to extract topic before date mentions
    if not filters['topic']:
        # Look for topic before date-related words
        before_date = re.search(r'(.*?)(?:\s+(?:from|between|during|in|released|published))', question_lower)
        if before_date:
            topic_part = before_date.group(1)
            # Remove common prefixes
            topic_part = re.sub(r'^(?:papers?|articles?|studies?|research)\s+(?:about|on|regarding|related\s+to)\s+', '', topic_part)
            if topic_part:
                filters['topic'] = topic_part.strip()
    
    # Extract date range
    date_patterns = [
        r'from\s+(\d{4})\s+to\s+(\d{4})',
        r'between\s+(\d{4})\s+and\s+(\d{4})',
        r'(\d{4})\s+to\s+(\d{4})',
        r'(\d{4})\s*-\s*(\d{4})',
        r'released from\s+(\d{4})\s+to\s+(\d{4})',
        r'published from\s+(\d{4})\s+to\s+(\d{4})',
    ]
    
    for pattern in date_patterns:
        match = re.search(pattern, question_lower)
        if match:
            filters['start_year'] = int(match.group(1))
            filters['end_year'] = int(match.group(2))
            break
    
    # Check for single year
    if not filters['start_year']:
        single_year_patterns = [
            r'in\s+(\d{4})',
            r'during\s+(\d{4})',
            r'from\s+(\d{4})',
        ]
        for pattern in single_year_patterns:
            match = re.search(pattern, question_lower)
            if match:
                year = int(match.group(1))
                filters['start_year'] = year
                filters['end_year'] = year
                break
    
    # Extract author if present
    if 'by' in question_lower:
        author_match = re.search(r'by\s+([^f]+?)(?:\s+(?:from|released|published)|\s*$)', question_lower)
        if author_match:
            filters['author'] = author_match.group(1).strip()
    
    print(f"Extracted filters: {filters}")
    return filters

def generate_sql_query(question: str) -> str:
    """
    Generate a SQL query for metadata-based questions
    """
    schema_info = """
    Database Schema:
    
    TABLE articles (
        id serial PRIMARY KEY,
        title text NOT NULL,
        article_url text,
        journal_id integer REFERENCES journals(id),
        article_type_id integer REFERENCES article_types(id),
        pdf_url text,
        full_text text,
        created_at timestamp with time zone DEFAULT now(),
        published_date date,
        embedding vector
    );

    TABLE authors (
        id serial PRIMARY KEY,
        name text NOT NULL
    );

    TABLE journals (
        id serial PRIMARY KEY,
        name text NOT NULL
    );

    TABLE article_types (
        id serial PRIMARY KEY,
        name text NOT NULL UNIQUE
    );

    TABLE article_authors (
        id serial PRIMARY KEY,
        article_id integer REFERENCES articles(id),
        author_id integer REFERENCES authors(id)
    );
    """
    
    # Handle author-based queries with a hardcoded template
    author_match = re.match(r"papers by ([\w\s\.]+)", question, re.IGNORECASE)
    if author_match:
        author_name = author_match.group(1).strip()
        sql_query = f"""
        SELECT a.id, a.title, a.published_date::text, j.name AS journal_name
        FROM articles a
        JOIN article_authors aa ON a.id = aa.article_id
        JOIN authors au ON aa.author_id = au.id
        JOIN journals j ON a.journal_id = j.id
        WHERE LOWER(au.name) ILIKE LOWER('%{author_name}%')
        ORDER BY a.published_date DESC
        LIMIT 5
        """
        print(f"Generated SQL query (hardcoded for author): {sql_query}")
        return sql_query.strip()

    # Fallback to LLM for other queries
    prompt = f"""
    You are a PostgreSQL expert. Convert this natural language question into a SQL query.

    Database Schema:
    {schema_info}

    Question: "{question}"

    IMPORTANT RULES:
    1. Always LIMIT results to 5
    2. For recent/latest queries, ORDER BY published_date DESC
    3. For date-specific queries, use proper date filtering (e.g., published_date BETWEEN 'YYYY-MM-DD' AND 'YYYY-MM-DD')
    4. Handle keywords case-insensitively when filtering (use LOWER() and ILIKE)
    5. Always cast date fields like 'published_date' to text using '::text'
    6. Always include the article id, title, published_date, and journal name in the SELECT clause
    7. Join with journals table to get journal name
    8. For author queries, join with article_authors and authors tables
    9. Do not include semicolons at the end of the query

    Examples:
    - Question: "papers by John Smith"
      SQL: SELECT a.id, a.title, a.published_date::text, j.name AS journal_name
           FROM articles a
           JOIN article_authors aa ON a.id = aa.article_id
           JOIN authors au ON aa.author_id = au.id
           JOIN journals j ON a.journal_id = j.id
           WHERE LOWER(au.name) ILIKE LOWER('%John Smith%')
           ORDER BY a.published_date DESC
           LIMIT 5

    - Question: "papers released from 2020 to 2022"
      SQL: SELECT a.id, a.title, a.published_date::text, j.name AS journal_name
           FROM articles a
           JOIN journals j ON a.journal_id = j.id
           WHERE a.published_date BETWEEN '2020-01-01' AND '2022-12-31'
           ORDER BY a.published_date DESC
           LIMIT 5

    Return ONLY the SQL query, no explanation or markdown formatting:
    """
    
    response = together_llm.generate(prompt).strip()
    print(f"Generated SQL query (via LLM): {response}")
    
    if response.startswith('```sql'):
        response = response[6:]
    if response.startswith('```'):
        response = response[3:]
    if response.endswith('```'):
        response = response[:-3]
    
    sql_query = response.strip()
    
    if not sql_query.upper().startswith('SELECT'):
        raise HTTPException(status_code=400, detail=f"Generated query doesn't start with SELECT: {sql_query}")
    if ';' in sql_query:
        sql_query = sql_query.replace(';', '')
    
    return sql_query

def execute_metadata_query(question: str):
    """
    Handle metadata-based questions using raw SQL queries with better error handling
    """
    print(f"Executing metadata query for: {question}")
    
    try:
        # Generate SQL query
        query = generate_sql_query(question)
        print(f"Generated SQL query: {query}")
        
        # Execute the query with detailed error handling
        try:
            results = supabase.rpc('execute_sql', {'query': query}).execute()
            print(f"RPC response: {results}")
            
            if hasattr(results, 'data') and results.data:
                print(f"Query returned {len(results.data)} results")
            else:
                print("No data returned from RPC call")
                return []
                
        except Exception as rpc_error:
            print(f"RPC call failed: {str(rpc_error)}")
            
            # Check if it's an RPC function not found error
            if "function execute_sql" in str(rpc_error).lower():
                raise HTTPException(
                    status_code=500, 
                    detail="RPC function 'execute_sql' not found. Please create it in Supabase."
                )
            
            # Check if it's a permission error
            if "permission" in str(rpc_error).lower():
                raise HTTPException(
                    status_code=500,
                    detail="Permission denied for RPC function execution."
                )
            
            # Re-raise with more context
            raise HTTPException(
                status_code=500,
                detail=f"RPC execution failed: {str(rpc_error)}"
            )
        
        # Process results
        enriched_results = []
        for item in results.data:
            try:
                # Check the structure of the returned data
                print(f"Processing item: {item}")
                
                # Handle different possible response structures
                if isinstance(item, dict):
                    if 'result' in item:
                        article = item['result']
                    else:
                        article = item
                else:
                    article = item
                
                article_id = article.get('id')
                if not article_id:
                    print(f"No article ID found in: {article}")
                    continue
                
                # Get authors
                authors = supabase.table('article_authors')\
                    .select('authors(name)')\
                    .eq('article_id', article_id)\
                    .execute()
                
                author_names = [a['authors']['name'] for a in authors.data] if authors.data else []
                
                enriched_article = {
                    **article,
                    'authors': author_names
                }
                
                enriched_results.append(enriched_article)
                
            except Exception as process_error:
                print(f"Error processing individual result: {str(process_error)}")
                continue
        
        return enriched_results
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Unexpected error in execute_metadata_query: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Metadata query execution failed: {str(e)}"
        )

# The following functions remain unchanged
def get_semantic_results(question: str, match_count: int = 5):
    question_embedding = model.encode([question], convert_to_numpy=True)[0].tolist()
    
    # Retrieve from article_chunks table for better granularity
    matched_chunks = supabase.rpc(
        "match_article_chunks", # Assuming you have a function like this for chunk embeddings
        {
            "query_embedding": question_embedding,
            "match_count": match_count,
            "min_similarity": 0.6
        }
    ).execute()

    if not matched_chunks.data:
        return None
    
    # Group chunks by article and get associated article metadata
    article_ids = list(set([chunk['article_id'] for chunk in matched_chunks.data]))
    
    # Fetch article details for these IDs
    articles_data = supabase.table('articles')\
        .select('id, title, article_url, pdf_url, published_date, full_text, journals(name)')\
        .in_('id', article_ids)\
        .execute()
    
    if not articles_data.data:
        return None

    articles_map = {article['id']: article for article in articles_data.data}
    
    enriched_results = []
    for chunk in matched_chunks.data:
        article_id = chunk['article_id']
        article = articles_map.get(article_id)
        if article:
            # Get authors
            authors = supabase.table('article_authors') \
                .select('authors(name)') \
                .eq('article_id', article_id) \
                .execute()
            
            enriched_article = article.copy()
            enriched_article['authors'] = [author['authors']['name'] for author in authors.data] if authors.data else []
            # Add the relevant chunk text and similarity score to the article object
            enriched_article['relevant_chunk'] = chunk['chunk_text']
            enriched_article['similarity_score'] = chunk['similarity']
            enriched_results.append(enriched_article)
    
    # Deduplicate articles if multiple chunks from the same article are returned,
    # prioritizing the one with the highest similarity.
    unique_articles = {}
    for article in enriched_results:
        art_id = article['id']
        if art_id not in unique_articles or article['similarity_score'] > unique_articles[art_id]['similarity_score']:
            unique_articles[art_id] = article
    
    # Sort by similarity score descending
    sorted_articles = sorted(unique_articles.values(), key=lambda x: x['similarity_score'], reverse=True)[:match_count]

    return sorted_articles

def get_hybrid_results(question: str, match_count: int = 5):
    """
    Handle hybrid queries that combine semantic search with metadata filtering.
    """
    filters = extract_topic_and_filters(question)
    
    if not filters['topic']:
        # Fallback to regular semantic search if no topic is found
        return get_semantic_results(question, match_count)
    
    # Get the embedding for the topic
    topic_embedding = model.encode([filters['topic']], convert_to_numpy=True)[0].tolist()
    
    # First, get all semantically similar chunks (from chunks table)
    matched_chunks = supabase.rpc(
        "match_article_chunks", # Using RPC for chunk-level matching
        {
            "query_embedding": topic_embedding,
            "match_count": 100,  # Get more to filter later
            "min_similarity": 0.5 # Lower threshold for filtering
        }
    ).execute()
    
    if not matched_chunks.data:
        return None, 0
    
    # Group chunks by article ID to avoid fetching article data multiple times
    article_ids_from_chunks = list(set([chunk['article_id'] for chunk in matched_chunks.data]))

    # Fetch full article details for all relevant articles
    articles_data_response = supabase.table('articles')\
        .select('id, title, article_url, pdf_url, published_date, full_text, journals(name)')\
        .in_('id', article_ids_from_chunks)\
        .execute()

    if not articles_data_response.data:
        return None, 0

    articles_map = {article['id']: article for article in articles_data_response.data}
    
    # Apply metadata filters
    filtered_articles_with_chunks = []
    for chunk in matched_chunks.data:
        article_id = chunk['article_id']
        article = articles_map.get(article_id)
        if not article:
            continue # Article data not found for this chunk, skip

        # Apply date filter
        if filters['start_year'] and filters['end_year']:
            if article.get('published_date'):
                try:
                    pub_date = datetime.fromisoformat(article['published_date'].replace('Z', '+00:00'))
                    pub_year = pub_date.year
                    if not (filters['start_year'] <= pub_year <= filters['end_year']):
                        continue
                except:
                    continue
            else:
                continue  # Skip articles without publication dates when date filter is specified
        
        # Apply author filter if specified
        if filters['author']:
            authors_result = supabase.table('article_authors')\
                .select('authors(name)')\
                .eq('article_id', article_id)\
                .execute()
            
            author_names = [a['authors']['name'].lower() for a in authors_result.data] if authors_result.data else []
            if not any(filters['author'].lower() in name for name in author_names):
                continue
        
        # Add relevant chunk and similarity to the article object
        article_with_chunk = article.copy()
        article_with_chunk['relevant_chunk'] = chunk['chunk_text']
        article_with_chunk['similarity_score'] = chunk['similarity']
        filtered_articles_with_chunks.append(article_with_chunk)
    
    # Deduplicate articles based on highest similarity for each unique article
    unique_filtered_articles = {}
    for article_item in filtered_articles_with_chunks:
        art_id = article_item['id']
        if art_id not in unique_filtered_articles or article_item['similarity_score'] > unique_filtered_articles[art_id]['similarity_score']:
            unique_filtered_articles[art_id] = article_item

    final_filtered_articles = list(unique_filtered_articles.values())

    # Get total count of filtered results before limiting
    total_count = len(final_filtered_articles)
    
    # Limit to requested count for response and sort by similarity
    limited_articles = sorted(final_filtered_articles, key=lambda x: x['similarity_score'], reverse=True)[:match_count]
    
    # Enrich with author information for the limited set
    enriched_articles = []
    for article in limited_articles:
        authors = supabase.table('article_authors')\
            .select('authors(name)')\
            .eq('article_id', article['id'])\
            .execute()
        
        enriched_article = article.copy()
        enriched_article['authors'] = [author['authors']['name'] for author in authors.data] if authors.data else []
        enriched_articles.append(enriched_article)
    
    return enriched_articles, total_count


# ***************************************************************
# START OF MODIFIED PROMPT FUNCTIONS
# ***************************************************************

def build_metadata_llm_prompt(question: str, articles: List[Dict]) -> str:
    """
    Builds a prompt for the LLM to provide a structured answer for metadata queries.
    This function will now focus on using the article metadata and a brief summary.
    """
    if not articles:
        return "No articles found matching your criteria."

    articles_summary = []
    for i, a in enumerate(articles):
        title = a.get('title', 'Unknown Title')
        authors = ', '.join(a.get('authors', ['Unknown']))
        journal = a.get('journal_name', 'Unknown Journal')
        published_date = a.get('published_date', 'Unknown')
        
        # For metadata queries, 'full_text' might not be fully fetched or relevant chunk is not provided.
        # Use a short snippet of the title or a general description if available.
        # If you want content summary for metadata, you'd need to fetch full_text and summarize here.
        # For now, let's keep it metadata-focused.
        articles_summary.append(
            f"  - Title: {title}\n"
            f"    Authors: {authors}\n"
            f"    Journal: {journal}\n"
            f"    Published Date: {published_date}"
        )

    return f"""
    USER QUERY: {question}

    Here are some articles found based on your query:
    {'\n'.join(articles_summary)}

    Based on the above information, respond to the user's query. Focus on presenting the metadata directly and clearly. Do not generate content beyond what is explicitly stated in the provided article information. If the articles are not directly relevant to a specific factual answer, simply list them.
    """

def build_semantic_llm_prompt(question: str, articles: List[Dict]) -> str:
    """
    Builds a prompt for the LLM to answer the user's question directly using the
    'relevant_chunk' from the semantically retrieved articles.
    """
    if not articles:
        return "No relevant articles found on this topic."

    # Concatenate relevant chunks into a single context for the LLM
    context_chunks = []
    for i, a in enumerate(articles):
        chunk = a.get('relevant_chunk', '')
        if chunk:
            context_chunks.append(f"ARTICLE TITLE: {a.get('title', 'N/A')}\nRELEVANT SNIPPET {i+1}:\n{chunk}\n---")

    context_text = "\n\n".join(context_chunks)

    return f"""
    You are a medical research expert. Your task is to answer the user's question concisely and accurately based ONLY on the provided relevant snippets from research articles.

    USER QUESTION: {question}

    RELEVANT SNIPPETS FROM ARTICLES:
    {context_text}

    INSTRUCTIONS:
    1. Answer the user's question directly and comprehensively using only the information present in the "RELEVANT SNIPPETS".
    2. Do NOT use any outside knowledge.
    3. If the provided snippets do not contain enough information to answer the question, state "I couldn't find enough information in the provided articles to fully answer this question."
    4. Cite the article title for each piece of information you use in your answer. Example: (Source: Article Title)
    5. Present the answer in a natural, conversational tone.
    6. If the snippets are not relevant to the question at all, respond with "No relevant information found on this topic in the provided articles."
    """

def build_hybrid_llm_prompt(question: str, articles: List[Dict], total_count: int, filters: dict) -> str:
    """
    Builds a prompt for LLM to answer the user's question using relevant chunks,
    while also mentioning the metadata filters and total count.
    """
    if not articles:
        return f"No articles found matching your criteria. Total count: 0"

    # Build filter description
    filter_desc = []
    if filters['topic']:
        filter_desc.append(f"topic: {filters['topic']}")
    if filters['start_year'] and filters['end_year']:
        if filters['start_year'] == filters['end_year']:
            filter_desc.append(f"year: {filters['start_year']}")
        else:
            filter_desc.append(f"years: {filters['start_year']}-{filters['end_year']}")
    if filters['author']:
        filter_desc.append(f"author: {filters['author']}")
    
    filter_text = ", ".join(filter_desc)

    context_chunks = []
    for i, a in enumerate(articles):
        chunk = a.get('relevant_chunk', '')
        if chunk:
            context_chunks.append(f"ARTICLE TITLE: {a.get('title', 'N/A')}\nRELEVANT SNIPPET {i+1}:\n{chunk}\n---")

    context_text = "\n\n".join(context_chunks)

    return f"""
    You are a medical research expert. Your task is to answer the user's question concisely and accurately based ONLY on the provided relevant snippets from research articles, considering the applied filters.

    USER QUESTION: {question}
    SEARCH FILTERS: {filter_text}
    TOTAL MATCHING ARTICLES FOUND (before final display limit): {total_count}
    SHOWING RESULTS FROM: Top {len(articles)} relevant articles

    RELEVANT SNIPPETS FROM ARTICLES:
    {context_text}

    INSTRUCTIONS:
    1. Start by mentioning the total number of articles found that match the criteria (e.g., "Found {total_count} articles matching your criteria...").
    2. Then, answer the user's question directly and comprehensively using only the information present in the "RELEVANT SNIPPETS".
    3. Do NOT use any outside knowledge.
    4. If the provided snippets do not contain enough information to answer the question, state "I couldn't find enough information in the provided articles to fully answer this question, but found {total_count} articles that partially match your criteria."
    5. Cite the article title for each piece of information you use in your answer. Example: (Source: Article Title)
    6. Present the answer in a natural, conversational tone.
    7. If the snippets are not relevant to the question at all, respond with "No relevant information found on this topic in the provided articles. Total count: 0"
    """
# ***************************************************************
# END OF MODIFIED PROMPT FUNCTIONS
# ***************************************************************


@app.get("/get-articles")
async def get_articles():
    try:
        # First, get articles with journal information
        articles_result = supabase.table('articles') \
            .select('id, title, article_url, pdf_url, published_date, created_at, journals(name)') \
            .order('published_date', desc=True) \
            .limit(500) \
            .execute()

        if not articles_result.data:
            return []

        # Get article types
        article_types_result = supabase.table('article_types').select('*').execute()
        article_types_dict = {at['id']: at['name'] for at in article_types_result.data} if article_types_result.data else {}

        articles_with_authors = []
        for article in articles_result.data:
            # Get authors for each article
            authors_result = supabase.table('article_authors') \
                .select('authors(name)') \
                .eq('article_id', article['id']) \
                .execute()

            # Use published_date if available, otherwise use created_at
            display_date = article['published_date'] if article['published_date'] else article['created_at']
            
            # Format the display_date properly
            if display_date:
                if isinstance(display_date, str):
                    # Parse the string date and format it consistently
                    try:
                        parsed_date = datetime.fromisoformat(display_date.replace('Z', '+00:00'))
                        formatted_date = parsed_date.strftime('%Y-%m-%d')
                    except:
                        formatted_date = display_date
                else:
                    formatted_date = display_date
            else:
                formatted_date = None
            
            # Format the article data
            article_data = {
                'id': article['id'],
                'title': article['title'],
                'article_url': article['article_url'],
                'pdf_url': article['pdf_url'],
                'published_date': article['published_date'],
                'created_at': article['created_at'],
                'display_date': formatted_date,  # Use formatted date
                'journal': article['journals']['name'] if article.get('journals') else 'Unknown Journal',
                'authors': [a['authors']['name'] for a in authors_result.data] if authors_result.data else []
            }
            articles_with_authors.append(article_data)

        return articles_with_authors

    except Exception as e:
        print(f"Error in get_articles: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/analyze-research")
async def analyze_research(request: QueryRequest):
    try:
        question = request.question.strip()
        query_type = detect_query_type(question)
        
        if query_type == 'metadata':
            metadata_results = execute_metadata_query(question)
            
            if not metadata_results:
                return {
                    "status": "success",
                    "answer": "No articles found matching your criteria.",
                    "sources": [],
                    "query_type": "metadata",
                    "total_count": 0
                }
            
            # Use the refined metadata prompt
            llm_response = together_llm.generate(build_metadata_llm_prompt(question, metadata_results))
            
            return {
                "status": "success",
                "answer": llm_response,
                "sources": metadata_results,
                "query_type": "metadata",
                "total_count": len(metadata_results)
            }
        
        elif query_type == 'hybrid':
            filters = extract_topic_and_filters(question)
            hybrid_results, total_count = get_hybrid_results(question) # Make sure get_hybrid_results returns two values
            
            if not hybrid_results:
                return {
                    "status": "success",
                    "answer": f"No articles found matching your criteria. Total count: 0",
                    "sources": [],
                    "query_type": "hybrid",
                    "total_count": 0,
                    "filters": filters
                }
            
            # Use the refined hybrid prompt
            llm_response = together_llm.generate(build_hybrid_llm_prompt(question, hybrid_results, total_count, filters))
            
            return {
                "status": "success",
                "answer": llm_response,
                "sources": hybrid_results,
                "query_type": "hybrid",
                "total_count": total_count,
                "filters": filters
            }
        
        else:  # semantic
            semantic_results = get_semantic_results(question)
            if not semantic_results:
                return {
                    "status": "success",
                    "answer": "No relevant articles found on this topic.",
                    "sources": [],
                    "query_type": "semantic",
                    "total_count": 0
                }

            # Use the new semantic prompt
            llm_response = together_llm.generate(build_semantic_llm_prompt(question, semantic_results))
            
            # Add a check for cases where LLM might still say no info
            if "i couldn't find enough information" in llm_response.lower() or \
               "no relevant information found" in llm_response.lower():
                return {
                    "status": "success",
                    "answer": "No relevant articles found on this topic.",
                    "sources": [], # Optionally, return semantic_results here if you still want to show them
                    "query_type": "semantic",
                    "total_count": 0 # Or len(semantic_results) if you return them
                }
            
            return {
                "status": "success",
                "answer": llm_response,
                "sources": semantic_results,
                "query_type": "semantic",
                "total_count": len(semantic_results)  # For semantic, this is limited by match_count
            }

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in analyze_research: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)