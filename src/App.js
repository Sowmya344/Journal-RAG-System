import React, { useState, useEffect } from "react";
import "./App.css";

function App() {
  const [userInput, setUserInput] = useState("");
  const [responses, setResponses] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [showTable, setShowTable] = useState(false);
  const [articlesData, setArticlesData] = useState([]);
  const [filteredArticlesData, setFilteredArticlesData] = useState([]);
  const [isLoadingArticles, setIsLoadingArticles] = useState(false);
  const [currentPage, setCurrentPage] = useState(1);
  const [totalArticles, setTotalArticles] = useState(0);
  const articlesPerPage = 10;

  // Filter states
  const [filters, setFilters] = useState({
    journal: "",
    articleType: "",
    yearFrom: "",
    yearTo: "",
    searchTitle: "",
    author: ""
  });

  const handleInputChange = (e) => setUserInput(e.target.value);

  const fetchArticles = async () => {
    setIsLoadingArticles(true);
    try {
      const res = await fetch("http://localhost:8000/get-articles", {
        method: "GET",
        headers: { 
          "Content-Type": "application/json",
          "Accept": "application/json"
        }
      });

      if (!res.ok) {
        throw new Error("Failed to fetch articles");
      }

      const data = await res.json();
      console.log("Fetched articles data:", data);
      
      if (data && data.articles && Array.isArray(data.articles)) {
        setArticlesData(data.articles);
        setFilteredArticlesData(data.articles);
        setTotalArticles(data.articles.length);
      } else {
        console.error("Invalid articles data structure:", data);
        setArticlesData([]);
        setFilteredArticlesData([]);
        setTotalArticles(0);
      }
    } catch (err) {
      console.error("Error fetching articles:", err);
      setArticlesData([]);
      setFilteredArticlesData([]);
      setTotalArticles(0);
    } finally {
      setIsLoadingArticles(false);
    }
  };

  // Filter articles based on current filters
  const applyFilters = () => {
    let filtered = [...articlesData];

    // Filter by journal
    if (filters.journal) {
      filtered = filtered.filter(article => 
        article.journal?.toLowerCase().includes(filters.journal.toLowerCase())
      );
    }

    // Filter by article type
    if (filters.articleType) {
      filtered = filtered.filter(article => 
        article.article_type?.toLowerCase().includes(filters.articleType.toLowerCase())
      );
    }

    // Filter by year range
    if (filters.yearFrom || filters.yearTo) {
      filtered = filtered.filter(article => {
        if (!article.published_date) return false;
        const year = new Date(article.published_date).getFullYear();
        const fromYear = filters.yearFrom ? parseInt(filters.yearFrom) : 0;
        const toYear = filters.yearTo ? parseInt(filters.yearTo) : 9999;
        return year >= fromYear && year <= toYear;
      });
    }

    // Filter by title search
    if (filters.searchTitle) {
      filtered = filtered.filter(article => 
        article.title?.toLowerCase().includes(filters.searchTitle.toLowerCase())
      );
    }

    // Filter by author
    if (filters.author) {
      filtered = filtered.filter(article => 
        article.authors?.some(author => 
          author.toLowerCase().includes(filters.author.toLowerCase())
        )
      );
    }

    setFilteredArticlesData(filtered);
    setCurrentPage(1); // Reset to first page when filters change
  };

  // Apply filters whenever filter values change
  useEffect(() => {
    applyFilters();
  }, [filters, articlesData]);

  const handleFilterChange = (filterName, value) => {
    setFilters(prev => ({
      ...prev,
      [filterName]: value
    }));
  };

  const clearFilters = () => {
    setFilters({
      journal: "",
      articleType: "",
      yearFrom: "",
      yearTo: "",
      searchTitle: "",
      author: ""
    });
  };

  // Get unique values for dropdown filters
  const getUniqueJournals = () => {
    const journals = articlesData
      .map(article => article.journal)
      .filter(journal => journal && journal.trim())
      .filter((value, index, self) => self.indexOf(value) === index)
      .sort();
    return journals;
  };

  const getUniqueArticleTypes = () => {
    const types = articlesData
      .map(article => article.article_type)
      .filter(type => type && type.trim())
      .filter((value, index, self) => self.indexOf(value) === index)
      .sort();
    return types;
  };

  useEffect(() => {
    if (showTable) {
      fetchArticles();
    }
  }, [showTable]);

  const handleSubmit = async () => {
    if (!userInput.trim()) return;

    setIsLoading(true);
    const newResponse = {
      question: userInput,
      answer: "Thinking...",
      sources: [],
      queryType: null,
    };
    setResponses(prev => [...prev, newResponse]);

    try {
      const res = await fetch("http://localhost:8000/analyze-research", {
        method: "POST",
        headers: { 
          "Content-Type": "application/json",
          "Accept": "application/json"
        },
        body: JSON.stringify({ 
          question: userInput,
          user_context: {}
        })
      });

      if (!res.ok) {
        const errorData = await res.json();
        throw new Error(errorData.error || "API request failed");
      }

      const data = await res.json();

      const sortedSources = data.sources ? 
        [...data.sources].sort((a, b) => (b.similarity || 0) - (a.similarity || 0)) : [];

      setResponses(prev =>
        prev.map((item, index) =>
          index === prev.length - 1
            ? {
                ...item,
                answer: data.answer || "No answer provided",
                sources: sortedSources,
                queryType: data.query_type,
                status: data.status
              }
            : item
        )
      );
    } catch (err) {
      console.error("Error:", err);
      setResponses(prev =>
        prev.map((item, index) =>
          index === prev.length - 1
            ? { 
                ...item, 
                answer: `Error: ${err.message || "Failed to retrieve information"}`,
                status: "error"
              }
            : item
        )
      );
    } finally {
      setIsLoading(false);
      setUserInput("");
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === "Enter") handleSubmit();
  };

  const handleSuggestionClick = (suggestion) => {
    setUserInput(suggestion);
    handleSubmit();
  };

  const formatDate = (dateStr) => {
    if (!dateStr) return "Date not available";
    try {
      const date = new Date(dateStr);
      return date.toLocaleDateString();
    } catch {
      return dateStr;
    }
  };

  const formatStructuredAnswer = (answer) => {
    if (!answer) return [];
    
    const sections = answer.split(/\n### /).filter(section => section.trim());
    
    return sections.map((section, index) => {
      const [title, ...content] = section.split('\n').filter(line => line.trim());
      const contentStr = content.join('\n');
      
      if (title.includes('Article Descriptions')) {
        const articles = contentStr.split('#### ').filter(article => article.trim());
        return {
          title,
          articles: articles.map(article => {
            const [articleTitle, ...articleContent] = article.split('\n').filter(line => line.trim());
            return {
              title: articleTitle,
              content: articleContent.join('\n')
            };
          })
        };
      }
      
      return {
        title: index === 0 ? title : `### ${title}`,
        content: contentStr
      };
    });
  };

  const safeArticlesData = Array.isArray(filteredArticlesData) ? filteredArticlesData : [];
  const indexOfLastArticle = currentPage * articlesPerPage;
  const indexOfFirstArticle = indexOfLastArticle - articlesPerPage;
  const currentArticles = safeArticlesData.slice(indexOfFirstArticle, indexOfLastArticle);
  const totalPages = Math.ceil(safeArticlesData.length / articlesPerPage);

  const paginate = (pageNumber) => setCurrentPage(pageNumber);

  const suggestions = [
    "Articles about diabetes treatments",
    "Latest research on COVID-19 vaccines",
    "Papers by John Smith published in 2020",
    "How does exercise affect heart disease?",
    "Clinical studies on Alzheimer's disease",
    "What are the effects of aspirin?",
    "Articles from Nature journal",
    "Research on cancer immunotherapy"
  ];

  return (
    <div className="app-container">
      {/* Header */}
      <header className="header">
        <div className="header-content">
          <div className="header-main">
            <div className="logo">
              <span className="logo-icon">🔬</span>
              <h1 className="title">Medical Research Assistant</h1>
            </div>
            <p className="subtitle">Find answers from medical research papers</p>
          </div>
          <button 
            className="database-toggle"
            onClick={() => setShowTable(!showTable)}
          >
            <span className="database-icon">📊</span>
            <span>Database</span>
          </button>
        </div>
      </header>

      {/* Search Section */}
      <section className="search-section">
        <div className="search-container">
          <div className="search-box">
            <input
              type="text"
              value={userInput}
              onChange={handleInputChange}
              onKeyPress={handleKeyPress}
              placeholder="Ask a medical research question..."
              disabled={isLoading}
              className="search-input"
            />
            <button
              onClick={handleSubmit}
              disabled={isLoading || !userInput.trim()}
              className="search-button"
            >
              {isLoading ? (
                <span className="loading-spinner"></span>
              ) : (
                <span>🔍</span>
              )}
            </button>
          </div>
        </div>

        {/* Suggestions */}
        <div className="suggestions-container">
          <p className="suggestions-label">Try these examples:</p>
          <div className="suggestions-grid">
            {suggestions.map((suggestion, index) => (
              <button
                key={index}
                onClick={() => handleSuggestionClick(suggestion)}
                className="suggestion-chip"
                disabled={isLoading}
              >
                {suggestion}
              </button>
            ))}
          </div>
        </div>
      </section>

      {/* Database Modal */}
      {showTable && (
        <div className="modal-overlay">
          <div className="modal-container">
            <div className="modal-header">
              <h2 className="modal-title">
                <span className="modal-icon">📊</span>
                Articles Database
                <span className="record-count">
                  {safeArticlesData.length} of {articlesData.length} records
                </span>
              </h2>
              <button 
                className="modal-close"
                onClick={() => setShowTable(false)}
              >
                ✕
              </button>
            </div>
            
            <div className="modal-body">
              {/* Filters Panel */}
              <div className="filters-panel">
                <div className="filters-header">
                  <h3>
                    <span className="filter-icon">🔍</span>
                    Filters
                  </h3>
                  <button 
                    className="clear-filters"
                    onClick={clearFilters}
                  >
                    Clear All
                  </button>
                </div>
                
                <div className="filters-grid">
                  <div className="filter-group">
                    <label className="filter-label">Search Title</label>
                    <input
                      type="text"
                      className="filter-input"
                      placeholder="Search in titles..."
                      value={filters.searchTitle}
                      onChange={(e) => handleFilterChange('searchTitle', e.target.value)}
                    />
                  </div>

                  <div className="filter-group">
                    <label className="filter-label">Author</label>
                    <input
                      type="text"
                      className="filter-input"
                      placeholder="Author name..."
                      value={filters.author}
                      onChange={(e) => handleFilterChange('author', e.target.value)}
                    />
                  </div>

                  <div className="filter-group">
                    <label className="filter-label">Journal</label>
                    <select
                      className="filter-select"
                      value={filters.journal}
                      onChange={(e) => handleFilterChange('journal', e.target.value)}
                    >
                      <option value="">All Journals</option>
                      {getUniqueJournals().map(journal => (
                        <option key={journal} value={journal}>{journal}</option>
                      ))}
                    </select>
                  </div>

                  <div className="filter-group">
                    <label className="filter-label">Article Type</label>
                    <select
                      className="filter-select"
                      value={filters.articleType}
                      onChange={(e) => handleFilterChange('articleType', e.target.value)}
                    >
                      <option value="">All Types</option>
                      {getUniqueArticleTypes().map(type => (
                        <option key={type} value={type}>{type}</option>
                      ))}
                    </select>
                  </div>

                  <div className="filter-group year-filter">
                    <label className="filter-label">Publication Year</label>
                    <div className="year-inputs">
                      <input
                        type="number"
                        className="filter-input"
                        placeholder="From"
                        min="1900"
                        max="2024"
                        value={filters.yearFrom}
                        onChange={(e) => handleFilterChange('yearFrom', e.target.value)}
                      />
                      <span className="year-separator">to</span>
                      <input
                        type="number"
                        className="filter-input"
                        placeholder="To"
                        min="1900"
                        max="2024"
                        value={filters.yearTo}
                        onChange={(e) => handleFilterChange('yearTo', e.target.value)}
                      />
                    </div>
                  </div>
                </div>

                {/* Active Filters */}
                {Object.entries(filters).filter(([key, value]) => value).length > 0 && (
                  <div className="active-filters">
                    <h4>Active Filters:</h4>
                    <div className="filter-tags">
                      {Object.entries(filters).map(([key, value]) => 
                        value && (
                          <span key={key} className="filter-tag">
                            {key}: {value}
                            <button 
                              className="remove-filter"
                              onClick={() => handleFilterChange(key, '')}
                            >
                              ×
                            </button>
                          </span>
                        )
                      )}
                    </div>
                  </div>
                )}
              </div>

              {/* Table Content */}
              <div className="table-section">
                {isLoadingArticles ? (
                  <div className="loading-state">
                    <span className="loading-spinner large"></span>
                    <p>Loading articles...</p>
                  </div>
                ) : safeArticlesData.length === 0 ? (
                  <div className="empty-state">
                    <span className="empty-icon">📄</span>
                    <p>No articles found matching your filters.</p>
                    <button className="clear-filters" onClick={clearFilters}>
                      Clear Filters
                    </button>
                  </div>
                ) : (
                  <>
                    <div className="table-container">
                      <table className="articles-table">
                        <thead>
                          <tr>
                            <th>ID</th>
                            <th>Title</th>
                            <th>Journal</th>
                            <th>Type</th>
                            <th>Published</th>
                            <th>Links</th>
                          </tr>
                        </thead>
                        <tbody>
                          {currentArticles.map((article, index) => (
                            <tr key={article.id || index}>
                              <td className="id-cell">{article.id}</td>
                              <td className="title-cell">
                                {article.title || 'Untitled'}
                              </td>
                              <td className="journal-cell">{article.journal || 'Unknown'}</td>
                              <td className="type-cell">{article.article_type || 'N/A'}</td>
                              <td className="date-cell">{formatDate(article.published_date)}</td>
                              <td className="links-cell">
                                <div className="link-buttons">
                                  {article.pdf_url && (
                                    <a 
                                      href={article.pdf_url} 
                                      target="_blank" 
                                      rel="noopener noreferrer"
                                      className="link-btn pdf-btn"
                                    >
                                      📄 PDF
                                    </a>
                                  )}
                                  {article.article_url && (
                                    <a 
                                      href={article.article_url} 
                                      target="_blank" 
                                      rel="noopener noreferrer"
                                      className="link-btn article-btn"
                                    >
                                      🔗 Article
                                    </a>
                                  )}
                                </div>
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                    
                    {/* Pagination */}
                    {totalPages > 1 && (
                      <div className="pagination">
                        <button
                          onClick={() => paginate(Math.max(1, currentPage - 1))}
                          disabled={currentPage === 1}
                          className="pagination-btn"
                        >
                          Previous
                        </button>
                        
                        <div className="page-numbers">
                          {Array.from({ length: Math.min(5, totalPages) }, (_, i) => {
                            let pageNum;
                            if (totalPages <= 5) {
                              pageNum = i + 1;
                            } else if (currentPage <= 3) {
                              pageNum = i + 1;
                            } else if (currentPage >= totalPages - 2) {
                              pageNum = totalPages - 4 + i;
                            } else {
                              pageNum = currentPage - 2 + i;
                            }
                            
                            return (
                              <button
                                key={pageNum}
                                onClick={() => paginate(pageNum)}
                                className={`page-btn ${pageNum === currentPage ? 'active' : ''}`}
                              >
                                {pageNum}
                              </button>
                            );
                          })}
                        </div>
                        
                        <button
                          onClick={() => paginate(Math.min(totalPages, currentPage + 1))}
                          disabled={currentPage === totalPages}
                          className="pagination-btn"
                        >
                          Next
                        </button>
                      </div>
                    )}
                  </>
                )}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Results */}
      <main className="results-section">
        {responses.map((response, index) => (
          <article key={index} className="result-card">
            <div className="result-header">
              <h3 className="result-question">
                <span className="question-icon">❓</span>
                {response.question}
              </h3>
              {response.queryType && (
                <span className={`query-badge ${response.queryType}`}>
                  {response.queryType === 'metadata' ? 'Database Query' : 'Semantic Search'}
                </span>
              )}
            </div>

            {/* Answer */}
            <div className="answer-section">
              <h4 className="section-title">
                <span className="section-icon">📋</span>
                Research Summary
              </h4>
              <div className="answer-content">
                {formatStructuredAnswer(response.answer).map((section, sIndex) => (
                  <div key={sIndex} className="content-section">
                    {section.title && (
                      <h5 className="content-title">
                        {section.title.replace('###', '').trim()}
                      </h5>
                    )}
                    
                    {section.title && section.title.includes('Comprehensive Review') && (
                      <div className="highlight-content">
                        {section.content.split('\n').filter(p => p.trim()).map((paragraph, pIndex) => (
                          <p key={pIndex} className="content-paragraph">
                            {paragraph}
                          </p>
                        ))}
                      </div>
                    )}
                    
                    {section.articles && (
                      <div className="articles-grid">
                        {section.articles.map((article, aIndex) => (
                          <div key={aIndex} className="article-card">
                            <h6 className="article-title">{article.title}</h6>
                            <div className="article-details">
                              {article.content
                                .replace(/\*+/g, '')
                                .split('\n')
                                .filter(line => line.trim())
                                .map((line, lIndex) => {
                                  if (line.includes(':')) {
                                    const [label, ...valueParts] = line.split(':');
                                    const value = valueParts.join(':').trim();
                                    return (
                                      <div key={lIndex} className="article-detail">
                                        <strong>{label.trim()}:</strong>
                                        <span>{value}</span>
                                      </div>
                                    );
                                  }
                                  return (
                                    <div key={lIndex} className="article-text">
                                      {line.trim()}
                                    </div>
                                  );
                                })}
                            </div>
                          </div>
                        ))}
                      </div>
                    )}
                    
                    {!section.title?.includes('Comprehensive Review') && 
                     !section.title?.includes('Article Descriptions') && (
                      section.content
                        .replace(/\*+/g, '')
                        .split('\n')
                        .filter(p => p.trim())
                        .map((paragraph, pIndex) => (
                          <p key={pIndex} className="content-paragraph">
                            {paragraph}
                          </p>
                        ))
                    )}
                  </div>
                ))}
              </div>
            </div>

            {/* Sources */}
            {response.sources?.length > 0 && (
              <div className="sources-section">
                <h4 className="section-title">
                  <span className="section-icon">📚</span>
                  {response.queryType === 'metadata' ? 'Matching Articles' : 'Relevant Sources'}
                  <span className="source-count">({response.sources.length} results)</span>
                  {response.sources[0]?.similarity && (
                    <span className="similarity-note">
                      - Sorted by relevance
                    </span>
                  )}
                </h4>
                <div className="sources-grid">
                  {response.sources.map((source, i) => (
                    <div key={i} className="source-card">
                      <div className="source-header">
                        <h5 className="source-title">{source.title || `Source ${i + 1}`}</h5>
                        {source.similarity && (
                          <div className="similarity-badge">
                            {(source.similarity * 100).toFixed(1)}% match
                          </div>
                        )}
                      </div>
                      
                      {source.authors?.length > 0 && (
                        <div className="source-authors">
                          <strong>Authors:</strong> {source.authors.join(', ')}
                        </div>
                      )}
                      
                      <div className="source-meta">
                        {source.journal && (
                          <span className="meta-item">
                            <strong>Journal:</strong> {source.journal}
                          </span>
                        )}
                        {source.published_date && (
                          <span className="meta-item">
                            <strong>Published:</strong> {formatDate(source.published_date)}
                          </span>
                        )}
                      </div>
                      
                      <div className="source-actions">
                        {source.pdf_url && (
                          <a
                            href={source.pdf_url}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="source-link pdf-link"
                          >
                            📄 View PDF
                          </a>
                        )}
                        {source.article_url && source.article_url !== source.pdf_url && (
                          <a
                            href={source.article_url}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="source-link article-link"
                          >
                            🔗 View Article
                          </a>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {response.status === "no_results" && (
              <div className="no-results">
                <span className="no-results-icon">🔍</span>
                <p>No relevant articles found matching your query.</p>
                <p>Try rephrasing your question or using different keywords.</p>
              </div>
            )}
          </article>
        ))}
      </main>
    </div>
  );
}

export default App;