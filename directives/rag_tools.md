# RAG Tools Usage Directive

## Purpose
This directive defines when and how to use the RAG (Retrieval Augmented Generation) tools for knowledge retrieval and context expansion.

## Architecture: Session ID Management

```
Prompt Optimiser Agent (parent)
    │
    ├── Creates session_id for Sub-Agent A ──→ passes to RAG tools
    ├── Creates session_id for Sub-Agent B ──→ passes to RAG tools
    └── Creates session_id for Sub-Agent C ──→ passes to RAG tools
```

- The **Prompt Optimiser agent** creates unique session IDs
- Each **sub-agent** receives its own session_id
- Sub-agents pass their session_id to RAG tool calls
- Session IDs track conversation context per sub-agent

## Architecture: Long-Term Memory (Zep)

```
Prompt Optimiser Agent
    │
    ├── Receives user query
    ├── Reads user_id from .env (ZEP_USER_ID)
    │
    ▼
Get Long Term Memories (Zep API)
    │
    ├── POST https://api.getzep.com/api/v2/graph/search
    ├── Returns temporal memories for user
    │
    ▼
Sub Agent
    │
    ├── Receives: user query + temporal memories + session_id
    ├── Executes using RAG tools
    └── Returns result to Prompt Optimiser Agent
```

- **Zep** provides temporal/long-term memory via graph search
- The **Prompt Optimiser agent** calls Zep **before** spawning sub-agents
- Retrieved memories are passed to sub-agents as additional context
- This enriches sub-agent responses with user-specific historical knowledge

## Endpoint Routing

The hybrid search and knowledge graph tools share the **same webhook URL**:
```
${N8N_WEBHOOK_BASE_URL}/231197f8-d72a-42da-8ddd-9ab0e19c5f8a
```

The `type` parameter routes the request to the correct workflow path:
| type value | Route |
|------------|-------|
| `"hybrid"` | Hybrid search (vector + lexical + ilike + fuzzy) |
| `"graph"` | Knowledge graph (entity relationships) |

## Available Tools

### 1. Dynamic Hybrid Search
**When to use:** Primary tool for querying the knowledge base with natural language queries.

**Configuration:**
- Tool config: `config/tools.yaml` → `dynamic_hybrid_search`
- Endpoint: n8n webhook at `${N8N_WEBHOOK_BASE_URL}/231197f8-d72a-42da-8ddd-9ab0e19c5f8a`

**Weight Guidelines:**

| Parameter | Default | Purpose |
|-----------|---------|---------|
| dense_weight | 0.5 | Vector/semantic similarity |
| sparse_weight | 0.5 | Lexical/keyword matching |
| ilike_weight | 0 | Exact wildcard matching |
| fuzzy_weight | 0 | Typo-tolerant matching |
| fuzzy_threshold | 0.8 | Fuzzy match strictness (higher = stricter, less latency) |

**When to use each weight:**

| Query Type | dense | sparse | ilike | fuzzy | Example |
|------------|-------|--------|-------|-------|---------|
| Semantic/natural language | 0.7 | 0.3 | 0 | 0 | "what AI projects has pierre built" |
| Technical terms/keywords | 0.3 | 0.7 | 0 | 0 | "RAG chunking strategies" |
| Exact ID/code lookup | 0 | 0.2 | 0.8 | 0 | "DOC-12345" |
| Typo recovery | 0.3 | 0.3 | 0 | 0.4 | "piere galet" (misspelled) |
| Balanced (default) | 0.5 | 0.5 | 0 | 0 | General queries |

**Important:**
- Weights must sum to 1.0
- For `ilike` and `fuzzy` searches, keep the query **short and focused** (e.g., exact ID or code) - long queries will likely return zero results
- `ilike` and `fuzzy` matching add latency - default to 0 unless specifically needed
- `fuzzy_threshold` adds significant latency - keep as high as possible (default 0.8)

---

### 2. Query Knowledge Graph
**When to use:** For entity relationship queries, finding connections between concepts.

**Configuration:**
- Tool config: `config/tools.yaml` → `query_knowledge_graph`
- Endpoint: Same n8n webhook as hybrid search
- Type: `"graph"` (routes to knowledge graph workflow path)

**Required parameters:**
| Parameter | Value | Notes |
|-----------|-------|-------|
| type | `"graph"` | Fixed - routes to graph path |
| dense_weight | `0` | Must be 0 for graph queries |
| sparse_weight | `0` | Must be 0 for graph queries |
| ilike_weight | `0` | Must be 0 for graph queries |
| fuzzy_weight | `0` | Must be 0 for graph queries |
| fuzzy_threshold | `0` | Must be 0 for graph queries |

**Use cases:**
- "What entities are related to X?"
- "Find connections between A and B"
- "Show the relationship network for concept C"

---

### 3. Context Expansion
**When to use:** After hybrid search returns chunks, use this to get surrounding context.

**Configuration:**
- Tool config: `config/tools.yaml` → `context_expansion`
- Endpoint: Supabase Edge Function

**Input format:**
```json
[
  {
    "doc_id": "doc-id-12345-abcde",
    "chunk_ranges": [[0, 5], [10, 15]]
  }
]
```

**Flow:**
1. Run hybrid search → get doc_ids and chunk indices
2. Call context_expansion with those doc_ids and desired chunk ranges
3. Receive expanded context with neighbouring chunks

---

### 4. Fetch Document Hierarchy
**When to use:** To understand document structure, get metadata, or navigate document tree.

**Configuration:**
- Tool config: `config/tools.yaml` → `fetch_document_hierarchy`
- Connection: Direct Supabase table query on `record_manager_v2`

**Use cases:**
- Get document metadata by doc_id
- Understand parent-child relationships between documents
- Retrieve document schema information

---

### 5. Query Tabular Rows
**When to use:** For structured data queries on tabular documents (CSV, Excel data).

**Configuration:**
- Tool config: `config/tools.yaml` → `query_tabular_rows`
- Connection: PostgreSQL direct query

**Important:**
- Always filter by `record_manager_id` to scope to specific document
- Use `row_data->>` operator to extract JSON field values
- First call `get_datasets_from_record_manager` to find available tables and their schemas

**Example query:**
```sql
SELECT row_data->>'column_name' as value
FROM tabular_document_rows
WHERE record_manager_id = 'uuid-here'
AND row_data->>'status' = 'active'
```

---

### 6. Short-Term Memory
**When to use:** To retrieve recent conversation context for multi-turn interactions.

**Configuration:**
- Tool config: `config/tools.yaml` → `short_term_memory`
- Connection: PostgreSQL query on `n8n_chat_histories`

**Parameters:**
- `session_id`: Required, identifies the conversation
- `limit`: Number of recent messages to retrieve (default: 10)

---

### 7. Get Datasets from Record Manager
**When to use:** Before querying tabular data, to discover available datasets and their schemas.

**Configuration:**
- Tool config: `config/tools.yaml` → `get_datasets_from_record_manager`
- Connection: PostgreSQL select on `record_manager_v2`

**Returns:** List of tabular documents with `id`, `document_title`, and `schema`

---

### 8. Get Long-Term Memories (Zep)
**When to use:** Called by Prompt Optimiser agent **before** spawning sub-agents to enrich context with user-specific temporal memories.

**Configuration:**
- Tool config: `config/tools.yaml` → `get_long_term_memories`
- Endpoint: Zep API at `${ZEP_API_URL}/api/v2/graph/search`

**Parameters:**

| Parameter | Type | Default | Source | Description |
|-----------|------|---------|--------|-------------|
| query | string | - | prompt_optimiser | User query to search memories for |
| user_id | string | - | env (`ZEP_USER_ID`) | User ID (e.g., `pierre_gallet_digital_twin`) |
| scope | string | `"edges"` | - | Memory scope to search |
| limit | integer | `5` | - | Max memories to return |
| min_relevance | number | `0.7` | - | Minimum relevance score filter |

**Flow:**
1. Prompt Optimiser receives user query
2. Calls `get_long_term_memories` with the query
3. Zep returns relevant temporal memories for the user
4. Memories are passed to sub-agent as additional context
5. Sub-agent executes with enriched context

**Use cases:**
- User asks about past conversations or preferences
- Query requires historical context about the user
- Personalizing responses based on user history

**Example request:**
```json
{
  "user_id": "pierre_gallet_digital_twin",
  "query": "what projects have I worked on",
  "scope": "edges",
  "limit": 5,
  "search_filters": {
    "min_relevance": 0.7
  }
}
```

---

## Decision Tree: Which Tool to Use

```
User Query (at Prompt Optimiser level)
    │
    ├── [FIRST] Need user-specific historical context?
    │   └── Use: get_long_term_memories (Zep)
    │       └── Pass memories to sub-agent
    │
    ▼
User Query (at Sub-Agent level)
    │
    ├── Natural language question about content?
    │   └── Use: dynamic_hybrid_search
    │           │
    │           └── Need more context around results?
    │               └── Use: context_expansion
    │
    ├── Question about entity relationships?
    │   └── Use: query_knowledge_graph
    │
    ├── Question about structured/tabular data?
    │   ├── Don't know what datasets exist?
    │   │   └── First: get_datasets_from_record_manager
    │   └── Know the dataset?
    │       └── Use: query_tabular_rows
    │
    ├── Need document metadata or structure?
    │   └── Use: fetch_document_hierarchy
    │
    └── Need conversation history?
        └── Use: short_term_memory
```

## Common Patterns

### Pattern 1: Deep Search
1. `dynamic_hybrid_search` with semantic weights
2. `context_expansion` on top results
3. Return combined context

### Pattern 2: Tabular Data Query
1. `get_datasets_from_record_manager` to find relevant dataset
2. Review schema in results
3. `query_tabular_rows` with appropriate SQL

### Pattern 3: Multi-Modal Search
1. `dynamic_hybrid_search` for unstructured content
2. `query_knowledge_graph` for relationships
3. Combine insights from both

## Error Handling

| Error | Likely Cause | Resolution |
|-------|--------------|------------|
| 401 Unauthorized | Invalid/missing API key | Check `SUPABASE_ANON_KEY` in `.env` |
| 404 Not Found | Wrong endpoint URL | Verify `N8N_WEBHOOK_BASE_URL` |
| Connection refused | Database not accessible | Check `POSTGRES_CONNECTION_STRING` |
| Empty results | Query too specific | Broaden search terms, adjust weights |

## Credentials Required

Before using these tools, ensure the following are set in `.env`:
- `SUPABASE_ANON_KEY` - Required for Supabase operations
- `POSTGRES_CONNECTION_STRING` - Required for direct DB queries
- `ZEP_API_KEY` - Required for Zep long-term memory operations
- `ZEP_USER_ID` - User ID for Zep graph (e.g., `pierre_gallet_digital_twin`)

## Learnings

- **2026-01-22:** The `dynamic_hybrid_search` endpoint requires the full parameter set to return results. Sending only `{"query": "..."}` returns a 500 error with "No item to return was found". Always include `type`, `dense_weight`, `sparse_weight`, `ilike_weight`, `fuzzy_weight`, and `fuzzy_threshold` parameters.
- **2026-01-22:** For `ilike` and `fuzzy` searches, queries must be extremely short and focused (e.g., an exact ID or code). Long natural language queries with these weights will return zero results. Default these weights to 0 unless specifically doing pattern/typo matching.
- **2026-01-22:** Hybrid search and knowledge graph share the same endpoint URL. The `type` parameter routes: `"hybrid"` for search, `"graph"` for knowledge graph. For graph queries, all weights and fuzzy_threshold must be 0.
- **2026-01-22:** Supabase uses pgbouncer in transaction mode, which requires `statement_cache_size=0` when connecting with asyncpg. Without this, you'll get `DuplicatePreparedStatementError`. Example: `await asyncpg.connect(connection_string, statement_cache_size=0)`
