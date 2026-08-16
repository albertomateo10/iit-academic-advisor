# IIT Academic Advising Agent

A proactive academic advising chatbot for the Illinois Institute of Technology, built with LangGraph, Claude, and hybrid retrieval over Elasticsearch.

Unlike a standard RAG chatbot, this system does not let the language model decide course eligibility on its own. A deterministic rule engine parses each course's AND/OR prerequisite logic and evaluates it in Python, so eligibility verdicts come from verifiable rules rather than model inference. The agent also asks clarifying questions when it lacks the context needed to answer safely.

Developed as a Master's Thesis (TFM) for a double degree program: the European Master in Software Engineering (EMSEP) at Universidad Politécnica de Madrid (UPM), combined with a Master in Artificial Intelligence at Illinois Institute of Technology (IIT).

---

## Features

- **Hybrid retrieval** combining BM25 keyword search with dense vector search (kNN) over Elasticsearch, so both exact course codes and natural-language paraphrases are matched reliably.
- **Deterministic prerequisite engine** that parses raw prerequisite text into a boolean tree and evaluates nested AND/OR conditions in pure Python.
- **Proactive clarification** — the agent asks for missing information (degree program, completed courses) instead of guessing.
- **Stateful conversations** via LangGraph, tracking completed courses across turns.
- **Automated data pipeline** that scrapes the IIT catalog and re-indexes it from scratch.
- **Three separate indices** for courses, department policies, and degree programs, each with its own mapping and filtering strategy.

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│  Data Layer          scraping/  →  data/scraped/*.json │
├─────────────────────────────────────────────────────┤
│  Knowledge Base      elastic_ingestion.py  →  Elasticsearch │
│                      (BGE embeddings + semantic chunking)   │
├─────────────────────────────────────────────────────┤
│  Agent Layer         agents/  (graph, retrievers,      │
│                      tools, state, rule engine)        │
├─────────────────────────────────────────────────────┤
│  Interface           app/app.py  (Streamlit)           │
└─────────────────────────────────────────────────────┘
```

**Stack:** Python · LangChain · LangGraph · Claude Haiku 4.5 · Elasticsearch 8.12 · BAAI/bge-large-en-v1.5 · Crawl4AI · Streamlit · Docker

---

## Prerequisites

- Python 3.10 or higher
- Docker and Docker Compose
- An Anthropic API key ([console.anthropic.com](https://console.anthropic.com))
- At least 8 GB of available RAM (Elasticsearch is configured with a 4 GB heap)

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Crawl4AI requires a one-time browser setup:

```bash
crawl4ai-setup
```

### 4. Configure environment variables

Create a `.env` file in the project root:

```env
ANTHROPIC_API_KEY=your_anthropic_api_key_here

ELASTIC_HOST=http://localhost:9200
ELASTIC_USER=elastic
ELASTIC_PASSWORD=your_chosen_password
```

The `ELASTIC_PASSWORD` value is read by both the application and Docker Compose, so it must be identical in both places.

---

## Running the system

The pipeline runs in four stages. Stages 1 to 3 only need to be repeated when the IIT catalog changes, typically once per semester.

### Stage 1 — Start Elasticsearch

```bash
docker compose up -d
```

Verify it is running:

```bash
curl -u elastic:your_chosen_password http://localhost:9200
```

Elasticsearch data persists in a named Docker volume, so it survives container restarts. To stop the container:

```bash
docker compose down
```

### Stage 2 — Scrape the IIT catalog

Run the three scrapers. Each one visits the IIT catalog with randomized delays between requests to avoid rate limiting, so expect them to take several minutes.

```bash
python scraping/courses_scraping.py     # ~40 departments, course-level data
python scraping/policies_scraping.py    # department policies (3 departments)
python scraping/programs_scraping.py    # degree programs and certificates
```

Output is written to `data/scraped/`.

> **Note:** the policy and program scrapers write to `all_policies.json` and `dynamic_programs.json`. The ingestion script expects `iit_policies.json` and `iit_programs.json`, so rename them before continuing:
>
> ```bash
> mv data/scraped/all_policies.json data/scraped/iit_policies.json
> mv data/scraped/dynamic_programs.json data/scraped/iit_programs.json
> ```

### Stage 3 — Build the knowledge base

```bash
python elastic_ingestion.py
```

This step embeds every document with `BAAI/bge-large-en-v1.5` and indexes it into Elasticsearch. On the first run it downloads the embedding model (roughly 1.3 GB).

The script **deletes and recreates all three indices** on every run, so a full ingestion always starts from a clean state.

Three indices are created:

| Index | Contents | Chunking |
|---|---|---|
| `iit_courses` | Course codes, credits, descriptions, prerequisites | None (descriptions indexed whole) |
| `iit_policies` | Department-level academic policies | Semantic chunking |
| `iit_programs` | Degree and certificate requirements | Semantic chunking + context header |

### Stage 4 — Launch the application

```bash
streamlit run app/app.py
```

The interface opens at `http://localhost:8501`.

---

## Usage

Ask the agent about courses, prerequisites, policies, or degree requirements. Some examples:

- `What are the prerequisites for ITMS 548 Cyber Security Technologies?`
- `I have already completed CS 411. Can I take CS 511 next semester?`
- `Could you list the core and elective requirements for the Master of Cyber Forensics and Security?`
- `I am enrolled in the Master of Cybersecurity and have completed CS 458. Which course should I select next semester?`

The agent tracks the courses you mention as completed during the conversation and uses them for eligibility checks. This context lives only in the browser session and is not persisted to any database.

---

## Project structure

```
.
├── agents/
│   ├── graph.py               # LangGraph workflow, system prompt, model config
│   ├── retrievers.py          # Hybrid search + deterministic prerequisite engine
│   ├── state.py               # AcademicAdvisorState (TypedDict)
│   └── tools.py               # The four tools exposed to the LLM
├── app/
│   └── app.py                 # Streamlit interface
├── scraping/
│   ├── courses_scraping.py
│   ├── policies_scraping.py
│   └── programs_scraping.py
├── mappings/
│   ├── course_mapping.json
│   ├── policy_mapping.json
│   └── program_mapping.json
├── data/
│   └── scraped/               # Scraper output (JSON)
├── elastic_ingestion.py       # Embedding + indexing pipeline
├── docker-compose.yml         # Elasticsearch service
├── requirements.txt
└── .env                       # Not committed
```

---

## How eligibility checking works

This is the part that distinguishes the system from a standard RAG chatbot.

When a student asks whether they can enroll in a course, the language model does **not** read the prerequisite text and decide for itself. Instead it calls `check_course_eligibility_tool`, which:

1. Fetches the course's raw prerequisite string from Elasticsearch.
2. Normalizes it, stripping grade qualifiers and boilerplate.
3. Parses it recursively into a boolean tree, handling nested parentheses and mixed AND/OR conditions.
4. Evaluates each leaf against the student's completed courses.
5. Returns an `ELIGIBLE` / `NOT ELIGIBLE` verdict along with the exact reason.

For example, `(CS 445 with min. grade of C or CS 487) and MATH 474` becomes an `AND` node with an `OR` subtree, evaluated exactly rather than interpreted.

The system prompt explicitly forbids the model from overriding this verdict with its own reading of the prerequisite text.

---

## Troubleshooting

**`ConnectionError: Cannot connect to Elasticsearch`**
Check the container is running with `docker ps`. Elasticsearch takes 30 to 60 seconds to become responsive after startup.

**Elasticsearch container exits immediately**
Usually a memory issue. Either raise Docker's memory limit or lower the heap size in `docker-compose.yml` (`ES_JAVA_OPTS=-Xms2g -Xmx2g`).

**`FileNotFoundError` during ingestion**
The scrapers have not been run yet, or the JSON files still carry their original names. See the rename note in Stage 2.

**Scraper returns no results**
The IIT catalog's HTML structure may have changed. The scrapers depend on specific CSS selectors, which need updating if the site is redesigned.

**Agent replies that it cannot find information that clearly exists**
Confirm ingestion completed successfully and that all three indices are populated:

```bash
curl -u elastic:your_password http://localhost:9200/_cat/indices?v
```

---

## Limitations

- Scope is limited to three IIT departments: Information Technology Management, Computer Science, and Applied Mathematics.
- Policy and program answers include a source URL, but course answers do not, since the scraped course data has no per-course URL field.
- Conversation state is not persisted between sessions.
- The scrapers depend on the current CSS structure of the IIT catalog and will need maintenance if the site changes.

---

## License

This project is licensed under the MIT License.

---

## Acknowledgements

Developed as a Master's Thesis (TFM) for a double degree master's program, the European Master in Software Engineering (UPM) and a Master in Artificial Intelligence (Illinois Institute of Technology).
