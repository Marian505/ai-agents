# LangChain examples project
This project is demonstration  of AI agents developed by LangChain framework.


## How to run
All comannds run from project root folder.

Porject prerequisite:
- python 3.11
- UV package manager

**Install dependecies**
```bash
uv sync
```

**Local run:**
```bash
langgraph dev
```

**Local run from docker or kubernetes:**
```bash
langgraph build -t image-name:latest .
docker compose up -d
```

**Tests run:**
```bash
pytest tests/unit_tests
pytest tests/integration_tests
pytest tests/integration_tests/test_lc_agent_eval.py::test_trajectory_quality -s -v
```

**Local running application is accessible from:**
- 🚀 API: http://127.0.0.1:2024
- 🎨 Studio UI: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
- 📚 API Docs: http://127.0.0.1:2024/docs

https://smith.langchain.com/studio/?baseUrl=http://192.168.100.98:8000

# Agents

## basic_agent
Simple chat agent with TavilySearch.

## cv_agent
Agent who can analyze CV in PDF, chat about it and call tool to generate new CV in PDF store it to file.

## deep_agent
Example of deep agent implementation with supagent. Subagent have web search tool. 

## lc_agent
Similar to cv_agent.
Agent who can analyze CV in PDF. Enhance propt in moddleweare function. Generate new CV in PDF. Agent have file management tools.

## lg_agent
Simple reactive AI agent implemented via LangGraph able to chat.

## lg_workflow
Prompt enhancer with multiple models implemented via LangGraph workflow.

## rag_agent
Add description

## simple_rag_agent
Add description

# Tests 
Add description

# uv 
uv python list
uv python pin 3.14
uv python install 3.14