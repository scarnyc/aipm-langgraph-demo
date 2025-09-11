# 🔬 LangGraph Multi-Agent Research System

A sophisticated AI research assistant powered by LangGraph, Claude Sonnet 4, and multiple specialized agents. This system can conduct comprehensive research using web search, Wikipedia, and intelligent agent coordination.

## 🏗️ System Architecture

```
                            ┌─────────────────┐
                            │   User Query    │
                            └─────────┬───────┘
                                      │
                            ┌─────────▼───────┐
                            │   Supervisor    │
                            │    Agent        │
                            └─────────┬───────┘
                                      │
              ┌───────────────────────┼───────────────────────┐
              │                       │                       │
    ┌─────────▼───────┐    ┌─────────▼───────┐    ┌─────────▼───────┐
    │  Planning       │    │    Search       │    │  Citation       │
    │   Agent         │    │    Agent        │    │   Agent         │
    └─────────────────┘    └─────────┬───────┘    └─────────────────┘
                                     │
                           ┌─────────▼───────┐
                           │   Tool Layer    │
                           │ • Tavily Search │
                           │ • Wikipedia     │
                           │ • DateTime      │
                           └─────────────────┘
              
    ┌─────────────────┐              ┌─────────────────┐
    │  Reflection     │              │   Synthesis     │
    │    Agent        │              │     Agent       │
    └─────────────────┘              └─────────────────┘
```

### 🤖 Agent Roles

1. **Supervisor Agent**: Orchestrates the entire research workflow and coordinates between agents
2. **Planning Agent**: Creates structured research plans and defines search strategies
3. **Search Agent**: Executes web searches using Tavily and Wikipedia APIs
4. **Citation Agent**: Validates sources and ensures proper citation formatting
5. **Reflection Agent**: Performs quality assurance and determines if more research is needed
6. **Synthesis Agent**: Creates comprehensive final reports with all findings

### 🛠️ Tool Layer

- **Tavily Search**: Real-time web search for current information
- **Wikipedia API**: Access to encyclopedic knowledge
- **DateTime Tool**: Current date/time context for time-sensitive queries

## 📁 Project Structure

```
langgraph-research-system/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── setup.py                     # Quick setup script
├── .env.example                # Environment variables template
├── .gitignore                  # Git ignore file
├── notebooks/
│   └── LangGraph_Demo.ipynb    # Interactive Jupyter notebook
├── src/
│   └── deep_research.py        # Multi-agent research system
└── docs/
    └── architecture.md         # Detailed architecture docs
```

## 🚀 Quick Start Guide

### For Python Beginners

#### 1. Install Python

**Windows:**
1. Download Python from [python.org](https://www.python.org/downloads/)
2. Run installer and **check "Add Python to PATH"**
3. Open Command Prompt and verify: `python --version`

**macOS:**
```bash
# Using Homebrew (recommended)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew install python

# Or download from python.org
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv
```

#### 2. Create Virtual Environment

A virtual environment keeps your project dependencies isolated:

```bash
# Create virtual environment
python -m venv langgraph_env

# Activate it
# Windows:
langgraph_env\Scripts\activate

# macOS/Linux:
source langgraph_env/bin/activate

# You should see (langgraph_env) in your terminal prompt
```

#### 3. Clone and Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd langgraph-research-system

# Install dependencies
pip install -r requirements.txt

# Run quick setup
python setup.py
```

#### 4. Install Jupyter (for notebook usage)

```bash
# Install Jupyter
pip install jupyter

# Launch Jupyter
jupyter notebook

# Navigate to notebooks/LangGraph_Demo.ipynb
```

### 🔧 Environment Configuration

1. Copy the environment template:
```bash
cp .env.example .env
```

2. Edit `.env` with your API keys:
```env
ANTHROPIC_API_KEY=your_anthropic_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

#### Getting API Keys

**Anthropic Claude API:**
1. Visit [console.anthropic.com](https://console.anthropic.com)
2. Sign up/login and go to API Keys
3. Create a new API key

**Tavily Search API:**
1. Visit [tavily.com](https://tavily.com)
2. Sign up for free tier
3. Get your API key from dashboard

## 🎯 Usage Examples

### 1. Interactive Notebook (Beginner-Friendly)

```bash
# Start Jupyter
jupyter notebook

# Open notebooks/LangGraph_Demo.ipynb
# Run cells step by step to see how each component works
```

### 2. Web Interface (Full Research System)

```bash
# Run the complete system
python src/deep_research.py

# Open browser to http://localhost:7860
# Enter your research question and get comprehensive results
```

### 3. Command Line Usage

```python
from src.deep_research import conduct_research

# Conduct research programmatically
result = conduct_research("What are the latest developments in quantum computing?")
print(result)
```

## 🔍 How It Works

### Step-by-Step Workflow

1. **Query Reception**: User submits a research question
2. **Planning Phase**: Planning agent creates a structured research strategy
3. **Search Execution**: Search agent uses tools to gather information
4. **Source Validation**: Citation agent verifies and formats sources
5. **Quality Check**: Reflection agent evaluates completeness
6. **Synthesis**: Final agent creates comprehensive report
7. **Result Delivery**: User receives formatted research report

### 🧠 Agent Communication

Agents communicate through LangGraph's message passing system:

```python
# Example message flow
messages = [
    HumanMessage(content="Research quantum computing trends"),
    AIMessage(content="Planning research strategy..."),
    ToolMessage(content="Search results from Tavily..."),
    AIMessage(content="Validated sources and citations..."),
    AIMessage(content="Final comprehensive report...")
]
```

### 🔄 Memory and State Management

- **Thread-based Memory**: Each conversation maintains context
- **State Persistence**: Research progress is tracked across agents
- **Error Recovery**: System handles API failures gracefully

## 📊 Features

### ✅ Current Capabilities

- **Multi-source Research**: Combines web search and encyclopedic knowledge
- **Source Validation**: Ensures credible, properly cited information
- **Quality Assurance**: Built-in fact-checking and completeness evaluation
- **Interactive Interface**: Both notebook and web UI options
- **Memory Persistence**: Maintains conversation context
- **Error Handling**: Robust error recovery and user feedback

### 🚧 Planned Enhancements

- [ ] PDF document ingestion
- [ ] Academic paper search integration
- [ ] Export to various formats (PDF, Word, etc.)
- [ ] Collaborative research features
- [ ] Custom agent creation interface

## 🛠️ Development

### Running Tests

```bash
# Install test dependencies
pip install pytest pytest-asyncio

# Run tests
pytest tests/
```

### Code Structure

```python
# Main components
conduct_research()      # Entry point for research
create_agents()        # Agent initialization
create_supervisor()    # Workflow orchestration
initialize_tools()     # Tool setup
```

### Adding New Tools

```python
@tool
def your_custom_tool(query: str) -> str:
    """Your tool description."""
    # Implementation
    return result

# Add to initialize_tools()
tools.append(your_custom_tool)
```

## 🔧 Troubleshooting

### Common Issues

**"Module not found" errors:**
```bash
# Make sure virtual environment is activated
source langgraph_env/bin/activate  # macOS/Linux
# or
langgraph_env\Scripts\activate     # Windows

# Reinstall dependencies
pip install -r requirements.txt
```

**API Key errors:**
- Verify `.env` file exists and contains correct keys
- Check API key quotas and billing status
- Ensure no extra spaces in `.env` file

**Jupyter not starting:**
```bash
# Install Jupyter explicitly
pip install jupyter notebook

# Try alternative start method
python -m jupyter notebook
```

### Performance Tips

- Use specific, focused research queries for better results
- Allow 30-90 seconds for complex research tasks
- Monitor API usage to stay within limits
- Use caching for repeated queries

## 📝 License

MIT License - see LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -am 'Add feature'`
4. Push to branch: `git push origin feature-name`
5. Submit Pull Request

## 📞 Support

- 📧 Email: your-email@domain.com
- 🐛 Issues: [GitHub Issues](https://github.com/your-repo/issues)
- 📖 Documentation: [Full Docs](https://your-docs-site.com)

---

**Happy Researching! 🔬✨**