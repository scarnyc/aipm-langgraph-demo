#!/usr/bin/env python3
"""
LangGraph Multi-Agent Research System with Gradio Interface
Run this script directly: python research_app.py
"""

import os
import time
import uuid
from typing import List, Union, Callable
from datetime import datetime, timezone
from urllib.parse import quote

# Environment and configuration
from dotenv import load_dotenv

# LangGraph and LangChain imports
from langgraph.graph import MessagesState, START, END, StateGraph
from langgraph.prebuilt import ToolNode, create_react_agent
from langgraph_supervisor import create_supervisor
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import Tool
from langchain.tools import tool
from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search.tool import TavilySearchResults
from langchain_community.utilities import WikipediaAPIWrapper

# Gradio for UI
import gradio as gr

# Load environment variables
load_dotenv()

# ============================================================================
# CONFIGURATION
# ============================================================================

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not ANTHROPIC_API_KEY:
    raise ValueError("ANTHROPIC_API_KEY environment variable not set.")

# Model configuration
MODEL = 'claude-sonnet-4-20250514'  # Updated to a stable model version

# ============================================================================
# INITIALIZE CLAUDE MODEL
# ============================================================================

claude = ChatAnthropic(
    model_name=MODEL,
    temperature=1,
    max_tokens=2000,
    
    # Enable thinking with budget_tokens as required by API  
    thinking={"type": "enabled", "budget_tokens": 1024},
    
    # Enable interleaved thinking for better tool use and reasoning
    extra_headers={
        "anthropic-beta": "interleaved-thinking-2025-05-14"
    },
    
    # Enable keep-alive as recommended by Anthropic
    timeout=300.0,  # 5 minute timeout
)

# ============================================================================
# TOOL DEFINITIONS
# ============================================================================

def create_tavily_search_tool(api_key):
    """Create the Tavily search tool."""
    if not api_key:
        print("⚠️ Tavily API key not found, search functionality will be limited")
        return None
    
    def tavily_search(query, *args, **kwargs):
        try:
            print(f"🔍 Searching for: {query[:50]}...")
            results = TavilySearchResults(
                api_key=api_key,
                max_results=3,
                include_answer=True,
                search_depth="basic"
            )(query, *args, **kwargs)
            return results
        except Exception as e:
            print(f"Error in Tavily search: {e}")
            return f"Search error: {str(e)}"
    
    return Tool(
        name="tavily_search_results",
        func=tavily_search,
        description="Search the web for current information. Use for recent events, news, and current data."
    )

@tool
def get_current_datetime() -> str:
    """Get the current date and time."""
    now_utc = datetime.now(timezone.utc)
    formatted_date = now_utc.strftime("%A, %B %d, %Y at %I:%M %p UTC")
    return f"Current date and time: {formatted_date}"

def create_wikipedia_tool():
    """Create a Wikipedia search tool."""
    try:
        api_wrapper = WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=2000)
        
        def wiki_query(query):
            if not query or not isinstance(query, str):
                return "Invalid query. Please provide a valid search term."
            
            query = query.strip()[:300]
            
            try:
                print(f"📚 Searching Wikipedia for: {query[:50]}...")
                result = api_wrapper.run(query)
                
                if len(result) > 3000:
                    result = result[:3000] + "... [content truncated]"
                
                wiki_url = f"https://en.wikipedia.org/wiki/{quote(query.replace(' ', '_'), safe='')}"
                result += f"\n\nSource: {wiki_url}"
                
                return result
            except Exception as e:
                print(f"Error in Wikipedia search: {e}")
                return "Wikipedia search encountered an error. Please try a different query."
        
        return Tool(
            name="wikipedia_query_run",
            func=wiki_query,
            description="Search Wikipedia for established knowledge and historical information."
        )
    except Exception as e:
        print(f"Failed to initialize Wikipedia tool: {e}")
        return None

# ============================================================================
# INITIALIZE TOOLS
# ============================================================================

def initialize_tools():
    """Initialize all available tools."""
    tools = []
    
    # Tavily search
    if TAVILY_API_KEY:
        tavily_tool = create_tavily_search_tool(TAVILY_API_KEY)
        if tavily_tool:
            tools.append(tavily_tool)
            print("✅ Tavily search tool initialized")
    
    # Wikipedia
    wiki_tool = create_wikipedia_tool()
    if wiki_tool:
        tools.append(wiki_tool)
        print("✅ Wikipedia tool initialized")
    
    # DateTime
    tools.append(get_current_datetime)
    print("✅ DateTime tool initialized")
    
    return tools

# ============================================================================
# AGENT DEFINITIONS
# ============================================================================

def create_agents(available_tools):
    """Create all specialized agents for the research system."""
    
    # Planning Agent
    planning_agent = create_react_agent(
        model=claude,
        tools=[],
        name="planning_expert",
        prompt="""You are a research planning specialist. Create concise, actionable research plans.
        
        OUTPUT FORMAT:
        RESEARCH PLAN
        - Query: [Core question]
        - Objectives: [Key goals]
        - Search Strategy: [Approach]
        - Success Criteria: [Completion conditions]
        """
    )
    
    # Search Agent (only one with tools)
    search_agent = create_react_agent(
        model=claude,
        tools=available_tools,
        name="search_expert",
        prompt="""You are a search specialist with access to web search and Wikipedia.
        
        Execute targeted searches and return key findings with sources.
        Focus on accuracy and source credibility.
        """
    )
    
    # Citation Agent
    citation_agent = create_react_agent(
        model=claude,
        tools=[],
        name="citation_expert",
        prompt="""You are a citation specialist. Validate sources and format citations properly.
        
        Ensure all claims are supported by credible sources.
        """
    )
    
    # Reflection Agent
    reflection_agent = create_react_agent(
        model=claude,
        tools=[],
        name="reflection_expert",
        prompt="""You are a quality assurance specialist.
        
        Evaluate if research adequately addresses the query.
        Either APPROVE for synthesis or request MORE RESEARCH with specific gaps.
        """
    )
    
    # Synthesis Agent
    synthesis_agent = create_react_agent(
        model=claude,
        tools=[],
        name="synthesis_expert",
        prompt="""You are a synthesis specialist. Create comprehensive final reports.
        
        OUTPUT FORMAT:
        RESEARCH REPORT
        
        Summary: [Direct answer to query]
        
        Key Findings:
        1. [Major finding with source]
        2. [Major finding with source]
        
        Details: [Expanded analysis]
        
        Sources: [All citations]
        """
    )
    
    return planning_agent, search_agent, citation_agent, reflection_agent, synthesis_agent

# ============================================================================
# SUPERVISOR SETUP
# ============================================================================

def create_research_supervisor(agents, claude_model):
    """Create the supervisor that orchestrates all agents."""
    
    supervisor = create_supervisor(
        agents,
        model=claude_model,
        prompt="""You are the research coordinator managing a multi-agent research system.
        
        WORKFLOW:
        1. Send query to planning_expert for research plan
        2. Send plan to search_expert for information gathering
        3. Send results to citation_expert for validation
        4. Send to reflection_expert for quality check
        5. If approved, send to synthesis_expert for final report
        
        Only search_expert has access to search tools.
        Keep responses concise and focused on the user's query.
        """
    )
    
    return supervisor

# ============================================================================
# RESEARCH FUNCTION
# ============================================================================

def conduct_research(query):
    """Conduct research using the multi-agent system."""
    if not query.strip():
        return "Please enter a research question."
    
    try:
        # Initialize tools
        print("\n🔧 Initializing tools...")
        available_tools = initialize_tools()
        
        # Create agents
        print("🤖 Creating agents...")
        agents = create_agents(available_tools)
        
        # Create supervisor
        print("👨‍💼 Creating supervisor...")
        supervisor = create_research_supervisor(agents, claude)
        
        # Compile with memory
        memory = MemorySaver()
        app = supervisor.compile(checkpointer=memory)
        
        # Create config
        thread_id = f"research_{uuid.uuid4().hex[:8]}"
        config = {"configurable": {"thread_id": thread_id}}
        
        print(f"\n🔍 Researching: {query}\n")
        print("=" * 50)
        
        # Run research
        result = app.invoke(
            {"messages": [HumanMessage(content=query)]},
            config
        )
        
        # Extract response
        if result and "messages" in result and result["messages"]:
            last_message = result["messages"][-1]
            if hasattr(last_message, 'content'):
                content = last_message.content
                if isinstance(content, str):
                    return content
                elif isinstance(content, list):
                    text_parts = []
                    for block in content:
                        if isinstance(block, dict) and block.get('type') == 'text':
                            text_parts.append(block.get('text', ''))
                    return '\n'.join(text_parts)
        
        return "Research completed but no results were generated."
        
    except Exception as e:
        error_msg = f"Error during research: {str(e)}"
        print(f"❌ {error_msg}")
        return error_msg

# ============================================================================
# GRADIO INTERFACE
# ============================================================================

def create_interface():
    """Create the Gradio interface."""
    
    with gr.Blocks(
        title="Deep Research Assistant",
        theme=gr.themes.Soft(),
        css="""
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        """
    ) as demo:
        
        gr.Markdown(
            """
            # 🔬 Deep Research Assistant
            
            Ask any research question and get a comprehensive, well-sourced report.
            
            **System Features:**
            - Multi-agent research system
            - Web search and Wikipedia integration
            - Source validation and citation
            - Quality assurance checks
            - Comprehensive synthesis
            """
        )
        
        with gr.Row():
            with gr.Column(scale=2):
                query_input = gr.Textbox(
                    label="Research Question",
                    placeholder="e.g., What are the latest developments in quantum computing?",
                    lines=3
                )
                
                submit_btn = gr.Button(
                    "🔍 Start Research",
                    variant="primary",
                    size="lg"
                )
                
                gr.Markdown("### Example Questions:")
                examples = [
                    "What are the latest AI regulation developments globally?",
                    "How is climate change affecting global food security?",
                    "What are the economic impacts of remote work?",
                ]
                
                for example in examples:
                    gr.Button(example, size="sm").click(
                        lambda x=example: x,
                        outputs=query_input
                    )
            
            with gr.Column(scale=3):
                output = gr.Textbox(
                    label="Research Results",
                    lines=25,
                    max_lines=40,
                    show_copy_button=True
                )
        
        gr.Markdown(
            """
            **Note:** Research typically takes 30-90 seconds depending on complexity.
            The system will plan, search, validate, and synthesize information automatically.
            """
        )
        
        # Connect the button
        submit_btn.click(
            fn=conduct_research,
            inputs=[query_input],
            outputs=[output]
        )
        
        query_input.submit(
            fn=conduct_research,
            inputs=[query_input],
            outputs=[output]
        )
    
    return demo

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main function to run the application."""
    print("=" * 50)
    print("🚀 Starting Deep Research Assistant")
    print("=" * 50)
    
    # Verify API keys
    if not ANTHROPIC_API_KEY:
        print("❌ Error: ANTHROPIC_API_KEY not found in environment variables")
        return
    
    if not TAVILY_API_KEY:
        print("⚠️ Warning: TAVILY_API_KEY not found - web search will be limited")
    
    print("\n📊 System Status:")
    print("   ✅ Claude model ready")
    print("   ✅ Multi-agent system configured")
    print("   ✅ Gradio interface initialized")
    
    # Create and launch interface
    demo = create_interface()
    
    print("\n🌐 Launching web interface...")
    print("=" * 50)
    
    # Launch with appropriate settings
    demo.launch(
        server_name="0.0.0.0",  # Allow external connections
        server_port=7860,        # Default Gradio port
        share=False,             # Set to True to create public link
        debug=False              # Set to True for debugging
    )

if __name__ == "__main__":
    main()