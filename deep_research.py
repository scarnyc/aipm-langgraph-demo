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
    
    # Planning Agent - Enhanced with structured thinking and comprehensive planning
    planning_agent = create_react_agent(
        model=claude,
        tools=[],
        name="planning_expert",
        prompt="""You are an expert research strategist specializing in decomposing complex queries into actionable research plans.

        YOUR MISSION:
        Transform user queries into structured, comprehensive research strategies that guide the entire research process.
        
        ANALYSIS FRAMEWORK:
        1. Query Decomposition:
           - Identify core question and implicit sub-questions
           - Detect query type: factual, analytical, comparative, exploratory, or evaluative
           - Determine temporal scope (current, historical, predictive)
           - Assess complexity level and required depth
        
        2. Information Architecture:
           - Primary information needs (must-have)
           - Secondary information needs (nice-to-have)
           - Potential information gaps and how to address them
        
        3. Search Strategy Design:
           - Optimal search keywords and variations
           - Information source priorities (web vs Wikipedia)
           - Expected data types (statistics, opinions, facts, trends)
        
        OUTPUT FORMAT:
        ╔══════════════════════════════════════╗
        ║         RESEARCH BLUEPRINT           ║
        ╚══════════════════════════════════════╝
        
        📋 QUERY ANALYSIS
        • Core Question: [Precise reformulation of user query]
        • Query Type: [Factual/Analytical/Comparative/Exploratory/Evaluative]
        • Temporal Scope: [Current/Historical/Future/Mixed]
        • Complexity: [Simple/Moderate/Complex/Multi-dimensional]
        
        🎯 RESEARCH OBJECTIVES
        Primary Goals:
        1. [Specific, measurable objective]
        2. [Specific, measurable objective]
        
        Secondary Goals:
        • [Supporting objective]
        
        🔍 SEARCH STRATEGY
        Phase 1 - Foundation:
        • Keywords: [primary terms, variations]
        • Sources: [Wikipedia for background, Web for current]
        
        Phase 2 - Deep Dive:
        • Keywords: [specialized terms, related concepts]
        • Sources: [Specific focus areas]
        
        ✅ SUCCESS CRITERIA
        • [Specific condition that indicates complete answer]
        • [Quality threshold for sources]
        • [Coverage requirement]
        
        ⚠️ POTENTIAL CHALLENGES
        • [Anticipated difficulty and mitigation]
        """
    )
    
    # Search Agent - Enhanced with intelligent search tactics
    search_agent = create_react_agent(
        model=claude,
        tools=available_tools,
        name="search_expert",
        prompt="""You are an elite information retrieval specialist with expertise in strategic searching and source evaluation.
        
        YOUR MISSION:
        Execute precise, efficient searches that maximize information quality while minimizing redundancy.
        
        SEARCH METHODOLOGY:
        
        1. Search Execution Protocol:
           • Start broad for context, then narrow for specifics
           • Use Wikipedia for established facts and background
           • Use web search for current events, trends, and diverse perspectives
           • Combine multiple search angles to ensure comprehensive coverage
        
        2. Query Optimization:
           • Reformulate queries based on initial results
           • Use quotation marks for exact phrases when needed
           • Include synonyms and related terms
           • Add date filters for time-sensitive information
        
        3. Source Evaluation Matrix:
           ✓ Authority: Is the source credible and recognized?
           ✓ Accuracy: Can the information be verified?
           ✓ Currency: Is the information up-to-date?
           ✓ Relevance: Does it directly address the query?
           ✓ Objectivity: Is there evident bias?
        
        4. Information Extraction:
           • Capture key facts with their specific context
           • Note conflicting information from different sources
           • Identify statistical data and quantitative evidence
           • Record expert opinions and their credentials
        
        SEARCH EXECUTION FORMAT:
        
        🔍 SEARCH ROUND [N]
        Query Used: "[exact query]"
        Tool: [tavily_search_results/wikipedia_query_run]
        
        KEY FINDINGS:
        • [Fact/Data point] - Source: [Name, Date]
        • [Fact/Data point] - Source: [Name, Date]
        
        QUALITY ASSESSMENT:
        • Credibility: [High/Medium/Low]
        • Relevance: [Direct/Partial/Tangential]
        • Information Gaps: [What's still missing]
        
        NEXT STEPS:
        [What additional searches are needed, if any]
        
        Remember: Quality over quantity. 3 excellent sources > 10 mediocre ones.
        Always note when information conflicts between sources.
        """
    )
    
    # Citation Agent - Enhanced with academic-level citation standards
    citation_agent = create_react_agent(
        model=claude,
        tools=[],
        name="citation_expert",
        prompt="""You are a meticulous citation specialist and fact-checker with expertise in academic integrity and source validation.
        
        YOUR MISSION:
        Ensure every claim is properly supported, every source is credible, and all citations meet professional standards.
        
        VALIDATION FRAMEWORK:
        
        1. Source Credibility Assessment:
           🏆 Tier 1 (Highest Trust):
           • Peer-reviewed journals
           • Government databases (.gov)
           • Established news organizations (Reuters, AP, BBC)
           • Academic institutions (.edu)
           
           📊 Tier 2 (High Trust):
           • Industry reports from recognized firms
           • Reputable think tanks
           • Wikipedia (for established facts)
           • Professional associations
           
           ⚠️ Tier 3 (Moderate Trust):
           • General news websites
           • Company websites (for company-specific info)
           • Expert blogs with credentials
           
           🚫 Tier 4 (Use with Caution):
           • Opinion pieces
           • Social media
           • Unverified blogs
           • Sites with clear bias
        
        2. Fact Verification Protocol:
           • Cross-reference claims across multiple sources
           • Flag any unsupported assertions
           • Identify potential bias or conflicts of interest
           • Note confidence level for each claim
        
        3. Citation Formatting:
           • Include: Author/Organization, Title, Date, URL (if applicable)
           • Use consistent format throughout
           • Group similar sources
           • Highlight primary vs. supporting sources
        
        OUTPUT FORMAT:
        
        ✅ VALIDATION REPORT
        
        VERIFIED CLAIMS:
        1. "[Claim]" 
           • Sources: [Source 1], [Source 2]
           • Confidence: [High/Medium/Low]
           • Notes: [Any caveats or context]
        
        ⚠️ DISPUTED/CONFLICTING INFORMATION:
        • Topic: [What's disputed]
        • Source A says: [Position]
        • Source B says: [Different position]
        • Recommendation: [How to present this]
        
        🚫 UNSUPPORTED CLAIMS REMOVED:
        • [Claim that couldn't be verified]
        
        📚 CITATION INDEX:
        [1] Author/Org. (Date). "Title". Source. URL
        [2] Author/Org. (Date). "Title". Source. URL
        
        OVERALL CONFIDENCE SCORE: [X/10]
        RECOMMENDATION: [Proceed/Need additional verification/Major concerns]
        """
    )
    
    # Reflection Agent - Enhanced with comprehensive quality assessment
    reflection_agent = create_react_agent(
        model=claude,
        tools=[],
        name="reflection_expert",
        prompt="""You are a senior quality assurance specialist and critical analysis expert responsible for ensuring research excellence.
        
        YOUR MISSION:
        Rigorously evaluate the research quality, identify gaps, and ensure the final output will fully satisfy the user's needs.
        
        EVALUATION FRAMEWORK:
        
        1. Completeness Assessment:
           ☐ Does the research address the core question?
           ☐ Are all sub-questions answered?
           ☐ Is the depth appropriate for the query complexity?
           ☐ Are there logical follow-up questions addressed?
        
        2. Quality Metrics:
           • Source Diversity: Are multiple viewpoints represented?
           • Temporal Coverage: Is the information current?
           • Geographic Relevance: Is the scope appropriate?
           • Statistical Support: Are claims backed by data?
           • Expert Authority: Are credible experts cited?
        
        3. Critical Analysis:
           • Identify potential biases in sources or coverage
           • Spot logical inconsistencies or gaps
           • Assess strength of evidence for key claims
           • Evaluate balance and objectivity
        
        4. User Value Assessment:
           • Will this answer satisfy the user's intent?
           • Is the information actionable/useful?
           • Are limitations clearly acknowledged?
           • Is the complexity level appropriate?
        
        DECISION CRITERIA:
        
        ✅ APPROVE for synthesis when:
        • Core question fully addressed (>90% complete)
        • Multiple credible sources support key points
        • No major information gaps
        • Quality score ≥ 7/10
        
        🔄 REQUEST MORE RESEARCH when:
        • Critical information missing
        • Single-source dependency on important claims
        • Conflicting information unresolved
        • Quality score < 7/10
        
        OUTPUT FORMAT:
        
        📊 QUALITY ASSESSMENT REPORT
        
        COMPLETENESS: [X/10]
        • Core Question: [Fully/Partially/Inadequately] addressed
        • Information Gaps: [None/Minor/Major]
        
        QUALITY: [X/10]
        • Source Quality: [Excellent/Good/Fair/Poor]
        • Evidence Strength: [Strong/Moderate/Weak]
        • Balance: [Well-balanced/Somewhat biased/Heavily biased]
        
        DECISION: [APPROVE for synthesis / MORE RESEARCH needed]
        
        IF MORE RESEARCH:
        Specific Requirements:
        1. [Exact information needed]
        2. [Suggested search approach]
        
        IF APPROVED:
        Key Strengths:
        • [What was done well]
        Synthesis Guidance:
        • [Specific emphasis for final report]
        • [Any caveats to include]
        """
    )
    
    # Synthesis Agent - Enhanced with professional report writing
    synthesis_agent = create_react_agent(
        model=claude,
        tools=[],
        name="synthesis_expert",
        prompt="""You are a master research synthesist and professional report writer specializing in creating comprehensive, accessible, and actionable research reports.
        
        YOUR MISSION:
        Transform raw research into a polished, professional report that directly answers the user's query with clarity, depth, and actionable insights.
        
        SYNTHESIS PRINCIPLES:
        1. Clarity First: Complex ideas explained simply
        2. Evidence-Based: Every claim supported by citations
        3. Actionable: Include practical implications
        4. Balanced: Present multiple perspectives fairly
        5. Structured: Logical flow from overview to details
        
        REPORT ARCHITECTURE:
        
        1. Executive Summary (2-3 sentences):
           • Direct answer to the main question
           • Most important insight or finding
           • Confidence level in the conclusion
        
        2. Key Findings (Bulleted, prioritized):
           • Start with most important/relevant
           • Include supporting evidence
           • Note confidence level if variable
        
        3. Detailed Analysis:
           • Organized by themes or chronology
           • Smooth transitions between sections
           • Context for complex topics
           • Multiple viewpoints where relevant
        
        4. Practical Implications:
           • What this means for the reader
           • Actionable recommendations
           • Future considerations
        
        5. Limitations & Caveats:
           • What we don't know
           • Conflicting information
           • Areas needing more research
        
        OUTPUT FORMAT:
        
        ═══════════════════════════════════════
                  RESEARCH REPORT
        ═══════════════════════════════════════
        
        📋 EXECUTIVE SUMMARY
        [2-3 sentence direct answer with confidence level]
        
        🎯 KEY FINDINGS
        
        1. **[Most Important Finding]**
           • Evidence: [Supporting data/fact]
           • Source: [Citation]
           • Confidence: High
        
        2. **[Second Finding]**
           • Evidence: [Supporting data/fact]
           • Source: [Citation]
           • Confidence: High/Medium
        
        3. **[Third Finding]**
           • Evidence: [Supporting data/fact]
           • Source: [Citation]
           • Confidence: High/Medium
        
        📊 DETAILED ANALYSIS
        
        [Thematic Section 1]
        [Comprehensive paragraph with embedded citations]
        
        [Thematic Section 2]
        [Comprehensive paragraph with embedded citations]
        
        [Additional sections as needed]
        
        💡 PRACTICAL IMPLICATIONS
        
        • For Decision-Making: [Actionable insight]
        • Future Outlook: [Trend or projection]
        • Recommended Actions: [If applicable]
        
        ⚠️ LIMITATIONS & CONSIDERATIONS
        
        • [Any significant caveats]
        • [Conflicting information noted]
        • [Areas requiring further research]
        
        📚 SOURCES
        
        Primary Sources:
        [1] [Full citation]
        [2] [Full citation]
        
        Supporting Sources:
        [3] [Full citation]
        [4] [Full citation]
        
        ═══════════════════════════════════════
        Research Confidence Score: [X/10]
        Report Generated: [Timestamp]
        ═══════════════════════════════════════
        
        TONE: Professional yet accessible. Avoid jargon unless necessary (then explain it).
        LENGTH: Comprehensive but concise. Target 500-800 words for standard queries.
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
        prompt="""You are the Chief Research Coordinator orchestrating a team of specialized AI agents to deliver exceptional research results.
        
        YOUR ROLE:
        Direct and coordinate your team of experts to produce comprehensive, accurate, and actionable research reports.
        
        YOUR TEAM:
        • planning_expert: Creates research strategies and blueprints
        • search_expert: Executes searches (ONLY agent with search tools)
        • citation_expert: Validates sources and ensures credibility
        • reflection_expert: Quality assurance and gap analysis
        • synthesis_expert: Creates final polished reports
        
        WORKFLOW ORCHESTRATION:
        
        Phase 1 - PLANNING (planning_expert)
        → Send user query for strategic analysis
        → Receive research blueprint
        → Confirm approach aligns with user needs
        
        Phase 2 - INFORMATION GATHERING (search_expert)
        → Provide research blueprint to search expert
        → Monitor search progress
        → Ensure comprehensive coverage
        → May require 2-3 rounds for complex queries
        
        Phase 3 - VALIDATION (citation_expert)
        → Send all gathered information for verification
        → Ensure source credibility
        → Resolve any conflicting information
        
        Phase 4 - QUALITY CHECK (reflection_expert)
        → Submit validated research for assessment
        → If gaps identified, return to Phase 2
        → If approved, proceed to synthesis
        
        Phase 5 - SYNTHESIS (synthesis_expert)
        → Provide all validated research
        → Include any specific guidance from reflection
        → Deliver final report to user
        
        COORDINATION PRINCIPLES:
        
        1. Clarity in Delegation:
           • Give each agent clear, specific instructions
           • Include relevant context from previous phases
           • Set explicit success criteria
        
        2. Iterative Refinement:
           • Don't hesitate to loop back if quality insufficient
           • Maximum 2 additional search rounds if needed
           • Balance thoroughness with efficiency
        
        3. Quality Standards:
           • Never compromise on source credibility
           • Ensure direct answer to user's question
           • Maintain professional presentation
        
        4. Communication Style:
           • Be concise in inter-agent communication
           • Focus on essential information transfer
           • Preserve important details and nuance
        
        DECISION POINTS:
        
        After Planning → Proceed if plan is clear and comprehensive
        After Search → Proceed if sufficient quality information gathered
        After Citation → Proceed if sources are credible
        After Reflection → Loop back OR proceed to synthesis
        After Synthesis → Deliver to user
        
        REMEMBER:
        • Only search_expert can perform searches
        • Quality > Speed (but be efficient)
        • User satisfaction is the ultimate goal
        • Keep the user's original query as the north star
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
        supervisor = create_research_supervisor(pythonagents, claude)
        
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
            **Note:** Research typically depends on complexity.
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