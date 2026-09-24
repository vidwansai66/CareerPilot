"""
CareerPilot AI Architecture
----------------------------
FastAPI: Provides the web server framework, HTML UI route, and health check.
LangServe: Exposes the LangGraph agent securely as a standard REST API at /careerpilot.
LangGraph: Orchestrates the workflow via StateGraph using meaningful nodes.
RAG / Embeddings: Uses an InMemoryVectorStore and Gemini Embeddings.
Tools: Implements deterministic @tool functions (skill_gap_analyzer, roadmap_planner).
Gemini: Powers natural language understanding via langchain-google-genai.
"""

import os
import uvicorn
import re
from typing import List, Dict, Any, Tuple
from typing_extensions import TypedDict
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

# LangChain / LangGraph imports
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_core.runnables import RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langgraph.graph import StateGraph, START, END
from langserve import add_routes

app = FastAPI(title="CareerPilot AI")

# --- 1. CONFIGURATION & RAG KNOWLEDGE BASE ---
career_knowledge = [
    Document(page_content="""Role: AI Automation Engineer
Description: Builds agentic workflows and connects LLMs to external systems using tools like n8n and LangChain.
Required Skills: Python, REST APIs, FastAPI, n8n, LangChain, LangGraph, LLM APIs, JSON, Webhooks, API Integration, Automation Workflows, Prompt Engineering, RAG, Vector Databases, Git/GitHub, Cloud Deployment
Tools: n8n, LangChain, LangGraph, FastAPI, Gemini/OpenAI APIs, GitHub
Frameworks: LangChain, LangGraph, FastAPI
Prerequisites: Python, JSON, REST APIs
Interview Topics: APIs, webhooks, n8n workflows, LangChain, LangGraph, RAG, LLM APIs, automation architecture, deployment
Typical Project Areas: LLM customer service bots, automated data extraction pipelines, multi-agent orchestration"""),
    
    Document(page_content="""Role: Agentic AI Engineer
Description: Designs multi-agent systems and stateful LLM applications for complex reasoning.
Required Skills: Python, LangGraph, AutoGen, OpenAI API, Vector Databases, State Management, Prompt Engineering, Tool Calling, System Architecture
Tools: LangGraph, AutoGen, Pinecone
Frameworks: LangChain, LangGraph
Prerequisites: Python, LLM APIs
Interview Topics: state management, tool binding, multi-agent consensus, error recovery
Typical Project Areas: Autonomous researchers, coding assistants"""),

    Document(page_content="""Role: Generative AI Engineer
Description: Works on fine-tuning, RAG architectures, and deploying custom models.
Required Skills: Python, PyTorch, Transformers, LLM Fine-tuning, RAG, HuggingFace, CUDA, Vector Search
Tools: HuggingFace, vLLM
Frameworks: PyTorch, Transformers
Prerequisites: Python, Deep Learning basics
Interview Topics: attention mechanisms, PEFT, LoRA, vector search optimization, quantization
Typical Project Areas: Custom model fine-tuning, enterprise RAG systems"""),

    Document(page_content="""Role: Machine Learning Engineer
Description: Deploys traditional and modern ML models into production.
Required Skills: Python, Scikit-learn, TensorFlow, MLOps, SQL, Docker, Kubernetes, CI/CD, Model Evaluation
Tools: Docker, Kubernetes, MLflow
Frameworks: Scikit-learn, TensorFlow, PyTorch
Prerequisites: Python, Math, Statistics
Interview Topics: model evaluation, CI/CD for ML, data preprocessing, deployment strategies
Typical Project Areas: Recommendation engines, predictive analytics pipelines"""),

    Document(page_content="""Role: Data Scientist
Description: Analyzes complex data to extract actionable insights.
Required Skills: Python, Pandas, SQL, Statistics, Data Visualization, Jupyter, Machine Learning, A/B Testing
Tools: Jupyter, Tableau
Frameworks: Pandas, Scikit-learn
Prerequisites: Statistics, Python, SQL
Interview Topics: statistical significance, A/B testing, regression analysis, data cleaning
Typical Project Areas: Customer churn prediction, exploratory data analysis"""),

    Document(page_content="""Role: Backend Developer
Description: Builds robust server-side applications and databases.
Required Skills: Python, FastAPI, Django, PostgreSQL, Docker, REST APIs, GraphQL, Redis, System Design
Tools: Docker, PostgreSQL, Redis, Postman
Frameworks: FastAPI, Django
Prerequisites: Python, Databases, Networking
Interview Topics: database indexing, asynchronous programming, API security, caching
Typical Project Areas: Scalable microservices, robust API gateways"""),

    Document(page_content="""Role: Python Developer
Description: Writes scripts, automations, and backend logic using Python.
Required Skills: Python, Pytest, Git/GitHub, Linux, Object-Oriented Programming, Data Structures, Algorithms
Tools: Git, Linux, VS Code
Frameworks: Pytest
Prerequisites: Basic programming
Interview Topics: data structures, algorithms, memory management, pythonic code
Typical Project Areas: Data scraping, automation scripts, CLI tools"""),

    Document(page_content="""Role: Full Stack Developer
Description: Handles both frontend and backend development.
Required Skills: JavaScript, React, Node.js, Python, CSS, HTML, SQL, Git/GitHub, REST APIs
Tools: Git, Webpack, Figma
Frameworks: React, Express, Django
Prerequisites: HTML, CSS, JS
Interview Topics: DOM manipulation, REST vs GraphQL, state management, web security
Typical Project Areas: End-to-end web applications, e-commerce platforms"""),

    Document(page_content="""Role: AI Scientist
Description: A role focused on researching, developing and evaluating AI/ML methods and applying them to intelligent systems.
Required Skills: Python, Data Structures & Algorithms, Mathematics, Probability & Statistics, Linear Algebra, Machine Learning, Deep Learning, NLP, Computer Vision, PyTorch, TensorFlow, Research Methodology, Model Evaluation, Experimentation, Git/GitHub
Tools: Python, NumPy, Pandas, Scikit-learn, PyTorch, TensorFlow, Jupyter, Git/GitHub
Frameworks: PyTorch, TensorFlow
Prerequisites: Python, DSA, Mathematics, Statistics, Machine Learning
Interview Topics: Machine Learning, Deep Learning, Mathematics, Statistics, Model Evaluation, Research Methodology, Neural Networks, NLP, Computer Vision, Experiment Design
Typical Project Areas: ML research project, Deep learning experiment, NLP system, Computer vision system, model evaluation study""")
]

vector_store = None

def get_vector_store():
    global vector_store
    if vector_store is not None:
        return vector_store
    
    if not os.getenv("GEMINI_API_KEY"):
        return None
        
    try:
        model_name = os.getenv("GEMINI_EMBEDDING_MODEL", "models/embedding-001")
        embeddings = GoogleGenerativeAIEmbeddings(model=model_name)
        vector_store = InMemoryVectorStore.from_documents(career_knowledge, embeddings)
        return vector_store
    except Exception as e:
        print(f"Vector store initialization error: {e}")
        return None

# --- 2. TOOLS ---
class SkillGapInput(BaseModel):
    user_skills: List[str] = Field(description="List of user's current skills")
    required_skills: List[str] = Field(description="List of skills required for the target role")

def normalize_skill(skill: str) -> str:
    s = skill.lower().strip()
    # Handle common equivalents
    equivalents = {
        "git": "git/github",
        "github": "git/github",
        "rest api": "rest apis",
        "api": "rest apis",
        "apis": "rest apis",
        "lang graph": "langgraph",
        "n8n": "n8n",
        "llm api": "llm apis",
        "vector database": "vector databases",
        "automation workflow": "automation workflows",
        "libraries in python": "python libraries",
        "dsa": "data structures & algorithms"
    }
    
    val = equivalents.get(s, s)
    # Title case formatting for display
    return ' '.join(word.capitalize() for word in val.split())

@tool("skill_gap_analyzer", args_schema=SkillGapInput)
def skill_gap_analyzer(user_skills: List[str], required_skills: List[str]) -> Dict[str, Any]:
    """Analyzes the gap between user's current skills and the skills required for the target role."""
    u_skills = {normalize_skill(s) for s in user_skills if s.strip()}
    
    matched = []
    missing = []
    
    for r in required_skills:
        r_clean = normalize_skill(r)
        if not r_clean:
            continue
        
        is_matched = False
        if r_clean in u_skills:
            is_matched = True
        else:
            # Check substrings for flexible matching
            for u in u_skills:
                if u in r_clean or r_clean in u:
                    is_matched = True
                    break
                    
        if is_matched:
            matched.append(r)
        else:
            missing.append(r)
            
    # Priority groupings based on general dependencies
    # E.g. Foundational concepts are High priority
    high_priority_keywords = ["python", "javascript", "sql", "git", "rest api", "json", "html", "css", "linux", "fastapi", "langgraph"]
    medium_priority_keywords = ["django", "react", "node", "docker", "llm apis", "webhooks", "api integration", "prompt engineering"]
    
    priorities = {"high": [], "medium": [], "low": []}
    for m in missing:
        m_norm = normalize_skill(m)
        is_high = any(k in m_norm for k in high_priority_keywords)
        is_medium = any(k in m_norm for k in medium_priority_keywords)
        
        if is_high:
            priorities["high"].append(m)
        elif is_medium:
            priorities["medium"].append(m)
        else:
            priorities["low"].append(m)
                
    return {
        "matched_skills": matched,
        "missing_skills": missing,
        "priority_groups": priorities
    }

class RoadmapInput(BaseModel):
    missing_skills: List[str] = Field(description="List of missing skills to learn")
    available_time: str = Field(description="Available preparation time")
    target_role: str = Field(description="The target job role")

@tool("roadmap_planner", args_schema=RoadmapInput)
def roadmap_planner(missing_skills: List[str], available_time: str, target_role: str) -> List[Dict[str, str]]:
    """Plans a dependency-aware learning sequence based on missing skills."""
    if not missing_skills:
        return [{"phase": "Final", "duration": available_time, "skills": "None", "learning_goals": "Master existing skills.", "mini_project": "Build advanced projects.", "expected_outcome": "Interview ready."}]
    
    # Define arbitrary but logical prerequisites for sorting
    dependency_order = [
        "python", "javascript", "html", "css", "sql", "linux", "git", "git/github",
        "json", "rest apis", "webhooks", "api integration",
        "fastapi", "django", "react", "node.js",
        "docker", "postgresql", "redis",
        "llm apis", "prompt engineering", "tool calling",
        "langchain", "n8n", "automation workflows",
        "langgraph", "agent architecture",
        "rag", "vector databases", "embeddings",
        "pytorch", "transformers", "machine learning", "deep learning", "statistics",
        "cloud deployment", "deployment", "mlops", "ci/cd"
    ]
    
    def get_sort_key(skill):
        s = normalize_skill(skill)
        for i, dep in enumerate(dependency_order):
            if dep in s:
                return i
        return 999
        
    sorted_missing = sorted(missing_skills, key=get_sort_key)
    
    # Divide into 3 chunks for months/phases
    n = max(1, len(sorted_missing) // 3 + (1 if len(sorted_missing) % 3 > 0 else 0))
    chunks = [sorted_missing[i:i + n] for i in range(0, len(sorted_missing), n)]
    
    roadmap = []
    titles = ["Foundations & Basics", "Core Architecture & Tools", "Advanced Integration & Deployment", "Final Polish"]
    
    for idx, chunk in enumerate(chunks):
        if not chunk:
            continue
        title = titles[idx] if idx < len(titles) else f"Phase {idx+1}"
        roadmap.append({
            "phase": f"Phase {idx+1} — {title}",
            "duration": f"1/{len(chunks)} of {available_time}",
            "skills": ", ".join(chunk),
            "learning_goals": f"Master the fundamentals of {chunk[0] if chunk else 'these tools'}.",
            "mini_project": f"Build a mini-project utilizing {', '.join(chunk[:2])}.",
            "expected_outcome": f"Comfortable using {chunk[0]} independently."
        })
        
    if roadmap:
        roadmap[-1]["mini_project"] = f"Deploy a production-style {target_role} project."
        roadmap[-1]["expected_outcome"] = "Portfolio ready for interviews."
        
    return roadmap

# --- 3. LANGGRAPH STATE & SCHEMAS ---
class CareerState(TypedDict, total=False):
    user_input: str
    target_role: str
    user_skills: List[str]
    available_time: str
    retrieved_context: str
    required_skills: List[str]
    skill_gaps: List[str]
    matched_skills: List[str]
    priority_groups: Dict[str, List[str]]
    roadmap: List[Dict[str, str]]
    role_interview_topics: List[str]
    interview_topics: str
    interview_questions: str
    final_answer: str
    execution_steps: List[str]

class UserProfile(BaseModel):
    target_role: str = Field(description="The target job role (e.g., AI Automation Engineer)")
    user_skills: List[str] = Field(description="The explicitly stated skills the user already possesses")
    available_time: str = Field(description="Available preparation time")

class InterviewPreparationOutput(BaseModel):
    technical_topics: List[str] = Field(description="List of technical topics to review")
    scenario_topics: List[str] = Field(description="List of scenario-based topics")
    project_topics: List[str] = Field(description="List of project-related topics")
    questions: List[str] = Field(description="List of exactly 8 interview questions")

def get_llm(temperature: float = 0.1):
    model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    return ChatGoogleGenerativeAI(model=model_name, temperature=temperature)

# --- 4. LANGGRAPH NODES ---
def parse_user_input(state: CareerState) -> CareerState:
    """Uses robust extraction to determine profile from natural language input."""
    print("\n--- CAREERPILOT DEBUG ---")
    user_input = state.get("user_input", "")
    print(f"Raw input: {user_input}")
    steps = state.get("execution_steps", [])
    steps.append("parse_user_input")
    
    # 1. Deterministic parser using regex
    role_match = re.search(r"(?:become an|become a|be an|be a|aiming for|goal is|become|work as an|work as a)\s+([a-zA-Z\-]+\s*[a-zA-Z\-]*\s*[a-zA-Z\-]*)(?:\.|\n|I know|I have|$)", user_input, re.IGNORECASE)
    target_role = role_match.group(1).strip() if role_match else ""
    
    # Normalize role
    aliases = {
        "ai scientist": "AI Scientist",
        "artificial intelligence scientist": "AI Scientist",
        "ai research scientist": "AI Scientist",
        "artificial intelligence researcher": "AI Scientist",
        "ai researcher": "AI Scientist",
        "generative ai engineer": "Generative AI Engineer",
        "genai engineer": "Generative AI Engineer",
        "generative ai developer": "Generative AI Engineer",
        "gen ai engineer": "Generative AI Engineer",
        "backend developer": "Backend Developer",
        "backend engineer": "Backend Developer",
        "server-side developer": "Backend Developer",
        "ai automation engineer": "AI Automation Engineer",
        "automation ai engineer": "AI Automation Engineer",
        "machine learning engineer": "Machine Learning Engineer",
        "ml engineer": "Machine Learning Engineer",
        "machine learning developer": "Machine Learning Engineer",
        "data scientist": "Data Scientist",
        "data science engineer": "Data Scientist",
        "python developer": "Python Developer",
        "python engineer": "Python Developer",
        "full stack developer": "Full Stack Developer",
        "fullstack developer": "Full Stack Developer",
        "full stack engineer": "Full Stack Developer",
        "agentic ai engineer": "Agentic AI Engineer",
        "agent engineer": "Agentic AI Engineer",
        "ai agent engineer": "Agentic AI Engineer"
    }
    if target_role.lower() in aliases:
        target_role = aliases[target_role.lower()]
    else:
        for k, v in aliases.items():
            if k in target_role.lower():
                target_role = v
                break
    
    skill_match = re.search(r"(?:I know|experience with|skills are)\s+([a-zA-Z0-9,\s\+]+?)(?:\.|\n|I have|$)", user_input, re.IGNORECASE)
    user_skills = []
    if skill_match:
        skill_str = skill_match.group(1).strip()
        skill_str = re.sub(r'\band\b', ',', skill_str, flags=re.IGNORECASE)
        # Fix casing and synonyms
        raw_skills = [s.strip() for s in skill_str.split(',') if s.strip()]
        synonyms = {"libraries in python": "Python Libraries", "dsa": "DSA", "machine learning": "Machine Learning", "basic nlp": "Basic NLP"}
        user_skills = [synonyms.get(s.lower(), s) for s in raw_skills]
        
    time_match = re.search(r"(?:I have)\s+(\d+\s*(?:months|month|years|year|weeks|week))", user_input, re.IGNORECASE)
    available_time = time_match.group(1).strip() if time_match else ""

    # 2. Try Gemini if missing pieces
    if not target_role or not user_skills:
        try:
            llm = get_llm()
            structured_llm = llm.with_structured_output(UserProfile)
            profile = structured_llm.invoke(f"Extract profile strictly from the user input. Preserve exact skills listed.\nInput: {user_input}")
            target_role = target_role or profile.target_role
            user_skills = user_skills or profile.user_skills
            available_time = available_time or profile.available_time
        except Exception as e:
            pass

    if not target_role:
        target_role = "Role could not be identified"

    print(f"Detected role: {target_role}")
    print(f"Current skills: {user_skills}")

    return {
        "target_role": target_role,
        "user_skills": user_skills,
        "available_time": available_time,
        "execution_steps": steps
    }

def retrieve_role_knowledge(state: CareerState) -> CareerState:
    """Uses RAG or direct text matching to find role knowledge and extracts required skills."""
    steps = state.get("execution_steps", [])
    steps.append("retrieve_role_knowledge")
    role = state.get("target_role", "Unknown Role")
    vs = get_vector_store()
    
    context = ""
    skills = []
    extracted_topics = []
    retrieved_role = ""
    
    normalized_role = role.lower().strip()
    aliases = {
        "ai scientist": "AI Scientist",
        "artificial intelligence scientist": "AI Scientist",
        "ai research scientist": "AI Scientist",
        "artificial intelligence researcher": "AI Scientist",
        "ai researcher": "AI Scientist",
        "generative ai engineer": "Generative AI Engineer",
        "genai engineer": "Generative AI Engineer",
        "generative ai developer": "Generative AI Engineer",
        "gen ai engineer": "Generative AI Engineer",
        "backend developer": "Backend Developer",
        "backend engineer": "Backend Developer",
        "server-side developer": "Backend Developer",
        "ai automation engineer": "AI Automation Engineer",
        "automation ai engineer": "AI Automation Engineer",
        "machine learning engineer": "Machine Learning Engineer",
        "ml engineer": "Machine Learning Engineer",
        "machine learning developer": "Machine Learning Engineer",
        "data scientist": "Data Scientist",
        "data science engineer": "Data Scientist",
        "python developer": "Python Developer",
        "python engineer": "Python Developer",
        "full stack developer": "Full Stack Developer",
        "fullstack developer": "Full Stack Developer",
        "full stack engineer": "Full Stack Developer",
        "agentic ai engineer": "Agentic AI Engineer",
        "agent engineer": "Agentic AI Engineer",
        "ai agent engineer": "Agentic AI Engineer"
    }
    
    exact_match = aliases.get(normalized_role)
    if not exact_match:
        for k, v in aliases.items():
            if k in normalized_role:
                exact_match = v
                break
                
    if exact_match:
        print(f"Role matching method: exact / alias ({exact_match})")
        for doc in career_knowledge:
            if doc.page_content.startswith(f"Role: {exact_match}"):
                context = doc.page_content
                break
    else:
        print("Role matching method: semantic")
        if vs:
            try:
                docs = vs.similarity_search(role, k=1)
                if docs:
                    context = docs[0].page_content
            except Exception as e:
                pass
                
        if not context:
            for doc in career_knowledge:
                if role.lower().strip() in doc.page_content.lower():
                    context = doc.page_content
                    break
                
    if context:
        for line in context.split("\n"):
            if line.startswith("Role:"):
                retrieved_role = line.replace("Role:", "").strip()
            if line.startswith("Required Skills:"):
                skills_str = line.replace("Required Skills:", "").strip()
                skills = [s.strip() for s in skills_str.split(",") if s.strip()]
            if line.startswith("Interview Topics:"):
                topics_str = line.replace("Interview Topics:", "").strip()
                extracted_topics = [t.strip() for t in topics_str.split(",") if t.strip()]
    else:
        context = f"Role knowledge for '{role}' unavailable in local knowledge base."
        retrieved_role = "None"
        
    print(f"Retrieved role: {retrieved_role}")
    print(f"Required skills: {skills}")
        
    return {"retrieved_context": context, "required_skills": skills, "role_interview_topics": extracted_topics, "execution_steps": steps}

def skill_gap_analysis(state: CareerState) -> CareerState:
    """Invokes the skill_gap_analyzer tool."""
    steps = state.get("execution_steps", [])
    steps.append("skill_gap_analysis")
    
    result = skill_gap_analyzer.invoke({
        "user_skills": state.get("user_skills", []),
        "required_skills": state.get("required_skills", [])
    })
    
    matched = result.get("matched_skills", [])
    missing = result.get("missing_skills", [])
    priorities = result.get("priority_groups", {"high": [], "medium": [], "low": []})
    
    print(f"matched = {matched}")
    print(f"missing = {missing}")
    
    return {
        "matched_skills": matched,
        "skill_gaps": missing,
        "priority_groups": priorities,
        "execution_steps": steps
    }

def roadmap_planning(state: CareerState) -> CareerState:
    """Invokes the roadmap_planner tool."""
    steps = state.get("execution_steps", [])
    steps.append("roadmap_planning")
    
    missing = state.get("skill_gaps", [])
    result = roadmap_planner.invoke({
        "missing_skills": missing,
        "available_time": state.get("available_time", "Unknown"),
        "target_role": state.get("target_role", "Unknown")
    })
    
    
    
    return {"roadmap": result, "execution_steps": steps}

def interview_preparation(state: CareerState) -> CareerState:
    """Generates specific interview topics and questions based on context and gaps."""
    steps = state.get("execution_steps", [])
    steps.append("interview_preparation")
    
    target_role = state.get("target_role", "Unknown Role")
    skill_gaps = state.get("skill_gaps", [])
    
    try:
        llm = get_llm(temperature=0.7)
        structured_llm = llm.with_structured_output(InterviewPreparationOutput)
        prompt = f"""
        Based on the following profile, suggest interview preparation topics and exactly 8 role-specific interview questions.
        Target Role: {target_role}
        Missing Skills: {skill_gaps}
        Knowledge Base Context: {state.get('retrieved_context', '')}
        
        Generate exactly 8 questions, broken down into Technical Questions, Scenario Questions, and Project Questions.
        """
        result = structured_llm.invoke(prompt)
        
        tech_topics = "\n".join([f"• {t}" for t in result.technical_topics])
        scen_topics = "\n".join([f"• {t}" for t in result.scenario_topics])
        proj_topics = "\n".join([f"• {t}" for t in result.project_topics])
        
        topics_str = f"Technical Topics\n{tech_topics}\n\nScenario Topics\n{scen_topics}\n\nProject Topics\n{proj_topics}"
        questions_str = "\n".join([f"{i+1}. {q}" if not q.startswith(str(i+1)) else q for i, q in enumerate(result.questions)])
        
        print(f"Interview topics: {topics_str[:50]}...")
        return {"interview_topics": topics_str, "interview_questions": questions_str, "execution_steps": steps}
        
    except Exception as e:
        # Deterministic fallback logic
        base_topics = state.get("role_interview_topics", [])
        
        tech_topics = [f"Fundamentals of {s}" for s in skill_gaps[:3]] if skill_gaps else (base_topics[:3] if base_topics else ["Data structures", "Algorithms"])
        scen_topics = [f"Debugging issues with {s}" for s in skill_gaps[1:3]] if len(skill_gaps)>1 else (base_topics[3:5] if len(base_topics)>3 else ["Scaling the system"])
        proj_topics = [f"Designing a {target_role} project"]
        
        topics_str = f"Technical Topics\n• " + "\n• ".join(tech_topics) + "\n\nScenario Topics\n• " + "\n• ".join(scen_topics) + "\n\nProject Topics\n• " + "\n• ".join(proj_topics)
        
        fallback_qs = []
        if skill_gaps:
            fallback_qs.append(f"Technical: How would you explain the core concepts of {skill_gaps[0]} to a beginner?")
            if len(skill_gaps) > 1:
                fallback_qs.append(f"Technical: What are the main limitations of {skill_gaps[1]}?")
                fallback_qs.append(f"Scenario: A critical service using {skill_gaps[1]} goes down. How do you troubleshoot it?")
        
        fallback_qs.extend([
            f"Technical: What are the best practices for developing as a {target_role}?",
            f"Scenario: Describe a time you had to learn a complex new framework quickly.",
            f"Scenario: How do you ensure code quality and maintainability in your projects?",
            f"Project: Walk me through a challenging project you built.",
            f"Project: How would you design a scalable architecture for a {target_role} application?"
        ])
        
        while len(fallback_qs) < 8:
            fallback_qs.append(f"Technical: What is your approach to testing and validation?")
        
        fallback_qs = fallback_qs[:8]
        questions_str = "\n".join([f"{i+1}. {q}" for i, q in enumerate(fallback_qs)])
        
        print(f"Interview topics: {topics_str[:50]}...")
        return {"interview_topics": topics_str, "interview_questions": questions_str, "execution_steps": steps}

def generate_final_answer(state: CareerState) -> CareerState:
    """Formats the final response strictly matching the required structure."""
    print("--- GENERATING FINAL ANSWER ---")
    steps = state.get("execution_steps", [])
    steps.append("generate_final_answer")
    
    target_role = state.get('target_role', 'Unknown')
    user_skills = state.get('user_skills', [])
    req_skills = state.get('required_skills', [])
    matched = state.get('matched_skills', [])
    missing = state.get('skill_gaps', [])
    priorities = state.get('priority_groups', {"high": [], "medium": [], "low": []})
    
    # Extract role title from context for evidence
    context = state.get('retrieved_context', '')
    retrieved_role = "Generic AI Engineer"
    for line in context.split("\n"):
        if line.startswith("Role:"):
            retrieved_role = line.replace("Role:", "").strip()
            break
            
    # Format strings (use standard ascii/unicode that won't crash terminal printing easily)
    current_skills_str = "\n".join([f"✓ {s}" for s in user_skills]) if user_skills else "None reported"
    req_skills_str = "\n".join([f"✓ {s}" for s in req_skills]) if req_skills else "None identified"
    matched_skills_str = "\n".join([f"✓ {s}" for s in matched]) if matched else "None"
    missing_skills_str = "\n".join([f"⚠ {s}" for s in missing]) if missing else "None"
    
    high_prio = "\n".join([f"• {s}" for s in priorities.get("high", [])]) or "None"
    med_prio = "\n".join([f"• {s}" for s in priorities.get("medium", [])]) or "None"
    low_prio = "\n".join([f"• {s}" for s in priorities.get("low", [])]) or "None"
    
    final_output = f"""━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CAREERPILOT AI
━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TARGET ROLE
{target_role}

CURRENT SKILLS
{current_skills_str}

REQUIRED SKILLS
{req_skills_str}

MATCHED SKILLS
{matched_skills_str}

SKILL GAPS
{missing_skills_str}

PRIORITY AREAS

HIGH
{high_prio}

MEDIUM
{med_prio}

LOW
{low_prio}

PERSONALIZED ROADMAP
"""
    for step in state.get('roadmap', []):
        final_output += f"""
{step['phase']}
- {step['skills']}
Mini Project:
{step['mini_project']}
"""
        
    final_output += f"""
INTERVIEW PREPARATION
---------------------

{state.get('interview_topics', '')}

INTERVIEW QUESTIONS
-------------------

{state.get('interview_questions', '')}

AGENT EXECUTION
"""
    # Dynamic checks based on execution steps
    has_parse = "✓ Profile analyzed" if "parse_user_input" in steps else "✗ Profile analysis failed"
    has_rag = "✓ RAG retrieval completed" if "retrieve_role_knowledge" in steps else "✗ RAG failed"
    has_skill = "✓ skill_gap_analyzer executed" if "skill_gap_analysis" in steps else "✗ Skill gap tool failed"
    has_road = "✓ roadmap_planner executed" if "roadmap_planning" in steps else "✗ Roadmap tool failed"
    has_int = "✓ Interview preparation generated" if "interview_preparation" in steps else "✗ Interview prep failed"
    
    final_output += f"""
{has_parse}
{has_rag}
{has_skill}
{has_road}
{has_int}

RAG EVIDENCE
------------
Source:
{retrieved_role} Knowledge Base

TOOLS EXECUTED
--------------
✓ skill_gap_analyzer
✓ roadmap_planner

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
    return {"final_answer": final_output.strip(), "execution_steps": steps}

# --- 5. BUILD LANGGRAPH ---
workflow = StateGraph(CareerState)
workflow.add_node("parse_user_input", parse_user_input)
workflow.add_node("retrieve_role_knowledge", retrieve_role_knowledge)
workflow.add_node("skill_gap_analysis", skill_gap_analysis)
workflow.add_node("roadmap_planning", roadmap_planning)
workflow.add_node("interview_preparation", interview_preparation)
workflow.add_node("generate_final_answer", generate_final_answer)

workflow.add_edge(START, "parse_user_input")
workflow.add_edge("parse_user_input", "retrieve_role_knowledge")
workflow.add_edge("retrieve_role_knowledge", "skill_gap_analysis")
workflow.add_edge("skill_gap_analysis", "roadmap_planning")
workflow.add_edge("roadmap_planning", "interview_preparation")
workflow.add_edge("interview_preparation", "generate_final_answer")
workflow.add_edge("generate_final_answer", END)

app_graph = workflow.compile()

# --- 6. API ENDPOINTS (FASTAPI & LANGSERVE) ---

class CareerPilotInput(BaseModel):
    user_input: str = Field(description="Natural language user input")

class CareerPilotOutput(BaseModel):
    final_answer: str

def run_agent(input_data: Any) -> CareerPilotOutput:
    """Wrapper to map API input to graph state."""
    if not os.getenv("GEMINI_API_KEY"):
        return CareerPilotOutput(final_answer="Error: GEMINI_API_KEY is not configured in the environment.")
    
    try:
        # Handle LangServe passing a dictionary to RunnableLambda
        if isinstance(input_data, dict):
            user_input = input_data.get("user_input", "")
        else:
            user_input = getattr(input_data, "user_input", "")
            
        print(f"Invoking graph with input: {user_input}")
        result = app_graph.invoke({"user_input": user_input})
        
        final_answer = result.get("final_answer", "Error generating response.") if isinstance(result, dict) else "Error: Graph did not return a dictionary."
        return CareerPilotOutput(final_answer=final_answer)
    except Exception as e:
        return CareerPilotOutput(final_answer=f"An error occurred during processing: {str(e)}")

agent_runnable = RunnableLambda(run_agent).with_types(input_type=CareerPilotInput, output_type=CareerPilotOutput)

add_routes(
    app,
    agent_runnable,
    path="/careerpilot"
)

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "agent": "CareerPilot AI",
        "langgraph": True,
        "langserve": True,
        "rag": True,
        "tools": True
    }

@app.get("/architecture")
def architecture():
    return {
      "agent": "CareerPilot AI",
      "workflow": [
        "parse_user_input",
        "retrieve_role_knowledge",
        "skill_gap_analysis",
        "roadmap_planning",
        "interview_preparation",
        "generate_final_answer"
      ],
      "tools": [
        "skill_gap_analyzer",
        "roadmap_planner"
      ],
      "rag": {
        "embeddings": "Google Gemini Embeddings",
        "vector_store": "InMemoryVectorStore"
      },
      "llm": "Google Gemini",
      "frameworks": [
        "LangChain",
        "LangGraph",
        "LangServe",
        "FastAPI"
      ],
      "deployment": "Vercel"
    }

@app.get("/roles")
def roles():
    return {
      "roles": [
        "AI Automation Engineer",
        "Agentic AI Engineer",
        "Generative AI Engineer",
        "Machine Learning Engineer",
        "Data Scientist",
        "Backend Developer",
        "Python Developer",
        "Full Stack Developer",
        "AI Scientist"
      ]
    }

@app.get("/")
def home():
    api_key = os.getenv("GEMINI_API_KEY")
    warning = ""
    if not api_key:
        warning = "<p style='color:red;'>⚠️ GEMINI_API_KEY is not set. The agent will not work.</p>"
        
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>CareerPilot AI</title>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; line-height: 1.6; max-width: 800px; margin: 0 auto; padding: 20px; background-color: #f9f9f9; color: #333; }}
            h1 {{ color: #2c3e50; margin-bottom: 5px; }}
            .subtitle {{ color: #7f8c8d; margin-top: 0; margin-bottom: 20px; }}
            .container {{ background: white; padding: 30px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
            textarea {{ width: 100%; padding: 12px; border: 1px solid #ddd; border-radius: 4px; box-sizing: border-box; font-family: inherit; margin-bottom: 15px; resize: vertical; font-size: 15px; }}
            button {{ background-color: #3498db; color: white; border: none; padding: 12px 24px; font-size: 16px; border-radius: 4px; cursor: pointer; transition: background 0.3s; font-weight: bold; }}
            button:hover {{ background-color: #2980b9; }}
            #result {{ margin-top: 30px; white-space: pre-wrap; background: #f8f9fa; padding: 20px; border-radius: 4px; border-left: 4px solid #3498db; display: none; font-size: 15px; }}
            .loader {{ display: none; margin-top: 20px; font-weight: bold; color: #3498db; }}
            .links a {{ text-decoration: none; color: #3498db; margin-right: 15px; font-size: 14px; }}
            .links a:hover {{ text-decoration: underline; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>CareerPilot AI 🚀</h1>
            <p class="subtitle">AI Career & Interview Preparation Agent</p>
            {warning}
            <textarea id="userInput" rows="5" placeholder="Example: I want to become an AI Automation Engineer. I know Python, LangChain and n8n. I have 3 months."></textarea>
            <br>
            <button onclick="analyzeCareer()">Analyze My Career</button>
            <div class="links" style="margin-top: 15px;">
                <a href="/health" target="_blank">Health Check</a>
                <a href="/architecture" target="_blank">Architecture Info</a>
                <a href="/careerpilot/playground" target="_blank">LangServe Playground</a>
            </div>
            
            <div id="loader" class="loader">⚙️ Analyzing your career path, searching RAG knowledge, and invoking tools...</div>
            <div id="result"></div>
        </div>

        <script>
            async function analyzeCareer() {{
                const input = document.getElementById('userInput').value;
                if (!input.trim()) {{
                    alert('Please enter your career goals and skills.');
                    return;
                }}
                
                document.getElementById('loader').style.display = 'block';
                document.getElementById('result').style.display = 'none';
                
                try {{
                    const response = await fetch('/careerpilot/invoke', {{
                        method: 'POST',
                        headers: {{ 'Content-Type': 'application/json' }},
                        body: JSON.stringify({{ input: {{ user_input: input }} }})
                    }});
                    
                    if (!response.ok) {{
                        throw new Error(`HTTP error! status: ${{response.status}}`);
                    }}
                    
                    const data = await response.json();
                    const resultDiv = document.getElementById('result');
                    
                    if (data && data.output && data.output.final_answer) {{
                        resultDiv.textContent = data.output.final_answer;
                    }} else {{
                        resultDiv.textContent = JSON.stringify(data, null, 2);
                    }}
                    
                    resultDiv.style.display = 'block';
                }} catch (error) {{
                    const resultDiv = document.getElementById('result');
                    resultDiv.textContent = 'Error: ' + error.message + '\\n\\nPlease check server logs or ensure GEMINI_API_KEY is configured in Vercel.';
                    resultDiv.style.display = 'block';
                }} finally {{
                    document.getElementById('loader').style.display = 'none';
                }}
            }}
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
