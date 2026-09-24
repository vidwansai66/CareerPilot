# 🚀 CareerPilot AI

**CareerPilot AI** is an intelligent, agentic career mapping and interview preparation assistant built with **LangGraph**, **LangServe**, **FastAPI**, and **Google Gemini**. 

It accepts a user's natural-language career goals, explicit skills, and available timeframe, then dynamically generates a personalized skill-gap analysis, learning roadmap, and role-specific interview questions.

---

## 🌟 Features

- **🧠 Natural Language Understanding**: Parses unstructured user inputs (e.g., *"I want to become an AI Automation Engineer. I know Python and LangChain. I have 3 months."*) using a combination of LLM structured parsing and deterministic regex fallbacks.
- **📚 Role-Specific Knowledge Base (RAG)**: Leverages an in-memory vector store to retrieve the exact requirements, frameworks, and prerequisites for various technical roles.
- **⚡ Dynamic Skill Gap Analysis**: Automatically computes missing skills by comparing what the user explicitly knows against industry-standard role requirements.
- **🗺️ Personalized Learning Roadmap**: Creates a prioritized, phase-by-phase learning sequence that respects the user's available timeline.
- **🎯 Interview Preparation Engine**: Dynamically generates targeted technical, scenario-based, and project-based interview questions addressing the user's specific skill gaps.
- **🛡️ High Reliability**: Built with comprehensive, deterministic fallbacks ensuring the agent never crashes or hallucinates if upstream LLM APIs (like Gemini) encounter outages or rate limits.

---

## 🛠️ Tech Stack

- **Framework**: [FastAPI](https://fastapi.tiangolo.com/)
- **AI Orchestration**: [LangGraph](https://python.langchain.com/docs/langgraph) & [LangChain](https://python.langchain.com/)
- **API Deployment**: [LangServe](https://python.langchain.com/docs/langserve)
- **LLM Engine**: Google Gemini (via `langchain-google-genai`)
- **Vector Search**: `InMemoryVectorStore`

---

## 📂 Supported Roles

CareerPilot natively understands and supports dynamic aliases for the following career tracks:
- AI Automation Engineer
- Agentic AI Engineer
- Generative AI Engineer
- Machine Learning Engineer
- AI Scientist
- Data Scientist
- Backend Developer
- Python Developer
- Full Stack Developer

---

## 🚀 Getting Started (Local Development)

### Prerequisites
- Python 3.9+
- A Google Gemini API Key

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/careerpilot-ai.git
   cd careerpilot-ai
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set your environment variable:**
   ```bash
   # Windows (PowerShell)
   $env:GEMINI_API_KEY="your_api_key_here"
   
   # Mac/Linux
   export GEMINI_API_KEY="your_api_key_here"
   ```

4. **Run the server:**
   ```bash
   uvicorn app:app --reload
   ```

5. **Test the agent:**
   Open the LangServe Playground in your browser:
   [http://localhost:8000/careerpilot/playground/](http://localhost:8000/careerpilot/playground/)

---

## 📡 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Basic health check and welcome message |
| `/roles` | GET | Returns a JSON list of supported career roles |
| `/architecture` | GET | Returns the JSON representation of the LangGraph state machine |
| `/careerpilot/invoke` | POST | Submits a query to the agent via LangServe API |
| `/careerpilot/playground/` | GET | Opens the interactive LangServe web UI |

---

## ☁️ Deployment (Vercel)

This project is specifically architected to be completely stateless and strictly bounded to `app.py` and `requirements.txt` to guarantee seamless serverless deployment on **Vercel**.

1. Install the Vercel CLI: `npm i -g vercel`
2. Run `vercel` in the project root.
3. Add your `GEMINI_API_KEY` to the Vercel project environment variables.

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! 
Feel free to check [issues page](https://github.com/yourusername/careerpilot-ai/issues).

## 📝 License

This project is [MIT](https://choosealicense.com/licenses/mit/) licensed.
