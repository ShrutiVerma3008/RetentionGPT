# 🚀 RetentionAI (RetentionGPT) — New Member Onboarding Guide

Welcome to the **RetentionAI** team! This guide will get you up to speed unconditionally on what we're building, the problem we're solving, our tech stack, and exactly how to run / extend the application without running into errors.

---

## 🎯 1. What Problem Are We Solving?

In the BFSI (Banking, Financial Services, and Insurance) sector, **customer churn** (when users close their accounts or switch to competitors) represents millions of dollars in lost lifetime value (CLV). 

Traditional banking tools are merely reactive—they tell relationship managers *who* has churned. **RetentionAI changes this from reactive to proactive, predictive, and prescriptive.** 

We aren't just predicting a churn percentage. Our platform answers four critical questions:
1. **Who is at risk?** (XGBoost Churn Prediction Model)
2. **Why are they at risk?** (SHAP Explainability - what factors pushed the score up?)
3. **What is their emotional state?** (BERT Sentiment Analysis on complaints/interactions)
4. **What is the Next Best Action?** (Digital Twin counterfactual simulations + LangChain outreach generation)

---

## 🛠️ 2. The Current Tech Stack

We have intentionally chosen a powerful but hackathon/demo-friendly tech stack:

### **Frontend (UI/UX)**
- **Vanilla HTML / CSS / JS**: We built a single file (`Retentiongpt_dashboard.html`) to keep it lightweight, lightning-fast, and heavily customized without framework bloat.
- **Lucide Icons**: Clean, premium SVG iconography.
- **Features**: Features an ultra-premium asymmetrical layout with 3 interactive modes: Overview, Action Mode, and Digital Twin Simulation.

### **Backend (API Layer)**
- **FastAPI**: A modern, exceptionally fast Python backend framework.
- **Uvicorn**: ASGI web server routing the requests to the frontend.
- **Architecture**: Contains 7 core REST API endpoints covering predictions, customer listings, digital twin simulations, and LangChain generation.

### **Machine Learning & AI Layer**
- **XGBoost**: Our core binary classification engine trained to output a churn probability (0 to 1). Handles class imbalances natively.
- **SHAP (SHapley Additive exPlanations)**: The interpreter. It breaks down *why* XGBoost gave a score so Relationship Managers can see exact timeline factors (e.g. `Transaction velocity dropped` → `+22% risk`).
- **DistilBERT (zero-shot classification)**: We use the HuggingFace transformers pipeline for offline, instant emotion detection (outputs tags like: *Frustrated*, *Confused*, *Price-Sensitive*).
- **LangChain + OpenAI**: A 4-module sequential pipeline (`Risk Analyst` → `Empathy Engine` → `Action Selector` → `Outreach Writer`) that drafts custom, highly personalized emails and WhatsApp messages based on the customer's exact data. 

---

## 🏃‍♀️ 3. How to Run the Project (Error-Free)

The current environment is already working perfectly! Follow these steps if your new team member is setting it up from scratch on their machine.

### **Step A: Environment Setup**
```bash
# Clone the repository
git clone <repo>
cd retentionAI

# Create a virtual environment and activate it 
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux

# Install all required data, ML, and server dependencies
pip install -r backend/requirements.txt
```

### **Step B: Environment Variables**
Copy `.env.template` to `.env`:
```bash
cp .env.template .env
```
Add your OpenAI API Key into the `.env` file (`OPENAI_API_KEY=sk-...`). 
> *Note: If you don't have an API key, the script is designed to fallback to "demo mode" seamlessly without throwing an error!*

### **Step C: Generate Data & Train the ML Model**
We intentionally generate our own synthetic BFSI dataset so we don't commit PII (Personal Identifiable Information) to GitHub.
```bash
# 1. Generate 500 rows of synthetic, highly correlated banking data
python ml/generate_dataset.py

# 2. Train the XGBoost model on that data. This creates /backend/models/xgb_churn.pkl
python ml/train_xgboost.py
```

### **Step D: Start the Backend & Frontend**
```bash
# Start the FastAPI backend
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```
Then, double-click the `Retentiongpt_dashboard.html` file to open it in your Chrome/Edge browser. (Or use VSCode Live Server). Because we have explicitly enabled CORS in `main.py`, the frontend will seamlessly talk to your local port 8000!

---

## 🧑‍💻 4. How to Continue Further (Development Next Steps)

If you are picking up development, here is where you can instantly add value:

1. **Database Integration**: Right now, the `backend/main.py` uses an in-memory `MOCK_CUSTOMERS` Python dictionary for speed. Transition this to **PostgreSQL** or **MongoDB**.
2. **Jupyter Notebook Presentation**: We've included `/ml/RetentionAI_ML_Pipeline.ipynb`. Use this if you are presenting to technical judges. It contains all the step-by-step logic and generates beautiful model evaluation graphs (ROC, AUC curves).
3. **Advanced Digital Twin**: The `services/digital_twin.py` simulates intervention scenarios (e.g., "What happens if we give a premier upgrade?"). Right now, it relies on static decay coefficients. We can explore using Reinforcement Learning here!
4. **Deploying (Docker)**: The project has a `Dockerfile` and `docker-compose.yml` pre-configured. To continuously test deployment, make sure `docker-compose up --build` continues to build perfectly.

> **Zero-Error Guarantee for Demo**: All error-prone services (like the OpenAI generation or HuggingFace DistilBERT) are wrapped in `try-except` fallback structures. If your teammate forgets an API key or a massive PyTorch download fails, the app politely falls back to hardcoded regex heuristics and templated strings so your frontend demo *never breaks in front of judges*.
