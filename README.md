# 6G Network Slice Orchestrator — LLM-Powered Agentic AI Framework

## Architecture

```
┌─────────────────────────────────────────────────┐
│                  Next.js Frontend                │
│         (Dashboard + Real-time Charts)           │
└───────────────────┬─────────────────────────────┘
                    │ WebSocket + REST
┌───────────────────▼─────────────────────────────┐
│                FastAPI Backend                    │
│  ┌───────────┐ ┌──────────┐ ┌────────────────┐  │
│  │ Simulator  │ │ DRL Agent│ │ Agentic Pipeline│ │
│  │ (Gymnasium)│ │ (PPO/SB3)│ │ (LLM + RAG)    │ │
│  └───────────┘ └──────────┘ └────────────────┘  │
└─────────────────────────────────────────────────┘
```

## Quick Start

### 1. Backend (FastAPI + Simulator)
```bash
cd backend
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn api.main:app --reload --port 8000
```

### 2. Frontend (Next.js Dashboard)
```bash
cd frontend
npm install
npm run dev    # runs on port 3000
```

### 3. Train the DRL Agent
```bash
cd backend
python -m rl.train --timesteps 500000
```

## Project Structure
```
backend/
├── simulator/
│   ├── network.py           # Core network state & physics
│   ├── slice_config.py      # Slice definitions (eMBB, URLLC, mMTC)
│   ├── traffic.py           # Traffic generators per slice type
│   ├── channel.py           # Wireless channel model (Rayleigh fading)
│   └── env.py               # Gymnasium environment wrapper
├── rl/
│   ├── train.py             # PPO training script
│   ├── policy.py            # Custom policy network
│   └── evaluate.py          # Evaluation & baseline comparisons
├── agents/
│   ├── monitor.py           # Monitor Agent — anomaly detection
│   ├── diagnosis.py         # Diagnosis Agent — RAG-powered root cause
│   └── optimizer.py         # Optimizer Agent — action selection
├── rag/
│   ├── knowledge_base/      # 6G standards & SLA docs
│   ├── indexer.py           # Document embedding & indexing
│   └── retriever.py         # Context retrieval
├── api/
│   ├── main.py              # FastAPI app + WebSocket
│   ├── routes.py            # REST endpoints
│   └── schemas.py           # Pydantic models
├── requirements.txt
└── config.py                # Global configuration

frontend/
├── src/app/
│   ├── page.tsx             # Main dashboard
│   ├── layout.tsx           # App layout
│   └── globals.css
├── src/components/
│   ├── SliceMetrics.tsx     # Per-slice KPI cards
│   ├── AllocationChart.tsx  # Resource allocation over time
│   ├── AgentLog.tsx         # Agent decision trace viewer
│   └── LiveChart.tsx        # Real-time line chart
├── src/hooks/
│   └── useWebSocket.ts     # WebSocket hook for live data
├── package.json
└── tailwind.config.ts
```
