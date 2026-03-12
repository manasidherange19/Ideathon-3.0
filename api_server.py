# api_server.py
from startup_evaluator_complete import AdvancedStartupEvaluator
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict
import uvicorn
import os

# Initialize FastAPI
app = FastAPI(title="StartupEvaluator AI API", version="2.0")

# Initialize evaluator
evaluator = AdvancedStartupEvaluator()

# Load or train models on startup
@app.on_event("startup")
async def startup_event():
    print("🚀 Starting Startup Evaluator API Server...")
    model_path = 'models/'
    if os.path.exists(model_path):
        try:
            evaluator.load_models(model_path)
            print("✅ Models loaded successfully!")
        except:
            print("⚠️ Error loading models, training new ones...")
            df = evaluator.generate_rich_synthetic_data(20000)
            evaluator.train_all_models(df)
            evaluator.save_models(model_path)
    else:
        print("🔄 No models found, training new ones...")
        df = evaluator.generate_rich_synthetic_data(20000)
        evaluator.train_all_models(df)
        evaluator.save_models(model_path)
    print("✅ Server ready!")

# Define input model
class StartupInput(BaseModel):
    idea: str
    market_size: str
    competitors: str

# Define response model
class PredictionResponse(BaseModel):
    success_probability: float
    confidence: float
    risk_level: str
    market_score: float
    team_score: float
    financial_health: float
    competitive_moat: float
    plan: Dict
    timestamp: str

# API endpoints
@app.get("/")
async def root():
    return {
        "message": "🚀 Startup Evaluator AI API is running!",
        "docs": "/docs",
        "endpoints": {
            "POST /predict": "Evaluate a startup idea",
            "GET /health": "Check API health"
        }
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_startup(input_data: StartupInput):
    """Complete startup evaluation endpoint"""
    try:
        result = evaluator.predict_from_input(
            input_data.idea,
            input_data.market_size,
            input_data.competitors
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "models_loaded": len(evaluator.models),
        "sectors_supported": len(evaluator.sectors)
    }

if __name__ == "__main__":
    uvicorn.run(
        "api_server:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True
    )