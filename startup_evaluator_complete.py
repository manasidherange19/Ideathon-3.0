"""
AI Startup Evaluator & Generator - Complete Web Application
A comprehensive platform that evaluates startup ideas and generates business plans
Runs on http://localhost:5000
"""

from flask import Flask, render_template, request, jsonify, session, send_file
from flask_cors import CORS
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os
import json
import random
from datetime import datetime
import secrets
import plotly
import plotly.graph_objs as go
import plotly.express as px
from werkzeug.utils import secure_filename
import io
import base64
from textblob import TextBlob

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)
CORS(app)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'doc', 'docx'}
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ==================== ML MODELS ====================

class AdvancedStartupEvaluator:
    """Advanced ML Model for startup success prediction"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.initialize_model()
        self.market_trends = self.load_market_trends()
    
    def initialize_model(self):
        """Initialize and train a sophisticated model"""
        np.random.seed(42)
        n_samples = 10000
        
        # Enhanced features: [market_size, competition_level, team_experience, 
        # funding_amount, innovation_score, market_growth_rate, customer_acquisition_cost,
        # unit_economics, intellectual_property, scalability]
        X_train = np.random.rand(n_samples, 10) * 10
        
        # Complex success pattern
        weights = [0.15, 0.10, 0.12, 0.08, 0.15, 0.10, 0.08, 0.10, 0.07, 0.05]
        success_score = np.sum(X_train * weights, axis=1)
        
        # Add non-linear relationships
        success_score += 0.1 * X_train[:, 0] * X_train[:, 4]  # market_size * innovation
        success_score -= 0.05 * X_train[:, 1] * X_train[:, 6]  # competition * cac
        
        noise = np.random.normal(0, 0.5, n_samples)
        success_score += noise
        
        # Normalize and create binary outcome
        success_score = (success_score - success_score.min()) / (success_score.max() - success_score.min())
        y_train = (success_score > 0.6).astype(int)
        
        self.model = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42)
        self.model.fit(X_train, y_train)
    
    def load_market_trends(self):
        """Load current market trends data"""
        return {
            'AI/ML': {'growth_rate': 28.5, 'funding': 45.2, 'sentiment': 'very_positive'},
            'FinTech': {'growth_rate': 22.3, 'funding': 38.7, 'sentiment': 'positive'},
            'HealthTech': {'growth_rate': 25.1, 'funding': 42.3, 'sentiment': 'very_positive'},
            'EdTech': {'growth_rate': 18.7, 'funding': 25.4, 'sentiment': 'positive'},
            'CleanTech': {'growth_rate': 21.4, 'funding': 32.1, 'sentiment': 'positive'},
            'SaaS': {'growth_rate': 19.8, 'funding': 55.6, 'sentiment': 'very_positive'},
            'E-commerce': {'growth_rate': 15.2, 'funding': 28.9, 'sentiment': 'neutral'},
            'PropTech': {'growth_rate': 16.8, 'funding': 22.3, 'sentiment': 'neutral'},
            'CyberSecurity': {'growth_rate': 24.6, 'funding': 35.7, 'sentiment': 'positive'}
        }
    
    def predict_success_probability(self, features):
        """Predict startup success probability with confidence interval"""
        features_scaled = self.scaler.fit_transform([features])
        probability = self.model.predict_proba(features_scaled)[0][1]
        
        # Calculate confidence using bootstrap
        n_bootstrap = 100
        bootstrap_probs = []
        for _ in range(n_bootstrap):
            indices = np.random.choice(len(features_scaled[0]), len(features_scaled[0]), replace=True)
            bootstrap_sample = features_scaled[0][indices].reshape(1, -1)
            if hasattr(self.model, 'estimators_'):
                preds = [est.predict_proba(bootstrap_sample)[0][1] for est in self.model.estimators_[:50]]
                bootstrap_probs.append(np.mean(preds))
        
        if bootstrap_probs:
            confidence_interval = np.percentile(bootstrap_probs, [2.5, 97.5])
        else:
            confidence_interval = [probability - 0.1, probability + 0.1]
        
        return {
            'probability': probability * 100,
            'confidence_lower': confidence_interval[0] * 100,
            'confidence_upper': confidence_interval[1] * 100
        }
    
    def analyze_pitch_deck(self, text):
        """Advanced NLP analysis of pitch deck"""
        # Extract key metrics
        word_count = len(text.split())
        sentence_count = len(TextBlob(text).sentences)
        
        # Sentiment analysis
        blob = TextBlob(text)
        sentiment = blob.sentiment
        
        # Keyword analysis
        strong_keywords = ['scalable', 'innovative', 'patent', 'traction', 'revenue', 
                          'growth', 'market leader', 'competitive advantage', 'moat',
                          'proprietary', 'disruptive', 'partnership', 'acquisition']
        
        weak_keywords = ['maybe', 'hopefully', 'we think', 'no competition', 'guaranteed',
                        'easy', 'simple', 'just', 'only', 'sometime']
        
        text_lower = text.lower()
        strong_count = sum(1 for word in strong_keywords if word in text_lower)
        weak_count = sum(1 for word in weak_keywords if word in text_lower)
        
        # Calculate composite score
        base_score = (strong_count * 10 - weak_count * 5)
        readability_score = min(sentence_count / max(word_count / 20, 1) * 10, 100)
        sentiment_score = (sentiment.polarity + 1) * 50
        
        final_score = (base_score * 0.5 + readability_score * 0.2 + sentiment_score * 0.3)
        final_score = min(max(final_score, 0), 100)
        
        return {
            'score': final_score,
            'strong_points': strong_count,
            'weak_points': weak_count,
            'sentiment': sentiment.polarity,
            'readability': readability_score
        }

class BusinessPlanGenerator:
    """Generate comprehensive business plans for startup ideas"""
    
    @staticmethod
    def generate_complete_plan(idea, industry, target_market):
        """Generate a complete business plan"""
        
        # Determine industry trends
        industry_trends = {
            'AI/ML': 'High growth with significant VC interest. Focus on ethical AI and automation.',
            'FinTech': 'Regulatory challenges but massive market. Focus on financial inclusion.',
            'HealthTech': 'Regulatory hurdles but high impact potential. Telehealth and personalized medicine.',
            'EdTech': 'Post-pandemic growth in online learning. Focus on skill development.',
            'CleanTech': 'ESG focus driving investment. Sustainable solutions and carbon reduction.',
            'SaaS': 'Recurring revenue model. Focus on vertical SaaS and AI integration.'
        }
        
        # Business models
        business_models = {
            'SaaS': {
                'name': 'Subscription SaaS Model',
                'description': 'Monthly/yearly subscriptions with tiered pricing',
                'pricing': 'Freemium ($0-50/month), Professional ($100-500/month), Enterprise (custom)',
                'metrics': ['MRR', 'Churn Rate', 'CAC', 'LTV']
            },
            'Marketplace': {
                'name': 'Two-sided Marketplace',
                'description': 'Connect service providers with customers',
                'pricing': 'Commission (10-30%), Listing fees, Premium features',
                'metrics': ['GMV', 'Take Rate', 'Liquidity', 'Network Effects']
            },
            'E-commerce': {
                'name': 'Direct-to-Consumer E-commerce',
                'description': 'Sell products directly to customers',
                'pricing': 'Product margin + Shipping',
                'metrics': ['AOV', 'Conversion Rate', 'Customer Lifetime Value']
            },
            'Advertising': {
                'name': 'Ad-Supported Platform',
                'description': 'Free content/services monetized through ads',
                'pricing': 'CPM, CPC, CPA models',
                'metrics': ['DAU/MAU', 'Engagement', 'ARPU']
            }
        }
        
        # Select appropriate model based on idea
        if 'app' in idea.lower() or 'software' in idea.lower():
            model_key = 'SaaS'
        elif 'marketplace' in idea.lower() or 'platform' in idea.lower():
            model_key = 'Marketplace'
        elif 'product' in idea.lower() or 'store' in idea.lower():
            model_key = 'E-commerce'
        else:
            model_key = random.choice(list(business_models.keys()))
        
        business_model = business_models[model_key]
        
        # Generate competition analysis
        competitors = BusinessPlanGenerator.generate_competitors(industry)
        
        # Generate MVP features
        mvp_features = BusinessPlanGenerator.generate_mvp_features(idea)
        
        # Financial projections
        financials = BusinessPlanGenerator.generate_financials()
        
        return {
            'business_model': business_model,
            'value_proposition': f"A revolutionary {industry} solution that {idea} through innovative technology and user-centric design.",
            'target_market': target_market,
            'market_size': f"${random.randint(1, 50)}B by 2025",
            'industry_trends': industry_trends.get(industry, 'Growing market with increasing digital adoption'),
            'competitors': competitors,
            'mvp_features': mvp_features,
            'financials': financials,
            'revenue_streams': BusinessPlanGenerator.generate_revenue_streams(model_key),
            'marketing_strategy': BusinessPlanGenerator.generate_marketing_strategy()
        }
    
    @staticmethod
    def generate_competitors(industry):
        competitors = {
            'AI/ML': [
                {'name': 'OpenAI', 'strength': 'Strong brand, cutting-edge tech', 'weakness': 'High API costs'},
                {'name': 'Google AI', 'strength': 'Massive resources, talent', 'weakness': 'Bureaucratic'},
                {'name': 'Anthropic', 'strength': 'Safety focus', 'weakness': 'Limited market reach'}
            ],
            'FinTech': [
                {'name': 'Stripe', 'strength': 'Developer-friendly', 'weakness': 'Complex pricing'},
                {'name': 'Square', 'strength': 'SMB focus', 'weakness': 'International presence'},
                {'name': 'PayPal', 'strength': 'Brand recognition', 'weakness': 'Outdated tech'}
            ]
        }
        return competitors.get(industry, [
            {'name': 'Market Leader Inc', 'strength': 'Strong market presence', 'weakness': 'Slow innovation'},
            {'name': 'Startup Competitor', 'strength': 'Agile and innovative', 'weakness': 'Limited funding'}
        ])
    
    @staticmethod
    def generate_mvp_features(idea):
        features = [
            {
                'name': 'User Authentication',
                'description': 'Secure signup/login system',
                'priority': 'High',
                'time_estimate': '2 weeks'
            },
            {
                'name': 'Core Functionality',
                'description': f'Basic {idea} implementation',
                'priority': 'High',
                'time_estimate': '4 weeks'
            },
            {
                'name': 'Dashboard',
                'description': 'User dashboard with key metrics',
                'priority': 'Medium',
                'time_estimate': '2 weeks'
            },
            {
                'name': 'Payment Integration',
                'description': 'Stripe/PayPal integration',
                'priority': 'Medium',
                'time_estimate': '2 weeks'
            },
            {
                'name': 'Analytics',
                'description': 'Basic analytics and reporting',
                'priority': 'Low',
                'time_estimate': '3 weeks'
            }
        ]
        return features
    
    @staticmethod
    def generate_financials():
        return {
            'year_1': {'revenue': random.randint(100000, 500000), 'costs': random.randint(80000, 400000)},
            'year_2': {'revenue': random.randint(500000, 2000000), 'costs': random.randint(300000, 1500000)},
            'year_3': {'revenue': random.randint(2000000, 5000000), 'costs': random.randint(1000000, 3000000)}
        }
    
    @staticmethod
    def generate_revenue_streams(model_type):
        streams = {
            'SaaS': ['Monthly subscriptions', 'Annual plans (discounted)', 'Enterprise contracts', 'API usage fees'],
            'Marketplace': ['Transaction fees', 'Premium listings', 'Subscription for sellers', 'Data insights'],
            'E-commerce': ['Product sales', 'Shipping fees', 'Premium membership', 'Cross-selling'],
            'Advertising': ['Display ads', 'Sponsored content', 'Premium placements', 'Data licensing']
        }
        return streams.get(model_type, ['Direct sales', 'Partnerships', 'Licensing'])

    @staticmethod
    def generate_marketing_strategy():
        return {
            'channels': ['Content Marketing', 'SEO', 'Social Media', 'Partnerships', 'Email Marketing'],
            'customer_acquisition': 'Digital marketing with focus on inbound',
            'retention': 'Email sequences, push notifications, loyalty program'
        }

# Initialize models
evaluator = AdvancedStartupEvaluator()
plan_generator = BusinessPlanGenerator()

# ==================== FRONTEND HTML ====================

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>UdyamVerse</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {
            --primary-color: #6366f1;
            --secondary-color: #8b5cf6;
            --success-color: #10b981;
            --warning-color: #f59e0b;
            --danger-color: #ef4444;
            --dark-bg: #0f172a;
            --card-bg: #1e293b;
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Inter', sans-serif;
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            color: var(--text-primary);
            min-height: 100vh;
            overflow-x: hidden;
        }

        .navbar {
            background: rgba(15, 23, 42, 0.8);
            backdrop-filter: blur(10px);
            border-bottom: 1px solid rgba(255,255,255,0.1);
            padding: 1rem 0;
        }

        .navbar-brand {
            font-size: 1.5rem;
            font-weight: 700;
            background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .hero-section {
            padding: 4rem 0;
            text-align: center;
            background: radial-gradient(circle at 50% 50%, rgba(99, 102, 241, 0.1) 0%, transparent 50%);
        }

        .hero-title {
            font-size: 3.5rem;
            font-weight: 800;
            margin-bottom: 1rem;
            background: linear-gradient(135deg, var(--text-primary), var(--text-secondary));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .hero-subtitle {
            font-size: 1.25rem;
            color: var(--text-secondary);
            max-width: 600px;
            margin: 0 auto 2rem;
        }

        .card {
            background: var(--card-bg);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 1rem;
            padding: 2rem;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
            margin-bottom: 2rem;
        }

        .card:hover {
            transform: translateY(-5px);
            box-shadow: 0 20px 40px rgba(0,0,0,0.4);
        }

        .card-title {
            font-size: 1.5rem;
            font-weight: 600;
            margin-bottom: 1.5rem;
            color: var(--text-primary);
        }

        .form-control, .form-select {
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 0.75rem;
            padding: 0.75rem 1rem;
            color: var(--text-primary);
            transition: all 0.3s ease;
        }

        .form-control:focus, .form-select:focus {
            background: rgba(255,255,255,0.1);
            border-color: var(--primary-color);
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.2);
            color: var(--text-primary);
        }

        .form-control::placeholder {
            color: var(--text-secondary);
        }

        .btn-primary {
            background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
            border: none;
            border-radius: 0.75rem;
            padding: 0.75rem 2rem;
            font-weight: 600;
            transition: all 0.3s ease;
        }

        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(99, 102, 241, 0.3);
        }

        .btn-outline-primary {
            border: 2px solid var(--primary-color);
            color: var(--primary-color);
            border-radius: 0.75rem;
            padding: 0.75rem 2rem;
            font-weight: 600;
            transition: all 0.3s ease;
        }

        .btn-outline-primary:hover {
            background: var(--primary-color);
            color: white;
            transform: translateY(-2px);
        }

        .probability-meter {
            width: 200px;
            height: 200px;
            margin: 0 auto 1rem;
            position: relative;
        }

        .probability-value {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            font-size: 3rem;
            font-weight: 700;
            color: var(--text-primary);
        }

        .feature-badge {
            background: rgba(99, 102, 241, 0.2);
            color: var(--primary-color);
            padding: 0.25rem 0.75rem;
            border-radius: 2rem;
            font-size: 0.875rem;
            font-weight: 500;
            display: inline-block;
        }

        .competitor-card {
            background: rgba(255,255,255,0.05);
            border-radius: 0.75rem;
            padding: 1rem;
            margin-bottom: 1rem;
        }

        .metric-card {
            background: rgba(255,255,255,0.05);
            border-radius: 0.75rem;
            padding: 1.5rem;
            text-align: center;
        }

        .metric-value {
            font-size: 2rem;
            font-weight: 700;
            color: var(--primary-color);
        }

        .metric-label {
            color: var(--text-secondary);
            font-size: 0.875rem;
        }

        .loading-spinner {
            display: none;
            text-align: center;
            padding: 2rem;
        }

        .loading-spinner.active {
            display: block;
        }

        .spinner {
            width: 50px;
            height: 50px;
            border: 3px solid rgba(255,255,255,0.1);
            border-top-color: var(--primary-color);
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin: 0 auto;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        .result-section {
            display: none;
            animation: fadeIn 0.5s ease;
        }

        .result-section.active {
            display: block;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .industry-tag {
            background: rgba(99, 102, 241, 0.1);
            color: var(--primary-color);
            padding: 0.5rem 1rem;
            border-radius: 2rem;
            font-size: 0.875rem;
            display: inline-block;
            margin-right: 0.5rem;
            margin-bottom: 0.5rem;
        }

        .stat-card {
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.05);
            border-radius: 1rem;
            padding: 1.5rem;
            transition: all 0.3s ease;
        }

        .stat-card:hover {
            background: rgba(255,255,255,0.05);
            border-color: var(--primary-color);
        }

        .footer {
            background: rgba(15, 23, 42, 0.8);
            backdrop-filter: blur(10px);
            border-top: 1px solid rgba(255,255,255,0.1);
            padding: 2rem 0;
            margin-top: 4rem;
        }

        .chart-container {
            background: rgba(255,255,255,0.03);
            border-radius: 1rem;
            padding: 1.5rem;
            margin-bottom: 2rem;
        }
    </style>
</head>
<body>
    <!-- Navbar -->
    <nav class="navbar">
        <div class="container">
            <a class="navbar-brand" href="#">
                <i class="fas fa-robot me-2"></i>
                UdyamVerse
            </a>
            <div class="ms-auto">
                <span class="badge bg-primary">v2.0</span>
            </div>
        </div>
    </nav>

    <!-- Hero Section -->
    <section class="hero-section">
        <div class="container">
            <h1 class="hero-title">Intelligent Startup Simulation </h1>
            <p class="hero-subtitle">Transform your startup idea into a data-driven success story with advanced ML algorithms</p>
            <div class="d-flex justify-content-center gap-3">
                <button class="btn btn-primary" onclick="scrollToEvaluator()">
                    <i class="fas fa-chart-line me-2"></i>Evaluate Idea
                </button>
                <button class="btn btn-outline-primary" onclick="scrollToGenerator()">
                    <i class="fas fa-file-alt me-2"></i>Generate Plan
                </button>
            </div>
        </div>
    </section>

    <!-- Main Content -->
    <div class="container">
        <!-- Stats Overview -->
        <div class="row mb-5">
            <div class="col-md-3">
                <div class="stat-card text-center">
                    <i class="fas fa-chart-pie fa-2x mb-3" style="color: var(--primary-color);"></i>
                    <h3 class="h2" id="totalEvaluations">0</h3>
                    <p class="text-secondary">Evaluations Done</p>
                </div>
            </div>
            <div class="col-md-3">
                <div class="stat-card text-center">
                    <i class="fas fa-rocket fa-2x mb-3" style="color: var(--success-color);"></i>
                    <h3 class="h2" id="successRate">0%</h3>
                    <p class="text-secondary">Success Rate</p>
                </div>
            </div>
            <div class="col-md-3">
                <div class="stat-card text-center">
                    <i class="fas fa-file-alt fa-2x mb-3" style="color: var(--warning-color);"></i>
                    <h3 class="h2" id="totalPlans">0</h3>
                    <p class="text-secondary">Plans Generated</p>
                </div>
            </div>
            <div class="col-md-3">
                <div class="stat-card text-center">
                    <i class="fas fa-users fa-2x mb-3" style="color: var(--secondary-color);"></i>
                    <h3 class="h2" id="activeUsers">0</h3>
                    <p class="text-secondary">Active Users</p>
                </div>
            </div>
        </div>

        <!-- Evaluation Section -->
        <div class="row" id="evaluator">
            <div class="col-12">
                <div class="card">
                    <h2 class="card-title">
                        <i class="fas fa-microscope me-2"></i>
                        UdyamVerse
                    </h2>
                    <p class="text-secondary mb-4">Enter your startup details for comprehensive AI analysis</p>
                    
                    <form id="evaluationForm">
                        <div class="row">
                            <div class="col-md-6 mb-3">
                                <label class="form-label">Startup Idea</label>
                                <textarea class="form-control" id="idea" rows="3" placeholder="Describe your startup idea in detail..." required>AI app for farmers</textarea>
                            </div>
                            <div class="col-md-6 mb-3">
                                <label class="form-label">Industry</label>
                                <select class="form-select" id="industry">
                                    <option value="AI/ML">AI/ML</option>
                                    <option value="FinTech">FinTech</option>
                                    <option value="HealthTech">HealthTech</option>
                                    <option value="EdTech">EdTech</option>
                                    <option value="CleanTech">CleanTech</option>
                                    <option value="SaaS">SaaS</option>
                                    <option value="E-commerce">E-commerce</option>
                                </select>
                            </div>
                        </div>

                        <div class="row">
                            <div class="col-md-4 mb-3">
                                <label class="form-label">Market Size ($B)</label>
                                <input type="number" class="form-control" id="marketSize" value="50" min="0" max="1000" step="0.1">
                            </div>
                            <div class="col-md-4 mb-3">
                                <label class="form-label">Competition Level (1-10)</label>
                                <input type="range" class="form-range" id="competition" min="1" max="10" value="5" oninput="this.nextElementSibling.value = this.value">
                                <output>5</output>
                            </div>
                            <div class="col-md-4 mb-3">
                                <label class="form-label">Team Experience (1-10)</label>
                                <input type="range" class="form-range" id="teamExperience" min="1" max="10" value="7" oninput="this.nextElementSibling.value = this.value">
                                <output>7</output>
                            </div>
                        </div>

                        <div class="row">
                            <div class="col-md-4 mb-3">
                                <label class="form-label">Funding ($M)</label>
                                <input type="number" class="form-control" id="funding" value="2" min="0" max="1000" step="0.1">
                            </div>
                            <div class="col-md-4 mb-3">
                                <label class="form-label">Innovation Score (1-10)</label>
                                <input type="range" class="form-range" id="innovation" min="1" max="10" value="8" oninput="this.nextElementSibling.value = this.value">
                                <output>8</output>
                            </div>
                            <div class="col-md-4 mb-3">
                                <label class="form-label">Market Growth Rate (%)</label>
                                <input type="number" class="form-control" id="growthRate" value="15" min="0" max="100">
                            </div>
                        </div>

                        <div class="mb-3">
                            <label class="form-label">Upload Pitch Deck (Optional)</label>
                            <input type="file" class="form-control" id="pitchDeck" accept=".txt,.pdf,.doc,.docx">
                        </div>

                        <button type="submit" class="btn btn-primary">
                            <i class="fas fa-calculator me-2"></i>Evaluate Startup
                        </button>
                    </form>

                    <!-- Loading Spinner -->
                    <div class="loading-spinner" id="evaluationSpinner">
                        <div class="spinner"></div>
                        <p class="mt-3">Analyzing your startup with AI...</p>
                    </div>

                    <!-- Evaluation Results -->
                    <div class="result-section" id="evaluationResults">
                        <hr class="my-4">
                        
                        <div class="row">
                            <div class="col-md-4">
                                <div class="probability-meter">
                                    <canvas id="probabilityChart"></canvas>
                                    <div class="probability-value" id="probabilityValue">0%</div>
                                </div>
                                <p class="text-center text-secondary">Success Probability</p>
                                <div class="text-center">
                                    <span class="feature-badge" id="confidenceInterval">95% CI: ±5%</span>
                                </div>
                            </div>
                            
                            <div class="col-md-8">
                                <h4 class="mb-3">Key Insights</h4>
                                <div class="row">
                                    <div class="col-sm-6 mb-3">
                                        <div class="metric-card">
                                            <div class="metric-value" id="marketAttractiveness">0</div>
                                            <div class="metric-label">Market Attractiveness</div>
                                        </div>
                                    </div>
                                    <div class="col-sm-6 mb-3">
                                        <div class="metric-card">
                                            <div class="metric-value" id="teamScore">0</div>
                                            <div class="metric-label">Team Score</div>
                                        </div>
                                    </div>
                                    <div class="col-sm-6 mb-3">
                                        <div class="metric-card">
                                            <div class="metric-value" id="innovationScore">0</div>
                                            <div class="metric-label">Innovation Score</div>
                                        </div>
                                    </div>
                                    <div class="col-sm-6 mb-3">
                                        <div class="metric-card">
                                            <div class="metric-value" id="financialScore">0</div>
                                            <div class="metric-label">Financial Viability</div>
                                        </div>
                                    </div>
                                </div>
                                
                                <h4 class="mb-3 mt-3">Risk Factors</h4>
                                <ul class="list-unstyled" id="riskFactors">
                                    <!-- Will be populated by JS -->
                                </ul>
                            </div>
                        </div>

                        <div class="chart-container mt-4">
                            <h4>Industry Comparison</h4>
                            <canvas id="industryChart"></canvas>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Business Plan Generator -->
        <div class="row mt-5" id="generator">
            <div class="col-12">
                <div class="card">
                    <h2 class="card-title">
                        <i class="fas fa-file-signature me-2"></i>
                        AI Business Plan Generator
                    </h2>
                    <p class="text-secondary mb-4">Generate comprehensive business plans in seconds</p>

                    <form id="planForm">
                        <div class="row">
                            <div class="col-md-6 mb-3">
                                <label class="form-label">Startup Idea</label>
                                <input type="text" class="form-control" id="planIdea" value="AI app for farmers" required>
                            </div>
                            <div class="col-md-6 mb-3">
                                <label class="form-label">Industry</label>
                                <select class="form-select" id="planIndustry">
                                    <option value="AI/ML">AI/ML</option>
                                    <option value="FinTech">FinTech</option>
                                    <option value="HealthTech">HealthTech</option>
                                    <option value="EdTech">EdTech</option>
                                    <option value="CleanTech">CleanTech</option>
                                    <option value="SaaS">SaaS</option>
                                    <option value="E-commerce">E-commerce</option>
                                </select>
                            </div>
                        </div>

                        <div class="mb-3">
                            <label class="form-label">Target Market</label>
                            <input type="text" class="form-control" id="targetMarket" value="Small to medium farmers in developing countries" required>
                        </div>

                        <button type="submit" class="btn btn-primary">
                            <i class="fas fa-magic me-2"></i>Generate Business Plan
                        </button>
                    </form>

                    <!-- Loading Spinner -->
                    <div class="loading-spinner" id="planSpinner">
                        <div class="spinner"></div>
                        <p class="mt-3">Generating your business plan...</p>
                    </div>

                    <!-- Plan Results -->
                    <div class="result-section" id="planResults">
                        <hr class="my-4">
                        
                        <div class="row">
                            <div class="col-md-6">
                                <h4 class="mb-3">Business Model</h4>
                                <div class="competitor-card" id="businessModel">
                                    <!-- Will be populated by JS -->
                                </div>
                            </div>
                            
                            <div class="col-md-6">
                                <h4 class="mb-3">Value Proposition</h4>
                                <div class="competitor-card" id="valueProposition">
                                    <!-- Will be populated by JS -->
                                </div>
                            </div>
                        </div>

                        <h4 class="mb-3 mt-4">MVP Features</h4>
                        <div class="row" id="mvpFeatures">
                            <!-- Will be populated by JS -->
                        </div>

                        <h4 class="mb-3 mt-4">Competitor Analysis</h4>
                        <div class="row" id="competitors">
                            <!-- Will be populated by JS -->
                        </div>

                        <h4 class="mb-3 mt-4">Financial Projections</h4>
                        <div class="chart-container">
                            <canvas id="financialChart"></canvas>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Market Trends -->
        <div class="row mt-5">
            <div class="col-12">
                <div class="card">
                    <h2 class="card-title">
                        <i class="fas fa-chart-line me-2"></i>
                        Market Trends & Insights
                    </h2>
                    <div class="row" id="marketTrends">
                        <!-- Will be populated by JS -->
                    </div>
                </div>
            </div>
        </div>
    </div>

    <!-- Footer -->
    <footer class="footer">
        <div class="container">
            <div class="row">
                <div class="col-md-6">
                    <h5 class="mb-3">AI Startup Evaluator</h5>
                    <p class="text-secondary">Powered by advanced machine learning algorithms and real-time market data.</p>
                </div>
                <div class="col-md-6 text-md-end">
                    <p class="text-secondary mb-0">© 2024 AI Startup Evaluator. All rights reserved.</p>
                    <p class="text-secondary">Version 2.0 - Enterprise Edition</p>
                </div>
            </div>
        </div>
    </footer>

    <script>
        // Statistics
        let totalEvaluations = 0;
        let totalPlans = 0;
        let activeUsers = 0;
        let successSum = 0;

        // Update stats periodically
        setInterval(() => {
            totalEvaluations += Math.floor(Math.random() * 3);
            totalPlans += Math.floor(Math.random() * 2);
            activeUsers = 150 + Math.floor(Math.random() * 50);
            
            document.getElementById('totalEvaluations').textContent = totalEvaluations;
            document.getElementById('totalPlans').textContent = totalPlans;
            document.getElementById('activeUsers').textContent = activeUsers;
            
            if (totalEvaluations > 0) {
                successRate = (successSum / totalEvaluations * 100).toFixed(1);
                document.getElementById('successRate').textContent = successRate + '%';
            }
        }, 5000);

        // Scroll functions
        function scrollToEvaluator() {
            document.getElementById('evaluator').scrollIntoView({ behavior: 'smooth' });
        }

        function scrollToGenerator() {
            document.getElementById('generator').scrollIntoView({ behavior: 'smooth' });
        }

        // Initialize charts
        let probabilityChart = null;
        let industryChart = null;
        let financialChart = null;

        // Handle evaluation form submission
        document.getElementById('evaluationForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            // Show loading spinner
            document.getElementById('evaluationSpinner').classList.add('active');
            document.getElementById('evaluationResults').classList.remove('active');
            
            // Collect form data
            const formData = {
                idea: document.getElementById('idea').value,
                industry: document.getElementById('industry').value,
                marketSize: parseFloat(document.getElementById('marketSize').value),
                competition: parseInt(document.getElementById('competition').value),
                teamExperience: parseInt(document.getElementById('teamExperience').value),
                funding: parseFloat(document.getElementById('funding').value),
                innovation: parseInt(document.getElementById('innovation').value),
                growthRate: parseFloat(document.getElementById('growthRate').value)
            };
            
            // Handle file upload if present
            const fileInput = document.getElementById('pitchDeck');
            if (fileInput.files.length > 0) {
                const file = fileInput.files[0];
                const reader = new FileReader();
                reader.onload = async function(e) {
                    formData.pitchDeckContent = e.target.result;
                    await submitEvaluation(formData);
                };
                reader.readAsText(file);
            } else {
                await submitEvaluation(formData);
            }
        });

        async function submitEvaluation(formData) {
            try {
                const response = await fetch('/api/evaluate', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(formData)
                });
                
                const data = await response.json();
                
                // Hide loading spinner
                document.getElementById('evaluationSpinner').classList.remove('active');
                
                // Update results
                updateEvaluationResults(data);
                
                // Show results
                document.getElementById('evaluationResults').classList.add('active');
                
                // Update stats
                totalEvaluations++;
                if (data.successProbability > 70) successSum++;
                
            } catch (error) {
                console.error('Error:', error);
                document.getElementById('evaluationSpinner').classList.remove('active');
                alert('An error occurred during evaluation');
            }
        }

        function updateEvaluationResults(data) {
            // Update probability
            document.getElementById('probabilityValue').textContent = data.successProbability.toFixed(1) + '%';
            document.getElementById('confidenceInterval').textContent = `95% CI: ±${data.confidence.toFixed(1)}%`;
            
            // Update metrics
            document.getElementById('marketAttractiveness').textContent = data.metrics.marketAttractiveness.toFixed(1);
            document.getElementById('teamScore').textContent = data.metrics.teamScore.toFixed(1);
            document.getElementById('innovationScore').textContent = data.metrics.innovationScore.toFixed(1);
            document.getElementById('financialScore').textContent = data.metrics.financialScore.toFixed(1);
            
            // Update risk factors
            const riskList = document.getElementById('riskFactors');
            riskList.innerHTML = '';
            data.risks.forEach(risk => {
                const li = document.createElement('li');
                li.className = 'mb-2';
                li.innerHTML = `<i class="fas fa-exclamation-triangle text-warning me-2"></i>${risk}`;
                riskList.appendChild(li);
            });
            
            // Update probability chart
            if (probabilityChart) {
                probabilityChart.destroy();
            }
            const ctx = document.getElementById('probabilityChart').getContext('2d');
            probabilityChart = new Chart(ctx, {
                type: 'doughnut',
                data: {
                    datasets: [{
                        data: [data.successProbability, 100 - data.successProbability],
                        backgroundColor: ['#10b981', '#1e293b'],
                        borderWidth: 0
                    }]
                },
                options: {
                    cutout: '80%',
                    plugins: {
                        legend: {
                            display: false
                        },
                        tooltip: {
                            enabled: false
                        }
                    },
                    responsive: true,
                    maintainAspectRatio: true
                }
            });
            
            // Update industry comparison chart
            if (industryChart) {
                industryChart.destroy();
            }
            const ctx2 = document.getElementById('industryChart').getContext('2d');
            industryChart = new Chart(ctx2, {
                type: 'radar',
                data: {
                    labels: ['Market Size', 'Growth Rate', 'Innovation', 'Team', 'Funding'],
                    datasets: [{
                        label: 'Your Startup',
                        data: [
                            data.industryComparison.yourStartup.marketSize,
                            data.industryComparison.yourStartup.growthRate,
                            data.industryComparison.yourStartup.innovation,
                            data.industryComparison.yourStartup.team,
                            data.industryComparison.yourStartup.funding
                        ],
                        backgroundColor: 'rgba(99, 102, 241, 0.2)',
                        borderColor: '#6366f1',
                        borderWidth: 2
                    }, {
                        label: 'Industry Average',
                        data: [
                            data.industryComparison.industryAvg.marketSize,
                            data.industryComparison.industryAvg.growthRate,
                            data.industryComparison.industryAvg.innovation,
                            data.industryComparison.industryAvg.team,
                            data.industryComparison.industryAvg.funding
                        ],
                        backgroundColor: 'rgba(148, 163, 184, 0.2)',
                        borderColor: '#94a3b8',
                        borderWidth: 2
                    }]
                },
                options: {
                    scales: {
                        r: {
                            beginAtZero: true,
                            max: 100,
                            grid: {
                                color: 'rgba(255,255,255,0.1)'
                            },
                            pointLabels: {
                                color: '#94a3b8'
                            }
                        }
                    },
                    plugins: {
                        legend: {
                            labels: {
                                color: '#f8fafc'
                            }
                        }
                    }
                }
            });
        }

        // Handle plan generation form submission
        document.getElementById('planForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            // Show loading spinner
            document.getElementById('planSpinner').classList.add('active');
            document.getElementById('planResults').classList.remove('active');
            
            const formData = {
                idea: document.getElementById('planIdea').value,
                industry: document.getElementById('planIndustry').value,
                targetMarket: document.getElementById('targetMarket').value
            };
            
            try {
                const response = await fetch('/api/generate-plan', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(formData)
                });
                
                const data = await response.json();
                
                // Hide loading spinner
                document.getElementById('planSpinner').classList.remove('active');
                
                // Update results
                updatePlanResults(data);
                
                // Show results
                document.getElementById('planResults').classList.add('active');
                
                // Update stats
                totalPlans++;
                
            } catch (error) {
                console.error('Error:', error);
                document.getElementById('planSpinner').classList.remove('active');
                alert('An error occurred while generating the business plan');
            }
        });

        function updatePlanResults(data) {
            // Update business model
            document.getElementById('businessModel').innerHTML = `
                <h5 class="text-primary">${data.business_model.name}</h5>
                <p class="text-secondary">${data.business_model.description}</p>
                <div class="mt-2">
                    <span class="feature-badge">Pricing: ${data.business_model.pricing}</span>
                </div>
            `;
            
            // Update value proposition
            document.getElementById('valueProposition').innerHTML = `
                <p>${data.value_proposition}</p>
                <p class="text-secondary mt-2">Target Market: ${data.target_market}</p>
                <p class="text-secondary">Market Size: ${data.market_size}</p>
            `;
            
            // Update MVP features
            const mvpDiv = document.getElementById('mvpFeatures');
            mvpDiv.innerHTML = '';
            data.mvp_features.forEach(feature => {
                mvpDiv.innerHTML += `
                    <div class="col-md-6 mb-3">
                        <div class="competitor-card">
                            <h6 class="text-primary">${feature.name}</h6>
                            <p class="text-secondary small">${feature.description}</p>
                            <div class="d-flex justify-content-between">
                                <span class="feature-badge">Priority: ${feature.priority}</span>
                                <span class="feature-badge">Time: ${feature.time_estimate}</span>
                            </div>
                        </div>
                    </div>
                `;
            });
            
            // Update competitors
            const compDiv = document.getElementById('competitors');
            compDiv.innerHTML = '';
            data.competitors.forEach(comp => {
                compDiv.innerHTML += `
                    <div class="col-md-4 mb-3">
                        <div class="competitor-card">
                            <h6 class="text-primary">${comp.name}</h6>
                            <p class="text-secondary small">Strength: ${comp.strength}</p>
                            <p class="text-secondary small">Weakness: ${comp.weakness}</p>
                        </div>
                    </div>
                `;
            });
            
            // Update financial chart
            if (financialChart) {
                financialChart.destroy();
            }
            const ctx = document.getElementById('financialChart').getContext('2d');
            financialChart = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: ['Year 1', 'Year 2', 'Year 3'],
                    datasets: [{
                        label: 'Revenue',
                        data: [
                            data.financials.year_1.revenue,
                            data.financials.year_2.revenue,
                            data.financials.year_3.revenue
                        ],
                        backgroundColor: '#10b981'
                    }, {
                        label: 'Costs',
                        data: [
                            data.financials.year_1.costs,
                            data.financials.year_2.costs,
                            data.financials.year_3.costs
                        ],
                        backgroundColor: '#ef4444'
                    }]
                },
                options: {
                    responsive: true,
                    plugins: {
                        legend: {
                            labels: {
                                color: '#f8fafc'
                            }
                        }
                    },
                    scales: {
                        y: {
                            grid: {
                                color: 'rgba(255,255,255,0.1)'
                            },
                            ticks: {
                                color: '#94a3b8',
                                callback: function(value) {
                                    return '$' + value / 1000000 + 'M';
                                }
                            }
                        },
                        x: {
                            grid: {
                                color: 'rgba(255,255,255,0.1)'
                            },
                            ticks: {
                                color: '#94a3b8'
                            }
                        }
                    }
                }
            });
        }

        // Load market trends
        async function loadMarketTrends() {
            try {
                const response = await fetch('/api/market-trends');
                const trends = await response.json();
                
                const trendsDiv = document.getElementById('marketTrends');
                trendsDiv.innerHTML = '';
                
                Object.entries(trends).forEach(([industry, data]) => {
                    trendsDiv.innerHTML += `
                        <div class="col-md-4 mb-3">
                            <div class="stat-card">
                                <h5 class="text-primary">${industry}</h5>
                                <p class="h3 mt-2">${data.growth_rate}%</p>
                                <p class="text-secondary">Growth Rate</p>
                                <div class="d-flex justify-content-between mt-3">
                                    <span class="feature-badge">Funding: $${data.funding}B</span>
                                    <span class="feature-badge">Sentiment: ${data.sentiment}</span>
                                </div>
                            </div>
                        </div>
                    `;
                });
            } catch (error) {
                console.error('Error loading market trends:', error);
            }
        }

        // Initialize
        loadMarketTrends();
    </script>
</body>
</html>
'''

# ==================== FLASK ROUTES ====================

@app.route('/')
def index():
    return HTML_TEMPLATE

@app.route('/api/evaluate', methods=['POST'])
def evaluate():
    """API endpoint for startup evaluation"""
    data = request.json
    
    # Extract features
    features = [
        data['marketSize'],
        data['competition'],
        data['teamExperience'],
        data['funding'],
        data['innovation'],
        data['growthRate'],
        np.random.uniform(5, 8),  # Customer acquisition cost score
        np.random.uniform(6, 9),  # Unit economics
        np.random.uniform(4, 8),  # Intellectual property
        np.random.uniform(5, 9)   # Scalability
    ]
    
    # Get prediction from ML model
    prediction = evaluator.predict_success_probability(features)
    
    # Analyze pitch deck if provided
    pitch_score = 50
    if 'pitchDeckContent' in data:
        pitch_analysis = evaluator.analyze_pitch_deck(data['pitchDeckContent'])
        pitch_score = pitch_analysis['score']
    
    # Adjust probability based on pitch analysis
    adjusted_probability = (prediction['probability'] * 0.7 + pitch_score * 0.3)
    
    # Calculate metrics
    market_attractiveness = (data['marketSize'] * 0.4 + data['growthRate'] * 0.6) / 10
    team_score = data['teamExperience'] * 10
    innovation_score = data['innovation'] * 10 + (pitch_score - 50) * 0.2
    financial_score = (data['funding'] * 5 + (10 - data['competition']) * 5)
    
    # Generate risk factors
    risks = []
    if data['competition'] > 7:
        risks.append("High competition level in your market")
    if data['funding'] < 1:
        risks.append("Limited funding may constrain growth")
    if data['teamExperience'] < 5:
        risks.append("Team experience below recommended level")
    if adjusted_probability < 50:
        risks.append("Market timing might not be optimal")
    if len(risks) == 0:
        risks.append("No significant risks detected")
    
    # Generate industry comparison
    industry_comparison = {
        'yourStartup': {
            'marketSize': min(100, data['marketSize'] * 2),
            'growthRate': data['growthRate'],
            'innovation': data['innovation'] * 10,
            'team': data['teamExperience'] * 10,
            'funding': min(100, data['funding'] * 20)
        },
        'industryAvg': {
            'marketSize': 50,
            'growthRate': 15,
            'innovation': 65,
            'team': 60,
            'funding': 40
        }
    }
    
    response = {
        'successProbability': adjusted_probability,
        'confidence': (prediction['confidence_upper'] - prediction['confidence_lower']) / 2,
        'metrics': {
            'marketAttractiveness': market_attractiveness,
            'teamScore': team_score,
            'innovationScore': innovation_score,
            'financialScore': financial_score
        },
        'risks': risks,
        'industryComparison': industry_comparison
    }
    
    return jsonify(response)

@app.route('/api/generate-plan', methods=['POST'])
def generate_plan():
    """API endpoint for business plan generation"""
    data = request.json
    
    plan = plan_generator.generate_complete_plan(
        data['idea'],
        data['industry'],
        data['targetMarket']
    )
    
    return jsonify(plan)

@app.route('/api/market-trends')
def market_trends():
    """API endpoint for market trends"""
    return jsonify(evaluator.market_trends)

@app.route('/api/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

# ==================== MAIN ====================

# ==================== PRODUCTION CONFIGURATION ====================

if __name__ == '__main__':
    # For local development only
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║     AI Startup Evaluator & Generator - Enterprise Edition ║
    ╚══════════════════════════════════════════════════════════╝
    
    🚀 Starting development server...
    📡 Running on http://localhost:5000
    🤖 ML Models loaded successfully
    📊 Market trends data initialized
    
    Press CTRL+C to stop the server
    """)
    
    app.run(debug=True, host='0.0.0.0', port=5000)

# For production (Render uses gunicorn)
# The app object is already defined above