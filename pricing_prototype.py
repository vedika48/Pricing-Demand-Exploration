# pricing_prototype.py
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings('ignore')
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
import os
from dotenv import load_dotenv
import json
import requests
from datetime import datetime
import time

# Load environment variables
load_dotenv()

# ============================================
# Part 1: AI Integration Layer
# ============================================

class AIGenerator:
    """Generate strategic insights using multiple AI providers"""
    
    def __init__(self):
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        self.huggingface_api_key = os.getenv("HUGGINGFACE_API_KEY")
        self.current_provider = "gemini"  # default
        self.last_call_time = 0
        self.rate_limit = 1  # 1 second between calls (free tier friendly)
    
    def _rate_limit(self):
        """Simple rate limiting for free tiers"""
        current_time = time.time()
        time_since_last = current_time - self.last_call_time
        if time_since_last < self.rate_limit:
            time.sleep(self.rate_limit - time_since_last)
        self.last_call_time = time.time()
    
    def set_provider(self, provider):
        """Switch between AI providers"""
        if provider in ["gemini", "huggingface"]:
            self.current_provider = provider
            return True
        return False
    
    def generate_insight(self, prompt, max_retries=2):
        """Generate insight using current provider with fallback"""
        
        for attempt in range(max_retries):
            try:
                self._rate_limit()
                
                if self.current_provider == "gemini" and self.gemini_api_key:
                    return self._call_gemini(prompt)
                elif self.current_provider == "huggingface" and self.huggingface_api_key:
                    return self._call_huggingface(prompt)
                else:
                    # Fallback to template-based generation
                    return self._fallback_template(prompt)
                    
            except Exception as e:
                if attempt == max_retries - 1:
                    return self._fallback_template(prompt)
                time.sleep(2)  # Wait before retry
    
    def _call_gemini(self, prompt):
        """Call Google Gemini API (free tier)"""
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={self.gemini_api_key}"
        
        payload = {
            "contents": [{
                "parts": [{"text": prompt}]
            }],
            "generationConfig": {
                "temperature": 0.7,
                "maxOutputTokens": 800,
                "topP": 0.8,
                "topK": 40
            }
        }
        
        headers = {'Content-Type': 'application/json'}
        
        try:
            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()
            result = response.json()
            
            if 'candidates' in result and len(result['candidates']) > 0:
                return result['candidates'][0]['content']['parts'][0]['text']
            else:
                return "⚠️ Gemini returned an empty response. Using fallback analysis."
                
        except Exception as e:
            raise Exception(f"Gemini API error: {str(e)}")
    
    def _call_huggingface(self, prompt):
        """Call Hugging Face Inference API (free tier)"""
        # Using a free, lightweight model suitable for business text
        API_URL = "https://api-inference.huggingface.co/models/gpt2"
        headers = {"Authorization": f"Bearer {self.huggingface_api_key}"}
        
        # Structure prompt for better results
        formatted_prompt = f"<|endoftext|>Business Analysis: {prompt}\n\nStrategic Insight:"
        
        payload = {
            "inputs": formatted_prompt,
            "parameters": {
                "max_length": 300,
                "temperature": 0.7,
                "top_p": 0.9,
                "do_sample": True,
                "return_full_text": False
            }
        }
        
        try:
            response = requests.post(API_URL, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            
            if isinstance(result, list) and len(result) > 0:
                return result[0].get('generated_text', '').strip()
            else:
                return self._fallback_template(prompt)
                
        except Exception as e:
            raise Exception(f"Hugging Face API error: {str(e)}")
    
    def _fallback_template(self, prompt):
        """Fallback template-based generation when APIs fail"""
        if "product analysis" in prompt.lower():
            return self._generate_product_fallback(prompt)
        elif "category" in prompt.lower():
            return self._generate_category_fallback(prompt)
        elif "trade-off" in prompt.lower() or "scenario" in prompt.lower():
            return self._generate_tradeoff_fallback(prompt)
        else:
            return "Based on the data analysis, consider optimizing pricing strategy while monitoring competitive landscape and customer feedback metrics."
    
    def _generate_product_fallback(self, prompt):
        """Template-based product analysis fallback"""
        return """
        PRODUCT ANALYSIS (AI service unavailable - showing template analysis):
        
        This product shows typical market patterns. Key considerations:
        
        1. Current pricing appears aligned with market position
        2. Rating suggests satisfactory customer experience
        3. Consider A/B testing small price adjustments
        4. Monitor competitor pricing in this segment
        5. Customer reviews indicate opportunities for feature communication
        
        For deeper AI-powered insights, please check your API configuration.
        """
    
    def _generate_category_fallback(self, prompt):
        return "Category analysis template - please configure AI API for enhanced insights."
    
    def _generate_tradeoff_fallback(self, prompt):
        return "Trade-off analysis template - please configure AI API for enhanced insights."


# ============================================
# Part 2: Pricing Optimizer Core
# ============================================

class PricingOptimizer:
    """
    Core ML engine for pricing and demand analysis
    """
    
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)
        self.model = None
        self.feature_importance = None
        self.ai_generator = AIGenerator()
        self.prepare_data()
    
    def clean_price(self, price_str):
        """Clean price string to float"""
        if pd.isna(price_str):
            return np.nan
        # Remove ₹, commas, and convert to float
        cleaned = re.sub(r'[₹,]', '', str(price_str))
        try:
            return float(cleaned)
        except:
            return np.nan
    
    def clean_percentage(self, pct_str):
        """Clean percentage string to float"""
        if pd.isna(pct_str):
            return np.nan
        # Remove % and convert to float
        cleaned = re.sub(r'[%]', '', str(pct_str))
        try:
            return float(cleaned)
        except:
            return np.nan
    
    def clean_rating_count(self, count_str):
        """Clean rating count string to float"""
        if pd.isna(count_str):
            return 0
        # Remove commas and convert to float
        cleaned = re.sub(r'[,]', '', str(count_str))
        try:
            return float(cleaned)
        except:
            return 0
    
    def prepare_data(self):
        """Clean and prepare dataset for modeling"""
        df = self.data.copy()
        
        # Clean price columns
        df['discounted_price'] = df['discounted_price'].apply(self.clean_price)
        df['actual_price'] = df['actual_price'].apply(self.clean_price)
        
        # Clean discount percentage
        df['discount_percentage'] = df['discount_percentage'].apply(self.clean_percentage)
        
        # Clean rating (convert to float)
        df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
        
        # Clean rating count
        df['rating_count'] = df['rating_count'].apply(self.clean_rating_count)
        
        # Calculate discount percentage if missing
        df['discount_percentage'] = df['discount_percentage'].fillna(
            ((df['actual_price'] - df['discounted_price']) / df['actual_price'] * 100).round()
        )
        
        # Extract main category
        df['main_category'] = df['category'].str.split('|').str[0]
        
        # Simple sentiment proxy from rating (normalized 0-1)
        df['sentiment_score'] = df['rating'] / 5.0
        
        # Create product type features
        df['is_braided'] = df['about_product'].str.contains('braided|nylon', case=False, na=False).astype(int)
        df['is_fast_charge'] = df['about_product'].str.contains('fast charging|quick charge', case=False, na=False).astype(int)
        df['has_warranty'] = df['about_product'].str.contains('warranty', case=False, na=False).astype(int)
        
        # Price bands
        df['price_band'] = pd.cut(df['discounted_price'], 
                                   bins=[0, 200, 500, 1000, 5000, 100000],
                                   labels=['Budget', 'Mid-Range', 'Premium', 'Luxury', 'Ultra-Luxury'])
        
        # Log transform target for better modeling (add small constant to avoid log(0))
        df['log_rating_count'] = np.log1p(df['rating_count'])
        
        # Drop rows with missing critical values
        df = df.dropna(subset=['discounted_price', 'rating', 'discount_percentage'])
        
        self.cleaned_data = df
        print(f"Data prepared: {len(df)} products after cleaning")
        return df
    
    def build_model(self):
        """Train demand prediction model"""
        # Select features
        feature_cols = ['discounted_price', 'discount_percentage', 'rating', 
                        'sentiment_score', 'is_braided', 'is_fast_charge', 'has_warranty']
        
        # Encode categoricals
        df_encoded = pd.get_dummies(self.cleaned_data, columns=['main_category'], prefix='cat')
        
        # Add encoded category columns to features
        cat_cols = [col for col in df_encoded.columns if col.startswith('cat_')]
        all_features = feature_cols + cat_cols
        
        # Prepare X and y
        X = df_encoded[all_features].fillna(0)
        y = df_encoded['log_rating_count']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        # Feature importance
        self.feature_importance = pd.DataFrame({
            'feature': all_features,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        self.feature_cols = all_features
        
        print(f"Model Performance: R² = {r2:.3f}, RMSE = {rmse:.3f}")
        print("\nTop 5 Features by Importance:")
        print(self.feature_importance.head(5).to_string(index=False))
        
        return self.model
    
    def predict_demand(self, product_features):
        """Predict demand for new product configuration"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Ensure all features present
        features_df = pd.DataFrame([product_features])
        
        # Add missing columns with zeros
        for col in self.feature_cols:
            if col not in features_df.columns:
                features_df[col] = 0
        
        # Scale and predict
        features_scaled = self.scaler.transform(features_df[self.feature_cols])
        log_pred = self.model.predict(features_scaled)[0]
        
        return np.expm1(log_pred)  # Reverse log transform
    
    def elasticity_analysis(self, product_id, price_range_pct=0.5):
        """Calculate price elasticity for a product"""
        product = self.cleaned_data[self.cleaned_data['product_id'] == product_id].iloc[0]
        base_price = product['discounted_price']
        
        # Create price variations
        price_factors = np.linspace(0.5, 1.5, 20)  # 50% to 150% of original price
        demands = []
        
        for factor in price_factors:
            new_price = base_price * factor
            # Create feature vector
            features = {
                'discounted_price': new_price,
                'discount_percentage': product['discount_percentage'],
                'rating': product['rating'],
                'sentiment_score': product['sentiment_score'],
                'is_braided': product['is_braided'],
                'is_fast_charge': product['is_fast_charge'],
                'has_warranty': product['has_warranty']
            }
            
            # Add category dummies
            cat_col = f"cat_{product['main_category']}"
            features[cat_col] = 1
            
            demands.append(self.predict_demand(features))
        
        # Calculate elasticity
        price_changes = (price_factors - 1) * 100
        demand_changes = (np.array(demands) / demands[10] - 1) * 100  # Index 10 is base price
        
        # Simple linear regression for average elasticity
        from scipy import stats
        slope, intercept, r_value, p_value, std_err = stats.linregress(price_changes, demand_changes)
        
        return {
            'product_name': product['product_name'],
            'base_price': base_price,
            'base_demand': demands[10],
            'elasticity': slope,  # % change in demand per 1% price change
            'price_factors': price_factors,
            'demands': demands,
            'price_changes': price_changes,
            'demand_changes': demand_changes
        }
    
    def what_if_scenario(self, product_id, modifications):
        """
        Run what-if scenario with multiple modifications
        
        modifications: dict with keys like 'price_change_pct', 'rating_change', etc.
        """
        product = self.cleaned_data[self.cleaned_data['product_id'] == product_id].iloc[0]
        
        # Base prediction
        base_features = {
            'discounted_price': product['discounted_price'],
            'discount_percentage': product['discount_percentage'],
            'rating': product['rating'],
            'sentiment_score': product['sentiment_score'],
            'is_braided': product['is_braided'],
            'is_fast_charge': product['is_fast_charge'],
            'has_warranty': product['has_warranty']
        }
        cat_col = f"cat_{product['main_category']}"
        base_features[cat_col] = 1
        
        base_demand = self.predict_demand(base_features)
        
        # Modified prediction
        modified_features = base_features.copy()
        
        if 'price_change_pct' in modifications:
            modified_features['discounted_price'] *= (1 + modifications['price_change_pct']/100)
        
        if 'rating_change' in modifications:
            new_rating = product['rating'] + modifications['rating_change']
            modified_features['rating'] = max(1, min(5, new_rating))  # Clamp between 1-5
            modified_features['sentiment_score'] = modified_features['rating'] / 5
        
        if 'discount_change_pct' in modifications:
            modified_features['discount_percentage'] += modifications['discount_change_pct']
        
        modified_demand = self.predict_demand(modified_features)
        
        return {
            'product_name': product['product_name'],
            'base_demand': base_demand,
            'modified_demand': modified_demand,
            'change_pct': ((modified_demand - base_demand) / base_demand) * 100,
            'base_config': base_features,
            'modified_config': modified_features
        }
    
    def generate_ai_product_insight(self, product_id):
        """Generate AI-powered product insight"""
        product = self.cleaned_data[self.cleaned_data['product_id'] == product_id].iloc[0]
        
        # Get category stats
        category_df = self.cleaned_data[self.cleaned_data['main_category'] == product['main_category']]
        category_stats = {
            'avg_price': category_df['discounted_price'].mean(),
            'avg_rating': category_df['rating'].mean(),
            'product_count': len(category_df),
            'top_performers': category_df.nlargest(3, 'rating_count')[['product_name', 'rating']].to_dict('records')
        }
        
        # Calculate price elasticity
        elasticity = self.elasticity_analysis(product_id)
        
        prompt = f"""
        You are a strategic pricing consultant analyzing an e-commerce product.
        
        PRODUCT DATA:
        - Name: {product['product_name']}
        - Category: {product['main_category']}
        - Price: ₹{product['discounted_price']:,.0f} (MRP: ₹{product['actual_price']:,.0f})
        - Discount: {product['discount_percentage']:.0f}%
        - Rating: {product['rating']}⭐ from {product['rating_count']:,.0f} customers
        - Features: {'Braided' if product['is_braided'] else 'Standard'}, 
                    {'Fast Charging' if product['is_fast_charge'] else 'Regular'}, 
                    {'Warranty' if product['has_warranty'] else 'No warranty'}
        
        PRICE SENSITIVITY:
        - Price Elasticity: {elasticity['elasticity']:.2f}
        - Classification: {'Highly elastic' if elasticity['elasticity'] < -1 else 'Moderately elastic' if elasticity['elasticity'] < 0 else 'Inelastic'}
        
        CATEGORY CONTEXT:
        - Category average price: ₹{category_stats['avg_price']:,.0f}
        - Category average rating: {category_stats['avg_rating']:.2f}⭐
        - Number of competitors: {category_stats['product_count']}
        
        Based on this data, provide a comprehensive strategic analysis including:
        1. Current market position assessment
        2. Optimal pricing strategy recommendation
        3. Specific actionable recommendations (max 3)
        4. Potential risks and mitigation strategies
        5. Key metrics to monitor
        
        Format your response as clear, professional business analysis with bullet points for recommendations.
        """
        
        return self.ai_generator.generate_insight(prompt)
    
    def generate_ai_category_insight(self, category):
        """Generate AI-powered category insight"""
        category_df = self.cleaned_data[self.cleaned_data['main_category'] == category]
        
        stats = {
            'product_count': len(category_df),
            'avg_price': category_df['discounted_price'].mean(),
            'avg_rating': category_df['rating'].mean(),
            'total_ratings': category_df['rating_count'].sum(),
            'price_range': [category_df['discounted_price'].min(), category_df['discounted_price'].max()],
            'top_products': category_df.nlargest(3, 'rating_count')[['product_name', 'rating', 'discounted_price']].to_dict('records'),
            'price_bands': category_df['price_band'].value_counts().to_dict()
        }
        
        prompt = f"""
        You are a category manager analyzing the {category} product category.
        
        CATEGORY STATISTICS:
        - Total products: {stats['product_count']}
        - Average price: ₹{stats['avg_price']:,.0f} (range: ₹{stats['price_range'][0]:,.0f} - ₹{stats['price_range'][1]:,.0f})
        - Average rating: {stats['avg_rating']:.2f}⭐
        - Total customer ratings: {stats['total_ratings']:,.0f}
        
        Price band distribution:
        {json.dumps(stats['price_bands'], indent=2)}
        
        Top performing products:
        {json.dumps(stats['top_products'], indent=2)}
        
        Provide strategic recommendations including:
        1. Category health assessment
        2. Pricing strategy recommendations
        3. Growth opportunities
        4. Competitive positioning advice
        5. Potential risks
        
        Format as a professional category strategy document.
        """
        
        return self.ai_generator.generate_insight(prompt)
    
    def generate_ai_scenario_insight(self, product_id, modifications, result):
        """Generate AI-powered scenario insight"""
        product = self.cleaned_data[self.cleaned_data['product_id'] == product_id].iloc[0]
        
        prompt = f"""
        You are analyzing a pricing scenario for {product['product_name']}.
        
        CURRENT STATE:
        - Price: ₹{product['discounted_price']:,.0f}
        - Rating: {product['rating']}⭐
        - Current demand: {result['base_demand']:,.0f} predicted ratings
        
        PROPOSED CHANGES:
        - Price change: {modifications.get('price_change_pct', 0)}%
        - Rating change: {modifications.get('rating_change', 0)}⭐
        - Discount change: {modifications.get('discount_change_pct', 0)} percentage points
        
        PREDICTED OUTCOME:
        - New demand: {result['modified_demand']:,.0f} ratings
        - Change: {result['change_pct']:+.1f}%
        
        Provide a strategic assessment including:
        1. Whether this is a good strategy
        2. Short-term vs long-term implications
        3. Implementation recommendations
        4. Risk assessment
        5. Alternative approaches to consider
        """
        
        return self.ai_generator.generate_insight(prompt)


# ============================================
# Part 3: Streamlit Dashboard Application
# ============================================

def run_dashboard():
    """Main dashboard application"""
    
    st.set_page_config(
        page_title="AI-Powered Pricing Optimizer",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 AI-Powered Pricing Strategy & Demand Simulator")
    st.markdown("---")
    
    # Sidebar for AI provider selection
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Check API keys
        gemini_key = os.getenv("GEMINI_API_KEY")
        huggingface_key = os.getenv("HUGGINGFACE_API_KEY")
        
        ai_provider = st.selectbox(
            "Select AI Provider",
            ["Gemini (Google)", "Hugging Face", "Template (Fallback)"],
            help="Choose which AI service to use for insights"
        )
        
        # Show API key status
        if gemini_key:
            st.success("✅ Gemini API key found")
        else:
            st.warning("⚠️ Gemini API key missing")
            
        if huggingface_key:
            st.success("✅ Hugging Face API key found")
        else:
            st.warning("⚠️ Hugging Face API key missing")
        
        st.markdown("---")
    
    # Initialize model
    with st.spinner("Loading data and training model..."):
        optimizer = PricingOptimizer("Pricing_dataset.csv")
        optimizer.build_model()
        
        # Set AI provider
        provider_map = {
            "Gemini (Google)": "gemini",
            "Hugging Face": "huggingface",
            "Template (Fallback)": "template"
        }
        if provider_map[ai_provider] in ["gemini", "huggingface"]:
            optimizer.ai_generator.set_provider(provider_map[ai_provider])
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["Market Overview", "Product Deep Dive", "What-If Simulator", "Category Analysis", "Strategic Recommendations"]
    )
    
    df = optimizer.cleaned_data
    
    if page == "Market Overview":
        st.header("📊 Market Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Products", f"{len(df):,}")
        with col2:
            st.metric("Avg Rating", f"{df['rating'].mean():.2f} ⭐")
        with col3:
            st.metric("Avg Discount", f"{df['discount_percentage'].mean():.1f}%")
        with col4:
            st.metric("Total Ratings", f"{df['rating_count'].sum():,.0f}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Price Distribution by Category")
            fig = px.box(df, x='main_category', y='discounted_price', 
                         title="Price Ranges Across Categories",
                         labels={'discounted_price': 'Price (₹)', 'main_category': 'Category'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Rating vs. Price")
            fig = px.scatter(df, x='discounted_price', y='rating', 
                             size='rating_count', color='main_category',
                             hover_name='product_name',
                             title="Product Performance Matrix",
                             labels={'discounted_price': 'Price (₹)', 'rating': 'Rating'})
            st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Category Performance")
        category_stats = df.groupby('main_category').agg({
            'product_id': 'count',
            'rating': 'mean',
            'rating_count': 'sum',
            'discounted_price': 'mean'
        }).round(2)
        category_stats.columns = ['Product Count', 'Avg Rating', 'Total Ratings', 'Avg Price']
        st.dataframe(category_stats, use_container_width=True)
        
        st.subheader("Feature Importance")
        fig = px.bar(optimizer.feature_importance.head(10), 
                     x='importance', y='feature', orientation='h',
                     title="Top 10 Factors Driving Demand",
                     labels={'importance': 'Importance Score', 'feature': 'Feature'})
        st.plotly_chart(fig, use_container_width=True)
    
    elif page == "Product Deep Dive":
        st.header("🔍 Product Deep Dive")
        
        # Product selector
        product_names = df['product_name'].tolist()
        selected_product = st.selectbox("Select a product", product_names)
        product = df[df['product_name'] == selected_product].iloc[0]
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Product Details")
            st.write(f"**Product:** {product['product_name']}")
            st.write(f"**Category:** {product['main_category']}")
            st.write(f"**Price:** ₹{product['discounted_price']:,.0f} (MRP: ₹{product['actual_price']:,.0f})")
            st.write(f"**Discount:** {product['discount_percentage']:.0f}%")
            st.write(f"**Rating:** {product['rating']} ⭐ ({product['rating_count']:,.0f} ratings)")
            st.write(f"**Features:**")
            st.write(f"- Braided: {'✅' if product['is_braided'] else '❌'}")
            st.write(f"- Fast Charge: {'✅' if product['is_fast_charge'] else '❌'}")
            st.write(f"- Warranty: {'✅' if product['has_warranty'] else '❌'}")
        
        with col2:
            st.subheader("Performance Metrics")
            
            # Similar products analysis
            same_category = df[df['main_category'] == product['main_category']]
            category_avg_price = same_category['discounted_price'].mean()
            category_avg_rating = same_category['rating'].mean()
            
            st.write(f"**vs Category Average:**")
            st.write(f"- Price: ₹{product['discounted_price']:,.0f} vs ₹{category_avg_price:,.0f} ({'Above' if product['discounted_price'] > category_avg_price else 'Below'} average)")
            st.write(f"- Rating: {product['rating']} vs {category_avg_rating:.1f} ({'Above' if product['rating'] > category_avg_rating else 'Below'} average)")
            
            # Price percentile
            price_percentile = (same_category['discounted_price'] < product['discounted_price']).mean() * 100
            st.write(f"- Price percentile: {price_percentile:.0f}th (more expensive than {price_percentile:.0f}% of category)")
            
            # Rating percentile
            rating_percentile = (same_category['rating'] < product['rating']).mean() * 100
            st.write(f"- Rating percentile: {rating_percentile:.0f}th")
            
            # Elasticity analysis
            if st.button("Calculate Price Elasticity"):
                with st.spinner("Analyzing..."):
                    try:
                        elasticity = optimizer.elasticity_analysis(product['product_id'])
                        
                        st.write(f"**Price Elasticity:** {elasticity['elasticity']:.2f}")
                        if elasticity['elasticity'] < -1:
                            st.write("→ Highly elastic (demand very sensitive to price)")
                        elif elasticity['elasticity'] < 0:
                            st.write("→ Inelastic (demand less sensitive to price)")
                        else:
                            st.write("→ Veblen effect (demand increases with price)")
                        
                        # Plot elasticity curve
                        fig = px.line(x=elasticity['price_factors']*100, y=elasticity['demands'],
                                      title="Demand vs. Price",
                                      labels={'x': 'Price (% of current)', 'y': 'Predicted Demand (ratings)'})
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error calculating elasticity: {str(e)}")
            
            # Generate AI insight button
            if st.button("🤖 Generate AI Product Analysis"):
                with st.spinner("AI is analyzing this product..."):
                    insight = optimizer.generate_ai_product_insight(product['product_id'])
                    st.info(insight)
        
        st.subheader("Similar Products")
        similar = same_category[
            (same_category['discounted_price'].between(product['discounted_price']*0.8, product['discounted_price']*1.2))
        ].head(5)
        st.dataframe(similar[['product_name', 'discounted_price', 'rating', 'rating_count']])
    
    elif page == "What-If Simulator":
        st.header("🎲 What-If Pricing Simulator")
        
        st.markdown("""
        Adjust pricing and product attributes below to see how demand would change.
        The model predicts the number of ratings (demand proxy) based on your changes.
        """)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            product_names = df['product_name'].tolist()
            selected_product = st.selectbox("Select product to simulate", product_names, key='sim_product')
            product = df[df['product_name'] == selected_product].iloc[0]
            
            st.write("---")
            st.subheader("Adjust Parameters")
            
            price_change = st.slider("Price Change (%)", -50, 100, 0, 
                                      help="Negative = discount, Positive = increase")
            
            rating_change = st.slider("Rating Change (★)", -1.0, 1.0, 0.0, step=0.1,
                                       help="Simulate improved/worsened customer perception")
            
            discount_change = st.slider("Discount Change (%-points)", -20, 30, 0,
                                         help="Increase or decrease discount percentage")
            
            simulate_btn = st.button("Run Simulation", type="primary")
        
        with col2:
            if simulate_btn:
                try:
                    modifications = {
                        'price_change_pct': price_change,
                        'rating_change': rating_change,
                        'discount_change_pct': discount_change
                    }
                    
                    result = optimizer.what_if_scenario(product['product_id'], modifications)
                    
                    st.subheader("Simulation Results")
                    
                    st.metric(
                        "Predicted Demand Change",
                        f"{result['change_pct']:+.1f}%",
                        delta=f"{result['change_pct']:+.1f}%",
                        delta_color="normal"
                    )
                    
                    st.write(f"**Current predicted demand:** {result['base_demand']:,.0f} ratings")
                    st.write(f"**New predicted demand:** {result['modified_demand']:,.0f} ratings")
                    
                    # Generate AI insight for this scenario
                    with st.spinner("🤖 AI is analyzing this scenario..."):
                        scenario_insight = optimizer.generate_ai_scenario_insight(
                            product['product_id'], modifications, result
                        )
                        st.info(scenario_insight)
                    
                except Exception as e:
                    st.error(f"Error running simulation: {str(e)}")
    
    elif page == "Category Analysis":
        st.header("📦 Category Performance Analysis")
        
        # Category selector
        categories = df['main_category'].unique()
        selected_category = st.selectbox("Select Category", categories)
        
        category_df = df[df['main_category'] == selected_category]
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Products", len(category_df))
        with col2:
            st.metric("Avg Rating", f"{category_df['rating'].mean():.2f}")
        with col3:
            st.metric("Avg Price", f"₹{category_df['discounted_price'].mean():,.0f}")
        with col4:
            st.metric("Total Ratings", f"{category_df['rating_count'].sum():,.0f}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Price vs Rating")
            fig = px.scatter(category_df, x='discounted_price', y='rating',
                             size='rating_count', hover_name='product_name',
                             title=f"{selected_category} - Price vs Performance")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Discount Effectiveness")
            fig = px.scatter(category_df, x='discount_percentage', y='rating_count',
                             hover_name='product_name',
                             title="Does higher discount drive more ratings?")
            st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Top Products in Category")
        top_products = category_df.nlargest(10, 'rating_count')[['product_name', 'discounted_price', 'rating', 'rating_count']]
        st.dataframe(top_products, use_container_width=True)
        
        # Price band analysis
        st.subheader("Performance by Price Band")
        price_band_stats = category_df.groupby('price_band').agg({
            'product_id': 'count',
            'rating': 'mean',
            'rating_count': 'sum'
        }).round(2)
        st.dataframe(price_band_stats, use_container_width=True)
        
        # Generate AI category insight
        if st.button(f"🤖 Generate AI Analysis for {selected_category}"):
            with st.spinner("AI is analyzing this category..."):
                insight = optimizer.generate_ai_category_insight(selected_category)
                st.info(insight)
    
    elif page == "Strategic Recommendations":
        st.header("💡 Strategic Recommendations")
        
        st.markdown("""
        ### AI-Generated Strategic Insights
        Based on the analysis of your product catalog and market performance, here are key recommendations:
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📌 Pricing Opportunities")
            
            # Find underpriced high-performers
            df_copy = df.copy()
            df_copy['price_performance_ratio'] = df_copy['rating_count'] / df_copy['discounted_price'].replace(0, np.nan)
            top_value = df_copy.nlargest(5, 'price_performance_ratio')[['product_name', 'discounted_price', 'rating', 'rating_count']]
            
            st.write("**High-Value Products (Potential for price increase)**")
            st.dataframe(top_value, use_container_width=True)
            
            # Find overpriced underperformers
            df_copy['rating_performance'] = df_copy['rating_count'] * df_copy['rating']
            df_copy['price_efficiency'] = df_copy['rating_performance'] / df_copy['discounted_price'].replace(0, np.nan)
            low_value = df_copy.nsmallest(5, 'price_efficiency')[['product_name', 'discounted_price', 'rating', 'rating_count']]
            
            st.write("**Underperformers (Consider discount or reposition)**")
            st.dataframe(low_value, use_container_width=True)
        
        with col2:
            st.subheader("📊 Category Gaps")
            
            # Category performance
            cat_perf = df.groupby('main_category').agg({
                'rating': 'mean',
                'rating_count': 'sum'
            }).reset_index()
            
            fig = px.scatter(cat_perf, x='rating_count', y='rating', text='main_category',
                             title="Category Performance Map",
                             labels={'rating_count': 'Total Ratings (Demand)', 'rating': 'Avg Rating'})
            fig.update_traces(textposition='top center')
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Generate overall strategic insight
        if st.button("🤖 Generate Overall Strategic Recommendations"):
            with st.spinner("AI is synthesizing overall strategy..."):
                prompt = f"""
                Based on the entire e-commerce dataset analysis, provide high-level strategic recommendations.
                
                Key metrics:
                - Total products: {len(df)}
                - Categories: {len(df['main_category'].unique())}
                - Average rating: {df['rating'].mean():.2f}⭐
                - Average discount: {df['discount_percentage'].mean():.1f}%
                - Total customer engagement: {df['rating_count'].sum():,.0f} ratings
                
                Top 3 demand drivers:
                {optimizer.feature_importance.head(3).to_string()}
                
                Provide strategic recommendations for:
                1. Overall pricing strategy
                2. Category prioritization
                3. Customer engagement improvement
                4. Competitive positioning
                5. Long-term growth initiatives
                """
                
                insight = optimizer.ai_generator.generate_insight(prompt)
                st.info(insight)


# ============================================
# Part 4: Run Application
# ============================================

if __name__ == "__main__":
    # Run Streamlit dashboard
    run_dashboard()