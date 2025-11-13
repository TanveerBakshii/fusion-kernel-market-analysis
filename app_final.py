# Fusion Kernel - Market Analysis Platform
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from datetime import datetime, timedelta

# Page configuration
st.set_page_config(
    page_title="Fusion Kernel - Market Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {font-size: 2.5rem; color: #1f77b4; text-align: center;}
    .metric-card {background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem;}
    .stock-card {border-left: 4px solid #1f77b4; padding-left: 1rem;}
</style>
""", unsafe_allow_html=True)

class FusionKernelApp:
    def __init__(self):
        self.stocks = {
            'RELIANCE': {'id': 1333, 'sector': 'Energy'},
            'TCS': {'id': 11536, 'sector': 'IT'},
            'INFY': {'id': 1594, 'sector': 'IT'},
            'HDFC BANK': {'id': 1333, 'sector': 'Banking'},
            'ICICI BANK': {'id': 4963, 'sector': 'Banking'},
            'SBIN': {'id': 3045, 'sector': 'Banking'}
        }
    
    def create_sample_data(self, symbol, days=30):
        """Create realistic sample market data"""
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
        
        # Generate realistic price data with trends
        np.random.seed(42)  # For consistent results
        base_price = np.random.randint(1000, 5000)
        prices = [base_price]
        
        for i in range(1, days):
            # Add some trend + noise
            trend = np.sin(i * 0.3) * 2  # Cyclical trend
            noise = np.random.normal(0, 10)
            new_price = prices[-1] + trend + noise
            prices.append(max(100, new_price))  # Prevent negative prices
        
        df = pd.DataFrame({
            'date': dates,
            'open': [p * (1 + np.random.uniform(-0.01, 0.01)) for p in prices],
            'high': [p * (1 + np.random.uniform(0, 0.02)) for p in prices],
            'low': [p * (1 - np.random.uniform(0, 0.02)) for p in prices],
            'close': prices,
            'volume': [np.random.randint(1000000, 5000000) for _ in range(days)]
        })
        df.set_index('date', inplace=True)
        return df
    
    def create_price_chart(self, symbol, days=30):
        """Create an interactive price chart"""
        data = self.create_sample_data(symbol, days)
        
        fig = go.Figure()
        # Candlestick chart
        fig.add_trace(go.Candlestick(
            x=data.index,
            open=data['open'],
            high=data['high'],
            low=data['low'],
            close=data['close'],
            name='Price'
        ))
        
        fig.update_layout(
            title=f'{symbol} - Price Chart',
            xaxis_title='Date',
            yaxis_title='Price (₹)',
            template='plotly_white',
            height=400
        )
        
        return fig
    
    def calculate_technical_indicators(self, symbol):
        """Calculate technical indicators"""
        data = self.create_sample_data(symbol, 50)
        
        # Simple moving averages
        sma_20 = data['close'].rolling(window=20).mean().iloc[-1]
        sma_50 = data['close'].rolling(window=50).mean().iloc[-1]
        
        # RSI calculation
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs)).iloc[-1]
        
        return {
            'sma_20': sma_20,
            'sma_50': sma_50,
            'rsi': rsi,
            'current_price': data['close'].iloc[-1],
            'trend': 'BULLISH' if sma_20 > sma_50 else 'BEARISH',
            'rsi_signal': 'OVERSOLD' if rsi < 30 else 'OVERBOUGHT' if rsi > 70 else 'NEUTRAL'
        }
    
    def display_header(self):
        """Display the main header"""
        st.markdown(
            """
            <div style='text-align: center;'>
                <h1 style='color: #1f77b4; margin-bottom: 0;'>🚀 Fusion Kernel</h1>
                <p style='color: #666; font-size: 1.2rem;'>Advanced Market Analysis Platform</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown("---")
    
    def display_metrics(self):
        """Display key metrics"""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📈 Market Trend", "BULLISH", "+1.2%")
        with col2:
            st.metric("💰 Portfolio Value", "₹1.5L", "+2.1%")
        with col3:
            st.metric("⚡ Active Stocks", "15/20", "75%")
        with col4:
            st.metric("🎯 Accuracy", "87.5%", "+5.2%")
    
    def display_stock_cards(self):
        """Display stock analysis cards"""
        st.markdown("### 📊 Stock Analysis")
        cols = st.columns(3)
        
        for idx, (symbol, info) in enumerate(list(self.stocks.items())[:3]):
            with cols[idx]:
                with st.expander(f"🔍 {symbol}", expanded=True):
                    indicators = self.calculate_technical_indicators(symbol)
                    
                    st.metric("Current Price", f"₹{indicators['current_price']:,.2f}")
                    st.metric("RSI", f"{indicators['rsi']:.1f}")
                    st.metric("Trend", indicators['trend'])
                    
                    # RSI indicator with color
                    rsi = indicators['rsi']
                    if rsi < 30:
                        st.success("📉 OVERSOLD - Potential Buy")
                    elif rsi > 70:
                        st.warning("📈 OVERBOUGHT - Consider Selling")
                    else:
                        st.info("➡️ NEUTRAL - Hold Position")
    
    def display_charts(self):
        """Display price charts"""
        st.markdown("### 📈 Live Market Charts")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### RELIANCE - Candlestick Chart")
            fig = self.create_price_chart('RELIANCE')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### TCS - Price Movement")
            fig = self.create_price_chart('TCS')
            st.plotly_chart(fig, use_container_width=True)
    
    def display_portfolio_overview(self):
        """Display portfolio overview"""
        st.markdown("### 💼 Portfolio Overview")
        
        # Sample portfolio data
        portfolio_data = {
            'Stock': ['RELIANCE', 'TCS', 'INFY', 'HDFC BANK', 'ICICI BANK'],
            'Quantity': [10, 5, 8, 15, 12],
            'Avg Price': [2450, 3450, 1450, 1650, 950],
            'Current Price': [2500, 3500, 1500, 1700, 1000],
            'P&L (%)': [2.04, 1.45, 3.45, 3.03, 5.26]
        }
        
        df = pd.DataFrame(portfolio_data)
        df['Value'] = df['Quantity'] * df['Current Price']
        df['Total P&L'] = df['Quantity'] * (df['Current Price'] - df['Avg Price'])
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Investment", "₹245,500")
        with col2:
            st.metric("Current Value", "₹254,200", "+3.5%")
        with col3:
            st.metric("Total P&L", "₹8,700", "+3.5%")
        
        # Display portfolio table
        st.dataframe(df, use_container_width=True)
    
    def display_market_news(self):
        """Display market news and updates"""
        st.markdown("### 📰 Market Intelligence")
        
        news_items = [
            {"title": "📈 Reliance Industries hits 52-week high", "impact": "Positive", "time": "2 hours ago"},
            {"title": "💰 RBI keeps repo rate unchanged at 6.5%", "impact": "Neutral", "time": "4 hours ago"},
            {"title": "🌍 Global markets show mixed signals", "impact": "Negative", "time": "6 hours ago"},
            {"title": "💼 TCS announces strong quarterly results", "impact": "Positive", "time": "8 hours ago"}
        ]
        
        for news in news_items:
            with st.container():
                col1, col2 = st.columns([0.9, 0.1])
                with col1:
                    st.write(f"**{news['title']}**")
                    st.caption(f"🕒 {news['time']} | Impact: {news['impact']}")
                st.markdown("---")

# Main app function
def main():
    app = FusionKernelApp()
    
    # Display header
    app.display_header()
    
    # Display metrics
    app.display_metrics()
    
    # Create tabs for better organization
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Dashboard", "🔍 Stock Analysis", "💼 Portfolio", "📰 Market News"])
    
    with tab1:
        st.markdown("### 🚀 Welcome to Fusion Kernel")
        st.info("""
        **Fusion Kernel** is an advanced market analysis platform that provides:
        - Real-time technical analysis
        - Portfolio tracking
        - Market intelligence
        - Trading signals
        """)
        
        app.display_charts()
    
    with tab2:
        app.display_stock_cards()
        
        # Additional technical analysis
        st.markdown("### 🔬 Advanced Technical Analysis")
        selected_stock = st.selectbox("Select Stock for Detailed Analysis", list(app.stocks.keys()))
        
        if selected_stock:
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(app.create_price_chart(selected_stock), use_container_width=True)
            with col2:
                indicators = app.calculate_technical_indicators(selected_stock)
                st.metric("RSI (14)", f"{indicators['rsi']:.1f}")
                st.metric("SMA (20)", f"₹{indicators['sma_20']:,.2f}")
                st.metric("SMA (50)", f"₹{indicators['sma_50']:,.2f}")
                st.metric("Trend", indicators['trend'])
    
    with tab3:
        app.display_portfolio_overview()
    
    with tab4:
        app.display_market_news()
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>"
        "🚀 <b>Fusion Kernel</b> - Advanced Market Analysis Platform | "
        "Data powered by Dhan API"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
