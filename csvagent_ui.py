import streamlit as st
import pandas as pd
import requests
import json
import io
from typing import List, Dict, Any
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt

class CSVAgent:
    def __init__(self, ollama_url: str = "http://localhost:11434"):
        self.ollama_url = ollama_url
        self.df = None
        self.model_name = "llama3.2"
        
    def load_csv(self, file_path: str) -> bool:
        try:
            self.df = pd.read_csv(file_path)
            return True
        except Exception as e:
            st.error(f"CSV dosyası yüklenirken hata: {e}")
            return False
    
    def load_csv_from_upload(self, uploaded_file) -> bool:
        try:
            self.df = pd.read_csv(uploaded_file)
            return True
        except Exception as e:
            st.error(f"CSV dosyası yüklenirken hata: {e}")
            return False
    
    def get_available_models(self) -> List[str]:
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                return [model['name'] for model in models]
            return []
        except:
            return []
    
    def ask_ollama(self, prompt: str) -> str:
        try:
            data = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False
            }
            
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', '').strip()
            return ""
        except Exception as e:
            st.error(f"Ollama ile iletişim hatası: {e}")
            return ""
    
    def analyze_columns(self, english_prompt: str) -> Dict[str, Any]:
        if self.df is None:
            return {"error": "CSV dosyası yüklenmemiş"}
        
        columns_info = f"Available columns: {list(self.df.columns)}"
        sample_data = self.df.head(3).to_string()
        
        full_prompt = f"""
You are a CSV data analyst. I have a CSV file with the following information:

{columns_info}

Sample data (first 3 rows):
{sample_data}

User request: {english_prompt}

Please analyze this request and:
1. Identify which columns the user wants
2. Explain your reasoning
3. Return the data in a clear format

If the user wants specific columns, extract and return them. If they want analysis, provide insights.

Response format:
- Column names: [list the relevant column names]
- Reasoning: [explain why these columns are relevant]
- Data: [return the requested data or analysis]
"""
        
        response = self.ask_ollama(full_prompt)
        relevant_columns = self._extract_columns_from_response(response)
        
        if relevant_columns:
            filtered_data = self.df[relevant_columns]
            return {
                "columns": relevant_columns,
                "data": filtered_data,
                "ollama_response": response,
                "shape": filtered_data.shape
            }
        else:
            return {
                "columns": [],
                "data": self.df,
                "ollama_response": response,
                "shape": self.df.shape,
                "note": "Could not identify specific columns, returning all data"
            }
    
    def _extract_columns_from_response(self, response: str) -> List[str]:
        if not response:
            return []
        
        available_columns = list(self.df.columns)
        found_columns = []
        
        for column in available_columns:
            if column.lower() in response.lower():
                found_columns.append(column)
        
        return found_columns
    
    def get_data_quality_metrics(self) -> Dict[str, Any]:
        """Veri kalitesi metriklerini hesapla"""
        if self.df is None:
            return {}
        
        total_cells = self.df.size
        missing_cells = self.df.isnull().sum().sum()
        duplicate_rows = self.df.duplicated().sum()
        
        return {
            "total_rows": len(self.df),
            "total_columns": len(self.df.columns),
            "total_cells": total_cells,
            "missing_cells": missing_cells,
            "missing_percentage": (missing_cells / total_cells) * 100,
            "duplicate_rows": duplicate_rows,
            "duplicate_percentage": (duplicate_rows / len(self.df)) * 100,
            "memory_usage": self.df.memory_usage(deep=True).sum() / 1024**2  # MB
        }
    
    def get_statistical_summary(self) -> pd.DataFrame:
        """İstatistiksel özet tablosu"""
        if self.df is None:
            return pd.DataFrame()
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return pd.DataFrame()
        
        stats_data = []
        for col in numeric_cols:
            col_data = self.df[col].dropna()
            if len(col_data) > 0:
                stats_data.append({
                    'Column': col,
                    'Count': len(col_data),
                    'Mean': col_data.mean(),
                    'Median': col_data.median(),
                    'Std': col_data.std(),
                    'Min': col_data.min(),
                    'Max': col_data.max(),
                    'Q1': col_data.quantile(0.25),
                    'Q3': col_data.quantile(0.75),
                    'Skewness': col_data.skew(),
                    'Kurtosis': col_data.kurtosis()
                })
        
        return pd.DataFrame(stats_data)
    
    def get_correlation_matrix(self) -> pd.DataFrame:
        """Korelasyon matrisi"""
        if self.df is None:
            return pd.DataFrame()
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 2:
            return pd.DataFrame()
        
        return self.df[numeric_cols].corr()
    
    def detect_outliers(self, column: str, method: str = 'iqr') -> Dict[str, Any]:
        """Outlier tespiti"""
        if self.df is None or column not in self.df.columns:
            return {}
        
        col_data = self.df[column].dropna()
        if len(col_data) == 0:
            return {}
        
        if method == 'iqr':
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(col_data))
            outliers = col_data[z_scores > 3]
        
        return {
            'outlier_count': len(outliers),
            'outlier_percentage': (len(outliers) / len(col_data)) * 100,
            'outliers': outliers.tolist(),
            'method': method
        }
    
    def get_data_types_analysis(self) -> Dict[str, Any]:
        """Veri tipleri analizi"""
        if self.df is None:
            return {}
        
        type_counts = self.df.dtypes.value_counts()
        type_info = {}
        
        for dtype in self.df.dtypes.unique():
            cols = self.df.select_dtypes(include=[dtype]).columns
            type_info[str(dtype)] = {
                'count': len(cols),
                'columns': cols.tolist()
            }
        
        return {
            'type_counts': type_counts.to_dict(),
            'type_info': type_info
        }

def main():
    st.set_page_config(
        page_title="CSV Agent - AI Powered Data Analysis",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">🤖 CSV Agent</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">AI-Powered CSV Data Analysis with Ollama</p>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'agent' not in st.session_state:
        st.session_state.agent = CSVAgent()
    if 'csv_loaded' not in st.session_state:
        st.session_state.csv_loaded = False
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Ollama connection check
        st.subheader("🔗 Ollama Connection")
        if st.button("Check Ollama Status"):
            models = st.session_state.agent.get_available_models()
            if models:
                st.success(f"✅ Connected! {len(models)} models available")
                st.session_state.available_models = models
            else:
                st.error("❌ Ollama not running or not accessible")
                st.info("Please start Ollama: `ollama serve`")
        
        # Model selection
        if 'available_models' in st.session_state and st.session_state.available_models:
            st.subheader("🤖 Select Model")
            selected_model = st.selectbox(
                "Choose Ollama Model:",
                st.session_state.available_models,
                index=0
            )
            if st.button("Set Model"):
                st.session_state.agent.model_name = selected_model
                st.success(f"Model set to: {selected_model}")
        
        # CSV Upload
        st.subheader("📁 Upload CSV")
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type="csv",
            help="Upload your CSV file for analysis"
        )
        
        if uploaded_file is not None:
            if st.button("Load CSV"):
                with st.spinner("Loading CSV..."):
                    if st.session_state.agent.load_csv_from_upload(uploaded_file):
                        st.session_state.csv_loaded = True
                        st.success("✅ CSV loaded successfully!")
                    else:
                        st.error("❌ Failed to load CSV")
        
        # Quick CSV selection
        st.subheader("📂 Quick CSV Selection")
        quick_csvs = [
            "kagglecomps/csvfiles/train.csv",
            "kagglecomps/csvfiles/sp500.csv", 
            "lama/70kdata(temiz).csv"
        ]
        
        selected_quick_csv = st.selectbox("Select from existing files:", quick_csvs)
        if st.button("Load Selected CSV"):
            with st.spinner("Loading CSV..."):
                if st.session_state.agent.load_csv(selected_quick_csv):
                    st.session_state.csv_loaded = True
                    st.success("✅ CSV loaded successfully!")
                else:
                    st.error("❌ Failed to load CSV")
    
    # Main content
    if st.session_state.csv_loaded and st.session_state.agent.df is not None:
        df = st.session_state.agent.df
        
        # Data overview
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📊 Rows", f"{df.shape[0]:,}")
        with col2:
            st.metric("📋 Columns", f"{df.shape[1]:,}")
        with col3:
            st.metric("💾 Memory", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        with col4:
            st.metric("🔍 Missing Values", f"{df.isnull().sum().sum():,}")
        
        # Data preview
        st.subheader("📋 Data Preview")
        st.dataframe(df.head(10), use_container_width=True)
        
        # Statistical Analysis Panel
        st.subheader("📊 Statistical Analysis Panel")
        
        # Create tabs for different analysis types
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📈 Data Quality", 
            "📊 Statistics", 
            "🔗 Correlations", 
            "🎯 Outliers", 
            "📋 Data Types"
        ])
        
        with tab1:
            st.markdown("### 📈 Data Quality Metrics")
            quality_metrics = st.session_state.agent.get_data_quality_metrics()
            
            if quality_metrics:
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("📊 Total Rows", f"{quality_metrics['total_rows']:,}")
                with col2:
                    st.metric("📋 Total Columns", f"{quality_metrics['total_columns']:,}")
                with col3:
                    st.metric("❌ Missing Values", f"{quality_metrics['missing_cells']:,}")
                with col4:
                    st.metric("🔄 Duplicate Rows", f"{quality_metrics['duplicate_rows']:,}")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("📉 Missing %", f"{quality_metrics['missing_percentage']:.2f}%")
                with col2:
                    st.metric("🔄 Duplicate %", f"{quality_metrics['duplicate_percentage']:.2f}%")
                with col3:
                    st.metric("💾 Memory Usage", f"{quality_metrics['memory_usage']:.2f} MB")
                with col4:
                    st.metric("📊 Total Cells", f"{quality_metrics['total_cells']:,}")
                
                # Missing values by column
                if quality_metrics['missing_cells'] > 0:
                    st.markdown("#### ❌ Missing Values by Column")
                    missing_by_col = df.isnull().sum()
                    missing_df = pd.DataFrame({
                        'Column': missing_by_col.index,
                        'Missing Count': missing_by_col.values,
                        'Missing %': (missing_by_col.values / len(df)) * 100
                    }).sort_values('Missing Count', ascending=False)
                    
                    st.dataframe(missing_df, use_container_width=True)
                    
                    # Missing values visualization
                    if len(missing_df) > 0:
                        fig = px.bar(missing_df, x='Column', y='Missing Count', 
                                   title="Missing Values by Column")
                        fig.update_xaxis(tickangle=45)
                        st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.markdown("### 📊 Statistical Summary")
            stats_summary = st.session_state.agent.get_statistical_summary()
            
            if not stats_summary.empty:
                st.dataframe(stats_summary.round(4), use_container_width=True)
                
                # Distribution plots for numeric columns
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    st.markdown("#### 📈 Distribution Plots")
                    
                    selected_cols = st.multiselect(
                        "Select columns for distribution analysis:",
                        numeric_cols,
                        default=numeric_cols[:3] if len(numeric_cols) >= 3 else numeric_cols
                    )
                    
                    if selected_cols:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Histogram
                            fig = px.histogram(df, x=selected_cols[0], 
                                             title=f"Distribution of {selected_cols[0]}")
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            if len(selected_cols) > 1:
                                # Box plot
                                fig = px.box(df, y=selected_cols[1], 
                                           title=f"Box Plot of {selected_cols[1]}")
                                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No numeric columns found for statistical analysis.")
        
        with tab3:
            st.markdown("### 🔗 Correlation Analysis")
            corr_matrix = st.session_state.agent.get_correlation_matrix()
            
            if not corr_matrix.empty:
                st.markdown("#### 📊 Correlation Matrix")
                st.dataframe(corr_matrix.round(3), use_container_width=True)
                
                # Correlation heatmap
                fig = px.imshow(corr_matrix, 
                              text_auto=True, 
                              aspect="auto",
                              title="Correlation Heatmap")
                st.plotly_chart(fig, use_container_width=True)
                
                # Strong correlations
                st.markdown("#### 🔥 Strong Correlations (|r| > 0.7)")
                strong_corr = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        corr_val = corr_matrix.iloc[i, j]
                        if abs(corr_val) > 0.7:
                            strong_corr.append({
                                'Column 1': corr_matrix.columns[i],
                                'Column 2': corr_matrix.columns[j],
                                'Correlation': corr_val
                            })
                
                if strong_corr:
                    strong_corr_df = pd.DataFrame(strong_corr)
                    st.dataframe(strong_corr_df, use_container_width=True)
                else:
                    st.info("No strong correlations found (|r| > 0.7)")
            else:
                st.info("Need at least 2 numeric columns for correlation analysis.")
        
        with tab4:
            st.markdown("### 🎯 Outlier Detection")
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) > 0:
                selected_col = st.selectbox("Select column for outlier analysis:", numeric_cols)
                method = st.selectbox("Detection method:", ["iqr", "zscore"])
                
                if st.button("Detect Outliers"):
                    outlier_info = st.session_state.agent.detect_outliers(selected_col, method)
                    
                    if outlier_info:
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Outlier Count", outlier_info['outlier_count'])
                        with col2:
                            st.metric("Outlier %", f"{outlier_info['outlier_percentage']:.2f}%")
                        with col3:
                            st.metric("Method", outlier_info['method'].upper())
                        
                        if outlier_info['outlier_count'] > 0:
                            st.markdown("#### 📊 Outlier Visualization")
                            
                            # Box plot
                            fig = px.box(df, y=selected_col, title=f"Box Plot - {selected_col}")
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Scatter plot if we have another numeric column
                            other_numeric = [col for col in numeric_cols if col != selected_col]
                            if other_numeric:
                                x_col = st.selectbox("X-axis for scatter plot:", other_numeric)
                                fig = px.scatter(df, x=x_col, y=selected_col, 
                                               title=f"{x_col} vs {selected_col}")
                                st.plotly_chart(fig, use_container_width=True)
                            
                            # Outlier values
                            st.markdown("#### 🔍 Outlier Values")
                            outlier_df = pd.DataFrame({
                                'Outlier Values': outlier_info['outliers']
                            })
                            st.dataframe(outlier_df, use_container_width=True)
                        else:
                            st.success("No outliers detected!")
            else:
                st.info("No numeric columns found for outlier analysis.")
        
        with tab5:
            st.markdown("### 📋 Data Types Analysis")
            type_analysis = st.session_state.agent.get_data_types_analysis()
            
            if type_analysis:
                # Data type counts
                st.markdown("#### 📊 Data Type Distribution")
                type_counts = type_analysis['type_counts']
                
                col1, col2 = st.columns(2)
                
                with col1:
                    type_df = pd.DataFrame(list(type_counts.items()), 
                                         columns=['Data Type', 'Count'])
                    st.dataframe(type_df, use_container_width=True)
                
                with col2:
                    fig = px.pie(type_df, values='Count', names='Data Type', 
                               title="Data Type Distribution")
                    st.plotly_chart(fig, use_container_width=True)
                
                # Detailed type information
                st.markdown("#### 📋 Detailed Type Information")
                for dtype, info in type_analysis['type_info'].items():
                    with st.expander(f"{dtype} ({info['count']} columns)"):
                        st.write(f"**Columns:** {', '.join(info['columns'])}")
        
        # Column information
        with st.expander("📊 Column Information"):
            col_info = pd.DataFrame({
                'Column': df.columns,
                'Type': df.dtypes,
                'Non-Null Count': df.count(),
                'Null Count': df.isnull().sum(),
                'Unique Values': df.nunique()
            })
            st.dataframe(col_info, use_container_width=True)
        
        # AI Analysis Section
        st.subheader("🤖 AI-Powered Analysis")
        
        # Prompt input
        prompt = st.text_area(
            "Enter your analysis request in English:",
            placeholder="e.g., 'Show me the name and age columns' or 'Get all sales-related data'",
            height=100
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            analyze_btn = st.button("🔍 Analyze", type="primary", use_container_width=True)
        with col2:
            if st.button("💡 Example Prompts", use_container_width=True):
                st.info("""
                Try these example prompts:
                - "Show me the name and age columns"
                - "Get all columns related to sales data"
                - "Extract the date and amount columns"
                - "Find columns with financial information"
                - "Show me customer data columns"
                """)
        
        if analyze_btn and prompt:
            with st.spinner("🤖 AI is analyzing your request..."):
                result = st.session_state.agent.analyze_columns(prompt)
                
                if "error" in result:
                    st.error(f"❌ Error: {result['error']}")
                else:
                    # Display results
                    st.success("✅ Analysis completed!")
                    
                    # Show AI response
                    with st.expander("🤖 AI Response", expanded=True):
                        st.markdown(result['ollama_response'])
                    
                    # Show selected columns
                    if result['columns']:
                        st.info(f"📋 Selected columns: {', '.join(result['columns'])}")
                    
                    # Display filtered data
                    if not result['data'].empty:
                        st.subheader("📊 Filtered Data")
                        st.dataframe(result['data'], use_container_width=True)
                        
                        # Data visualization
                        if len(result['data'].columns) >= 2:
                            st.subheader("📈 Data Visualization")
                            
                            # Auto-detect numeric columns for visualization
                            numeric_cols = result['data'].select_dtypes(include=['number']).columns
                            categorical_cols = result['data'].select_dtypes(include=['object']).columns
                            
                            if len(numeric_cols) >= 1:
                                viz_col1, viz_col2 = st.columns(2)
                                
                                with viz_col1:
                                    if len(numeric_cols) >= 2:
                                        # Scatter plot
                                        x_col = st.selectbox("X-axis:", numeric_cols)
                                        y_col = st.selectbox("Y-axis:", numeric_cols)
                                        fig = px.scatter(result['data'], x=x_col, y=y_col, title=f"{x_col} vs {y_col}")
                                        st.plotly_chart(fig, use_container_width=True)
                                
                                with viz_col2:
                                    if len(numeric_cols) >= 1:
                                        # Histogram
                                        hist_col = st.selectbox("Histogram column:", numeric_cols)
                                        fig = px.histogram(result['data'], x=hist_col, title=f"Distribution of {hist_col}")
                                        st.plotly_chart(fig, use_container_width=True)
                            
                            # Categorical data visualization
                            if len(categorical_cols) >= 1:
                                cat_col = st.selectbox("Categorical column:", categorical_cols)
                                value_counts = result['data'][cat_col].value_counts().head(10)
                                fig = px.bar(x=value_counts.index, y=value_counts.values, 
                                           title=f"Top 10 values in {cat_col}")
                                st.plotly_chart(fig, use_container_width=True)
                        
                        # Download button
                        csv_buffer = io.StringIO()
                        result['data'].to_csv(csv_buffer, index=False)
                        csv_data = csv_buffer.getvalue()
                        
                        st.download_button(
                            label="📥 Download Filtered Data as CSV",
                            data=csv_data,
                            file_name="filtered_data.csv",
                            mime="text/csv"
                        )
                    else:
                        st.warning("No data found matching your criteria.")
    
    else:
        # Welcome screen
        st.markdown("""
        <div class="info-box">
        <h3>👋 Welcome to CSV Agent!</h3>
        <p>This AI-powered tool helps you analyze CSV data using natural language prompts.</p>
        <h4>🚀 Getting Started:</h4>
        <ol>
        <li>Make sure Ollama is running: <code>ollama serve</code></li>
        <li>Upload a CSV file using the sidebar</li>
        <li>Enter your analysis request in English</li>
        <li>Get instant AI-powered insights!</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)
        
        # Example data
        st.subheader("📊 Example Data")
        example_data = pd.DataFrame({
            'Name': ['John', 'Jane', 'Bob', 'Alice', 'Charlie'],
            'Age': [25, 30, 35, 28, 42],
            'City': ['New York', 'London', 'Paris', 'Tokyo', 'Sydney'],
            'Salary': [50000, 60000, 55000, 70000, 80000],
            'Department': ['Sales', 'Marketing', 'IT', 'HR', 'Finance']
        })
        st.dataframe(example_data, use_container_width=True)
        
        st.info("💡 Try uploading your own CSV file to get started!")

if __name__ == "__main__":
    main()
