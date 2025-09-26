# CSV Agent: AI-Powered Data Analysis Tool

_Analyze Your CSV Files Using Natural Language with Ollama and Streamlit_

---

## Introduction

One of the biggest challenges in data analysis is understanding complex datasets and quickly extracting the information you need. Traditional methods can take hours and require technical expertise. This is where **CSV Agent** comes in - a powerful tool that uses artificial intelligence to analyze your CSV files through natural language.

In this article, we'll explore in detail how the CSV Agent application I developed using Python, Streamlit, and Ollama works, its features, and use cases.

## 🤖 What is CSV Agent?

CSV Agent is a web application that allows you to "talk" to your CSV files using natural language processing (NLP) technologies. With a simple sentence like "show me the age and salary columns," you can analyze your data, create visualizations, and generate statistical reports.

### Key Features

- **🧠 AI-Powered Analysis**: Uses Ollama's local language models
- **💬 Natural Language Queries**: Perform data analysis with English sentences
- **📊 Comprehensive Statistical Analysis**: Correlation, outlier detection, data quality metrics
- **🎨 Interactive Visualizations**: Dynamic charts with Plotly
- **⚡ Real-time Analysis**: Instant results and visualizations

## 🛠️ Technical Infrastructure

### Technologies Used

```python
# Main dependencies
pandas>=1.5.0          # Data manipulation
streamlit>=1.28.0       # Web interface
plotly>=5.15.0         # Interactive visualizations
requests>=2.28.0       # API communication
numpy, scipy, seaborn  # Statistical analysis
```

### Architecture

CSV Agent is designed with a modular structure:

```python
class CSVAgent:
    def __init__(self, ollama_url: str = "http://localhost:11434"):
        self.ollama_url = ollama_url
        self.df = None
        self.model_name = "llama3.2"
```

## 🚀 Installation and Setup

### 1. Prerequisites

```bash
# Python 3.8+ required
python --version

# Ollama installation
# Download from https://ollama.ai
ollama serve
ollama pull llama3.2
```

### 2. Project Setup

```bash
# Clone the project
git clone https://github.com/yourusername/csv-agent.git
cd csv-agent

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run csvagent_ui.py
```

### 3. Open in Browser

Navigate to `http://localhost:8501` to start using the application.

## 📊 Main Features and Usage

### 1. Data Loading and Preview

CSV Agent allows you to load data in two different ways:

- **File Upload**: Upload your CSV files with drag & drop
- **Quick Selection**: Choose from predefined files

```python
def load_csv_from_upload(self, uploaded_file) -> bool:
    try:
        self.df = pd.read_csv(uploaded_file)
        return True
    except Exception as e:
        st.error(f"Error loading CSV file: {e}")
        return False
```

### 2. AI-Powered Analysis

One of its most powerful features is the ability to perform data analysis using natural language:

```python
def analyze_columns(self, english_prompt: str) -> Dict[str, Any]:
    # Prepare prompt to send to Ollama
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
    """

    response = self.ask_ollama(full_prompt)
    return self._process_response(response)
```

**Example Queries:**

- "Show me the name and age columns"
- "Get all columns related to sales data"
- "Extract the date and amount columns"
- "Find columns with financial information"

### 3. Statistical Analysis Panel

The application offers five different analysis tabs:

#### 📈 Data Quality Analysis

```python
def get_data_quality_metrics(self) -> Dict[str, Any]:
    return {
        "total_rows": len(self.df),
        "total_columns": len(self.df.columns),
        "missing_cells": self.df.isnull().sum().sum(),
        "duplicate_rows": self.df.duplicated().sum(),
        "memory_usage": self.df.memory_usage(deep=True).sum() / 1024**2
    }
```

#### 📊 Statistical Summary

- Mean, median, standard deviation
- Skewness and kurtosis
- Quartile values (Q1, Q3)

#### 🔗 Correlation Analysis

```python
def get_correlation_matrix(self) -> pd.DataFrame:
    numeric_cols = self.df.select_dtypes(include=[np.number]).columns
    return self.df[numeric_cols].corr()
```

#### 🎯 Outlier Detection

- IQR (Interquartile Range) method
- Z-score method
- Visual outlier analysis

#### 📋 Data Type Analysis

- Column type distribution
- Detailed type information

### 4. Interactive Visualizations

Powerful visualizations with Plotly integration:

```python
# Histogram
fig = px.histogram(df, x=selected_cols[0],
                   title=f"Distribution of {selected_cols[0]}")

# Box Plot
fig = px.box(df, y=selected_cols[1],
             title=f"Box Plot of {selected_cols[1]}")

# Correlation Heatmap
fig = px.imshow(corr_matrix,
                text_auto=True,
                aspect="auto",
                title="Correlation Heatmap")
```

## 💡 Use Cases

### 1. Business Intelligence and Analytics

**Sales Analysis:**

```
"Show me sales data for the last quarter"
"Analyze customer demographics"
"Find the top 10 products by revenue"
```

**Customer Segmentation:**

```
"Group customers by age and income"
"Show me customer behavior patterns"
"Identify high-value customers"
```

### 2. Data Science and EDA

**Exploratory Data Analysis:**

```
"Perform comprehensive EDA on this dataset"
"Identify data quality issues"
"Show me the distribution of all numeric columns"
```

**Feature Engineering:**

```
"Find correlations between features"
"Identify potential new features"
"Analyze feature importance"
```

### 3. Research and Reporting

**Academic Research:**

```
"Generate statistical summary for research paper"
"Perform hypothesis testing on the data"
"Create publication-ready visualizations"
```

**Report Generation:**

```
"Create executive summary of key metrics"
"Generate monthly performance report"
"Analyze trends and patterns"
```

## 🎨 User Interface Features

### Modern and Responsive Design

```css
.main-header {
  font-size: 3rem;
  font-weight: bold;
  text-align: center;
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
}
```

### User-Friendly Features

- **Real-time Feedback**: Process status indicators
- **Error Management**: Comprehensive error messages and solution suggestions
- **Data Download**: Download filtered data as CSV
- **Model Selection**: Switch between different Ollama models

## 🔧 Advanced Features

### 1. Smart Column Selection

```python
def _extract_columns_from_response(self, response: str) -> List[str]:
    available_columns = list(self.df.columns)
    found_columns = []

    for column in available_columns:
        if column.lower() in response.lower():
            found_columns.append(column)

    return found_columns
```

### 2. Automatic Visualization

```python
# Automatic chart selection for numeric columns
numeric_cols = result['data'].select_dtypes(include=['number']).columns
categorical_cols = result['data'].select_dtypes(include=['object']).columns

if len(numeric_cols) >= 2:
    # Scatter plot
    fig = px.scatter(result['data'], x=x_col, y=y_col)
elif len(numeric_cols) >= 1:
    # Histogram
    fig = px.histogram(result['data'], x=hist_col)
```

### 3. Data Quality Monitoring

```python
def get_data_quality_metrics(self) -> Dict[str, Any]:
    total_cells = self.df.size
    missing_cells = self.df.isnull().sum().sum()
    duplicate_rows = self.df.duplicated().sum()

    return {
        "missing_percentage": (missing_cells / total_cells) * 100,
        "duplicate_percentage": (duplicate_rows / len(self.df)) * 100,
        "memory_usage": self.df.memory_usage(deep=True).sum() / 1024**2
    }
```

## 🚀 Performance and Optimization

### Memory Management

```python
# Data size control
memory_usage = self.df.memory_usage(deep=True).sum() / 1024**2
if memory_usage > 100:  # Warning above 100MB
    st.warning(f"Large dataset detected: {memory_usage:.2f} MB")
```

### Error Handling

```python
try:
    response = requests.post(
        f"{self.ollama_url}/api/generate",
        json=data,
        timeout=30
    )
except Exception as e:
    st.error(f"Ollama communication error: {e}")
    return ""
```

## 🔮 Future Plans

### Short-term Goals

- [ ] Excel and JSON file support
- [ ] More visualization types
- [ ] Data cleaning suggestions
- [ ] Automatic report generation

### Long-term Goals

- [ ] Cloud deployment options
- [ ] API endpoints
- [ ] Docker containerization
- [ ] Real-time data streaming

## 📈 Conclusion

CSV Agent is a powerful tool that democratizes the data analysis process and enables everyone to analyze their data using natural language. By combining Ollama's local AI models with Streamlit's user-friendly interface, you can perform comprehensive data analysis without requiring technical expertise.

### Advantages

✅ **Easy to Use**: Data analysis with natural language  
✅ **Local AI**: Your data is secure, no internet connection required  
✅ **Comprehensive Analysis**: Everything from statistical analysis to visualization  
✅ **Open Source**: Completely free and customizable  
✅ **Fast Results**: Instant analysis and visualization

### Who Can Use It?

- **Data Analysts**: Quick EDA and reporting
- **Business Analysts**: KPI analysis and trend detection
- **Researchers**: Academic data analysis
- **Students**: Data science learning
- **Entrepreneurs**: Understanding business data

With CSV Agent, data analysis is no longer just for experts - it becomes accessible to everyone. Use natural language to talk to your data, discover insights, and share your findings!

---

**GitHub Repository**: [CSV Agent](https://github.com/yourusername/csv-agent)  
**Demo**: [Live Demo](https://your-demo-link.com)  
**Documentation**: [Detailed Usage Guide](https://your-docs-link.com)

_This article comprehensively covers the technical details and use cases of the CSV Agent project. Visit the GitHub repository to experience the project and start your own data analysis journey._

---

**#DataScience #Python #AI #Streamlit #Ollama #DataAnalysis #MachineLearning #OpenSource**
