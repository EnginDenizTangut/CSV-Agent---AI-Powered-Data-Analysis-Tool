# 🤖 CSV Agent: AI-Powered Data Analysis with Ollama

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)
[![Ollama](https://img.shields.io/badge/Ollama-LLM-green.svg)](https://ollama.ai)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A powerful, AI-driven CSV data analysis tool that leverages Ollama's local language models to provide natural language data exploration and analysis capabilities. Perfect for data scientists, analysts, and anyone who wants to interact with their data using conversational AI.

## 🌟 Key Features

### 🧠 AI-Powered Analysis

- **Natural Language Queries**: Ask questions about your data in plain English
- **Intelligent Column Selection**: AI automatically identifies relevant columns
- **Smart Data Filtering**: Extract specific data subsets based on your requirements
- **Automated Visualizations**: Generate charts and graphs based on your data

### 📊 Comprehensive Data Analysis

- **Statistical Analysis**: Mean, median, standard deviation, skewness, kurtosis
- **Correlation Analysis**: Interactive heatmaps and relationship insights
- **Outlier Detection**: IQR and Z-score methods with visualizations
- **Data Quality Metrics**: Missing values, duplicates, memory usage analysis
- **Data Type Analysis**: Comprehensive type distribution and categorization

### 🎨 Interactive Visualizations

- **Plotly Integration**: Interactive charts with zoom, pan, and hover capabilities
- **Multiple Chart Types**: Histograms, box plots, scatter plots, bar charts, heatmaps
- **Responsive Design**: Works seamlessly on desktop and mobile devices
- **Export Capabilities**: Download filtered data and visualizations

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Ollama installed and running

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/csv-agent.git
   cd csv-agent
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Setup Ollama**

   ```bash
   # Install Ollama from https://ollama.ai
   ollama serve
   ollama pull llama3.2
   ```

4. **Run the application**

   ```bash
   streamlit run src/csvagent_ui.py
   ```

5. **Open your browser**
   Navigate to `http://localhost:8501`

## 📁 Project Structure

```
csv-agent/
├── README.md                 # This file
├── LICENSE                   # MIT License
├── requirements.txt          # Python dependencies
├── .gitignore               # Git ignore rules
├── .env.example             # Environment variables template
├── config/
│   ├── __init__.py
│   ├── settings.py          # Configuration settings
│   └── models.py            # Data models
├── src/
│   ├── __init__.py
│   ├── csvagent_ui.py       # Main Streamlit application
│   ├── core/
│   │   ├── __init__.py
│   │   ├── agent.py         # Core CSV Agent class
│   │   ├── analyzer.py      # Data analysis functions
│   │   └── visualizer.py    # Visualization functions
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── data_loader.py   # Data loading utilities
│   │   └── helpers.py       # Helper functions
│   └── api/
│       ├── __init__.py
│       └── ollama_client.py # Ollama API client
├── examples/
│   ├── sample_data/         # Sample CSV files
│   ├── notebooks/           # Jupyter notebooks
│   └── use_cases/           # Example use cases
├── tests/
│   ├── __init__.py
│   ├── test_agent.py
│   └── test_analyzer.py
├── docs/
│   ├── installation.md
│   ├── usage.md
│   └── api_reference.md
└── scripts/
    ├── setup.sh             # Setup script
    └── run_demo.py          # Demo script
```

## 🎯 Use Cases

### Business Intelligence

- **Sales Analysis**: Analyze sales data, identify trends, and generate insights
- **Customer Segmentation**: Group customers based on behavior and demographics
- **Performance Metrics**: Track KPIs and business performance indicators
- **Market Research**: Analyze survey data and market trends

### Data Science

- **Exploratory Data Analysis (EDA)**: Comprehensive data exploration and profiling
- **Data Quality Assessment**: Identify data issues and quality problems
- **Feature Engineering**: Discover relationships and create new features
- **Model Preparation**: Prepare data for machine learning models

### Research & Analytics

- **Academic Research**: Analyze research data and generate statistical insights
- **Survey Analysis**: Process and analyze survey responses
- **Statistical Analysis**: Perform comprehensive statistical tests
- **Report Generation**: Create automated reports and visualizations

## 🔧 Configuration

### Environment Variables

Create a `.env` file based on `.env.example`:

```bash
OLLAMA_URL=http://localhost:11434
DEFAULT_MODEL=llama3.2
MAX_FILE_SIZE=100MB
ENABLE_CACHING=true
```

### Model Configuration

The application supports various Ollama models:

- `llama3.2` (recommended)
- `llama3.1`
- `codellama`
- `mistral`
- `phi3`

## 📊 Example Queries

Try these natural language queries with your data:

```
"Show me the name and age columns"
"Get all columns related to sales data"
"Extract the date and amount columns"
"Find columns with financial information"
"Show me customer data columns"
"Analyze the relationship between price and sales"
"Identify outliers in the revenue column"
"Create a correlation matrix for numeric columns"
```

## 🛠️ Development

### Running Tests

```bash
pytest tests/
```

### Code Formatting

```bash
black src/
isort src/
```

### Type Checking

```bash
mypy src/
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Streamlit](https://streamlit.io) for the amazing web framework
- [Ollama](https://ollama.ai) for local AI capabilities
- [Plotly](https://plotly.com) for interactive visualizations
- [Pandas](https://pandas.pydata.org) for data manipulation
- [NumPy](https://numpy.org) for numerical computing
- [SciPy](https://scipy.org) for scientific computing

## 📞 Support

- 📧 Email: support@csvagent.com
- 🐛 Issues: [GitHub Issues](https://github.com/yourusername/csv-agent/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/yourusername/csv-agent/discussions)

## 📈 Roadmap

- [ ] Support for more data formats (Excel, JSON, Parquet)
- [ ] Advanced AI models integration (GPT-4, Claude)
- [ ] Real-time data streaming capabilities
- [ ] Collaborative analysis features
- [ ] API endpoints for programmatic access
- [ ] Docker containerization
- [ ] Cloud deployment options

---

**Made with ❤️ for the data science community**

_Happy Analyzing! 🎉_
