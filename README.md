# CSV Agent - AI-Powered Data Analysis Tool

A powerful Streamlit-based web application that uses AI (Ollama) to analyze CSV data through natural language queries. This tool provides comprehensive data analysis capabilities with an intuitive interface.

## 🚀 Features

### Core Functionality

- **AI-Powered Analysis**: Uses Ollama models to understand natural language queries about your data
- **CSV Upload & Management**: Upload CSV files or select from existing datasets
- **Interactive Data Exploration**: Browse and preview your data with an intuitive interface
- **Real-time Data Quality Metrics**: Get instant insights about data completeness and quality

### Advanced Analytics

- **Statistical Analysis**: Comprehensive statistical summaries including mean, median, standard deviation, skewness, and kurtosis
- **Correlation Analysis**: Visualize relationships between numeric variables with heatmaps
- **Outlier Detection**: Identify outliers using IQR and Z-score methods
- **Data Type Analysis**: Understand your data structure and column types
- **Missing Value Analysis**: Detailed breakdown of missing data patterns

### AI Capabilities

- **Natural Language Queries**: Ask questions about your data in plain English
- **Intelligent Column Selection**: AI automatically identifies relevant columns based on your query
- **Smart Data Filtering**: Extract specific data subsets based on your requirements
- **Automated Visualizations**: Generate charts and graphs based on your data

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- Ollama installed and running

### Setup Ollama

1. Install Ollama from [ollama.ai](https://ollama.ai)
2. Start Ollama service:
   ```bash
   ollama serve
   ```
3. Pull a model (e.g., Llama 3.2):
   ```bash
   ollama pull llama3.2
   ```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Required Packages

```
streamlit
pandas
requests
plotly
numpy
scipy
seaborn
matplotlib
```

## 🚀 Usage

### Starting the Application

```bash
streamlit run csvagent_ui.py
```

The application will open in your browser at `http://localhost:8501`

### Basic Workflow

1. **Connect to Ollama**: Click "Check Ollama Status" in the sidebar
2. **Select Model**: Choose your preferred Ollama model
3. **Upload CSV**: Either upload a file or select from existing datasets
4. **Analyze Data**: Use the statistical analysis tabs for comprehensive insights
5. **Ask AI**: Enter natural language queries to get AI-powered analysis

### Example Queries

- "Show me the name and age columns"
- "Get all columns related to sales data"
- "Extract the date and amount columns"
- "Find columns with financial information"
- "Show me customer data columns"

## 📊 Features Overview

### Data Quality Panel

- Total rows and columns count
- Missing values analysis with percentages
- Duplicate rows detection
- Memory usage statistics
- Missing values visualization by column

### Statistical Analysis

- Comprehensive statistical summaries
- Distribution plots (histograms, box plots)
- Column selection for targeted analysis
- Real-time statistical calculations

### Correlation Analysis

- Correlation matrix visualization
- Interactive heatmaps
- Strong correlation identification (|r| > 0.7)
- Relationship insights between variables

### Outlier Detection

- IQR and Z-score methods
- Outlier count and percentage
- Visual outlier identification
- Scatter plot analysis for context

### Data Types Analysis

- Data type distribution
- Column categorization by type
- Detailed type information
- Visual type distribution charts

## 🎯 Use Cases

### Business Intelligence

- Sales data analysis
- Customer segmentation
- Performance metrics evaluation
- Market research insights

### Data Science

- Exploratory data analysis (EDA)
- Data quality assessment
- Feature engineering insights
- Model preparation

### Research & Analytics

- Survey data analysis
- Statistical research
- Academic data exploration
- Report generation

## 🔧 Configuration

### Ollama Settings

- Default URL: `http://localhost:11434`
- Default model: `llama3.2`
- Configurable model selection
- Connection status monitoring

### Quick CSV Access

Pre-configured access to common datasets:

- Kaggle competition data
- S&P 500 data
- Sample datasets

## 📈 Data Visualization

### Supported Chart Types

- **Histograms**: Distribution analysis
- **Box Plots**: Outlier identification
- **Scatter Plots**: Relationship analysis
- **Bar Charts**: Categorical data visualization
- **Heatmaps**: Correlation matrices
- **Pie Charts**: Data type distribution

### Interactive Features

- Zoom and pan capabilities
- Hover information
- Responsive design
- Export capabilities

## 🚨 Troubleshooting

### Common Issues

**Ollama Connection Failed**

- Ensure Ollama is running: `ollama serve`
- Check if Ollama is accessible at `http://localhost:11434`
- Verify model is installed: `ollama list`

**CSV Upload Issues**

- Check file format (must be CSV)
- Ensure file is not corrupted
- Verify file size limits

**Memory Issues**

- Large datasets may require more RAM
- Consider data sampling for very large files
- Monitor memory usage in the metrics panel

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Development Setup

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- [Streamlit](https://streamlit.io) for the web framework
- [Ollama](https://ollama.ai) for AI capabilities
- [Plotly](https://plotly.com) for interactive visualizations
- [Pandas](https://pandas.pydata.org) for data manipulation

## 📞 Support

For support, please open an issue in the repository or contact the development team.

---

**Happy Analyzing! 🎉**
