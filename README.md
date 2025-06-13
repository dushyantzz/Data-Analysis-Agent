# 🤖 Data Analyst Agent

## Project Overview

The **Data Analyst Agent** is an advanced AI-powered data analysis tool that leverages the **Llama-4-Maverick-17B-128E-Instruct-FP8** model from Together.ai to provide comprehensive data analysis capabilities. This project offers both a Jupyter notebook interface and a modern Streamlit web application for analyzing various types of data files and generating actionable insights.

## 🎯 Key Features

### 📁 Multi-Format File Support
- **CSV Files**: Complete statistical analysis with automatic visualizations
- **Excel Files (.xlsx, .xls)**: Full spreadsheet data processing and analysis
- **Text Files (.txt)**: Content analysis, theme extraction, and summarization
- **Word Documents (.docx)**: Document analysis and key insights extraction
- **PDF Files (.pdf)**: Text extraction and comprehensive content analysis
- **Images (.jpg, .png, .bmp, .gif)**: Basic image metadata and analysis

### 🧠 AI-Powered Analysis Engine
- **Advanced Data Insights**: Utilizes the specified Llama-4-Maverick model for deep data understanding
- **Pattern Recognition**: Automatically identifies trends, correlations, and anomalies
- **Statistical Analysis**: Comprehensive descriptive statistics and data quality assessment
- **Business Intelligence**: Generates actionable recommendations and strategic insights
- **Context-Aware Responses**: Maintains conversation history for follow-up questions

### 📊 Automatic Visualization Generation
- **Distribution Plots**: Histograms for numerical data distribution analysis
- **Correlation Matrices**: Heatmaps showing relationships between variables
- **Missing Data Analysis**: Visual representation of data completeness
- **Interactive Charts**: Plotly-powered interactive scatter plots and visualizations
- **Data Type Overview**: Pie charts showing dataset composition

### 💬 Conversational Interface
- **Question-Answer System**: Ask specific questions about your data
- **Follow-up Capabilities**: Continue conversations with context retention
- **Natural Language Processing**: Understands complex analytical queries
- **Detailed Explanations**: Provides clear, business-oriented insights

## 🏗️ Project Architecture

### Core Components

#### 1. DataAnalystAgent Class
The main engine of the application with the following methods:

- **`__init__(api_key, model_name)`**: Initializes the agent with Together.ai credentials
- **`query_llm(prompt, system_message)`**: Interfaces with the Llama model for analysis
- **`read_file(file_path)`**: Handles multiple file format reading and processing
- **`analyze_data(data, question)`**: Core analysis engine for data interpretation
- **`generate_data_summary(df)`**: Creates comprehensive dataset summaries
- **`create_visualizations(df)`**: Generates automatic charts and plots
- **`upload_and_analyze(file_path, question)`**: Main workflow for file analysis
- **`ask_followup(question)`**: Handles follow-up questions with context

#### 2. File Processing Engine
Robust file handling system supporting:
- **Pandas Integration**: For CSV and Excel file processing
- **Document Processing**: Using python-docx for Word documents
- **PDF Text Extraction**: PyPDF2 for PDF content analysis
- **Image Processing**: PIL for image metadata extraction
- **Error Handling**: Comprehensive exception management

#### 3. Visualization System
Multi-layered visualization approach:
- **Matplotlib**: For static statistical plots
- **Seaborn**: For enhanced statistical visualizations
- **Plotly**: For interactive charts and dashboards
- **Automatic Chart Selection**: Based on data types and structure

## 🚀 Installation and Setup

### Prerequisites
- Python 3.7 or higher
- Together.ai API account and key
- Virtual environment (recommended)

### Step-by-Step Installation

1. **Clone or Download the Project**
   ```bash
   cd Desktop/int_2_project
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv data_analyst_env
   ```

3. **Activate Virtual Environment**
   ```bash
   # Windows
   data_analyst_env\Scripts\activate.bat
   
   # Linux/Mac
   source data_analyst_env/bin/activate
   ```

4. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Required Packages
- **together**: Together.ai API client
- **streamlit**: Web application framework
- **pandas**: Data manipulation and analysis
- **matplotlib**: Static plotting library
- **seaborn**: Statistical data visualization
- **plotly**: Interactive plotting library
- **openpyxl**: Excel file support
- **python-docx**: Word document processing
- **PyPDF2**: PDF text extraction
- **pillow**: Image processing
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning utilities

## 🎮 Usage Guide

### Jupyter Notebook Interface

1. **Launch Jupyter**
   ```bash
   jupyter notebook data_analyst_agent.ipynb
   ```

2. **Run Installation Cell**
   Execute the first cell to install required packages

3. **Initialize Agent**
   Run the configuration and agent initialization cells

4. **Analyze Data**
   ```python
   # Basic analysis
   result = agent.upload_and_analyze('your_file.csv')
   print(result)
   
   # Question-specific analysis
   result = agent.upload_and_analyze('your_file.csv', "What are the main trends?")
   print(result)
   
   # Follow-up questions
   followup = agent.ask_followup("Are there any outliers?")
   print(followup)
   ```

### Streamlit Web Application

1. **Launch Streamlit App**
   ```bash
   streamlit run streamlit_app.py
   ```

2. **Access Web Interface**
   Open your browser to `http://localhost:8501`

3. **Upload and Analyze**
   - Use the file uploader in the sidebar
   - Choose analysis type (General or Specific Question)
   - View results and interactive visualizations
   - Ask follow-up questions in real-time

## 🔧 Configuration

### API Key Setup
The project is pre-configured with the Together.ai API key:
```python
TOGETHER_API_KEY = "cdd020e6a49733d3836952dbd379ccc3476264c76c0a510fcd9a6b490b588588"
MODEL_NAME = "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"
```

### Model Configuration
The system uses the **meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8** model with:
- **Max Tokens**: 2048 for comprehensive responses
- **Temperature**: 0.7 for balanced creativity and accuracy
- **System Message**: Optimized for senior data analyst expertise

## 📈 Analysis Capabilities

### Structured Data Analysis (CSV/Excel)
- **Descriptive Statistics**: Mean, median, mode, standard deviation
- **Data Quality Assessment**: Missing values, data types, outliers
- **Correlation Analysis**: Relationships between variables
- **Distribution Analysis**: Data spread and normality
- **Categorical Analysis**: Frequency distributions and patterns

### Text Data Analysis (TXT/DOCX/PDF)
- **Content Summarization**: Key themes and main points
- **Sentiment Analysis**: Tone and emotional content
- **Topic Extraction**: Main subjects and concepts
- **Pattern Recognition**: Recurring themes and structures

### Visual Analytics
- **Automatic Chart Generation**: Based on data characteristics
- **Interactive Dashboards**: User-controlled visualizations
- **Statistical Plots**: Histograms, scatter plots, heatmaps
- **Missing Data Visualization**: Data completeness assessment

## 🎯 Use Cases

### Business Intelligence
- Sales data analysis and trend identification
- Customer behavior pattern recognition
- Market research data interpretation
- Performance metrics evaluation

### Research and Academia
- Survey data analysis and insights
- Experimental data interpretation
- Literature review and document analysis
- Statistical hypothesis testing

### Data Science Projects
- Exploratory data analysis (EDA)
- Feature engineering insights
- Data quality assessment
- Preliminary model recommendations

## 🔒 Security and Privacy

- **API Key Management**: Secure handling of Together.ai credentials
- **Local Processing**: Data analysis performed locally
- **No Data Storage**: Files processed in memory without persistence
- **Error Handling**: Robust exception management for data protection

## 🚀 Performance Optimization

- **Streamlit Caching**: Efficient agent initialization with `@st.cache_resource`
- **Memory Management**: Optimized data processing for large files
- **Async Processing**: Non-blocking UI during analysis
- **Error Recovery**: Graceful handling of processing failures

## 📝 Project Structure

```
Desktop/int_2_project/
├── data_analyst_agent.ipynb    # Jupyter notebook interface
├── streamlit_app.py           # Streamlit web application
├── requirements.txt           # Python dependencies
└── README.md                 # This documentation file
```

## 🤝 Contributing

This project demonstrates advanced AI integration for data analysis. Key areas for enhancement:
- Additional file format support
- Advanced statistical methods
- Machine learning model recommendations
- Enhanced visualization options
- Real-time data streaming capabilities

## 📄 License

This project is designed for educational and professional use, showcasing the integration of modern AI models with data analysis workflows.

---

**Built with ❤️ using Llama-4-Maverick-17B-128E-Instruct-FP8 from Together.ai**
#
