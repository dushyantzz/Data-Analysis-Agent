import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')
import docx
import PyPDF2
from PIL import Image
import json
import re
import os
from io import StringIO, BytesIO
from together import Together

TOGETHER_API_KEY = "cdd020e6a49733d3836952dbd379ccc3476264c76c0a510fcd9a6b490b588588"
MODEL_NAME = "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"

class DataAnalystAgent:
    def __init__(self, api_key, model_name):
        self.client = Together(api_key=api_key)
        self.model_name = model_name
        self.current_data = None
        self.data_info = {}
        self.conversation_history = []
        
    def query_llm(self, prompt, system_message="You are a senior data analyst with over 15 years of experience in advanced analytics, statistical modeling, and business intelligence across multiple industries. Your role is to thoroughly analyze the provided dataset, uncover meaningful patterns, trends, and anomalies, and generate actionable, descriptive insights that can drive strategic decision-making. For each insight, provide clear explanations, relevant statistical measures, and, where appropriate, suggest advanced data visualization techniques to communicate findings effectively to both technical and non-technical stakeholders. Anticipate potential business questions, highlight key opportunities or risks, and recommend next steps or further analyses. If any information or data context is missing, ask clarifying questions before proceeding. Present your findings in a structured, comprehensive, and business-oriented manner."):
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2048,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error querying model: {str(e)}"

    def read_uploaded_file(self, uploaded_file):
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        try:
            if file_extension == '.csv':
                return pd.read_csv(uploaded_file)
            elif file_extension in ['.xlsx', '.xls']:
                return pd.read_excel(uploaded_file)
            elif file_extension == '.txt':
                content = str(uploaded_file.read(), "utf-8")
                return content
            elif file_extension == '.docx':
                doc = docx.Document(uploaded_file)
                content = '\n'.join([paragraph.text for paragraph in doc.paragraphs])
                return content
            elif file_extension == '.pdf':
                reader = PyPDF2.PdfReader(uploaded_file)
                content = ''
                for page in reader.pages:
                    content += page.extract_text() + '\n'
                return content
            elif file_extension in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']:
                img = Image.open(uploaded_file)
                return f"Image loaded: {img.size} pixels, Mode: {img.mode}"
            else:
                return f"Unsupported file format: {file_extension}"
        except Exception as e:
            return f"Error reading file: {str(e)}"
    
    def analyze_data(self, data, question=None):
        if isinstance(data, pd.DataFrame):
            summary = self.generate_data_summary(data)
            if question:
                prompt = f"""I have a dataset with the following summary:
                {summary}
                Dataset preview:
                {data.head().to_string()}
                Question: {question}
                Please provide a detailed analysis and answer to this question based on the data."""
            else:
                prompt = f"""I have a dataset with the following summary:
                {summary}
                Dataset preview:
                {data.head().to_string()}
                Please provide a comprehensive analysis of this dataset including:
                1. Key insights and patterns
                2. Data quality observations
                3. Potential areas for further investigation
                4. Suggested visualizations"""
            system_message = """You are an expert data analyst. Provide clear, actionable insights based on the data provided. Focus on practical findings and recommendations."""
            return self.query_llm(prompt, system_message)
        elif isinstance(data, str):
            if question:
                prompt = f"""Text content: {data[:2000]}...
                Question: {question}
                Please analyze this text and answer the question."""
            else:
                prompt = f"""Text content: {data[:2000]}...
                Please provide a comprehensive analysis of this text including:
                1. Key themes and topics
                2. Summary of main points
                3. Any patterns or insights"""
            return self.query_llm(prompt)

    def generate_data_summary(self, df):
        summary = f"""Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns
        Columns and Data Types:
        {df.dtypes.to_string()}
        Missing Values:
        {df.isnull().sum().to_string()}
        Numerical Columns Summary:
        {df.describe().to_string() if len(df.select_dtypes(include=[np.number]).columns) > 0 else 'No numerical columns'}
        Categorical Columns:
        {list(df.select_dtypes(include=['object']).columns)}"""
        return summary
    
    def create_visualizations(self, df):
        if not isinstance(df, pd.DataFrame):
            return
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        if len(numerical_cols) > 0:
            st.subheader("📊 Data Overview")
            col1, col2 = st.columns(2)
            with col1:
                if len(numerical_cols) >= 1:
                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.hist(df[numerical_cols[0]].dropna(), bins=30, alpha=0.7, color='skyblue')
                    ax.set_title(f'Distribution of {numerical_cols[0]}')
                    ax.set_xlabel(numerical_cols[0])
                    ax.set_ylabel('Frequency')
                    st.pyplot(fig)
            with col2:
                missing_data = df.isnull().sum()
                missing_data = missing_data[missing_data > 0]
                if len(missing_data) > 0:
                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.bar(range(len(missing_data)), missing_data.values, color='coral')
                    ax.set_xticks(range(len(missing_data)))
                    ax.set_xticklabels(missing_data.index, rotation=45)
                    ax.set_title('Missing Values by Column')
                    ax.set_ylabel('Count')
                    st.pyplot(fig)
                else:
                    st.info("No missing values found!")
            if len(numerical_cols) > 1:
                st.subheader("🔥 Correlation Matrix")
                corr_matrix = df[numerical_cols].corr()
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
                ax.set_title('Correlation Matrix')
                st.pyplot(fig)
            if len(numerical_cols) >= 2:
                st.subheader("📈 Interactive Scatter Plot")
                col1_select = st.selectbox("Select X-axis:", numerical_cols, key="x_axis")
                col2_select = st.selectbox("Select Y-axis:", numerical_cols, index=1, key="y_axis")
                if categorical_cols:
                    color_col = st.selectbox("Color by (optional):", ["None"] + categorical_cols, key="color_by")
                    if color_col != "None":
                        fig = px.scatter(df, x=col1_select, y=col2_select, color=color_col, title=f"{col1_select} vs {col2_select}")
                    else:
                        fig = px.scatter(df, x=col1_select, y=col2_select, title=f"{col1_select} vs {col2_select}")
                else:
                    fig = px.scatter(df, x=col1_select, y=col2_select, title=f"{col1_select} vs {col2_select}")
                st.plotly_chart(fig, use_container_width=True)

@st.cache_resource
def get_agent():
    return DataAnalystAgent(TOGETHER_API_KEY, MODEL_NAME)

def main():
    st.set_page_config(
        page_title="Data Analyst Agent",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown('<h1 class="main-header">🤖 Data Analyst Agent</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem;">Powered by Llama-4-Maverick-17B-128E-Instruct-FP8 from Together.ai</p>', unsafe_allow_html=True)

    # Initialize agent
    agent = get_agent()

    # Sidebar
    with st.sidebar:
        st.header("📁 File Upload")
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['csv', 'xlsx', 'xls', 'txt', 'docx', 'pdf', 'jpg', 'jpeg', 'png', 'bmp', 'gif'],
            help="Supported formats: CSV, Excel, TXT, DOCX, PDF, Images"
        )

        st.header("❓ Analysis Options")
        analysis_type = st.radio(
            "Choose analysis type:",
            ["General Analysis", "Specific Question"]
        )

        if analysis_type == "Specific Question":
            user_question = st.text_area(
                "Enter your question:",
                placeholder="e.g., What are the main trends in this data?"
            )
        else:
            user_question = None

        st.header("📊 Model Info")
        st.info(f"**Model**: {MODEL_NAME}")
        st.success("✅ Agent Ready!")

    if uploaded_file is not None:
        st.success(f"📁 File uploaded: {uploaded_file.name}")

        with st.spinner("🔄 Processing file..."):
            data = agent.read_uploaded_file(uploaded_file)

        if isinstance(data, str) and data.startswith('Error'):
            st.error(data)
            return

        st.session_state.current_data = data
        st.session_state.agent = agent

        if isinstance(data, pd.DataFrame):
            st.subheader("📋 Data Preview")
            st.dataframe(data.head(10), use_container_width=True)

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Rows", data.shape[0])
            with col2:
                st.metric("Columns", data.shape[1])
            with col3:
                st.metric("Missing Values", data.isnull().sum().sum())
            with col4:
                st.metric("Memory Usage", f"{data.memory_usage(deep=True).sum() / 1024:.1f} KB")

        elif isinstance(data, str):
            st.subheader("📄 Text Preview")
            st.text_area("Content preview:", data[:1000] + "..." if len(data) > 1000 else data, height=200)

        st.subheader("🔍 AI Analysis")

        if st.button("🚀 Start Analysis", type="primary"):
            with st.spinner("🤖 AI is analyzing your data..."):
                analysis_result = agent.analyze_data(data, user_question)

            st.markdown("### 📊 Analysis Results")
            st.markdown(analysis_result)

            st.session_state.last_analysis = analysis_result

            if isinstance(data, pd.DataFrame):
                st.markdown("### 📈 Visualizations")
                agent.create_visualizations(data)

        if 'last_analysis' in st.session_state:
            st.markdown("---")
            st.subheader("💬 Follow-up Questions")

            followup_question = st.text_input(
                "Ask a follow-up question:",
                placeholder="e.g., Can you identify any outliers?"
            )

            if st.button("Ask Follow-up") and followup_question:
                with st.spinner("🤖 Processing follow-up question..."):
                    followup_result = agent.analyze_data(st.session_state.current_data, followup_question)

                st.markdown("### 💡 Follow-up Answer")
                st.markdown(followup_result)

    else:
        st.markdown("""
        <div class="info-box">
        <h2>🎯 Welcome to the Data Analyst Agent!</h2>
        <p>This AI-powered tool can analyze various types of files and answer questions about your data.</p>
        </div>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("""
            ### 📁 Supported Files
            - **CSV/Excel**: Full analysis with visualizations
            - **Text/Word/PDF**: Content analysis
            - **Images**: Basic image analysis
            """)

        with col2:
            st.markdown("""
            ### 🤖 AI Capabilities
            - Data summarization
            - Pattern identification
            - Question answering
            - Insight generation
            """)

        with col3:
            st.markdown("""
            ### 📊 Visualizations
            - Automatic chart generation
            - Interactive plots
            - Correlation analysis
            - Distribution analysis
            """)

        st.markdown("---")
        st.subheader("🧪 Try with Sample Data")

        if st.button("Generate Sample Sales Data"):
            np.random.seed(42)
            n_records = 1000

            sample_data = {
                'date': pd.date_range('2023-01-01', periods=n_records, freq='D'),
                'product': np.random.choice(['Product A', 'Product B', 'Product C', 'Product D'], n_records),
                'sales': np.random.normal(1000, 200, n_records),
                'region': np.random.choice(['North', 'South', 'East', 'West'], n_records),
                'customer_satisfaction': np.random.uniform(1, 5, n_records)
            }

            sample_df = pd.DataFrame(sample_data)
            sample_df['sales'] = sample_df['sales'].round(2)
            sample_df['customer_satisfaction'] = sample_df['customer_satisfaction'].round(1)

            st.session_state.current_data = sample_df
            st.session_state.agent = agent

            st.success("✅ Sample data generated!")
            st.dataframe(sample_df.head(), use_container_width=True)

            with st.spinner("🤖 Analyzing sample data..."):
                analysis_result = agent.analyze_data(sample_df)

            st.markdown("### 📊 Sample Data Analysis")
            st.markdown(analysis_result)

            st.markdown("### 📈 Sample Data Visualizations")
            agent.create_visualizations(sample_df)

if __name__ == "__main__":
    main()
