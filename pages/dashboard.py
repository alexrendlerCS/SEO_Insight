"""
Streamlit dashboard for SEO analysis and visualization.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
from typing import Dict, List
import json
import logging

from services.google_ads import GoogleAdsService
from services.llm_generator import LLMGenerator
from utils.keyword_utils import (
    KeywordClusterer,
    cluster_low_performance_keywords,
    prepare_case_study_data,
    export_case_study_report
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="SEO Analysis Dashboard",
    page_icon="🔍",
    layout="wide"
)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'clustered_data' not in st.session_state:
    st.session_state.clustered_data = None
if 'suggestions' not in st.session_state:
    st.session_state.suggestions = None
if 'estimated_ctrs' not in st.session_state:
    st.session_state.estimated_ctrs = None
if 'case_study_data' not in st.session_state:
    st.session_state.case_study_data = None

def load_data():
    """Load and process keyword data from file or mock data."""
    st.sidebar.header("Data Source")
    
    # Option to use mock data
    use_mock = st.sidebar.checkbox("Use Mock Data", value=True)
    
    if use_mock:
        try:
            mock_data_path = os.path.join('data', 'mock_keyword_data.csv')
            if not os.path.exists(mock_data_path):
                mock_data_path = 'mock_keyword_data.csv'
            df = pd.read_csv(mock_data_path)
            st.sidebar.success("✅ Loaded mock data successfully!")
            return df
        except Exception as e:
            logger.error(f"Error loading mock data: {str(e)}")
            st.sidebar.error("❌ Error loading mock data. Please check the file exists.")
            return None
    
    # File upload option
    uploaded_file = st.sidebar.file_uploader("Upload Keyword Data (CSV)", type=['csv'])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.sidebar.success("✅ File uploaded successfully!")
            return df
        except Exception as e:
            logger.error(f"Error reading file: {str(e)}")
            st.sidebar.error("❌ Error reading file. Please check the format.")
            return None
    
    return None

def display_metrics(df: pd.DataFrame):
    """Display key performance metrics and data table."""
    st.header("Keyword Performance Metrics")
    
    # Calculate metrics
    total_keywords = len(df)
    total_impressions = df['impressions'].sum()
    total_clicks = df['clicks'].sum()
    avg_ctr = (total_clicks / total_impressions * 100) if total_impressions > 0 else 0
    
    # Display metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Keywords", total_keywords)
    with col2:
        st.metric("Total Impressions", f"{total_impressions:,.0f}")
    with col3:
        st.metric("Total Clicks", f"{total_clicks:,.0f}")
    with col4:
        st.metric("Average CTR", f"{avg_ctr:.2f}%")
    
    # Display data table
    st.dataframe(
        df.style.format({
            'CTR': '{:.2f}%',
            'cost': '${:.2f}',
            'avg_position': '{:.1f}'
        }),
        use_container_width=True
    )

def perform_clustering(df: pd.DataFrame):
    """Perform keyword clustering with adjustable parameters."""
    st.header("Keyword Clustering")
    
    # Clustering section
    if st.session_state.data is not None:
        st.markdown("### 🔄 Keyword Clustering")
        st.markdown("""
        🎯 *Adjust the CTR threshold to identify low-performing keywords, then group them into clusters 
        based on similarity to discover common themes and improvement opportunities.*
        """)
        
        col1, col2 = st.columns(2)
        with col1:
            ctr_threshold = st.slider(
                "CTR Threshold (%)",
                min_value=0.0,
                max_value=10.0,
                value=1.0,
                step=0.1,
                help="Keywords with CTR below this threshold will be considered low-performing"
            )
        
        with col2:
            n_clusters = st.slider(
                "Number of Clusters",
                min_value=2,
                max_value=10,
                value=5,
                help="Number of clusters to group similar keywords"
            )
    
    # Option to use transformer
    use_transformer = st.checkbox("Use Advanced Embeddings (SentenceTransformer)", value=False)
    
    if st.button("Run Clustering"):
        with st.spinner("🔄 Clustering keywords..."):
            try:
                clustered_df = cluster_low_performance_keywords(
                    df,
                    ctr_threshold=ctr_threshold,
                    n_clusters=n_clusters,
                    use_transformer=use_transformer
                )
                st.session_state.clustered_data = clustered_df
                st.success("✅ Clustering completed successfully!")
                
                # Display low-performing keywords
                low_perf_df = clustered_df[clustered_df['cluster_label'] != -1]
                if not low_perf_df.empty:
                    st.subheader("Low-Performing Keywords")
                    st.dataframe(
                        low_perf_df.style.format({
                            'CTR': '{:.2f}%',
                            'cost': '${:.2f}',
                            'avg_position': '{:.1f}'
                        }),
                        use_container_width=True
                    )
            except Exception as e:
                logger.error(f"Error during clustering: {str(e)}")
                st.error("❌ Error during clustering. Please try again.")
                return
    
    if st.session_state.clustered_data is not None:
        # Display cluster visualization
        try:
            clusterer = KeywordClusterer(n_clusters=n_clusters, use_transformer=use_transformer)
            keywords = st.session_state.clustered_data['keyword'].tolist()
            embeddings, labels = clusterer.cluster_keywords(keywords)
            
            fig = clusterer.visualize_clusters(embeddings, labels, keywords)
            st.plotly_chart(fig, use_container_width=True)
            
            # Add descriptive section
            st.subheader("🔍 Clusters")
            st.markdown("""
            Each cluster groups similar keywords based on their meaning. 
            Use clusters to identify keyword themes and replace underperformers more effectively.
            """)
            
            # Show low-performing keywords
            show_low_performing_keywords()
        except Exception as e:
            logger.error(f"Error visualizing clusters: {str(e)}")
            st.error("❌ Error visualizing clusters. Please try again.")

def generate_suggestions():
    """Generate and display LLM suggestions for low-performing keywords."""
    st.header("AI-Generated Suggestions")
    
    if st.session_state.clustered_data is None:
        st.warning("⚠️ Please run clustering first to identify low-performing keywords.")
        return
    
    if st.button("Generate Suggestions"):
        with st.spinner("🔄 Generating suggestions..."):
            try:
                # Initialize LLM
                llm = LLMGenerator()
                
                # Get low-performing keywords
                low_perf_keywords = st.session_state.clustered_data[
                    st.session_state.clustered_data['cluster_label'] != -1
                ]['keyword'].tolist()
                
                # Generate suggestions
                suggestions = llm.generate_keyword_suggestions(low_perf_keywords)
                meta_descriptions = {
                    keyword: llm.generate_meta_description(keyword)
                    for keyword in low_perf_keywords
                }
                
                # Store suggestions
                st.session_state.suggestions = {
                    'keywords': suggestions,
                    'descriptions': meta_descriptions
                }
                
                st.success("✅ Suggestions generated successfully!")
            except Exception as e:
                logger.error(f"Error generating suggestions: {str(e)}")
                st.error("❌ Error generating suggestions. Please try again.")
                return
    
    if st.session_state.suggestions:
        # Display suggestions and collect estimated CTRs
        st.subheader("Keyword Suggestions and CTR Estimates")
        
        for keyword, alternatives in st.session_state.suggestions['keywords'].items():
            with st.expander(f"Suggestions for: {keyword}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write("Alternative Keywords:")
                    for alt in alternatives:
                        st.write(f"- {alt}")
                with col2:
                    st.write("Meta Description:")
                    st.write(st.session_state.suggestions['descriptions'][keyword])
                
                # Add CTR estimate input
                estimated_ctr = st.number_input(
                    f"Estimated New CTR for {keyword} (%)",
                    min_value=0.0,
                    max_value=100.0,
                    value=float(st.session_state.clustered_data[
                        st.session_state.clustered_data['keyword'] == keyword
                    ]['CTR'].iloc[0]),
                    step=0.1,
                    key=f"ctr_{keyword}"
                )
                st.session_state.estimated_ctrs[keyword] = estimated_ctr

def prepare_case_study():
    """Prepare and display case study data."""
    st.header("Case Study Analysis")
    
    if not st.session_state.suggestions or not st.session_state.estimated_ctrs:
        st.warning("⚠️ Please generate suggestions and provide CTR estimates first.")
        return
    
    if st.button("Generate Case Study Report"):
        with st.spinner("🔄 Preparing case study data..."):
            try:
                case_study_df = prepare_case_study_data(
                    st.session_state.clustered_data,
                    st.session_state.suggestions['keywords'],
                    st.session_state.estimated_ctrs
                )
                st.session_state.case_study_data = case_study_df
                
                # Display case study data
                st.dataframe(
                    case_study_df.style.format({
                        'original_ctr': '{:.2f}%',
                        'estimated_ctr': '{:.2f}%',
                        'ctr_uplift': '{:.2f}%'
                    }),
                    use_container_width=True
                )
                
                # Create CTR comparison bar chart
                st.subheader("CTR Comparison")
                
                # Prepare data for plotting
                plot_df = case_study_df.melt(
                    id_vars=['keyword'],
                    value_vars=['original_ctr', 'estimated_ctr'],
                    var_name='CTR Type',
                    value_name='CTR'
                )
                
                # Create bar chart
                fig = px.bar(
                    plot_df,
                    x='keyword',
                    y='CTR',
                    color='CTR Type',
                    barmode='group',
                    title='Original vs Estimated CTR Comparison',
                    labels={
                        'keyword': 'Keywords',
                        'CTR': 'CTR (%)',
                        'CTR Type': 'CTR Type'
                    }
                )
                
                # Update layout
                fig.update_layout(
                    xaxis_tickangle=-45,
                    showlegend=True,
                    legend_title='',
                    yaxis_title='CTR (%)',
                    xaxis_title='Keywords'
                )
                
                # Display chart
                st.plotly_chart(fig, use_container_width=True)
                
                # Export button
                if st.button("Export Case Study Report"):
                    export_case_study_report(case_study_df)
                    st.success("✅ Case study report exported successfully!")
            except Exception as e:
                logger.error(f"Error preparing case study: {str(e)}")
                st.error("❌ Error preparing case study. Please try again.")

def show_progress_tracker():
    """Display progress tracker in the sidebar."""
    st.sidebar.markdown("### 🧭 Progress Tracker")
    
    # Define steps and their completion conditions
    steps = [
        {
            "name": "Data Loaded",
            "condition": st.session_state.data is not None,
            "description": "Upload or load keyword performance data"
        },
        {
            "name": "Clustering Completed",
            "condition": st.session_state.clustered_data is not None,
            "description": "Run keyword clustering analysis"
        },
        {
            "name": "Suggestions Generated",
            "condition": (
                'suggestions' in st.session_state 
                and st.session_state.suggestions is not None
            ),
            "description": "Generate AI keyword suggestions"
        },
        {
            "name": "Estimated CTRs Entered",
            "condition": (
                'estimated_ctrs' in st.session_state 
                and st.session_state.estimated_ctrs is not None
            ),
            "description": "Enter estimated CTRs for suggestions"
        },
        {
            "name": "Case Study Created",
            "condition": st.session_state.case_study_data is not None,
            "description": "Generate case study report"
        }
    ]
    
    # Display steps with status indicators
    for step in steps:
        status = "✅" if step["condition"] else "⏳"
        st.sidebar.markdown(
            f"{status} **{step['name']}**  \n"
            f"<span style='color: gray; font-size: 0.8em;'>{step['description']}</span>",
            unsafe_allow_html=True
        )

def show_low_performing_keywords():
    """Display low-performing keywords table with suggestions."""
    if st.session_state.clustered_data is not None:
        st.markdown("### 🔍 Low-Performing Keywords")
        
        # Filter low-performing keywords
        low_perf = st.session_state.clustered_data[
            st.session_state.clustered_data['cluster_label'] != -1
        ].copy()
        
        # Add suggestions column if available
        if (
            'suggestions' in st.session_state 
            and st.session_state.suggestions is not None
            and 'keywords' in st.session_state.suggestions
        ):
            low_perf['top_suggestion'] = low_perf['keyword'].apply(
                lambda x: st.session_state.suggestions['keywords'].get(x, ['—'])[0]
            )
        else:
            low_perf['top_suggestion'] = 'Pending'
        
        # Display table with tooltips
        st.dataframe(
            low_perf[['keyword', 'CTR', 'cluster_label', 'top_suggestion']],
            use_container_width=True,
            column_config={
                'keyword': st.column_config.TextColumn(
                    'Keyword',
                    help='Original low-performing keyword'
                ),
                'CTR': st.column_config.NumberColumn(
                    'CTR (%)',
                    help='Current Click-Through Rate',
                    format='%.2f%%'
                ),
                'cluster_label': st.column_config.NumberColumn(
                    'Cluster',
                    help='Group of similar keywords',
                    format='%d'
                ),
                'top_suggestion': st.column_config.TextColumn(
                    'Top Suggested Keyword',
                    help='AI-generated alternative keyword'
                )
            }
        )

def main():
    """Main dashboard function."""
    st.title("SEO Analysis Tool")
    
    # Show progress tracker in sidebar
    show_progress_tracker()
    
    # File upload section
    st.markdown("### 📂 Upload Data")
    uploaded_file = st.file_uploader(
        "Upload your keyword performance data (CSV)",
        type=['csv'],
        help="Upload a CSV file with columns: keyword, CTR, and other metrics"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.data = df
            st.success("Data loaded successfully!")
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return
    
    # Clustering section
    if st.session_state.data is not None:
        st.markdown("### 🔄 Keyword Clustering")
        st.markdown("""
        🎯 *Adjust the CTR threshold to identify low-performing keywords, then group them into clusters 
        based on similarity to discover common themes and improvement opportunities.*
        """)
        
        col1, col2 = st.columns(2)
        with col1:
            ctr_threshold = st.slider(
                "CTR Threshold (%)",
                min_value=0.0,
                max_value=10.0,
                value=1.0,
                step=0.1,
                help="Keywords with CTR below this threshold will be considered low-performing"
            )
        
        with col2:
            n_clusters = st.slider(
                "Number of Clusters",
                min_value=2,
                max_value=10,
                value=5,
                help="Number of clusters to group similar keywords"
            )
        
        use_transformer = st.checkbox(
            "Use Advanced Embeddings",
            help="Use more sophisticated text embeddings for better clustering (requires more processing time)"
        )
        
        # Style the Run Clustering button
        st.markdown(
            """
            <style>
            div.stButton > button:first-child {
                background-color: #28a745;
                color: white;
                font-weight: bold;
                font-size: 1.1em;
                border: none;
                padding: 0.5rem 1rem;
                border-radius: 0.5rem;
                transition: all 0.3s ease;
            }
            div.stButton > button:first-child:hover {
                background-color: #218838;
                border: none;
                transform: translateY(-1px);
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        
        if st.button("🚀 Run Clustering"):
            try:
                clustered_data = cluster_low_performance_keywords(
                    st.session_state.data,
                    ctr_threshold=ctr_threshold,
                    n_clusters=n_clusters,
                    use_transformer=use_transformer
                )
                st.session_state.clustered_data = clustered_data
                st.success("Clustering completed successfully!")
            except Exception as e:
                st.error(f"Error during clustering: {str(e)}")
                return

if __name__ == "__main__":
    main() 