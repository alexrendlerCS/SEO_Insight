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
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

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

def load_data():
    """Load and process keyword data from file or mock data."""
    st.markdown("### 📂 Upload Data")
    st.markdown("""
    Start by either uploading your own keyword performance data or using our sample data to explore the tool.
    The data should be in CSV format with columns for keyword, CTR, and other performance metrics.
    """)
    
    # Option to use mock data
    use_mock = st.checkbox(
        "Use Sample Data",
        value=False,
        help="Use our sample dataset to explore the tool's features"
    )
    
    if use_mock:
        try:
            mock_data_path = os.path.join('data', 'mock_keyword_data.csv')
            if not os.path.exists(mock_data_path):
                mock_data_path = 'mock_keyword_data.csv'
            df = pd.read_csv(mock_data_path)
            st.success("✅ Sample data loaded successfully!")
            st.session_state.data = df
            return df
        except Exception as e:
            logger.error(f"Error loading mock data: {str(e)}")
            st.error("❌ Error loading sample data. Please check the file exists.")
            return None
    
    # File upload option
    uploaded_file = st.file_uploader(
        "Upload your keyword performance data (CSV)",
        type=['csv'],
        help="Upload a CSV file with columns: keyword, CTR, and other metrics"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("✅ File uploaded successfully!")
            st.session_state.data = df
            return df
        except Exception as e:
            logger.error(f"Error reading file: {str(e)}")
            st.error("❌ Error reading file. Please check the format.")
            return None
    
    return None

def display_metrics(df: pd.DataFrame):
    """Display key performance metrics and data table."""
    st.header("Keyword Performance Metrics")
    
    # Convert numeric columns, handling errors gracefully
    numeric_columns = ['search_volume', 'competition_index', 'impressions', 'clicks', 'CTR', 'cost', 'avg_position', 'avg_monthly_searches']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Define metrics configuration
    metrics = {
        'search_volume': {
            'fallback': 0,
            'label': 'Total Search Volume',
            'format': ',.0f',
            'aggregation': 'sum'
        },
        'competition_index': {
            'fallback': 0.0,
            'label': 'Average Competition',
            'format': '.2f',
            'aggregation': 'mean'
        },
        # Optional metrics (no warnings if missing)
        'impressions': {
            'fallback': 0,
            'label': 'Total Impressions',
            'format': ',.0f',
            'aggregation': 'sum',
            'optional': True
        },
        'clicks': {
            'fallback': 0,
            'label': 'Total Clicks',
            'format': ',.0f',
            'aggregation': 'sum',
            'optional': True
        },
        'CTR': {
            'fallback': 0.0,
            'label': 'Average CTR',
            'format': '.2%',
            'aggregation': 'mean',
            'optional': True
        },
        'cost': {
            'fallback': 0.0,
            'label': 'Total Cost',
            'format': '$,.2f',
            'aggregation': 'sum',
            'optional': True
        },
        'avg_position': {
            'fallback': 0.0,
            'label': 'Average Position',
            'format': '.1f',
            'aggregation': 'mean',
            'optional': True
        },
        'avg_monthly_searches': {
            'fallback': 0,
            'label': 'Total Monthly Searches',
            'format': ',.0f',
            'aggregation': 'sum',
            'optional': True
        }
    }
    
    # Calculate metrics with safe fallbacks
    calculated_metrics = {}
    for metric, config in metrics.items():
        if metric in df.columns and not df[metric].isna().all():
            if config['aggregation'] == 'mean':
                calculated_metrics[metric] = df[metric].mean()
            else:
                calculated_metrics[metric] = df[metric].sum()
        elif not config.get('optional', False):
            st.warning(f"⚠️ {config['label']} data is not available for this dataset.")
            calculated_metrics[metric] = config['fallback']
    
    # Display metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    # Display total keywords (always available)
    with col1:
        st.metric("Total Keywords", len(df))
    
    # Display other metrics if available
    metric_index = 1
    for metric, config in metrics.items():
        if metric in calculated_metrics and not config.get('optional', False):
            with [col2, col3, col4][metric_index % 3]:
                value = calculated_metrics[metric]
                if pd.notna(value):  # Only format if value is not NaN
                    formatted = f"{value:{config['format']}}"
                    st.metric(
                        config['label'],
                        formatted
                    )
                else:
                    st.metric(config['label'], "N/A")
            metric_index += 1
    
    # Display data table with safe formatting
    st.subheader("Data Preview")
    
    # Define column configurations with safe formatting
    column_config = {}
    
    # Add keyword column if present
    if 'keyword' in df.columns:
        column_config['keyword'] = st.column_config.TextColumn("Keyword")
    
    # Add competition column if present
    if 'competition' in df.columns:
        column_config['competition'] = st.column_config.TextColumn("Competition Level")
    
    # Add numeric columns with formatting
    for metric, config in metrics.items():
        if metric in df.columns and pd.api.types.is_numeric_dtype(df[metric]):
            column_config[metric] = st.column_config.NumberColumn(
                label=config['label'],
                help=f"{config['label']} for this keyword"
            )

    # Display the dataframe with configured columns
    if column_config:
        st.dataframe(
            df,
            column_config=column_config,
            use_container_width=True
        )
    else:
        st.warning("No valid columns found in the dataset.")

def perform_clustering(df: pd.DataFrame):
    """Perform keyword clustering with adjustable parameters."""
    st.header("Keyword Clustering")
    st.markdown("""
    ### Understanding Keyword Clustering
    
    This section helps you identify and group similar keywords that aren't performing well. Here's what each setting does:
    
    - **CTR Threshold**: This is the minimum Click-Through Rate (CTR) you expect from your keywords. 
      - Setting it lower (e.g., 0.5%) will identify more keywords as underperforming
      - Setting it higher (e.g., 2%) will be more selective, focusing only on the worst performers
    
    - **Number of Clusters**: This determines how many groups your keywords will be divided into.
      - Fewer clusters (2-3): Broader groups, good for identifying major themes
      - More clusters (6-10): More specific groups, better for detailed analysis
    
    - **Advanced Embeddings**: This option uses more sophisticated text analysis.
      - Off: Faster processing, good for basic keyword similarity
      - On: More accurate grouping, better for complex keyword relationships
    """)
    
    # Clustering parameters
    col1, col2 = st.columns(2)
    with col1:
        ctr_threshold = st.slider(
            "CTR Threshold (%)",
            min_value=0.0,
            max_value=5.0,
            value=1.0,
            step=0.1,
            help="Keywords with CTR below this threshold will be considered low-performing. Lower values will identify more keywords as underperforming."
        )
    with col2:
        n_clusters = st.slider(
            "Number of Clusters",
            min_value=2,
            max_value=10,
            value=5,
            help="Number of groups to divide keywords into. More clusters = more specific groupings."
        )
    
    # Option to use transformer
    use_transformer = st.checkbox(
        "Use Advanced Embeddings",
        help="Enable more sophisticated text analysis for better keyword grouping (takes longer to process)"
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
                
                # Secondary warning if no real CTR data exists or all values are identical
                ctr_col = None
                for col in ['CTR', 'ctr', 'ctr_pct']:
                    if col in clustered_df.columns:
                        ctr_col = col
                        break
                show_ctr_warning = False
                if ctr_col is None:
                    show_ctr_warning = True
                else:
                    ctr_vals = clustered_df[ctr_col]
                    if ctr_vals.isnull().all() or ctr_vals.nunique() == 1:
                        show_ctr_warning = True
                if show_ctr_warning:
                    st.warning("⚠️ Note: No CTR data was found. You can identify high-potential keywords, but cannot evaluate your own keyword performance without live ad data.")
                
                # Display cluster visualization
                try:
                    clusterer = KeywordClusterer(n_clusters=n_clusters, use_transformer=use_transformer)
                    keywords = clustered_df['keyword'].tolist()
                    embeddings, labels = clusterer.cluster_keywords(keywords)
                    
                    fig, summary_df = clusterer.visualize_clusters(embeddings, labels, keywords, clustered_df)
                    st.plotly_chart(fig, use_container_width=True)
                    st.subheader("🔢 Cluster Summary")
                    st.dataframe(summary_df, use_container_width=True)
                    
                    # Add descriptive section
                    st.subheader("🔍 Understanding Your Clusters")
                    st.markdown("""
                    Each cluster represents a group of similar keywords. The visualization above shows how your keywords are related:
                    
                    - **Points**: Each point represents a keyword
                    - **Colors**: Different colors indicate different clusters
                    - **Distance**: Keywords closer together are more similar
                    
                    Use this visualization to:
                    1. Identify common themes in underperforming keywords
                    2. Find opportunities to consolidate similar keywords
                    3. Discover patterns in your keyword strategy
                    """)
                    
                    # Show low-performing keywords
                    show_low_performing_keywords()
                except Exception as e:
                    logger.error(f"Error visualizing clusters: {str(e)}")
                    st.error("❌ Error visualizing clusters. Please try again.")
            except Exception as e:
                logger.error(f"Error during clustering: {str(e)}")
                st.error("❌ Error during clustering. Please try again.")
                return

def generate_suggestions():
    """Generate and display LLM suggestions for low-performing keywords."""
    st.header("AI-Generated Suggestions")
    st.markdown("""
    ### Understanding AI Suggestions
    
    This section uses artificial intelligence to analyze your underperforming keywords and suggest improvements:
    
    1. **Alternative Keywords**: The AI suggests similar keywords that might perform better
    2. **Meta Descriptions**: AI-generated descriptions optimized for search engines
    3. **CTR Estimates**: You can adjust these to simulate potential improvements
    
    The suggestions are based on:
    - Similarity to your existing keywords
    - Common search patterns
    - Current performance trends
    """)
    
    if st.session_state.clustered_data is None:
        st.warning("⚠️ Please run clustering first to identify low-performing keywords.")
        return
    
    if st.button("Generate Suggestions"):
        with st.spinner("🔄 Generating suggestions..."):
            try:
                logger.info("Initializing LLMGenerator()...")
                llm = LLMGenerator()
                logger.info("LLMGenerator initialized.")

                # Get low-performing keywords
                low_perf_keywords = st.session_state.clustered_data[
                    st.session_state.clustered_data['cluster_label'] != -1
                ]['keyword'].tolist()
                logger.info(f"Generating suggestions for {len(low_perf_keywords)} keywords")
                if len(low_perf_keywords) > 10:
                    logger.warning("Too many keywords. Limiting to 10 for debugging.")
                    low_perf_keywords = low_perf_keywords[:10]

                # Generate suggestions
                try:
                    logger.info("Calling llm.generate_keyword_suggestions()...")
                    start_time = time.time()
                    suggestions = llm.generate_keyword_suggestions(low_perf_keywords)
                    logger.info(f"llm.generate_keyword_suggestions() completed in {time.time() - start_time:.2f}s.")
                except Exception as e:
                    logger.error(f"Error in generate_keyword_suggestions: {e}")
                    raise

                # Parallel meta description generation
                meta_descriptions = {}
                logger.info("Starting parallel meta description generation...")
                start_time = time.time()
                with ThreadPoolExecutor(max_workers=5) as executor:
                    future_to_keyword = {
                        executor.submit(llm.generate_meta_description, keyword): keyword
                        for keyword in low_perf_keywords
                    }
                    for future in as_completed(future_to_keyword):
                        keyword = future_to_keyword[future]
                        try:
                            desc = future.result()
                            meta_descriptions[keyword] = desc
                            logger.info(f"Meta description for '{keyword}' completed.")
                        except Exception as e:
                            logger.error(f"Error in generate_meta_description for '{keyword}': {e}")
                            meta_descriptions[keyword] = "Error generating meta description."
                logger.info(f"All meta descriptions generated in {time.time() - start_time:.2f}s.")

                # Store suggestions
                st.session_state.suggestions = {
                    'keywords': suggestions,
                    'descriptions': meta_descriptions
                }

                # Initialize estimated_ctrs dictionary
                st.session_state.estimated_ctrs = {}

                st.success("✅ Suggestions generated successfully!")
            except Exception as e:
                logger.error(f"Error generating suggestions: {str(e)}")
                st.error("❌ Error generating suggestions. Please try again.")
                return
    
    if st.session_state.suggestions:
        # Display suggestions and collect estimated CTRs
        st.subheader("Review and Adjust Suggestions")
        st.markdown("""
        For each keyword, you can:
        - Review alternative keyword suggestions
        - Read the AI-generated meta description
        - Adjust the estimated CTR to see potential improvements
        """)
        
        # Initialize estimated_ctrs if not already done
        if 'estimated_ctrs' not in st.session_state:
            st.session_state.estimated_ctrs = {}
        
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
                
                # Get current CTR from clustered data
                current_ctr = float(st.session_state.clustered_data[
                    st.session_state.clustered_data['keyword'] == keyword
                ]['CTR'].iloc[0])
                
                # Initialize estimated CTR with current CTR if not set
                if keyword not in st.session_state.estimated_ctrs:
                    st.session_state.estimated_ctrs[keyword] = current_ctr
                
                # Add CTR estimate input
                estimated_ctr = st.number_input(
                    f"Estimated New CTR for {keyword} (%)",
                    min_value=0.0,
                    max_value=100.0,
                    value=st.session_state.estimated_ctrs[keyword],
                    step=0.1,
                    key=f"ctr_{keyword}"
                )
                st.session_state.estimated_ctrs[keyword] = estimated_ctr

def prepare_case_study():
    """Prepare and display case study data."""
    st.header("Case Study Analysis")
    st.markdown("""
    ### Understanding the Case Study
    
    This section helps you quantify the potential impact of implementing the suggested changes:
    
    1. **Original vs Estimated CTR**: Compare current performance with projected improvements
    2. **CTR Uplift**: See the percentage improvement for each keyword
    3. **Visual Comparison**: Bar chart showing before and after scenarios
    
    Use this analysis to:
    - Prioritize which keywords to update first
    - Estimate the overall impact on your campaign
    - Make data-driven decisions about keyword changes
    """)
    
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
                
                # Export options
                st.subheader("Export Options")
                col1, col2 = st.columns(2)
                
                with col1:
                    # Export CSV
                    csv = case_study_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download CSV Report",
                        data=csv,
                        file_name="seo_case_study.csv",
                        mime="text/csv",
                        help="Download the case study data in CSV format"
                    )
                
                with col2:
                    # Export HTML Report
                    html_report = export_case_study_report(
                        case_study_df,
                        st.session_state.suggestions
                    )
                    st.download_button(
                        label="📥 Download HTML Report",
                        data=html_report,
                        file_name="seo_case_study_report.html",
                        mime="text/html",
                        help="Download a comprehensive HTML report with analysis and recommendations"
                    )
                
                st.success("✅ Case study report generated successfully!")
            except Exception as e:
                logger.error(f"Error preparing case study: {str(e)}")
                st.error("❌ Error preparing case study. Please try again.")

def show_low_performing_keywords():
    """Display low-performing keywords table with suggestions."""
    df = st.session_state.clustered_data if 'clustered_data' in st.session_state else None
    if df is not None:
        # --- CTR warning above section ---
        ctr_col = None
        for col in ['CTR', 'ctr', 'ctr_pct']:
            if col in df.columns:
                ctr_col = col
                break
        show_ctr_warning = False
        if ctr_col is None:
            show_ctr_warning = True
        else:
            ctr_vals = df[ctr_col]
            if ctr_vals.isnull().all() or (ctr_vals.nunique() == 1 and (ctr_vals == 0.5).all()):
                show_ctr_warning = True
        if show_ctr_warning:
            st.warning("⚠️ Warning: No CTR data detected. The keywords shown below may not be truly 'low-performing' — simulated or default values are used instead.")

        st.markdown("### 🔍 Low-Performing Keywords")
        
        # Filter low-performing keywords
        low_perf = df[df['cluster_label'] != -1].copy()
        
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
    st.title("SEO Analysis Dashboard")
    
    # Show progress tracker
    show_progress_tracker()
    
    # Load data
    df = load_data()
    
    if df is not None:
        # Display data preview
        with st.expander("Data Preview"):
            st.dataframe(df.head())
        
        # Display metrics
        display_metrics(df)
        
        # Perform clustering
        perform_clustering(df)
        
        # Generate suggestions
        generate_suggestions()
        
        # Prepare case study
        prepare_case_study()

if __name__ == "__main__":
    main() 