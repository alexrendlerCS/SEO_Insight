"""
Keyword analysis and clustering utilities.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Dict
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import plotly.express as px
from umap import UMAP

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KeywordClusterer:
    def __init__(self, n_clusters: int = 5, use_transformer: bool = False):
        """
        Initialize keyword clusterer.
        
        Args:
            n_clusters (int): Number of clusters to create
            use_transformer (bool): Whether to use SentenceTransformer for embeddings
        """
        self.n_clusters = n_clusters
        self.use_transformer = use_transformer
        self.vectorizer = None
        self.kmeans = None  # Initialize in cluster_keywords
        self.umap = None    # Initialize in cluster_keywords
        
        # Only import embedding helpers if transformer is requested
        if self.use_transformer:
            try:
                from .embedding_helpers import initialize_transformer, is_transformer_available
                if not initialize_transformer():
                    logger.warning("SentenceTransformer not available. Falling back to TF-IDF.")
                    self.use_transformer = False
            except ImportError:
                logger.warning("Embedding helpers not available. Falling back to TF-IDF.")
                self.use_transformer = False
    
    def _get_embeddings(self, keywords: List[str]) -> np.ndarray:
        """Get embeddings for keywords using either TF-IDF or SentenceTransformer."""
        if self.use_transformer:
            try:
                from .embedding_helpers import get_transformer_embeddings
                embeddings = get_transformer_embeddings(keywords)
                if embeddings is not None:
                    return embeddings
                logger.warning("Failed to get transformer embeddings. Falling back to TF-IDF.")
                self.use_transformer = False
            except Exception as e:
                logger.error(f"Error getting transformer embeddings: {str(e)}")
                self.use_transformer = False
        
        # Fallback to TF-IDF
        if self.vectorizer is None:
            self.vectorizer = TfidfVectorizer(max_features=1000)
        return self.vectorizer.fit_transform(keywords).toarray()
    
    def cluster_keywords(self, keywords: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Cluster keywords using embeddings and KMeans.
        
        Args:
            keywords (List[str]): List of keywords to cluster
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Embeddings and cluster labels
            
        Raises:
            ValueError: If there are fewer than 2 keywords to cluster
        """
        if len(keywords) < 2:
            raise ValueError("At least 2 keywords are required for clustering")
        
        try:
            # Adjust number of clusters if needed
            n_clusters = min(self.n_clusters, len(keywords))
            if n_clusters < self.n_clusters:
                logger.warning(f"Reducing number of clusters from {self.n_clusters} to {n_clusters} due to dataset size")
            
            # Initialize models with adjusted parameters
            self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            self.umap = UMAP(
                n_components=2,
                n_neighbors=min(15, len(keywords) - 1),
                min_dist=0.1,
                random_state=42
            )
            
            # Get embeddings
            embeddings = self._get_embeddings(keywords)
            
            # Reduce dimensionality
            reduced_embeddings = self.umap.fit_transform(embeddings)
            
            # Cluster
            labels = self.kmeans.fit_predict(reduced_embeddings)
            
            return reduced_embeddings, labels
            
        except Exception as e:
            logger.error(f"Error clustering keywords: {str(e)}")
            raise
    
    def visualize_clusters(self, embeddings, labels, keywords, df=None):
        """
        Create interactive visualization of keyword clusters.
        Only shows hover tooltips, no visible labels.
        Also returns a summary DataFrame per cluster (excluding -1).
        
        Args:
            embeddings: 2D array of keyword embeddings
            labels: Cluster labels for each keyword
            keywords: List of keywords
            df: Optional DataFrame with 'CTR' and 'search_volume' columns
        Returns:
            (fig, summary_df): Plotly figure and summary DataFrame
        """
        import pandas as pd
        import plotly.graph_objects as go
        import numpy as np
        
        # Create DataFrame for plotting
        plot_df = pd.DataFrame({
            'x': embeddings[:, 0],
            'y': embeddings[:, 1],
            'cluster': labels,
            'keyword': keywords
        })
        if df is not None:
            if 'CTR' in df.columns:
                plot_df['CTR'] = df['CTR'].values
            if 'search_volume' in df.columns:
                plot_df['search_volume'] = df['search_volume'].values
        
        # Create scatter plot
        fig = go.Figure()
        for cluster in range(max(labels) + 1):
            cluster_data = plot_df[plot_df['cluster'] == cluster].copy()
            fig.add_trace(go.Scatter(
                x=cluster_data['x'],
                y=cluster_data['y'],
                mode='markers',
                name=f'Cluster {cluster}',
                hovertext=cluster_data['keyword'],
                hoverinfo='text',
                marker=dict(size=10)
            ))
        fig.update_layout(
            title='Keyword Clusters',
            xaxis_title='Embedding X',
            yaxis_title='Embedding Y',
            legend_title='Cluster'
        )
        # Build summary DataFrame (exclude cluster -1)
        summary_rows = []
        for cluster in sorted(set(labels)):
            if cluster == -1:
                continue
            cluster_df = plot_df[plot_df['cluster'] == cluster]
            n_keywords = len(cluster_df)
            avg_ctr = float(np.nanmean(cluster_df['CTR'])) * 100 if 'CTR' in cluster_df.columns else None
            avg_ctr = round(avg_ctr, 2) if avg_ctr is not None else None
            avg_sv = int(np.nanmean(cluster_df['search_volume'])) if 'search_volume' in cluster_df.columns else None
            avg_sv_fmt = f"{avg_sv:,}" if avg_sv is not None else None
            summary_rows.append({
                'Cluster ID': cluster,
                'Num Keywords': n_keywords,
                'Avg CTR (%)': avg_ctr,
                'Avg Search Volume': avg_sv_fmt
            })
        summary_df = pd.DataFrame(summary_rows)
        return fig, summary_df


def cluster_low_performance_keywords(
    df: pd.DataFrame,
    ctr_threshold: float = 1.0,
    n_clusters: int = 5,
    use_transformer: bool = False
) -> pd.DataFrame:
    """
    Cluster low-performing keywords based on CTR and text similarity.
    If 'CTR' column is missing or invalid, fill with default value 0.5 and log a warning.
    
    Args:
        df (pd.DataFrame): DataFrame with keyword performance data
        ctr_threshold (float): CTR threshold for low-performing keywords
        n_clusters (int): Number of clusters to create
        use_transformer (bool): Whether to use SentenceTransformer for embeddings
        
    Returns:
        pd.DataFrame: Original DataFrame with added cluster_label column
    """
    try:
        # Handle missing or invalid 'CTR' column
        if 'CTR' not in df.columns:
            logger.warning("'CTR' column missing from input DataFrame. Filling with default value 0.5.")
            df['CTR'] = 0.5
        else:
            ctr_col = df['CTR']
            if ctr_col.isnull().all() or (ctr_col == 0).all():
                logger.warning("'CTR' column is all nulls or all zeroes. Filling with default value 0.5.")
                df['CTR'] = 0.5
            else:
                # If some values are null, fill only those
                if ctr_col.isnull().any():
                    logger.info("'CTR' column contains nulls. Filling nulls with default value 0.5.")
                    df['CTR'] = ctr_col.fillna(0.5)
        
        # Filter low-performing keywords
        low_perf_df = df[df['CTR'] < ctr_threshold].copy()
        
        if len(low_perf_df) < 2:
            logger.warning(f"Not enough low-performing keywords to cluster (found {len(low_perf_df)}). Try lowering the CTR threshold.")
            df['cluster_label'] = -1
            return df
        
        # Initialize clusterer
        clusterer = KeywordClusterer(n_clusters=n_clusters, use_transformer=use_transformer)
        
        try:
            # Get embeddings and cluster labels
            keywords = low_perf_df['keyword'].tolist()
            embeddings, labels = clusterer.cluster_keywords(keywords)
            
            # Add cluster labels to DataFrame
            low_perf_df['cluster_label'] = labels
            
            # Add -1 label for high-performing keywords
            high_perf_df = df[df['CTR'] >= ctr_threshold].copy()
            high_perf_df['cluster_label'] = -1
            
            # Combine results
            result_df = pd.concat([low_perf_df, high_perf_df])
            
            return result_df
            
        except ValueError as ve:
            logger.warning(f"Clustering skipped: {str(ve)}")
            df['cluster_label'] = -1
            return df
            
    except Exception as e:
        logger.error(f"Error clustering low-performance keywords: {str(e)}")
        raise

def prepare_case_study_data(
    df: pd.DataFrame,
    suggestions: Dict[str, List[str]],
    estimated_ctrs: Dict[str, float]
) -> pd.DataFrame:
    """
    Prepare case study data comparing original and suggested keywords.
    
    Args:
        df (pd.DataFrame): Original keyword data
        suggestions (Dict[str, List[str]]): Suggested keywords for each original keyword
        estimated_ctrs (Dict[str, float]): Estimated CTRs for suggested keywords
        
    Returns:
        pd.DataFrame: Case study data with comparisons
    """
    try:
        case_study_data = []
        
        for keyword, alternatives in suggestions.items():
            original_row = df[df['keyword'] == keyword].iloc[0]
            original_ctr = original_row['CTR']
            estimated_ctr = estimated_ctrs.get(keyword, original_ctr)
            
            case_study_data.append({
                'keyword': keyword,
                'original_ctr': original_ctr,
                'suggested_keywords': ' | '.join(alternatives),
                'estimated_ctr': estimated_ctr,
                'ctr_uplift': estimated_ctr - original_ctr
            })
        
        return pd.DataFrame(case_study_data)
        
    except Exception as e:
        logger.error(f"Error preparing case study data: {str(e)}")
        raise

def export_case_study_report(case_study_df: pd.DataFrame, suggestions: Dict = None) -> str:
    """
    Generate a comprehensive case study report in HTML format.
    
    Args:
        case_study_df: DataFrame containing the case study data
        suggestions: Dictionary containing keyword suggestions and meta descriptions
    
    Returns:
        str: HTML content of the report
    """
    # Calculate summary metrics
    total_keywords = len(case_study_df)
    avg_ctr_uplift = case_study_df['ctr_uplift'].mean()
    max_uplift = case_study_df['ctr_uplift'].max()
    min_uplift = case_study_df['ctr_uplift'].min()
    
    # Generate HTML report
    html_content = f"""
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1 {{ color: #2c3e50; }}
            h2 {{ color: #34495e; margin-top: 30px; }}
            .summary {{ background-color: #f8f9fa; padding: 20px; border-radius: 5px; }}
            .metric {{ margin: 10px 0; }}
            .keyword-section {{ margin: 20px 0; padding: 15px; border: 1px solid #dee2e6; border-radius: 5px; }}
            .suggestion {{ color: #28a745; }}
            .meta-desc {{ color: #6c757d; font-style: italic; }}
            table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
            th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #dee2e6; }}
            th {{ background-color: #f8f9fa; }}
            .highlight {{ background-color: #e8f4f8; }}
        </style>
    </head>
    <body>
        <h1>SEO Optimization Case Study Report</h1>
        <div class="summary">
            <h2>Executive Summary</h2>
            <div class="metric">Total Keywords Analyzed: {total_keywords}</div>
            <div class="metric">Average CTR Uplift: {avg_ctr_uplift:.2f}%</div>
            <div class="metric">Maximum CTR Uplift: {max_uplift:.2f}%</div>
            <div class="metric">Minimum CTR Uplift: {min_uplift:.2f}%</div>
        </div>
        
        <h2>Detailed Analysis</h2>
        <table>
            <tr>
                <th>Keyword</th>
                <th>Original CTR</th>
                <th>Estimated CTR</th>
                <th>CTR Uplift</th>
            </tr>
    """
    
    # Add keyword rows
    for _, row in case_study_df.iterrows():
        html_content += f"""
            <tr>
                <td>{row['keyword']}</td>
                <td>{row['original_ctr']:.2f}%</td>
                <td>{row['estimated_ctr']:.2f}%</td>
                <td class="highlight">{row['ctr_uplift']:.2f}%</td>
            </tr>
        """
    
    html_content += """
        </table>
        
        <h2>Keyword Recommendations</h2>
    """
    
    # Add detailed recommendations if suggestions are provided
    if suggestions:
        for keyword, alts in suggestions['keywords'].items():
            html_content += f"""
            <div class="keyword-section">
                <h3>{keyword}</h3>
                <div class="suggestion">
                    <strong>Suggested Alternatives:</strong><br>
                    {', '.join(alts)}
                </div>
                <div class="meta-desc">
                    <strong>Meta Description:</strong><br>
                    {suggestions['descriptions'][keyword]}
                </div>
            </div>
            """
    
    html_content += """
        <h2>Implementation Recommendations</h2>
        <ol>
            <li>Prioritize keywords with the highest CTR uplift potential</li>
            <li>Implement suggested meta descriptions for improved click-through rates</li>
            <li>Monitor performance after implementing changes</li>
            <li>Regularly review and update keyword strategy based on results</li>
        </ol>
        
        <h2>Next Steps</h2>
        <ul>
            <li>Review and approve suggested keyword changes</li>
            <li>Update meta descriptions for selected keywords</li>
            <li>Set up tracking for new keyword performance</li>
            <li>Schedule follow-up analysis in 30 days</li>
        </ul>
    </body>
    </html>
    """
    
    return html_content 