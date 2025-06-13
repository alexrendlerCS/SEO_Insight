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

# Lazy loading for SentenceTransformer
TRANSFORMER_AVAILABLE = False
try:
    from sentence_transformers import SentenceTransformer
    TRANSFORMER_AVAILABLE = True
except ImportError:
    logger.warning("SentenceTransformer not available. Will use TF-IDF for embeddings.")

class KeywordClusterer:
    def __init__(self, n_clusters: int = 5, use_transformer: bool = False):
        """
        Initialize keyword clusterer.
        
        Args:
            n_clusters (int): Number of clusters to create
            use_transformer (bool): Whether to use SentenceTransformer for embeddings
        """
        self.n_clusters = n_clusters
        self.use_transformer = use_transformer and TRANSFORMER_AVAILABLE
        self.vectorizer = None
        self.transformer = None
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.umap = UMAP(n_components=2, random_state=42)
        
        if self.use_transformer and not TRANSFORMER_AVAILABLE:
            logger.warning("SentenceTransformer requested but not available. Falling back to TF-IDF.")
            self.use_transformer = False
    
    def _get_embeddings(self, keywords: List[str]) -> np.ndarray:
        """Get embeddings for keywords using either TF-IDF or SentenceTransformer."""
        if self.use_transformer:
            if self.transformer is None:
                self.transformer = SentenceTransformer('all-MiniLM-L6-v2')
            return self.transformer.encode(keywords)
        else:
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
        """
        try:
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
    
    def visualize_clusters(self, embeddings: np.ndarray, labels: np.ndarray, keywords: List[str]) -> go.Figure:
        """
        Create interactive visualization of keyword clusters.
        
        Args:
            embeddings (np.ndarray): 2D embeddings from UMAP
            labels (np.ndarray): Cluster labels
            keywords (List[str]): Original keywords
            
        Returns:
            go.Figure: Interactive Plotly figure
        """
        try:
            # Create DataFrame for plotting
            plot_df = pd.DataFrame({
                'x': embeddings[:, 0],
                'y': embeddings[:, 1],
                'cluster': labels,
                'keyword': keywords
            })
            
            # Create scatter plot
            fig = go.Figure()
            
            # Add traces for each cluster
            for cluster in range(self.n_clusters):
                cluster_data = plot_df[plot_df['cluster'] == cluster]
                fig.add_trace(go.Scatter(
                    x=cluster_data['x'],
                    y=cluster_data['y'],
                    mode='markers+text',
                    name=f'Cluster {cluster}',
                    text=cluster_data['keyword'],
                    textposition="top center",
                    marker=dict(size=10)
                ))
            
            # Update layout
            fig.update_layout(
                title='Keyword Clusters',
                xaxis_title='UMAP 1',
                yaxis_title='UMAP 2',
                showlegend=True,
                hovermode='closest'
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error visualizing clusters: {str(e)}")
            raise

def cluster_low_performance_keywords(
    df: pd.DataFrame,
    ctr_threshold: float = 1.0,
    n_clusters: int = 5,
    use_transformer: bool = False
) -> pd.DataFrame:
    """
    Cluster low-performing keywords based on CTR threshold.
    
    Args:
        df (pd.DataFrame): DataFrame with keyword performance data
        ctr_threshold (float): CTR threshold for low-performing keywords
        n_clusters (int): Number of clusters to create
        use_transformer (bool): Whether to use SentenceTransformer for embeddings
        
    Returns:
        pd.DataFrame: Original DataFrame with added cluster_label column
    """
    try:
        # Filter low-performing keywords
        low_perf_df = df[df['CTR'] < ctr_threshold].copy()
        
        if len(low_perf_df) == 0:
            logger.warning(f"No keywords found below {ctr_threshold}% CTR")
            df['cluster_label'] = -1
            return df
        
        # Initialize clusterer
        clusterer = KeywordClusterer(n_clusters=n_clusters, use_transformer=use_transformer)
        
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

def export_case_study_report(df: pd.DataFrame, output_path: str = 'case_study_results.csv') -> None:
    """
    Export case study results to CSV.
    
    Args:
        df (pd.DataFrame): Case study data
        output_path (str): Path to save the CSV file
    """
    try:
        df.to_csv(output_path, index=False)
        logger.info(f"Case study results exported to {output_path}")
    except Exception as e:
        logger.error(f"Error exporting case study report: {str(e)}")
        raise 