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
    
    def visualize_clusters(self, embeddings, labels, keywords, df):
        """
        Visualize keyword clusters using UMAP and Plotly.
        """
        import plotly.express as px
        import pandas as pd
        import numpy as np

        # Create DataFrame for plotting
        plot_df = pd.DataFrame({
            'x': embeddings[:, 0],
            'y': embeddings[:, 1],
            'cluster': labels,
            'keyword': keywords
        })

        # Add CTR if available
        if 'CTR' in df.columns:
            plot_df['CTR'] = df['CTR'].values
        elif 'original_ctr' in df.columns:
            plot_df['CTR'] = df['original_ctr'].values

        # Create scatter plot
        fig = px.scatter(
            plot_df,
            x='x',
            y='y',
            color='cluster',
            hover_data=['keyword', 'CTR'],
            title='Keyword Clusters Visualization'
        )

        # Update layout
        fig.update_layout(
            xaxis_title='UMAP Dimension 1',
            yaxis_title='UMAP Dimension 2',
            showlegend=True
        )

        # Create summary DataFrame
        summary_df = plot_df.groupby('cluster').agg({
            'keyword': 'count',
            'CTR': 'mean'
        }).rename(columns={'keyword': 'count'})

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

def prepare_case_study_data(clustered_df, suggestions, estimated_ctrs):
    """
    Prepare case study data with CTR uplift calculations and opportunity score.
    """
    import numpy as np

    # Create a copy of the clustered data
    case_study_df = clustered_df.copy()

    # Ensure 'original_ctr' is correctly calculated from 'CTR'
    if 'CTR' in case_study_df.columns:
        case_study_df['original_ctr'] = case_study_df['CTR']
    else:
        case_study_df['original_ctr'] = np.nan

    # Add estimated CTRs with smart fallback
    def get_estimated_ctr(row):
        keyword = row['keyword']
        original_ctr = row['original_ctr']
        
        # If keyword has a manual estimate, use it
        if keyword in estimated_ctrs and estimated_ctrs[keyword] is not None:
            return estimated_ctrs[keyword]
        
        # Otherwise, compute smart estimate
        if pd.notna(original_ctr):
            if original_ctr < 0.5:
                smart_est = original_ctr + 0.3
            elif original_ctr < 1.0:
                smart_est = original_ctr + 0.2
            elif original_ctr < 2.0:
                smart_est = original_ctr + 0.1
            elif original_ctr >= 3.0:
                smart_est = original_ctr
            else:
                smart_est = original_ctr + 0.05  # fallback for 2.0 <= CTR < 3.0

            return smart_est
        else:
            return 0.5  # Default estimate

    case_study_df['estimated_ctr'] = case_study_df.apply(get_estimated_ctr, axis=1)

    # Calculate CTR uplift
    case_study_df['ctr_uplift'] = case_study_df['estimated_ctr'] - case_study_df['original_ctr']

    # Calculate opportunity score
    case_study_df['opportunity_score'] = case_study_df['ctr_uplift'] * np.log1p(case_study_df['search_volume']) * np.log1p(case_study_df['impressions'])

    return case_study_df

def export_case_study_report(case_study_df, suggestions):
    """
    Generate an HTML report summarizing the case study results with modern styling.
    """
    from io import StringIO
    import numpy as np
    from datetime import datetime

    html = StringIO()
    html.write("""
    <html>
    <head>
    <title>SEO Case Study Report</title>
    <style>
        body {
            background: #f7f9fa;
            font-family: 'Segoe UI', Arial, sans-serif;
            color: #222;
            margin: 0;
            padding: 0;
        }

        .container {
            max-width: 100%;
            margin: 30px auto;
            background: #fff;
            border-radius: 12px;
            box-shadow: 0 2px 12px rgba(0,0,0,0.07);
            padding: 32px 24px;
            overflow-x: hidden;
        }

        h1 {
            font-size: 2.5em;
            margin-bottom: 0.2em;
            color: #0072C6;
            background: linear-gradient(90deg, #e3f0fc 60%, #fff 100%);
            padding: 18px 0 18px 18px;
            border-bottom: 3px solid #0072C6;
            border-radius: 12px 12px 0 0;
            text-align: center;
        }

        .problem-statement {
            font-size: 1.05em;
            line-height: 1.6em;
            background-color: #f9f9f9;
            padding: 15px;
            border-radius: 8px;
            border-left: 5px solid #0072C6;
            margin-bottom: 20px;
            max-width: 800px;
            margin-left: auto;
            margin-right: auto;
        }

        /* Responsive grid for key insights sections */
        .grid-section-container {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 24px;
            margin-top: 20px;
            margin-bottom: 40px;
        }

        .grid-section {
            background-color: #f3f6fa;
            border-radius: 8px;
            padding: 24px 20px;
            box-shadow: 0 1px 4px rgba(0,0,0,0.03);
        }

        .grid-section h2 {
            font-size: 1.4em;
            color: #0072C6;
            margin-bottom: 12px;
            border-left: 6px solid #0072C6;
            padding-left: 12px;
            background: #eaf3fa;
            border-radius: 6px;
        }

        .section {
            background: #f3f6fa;
            border-radius: 8px;
            margin: 0 auto 32px auto;
            padding: 24px 20px;
            box-shadow: 0 1px 4px rgba(0,0,0,0.03);
            overflow-x: auto;
            max-width: 95%;
        }

        .section table {
            min-width: 900px;
            margin: 0 auto;
        }

        ul {
            margin-top: 0.5em;
            margin-bottom: 0.5em;
        }

        .summary-list li {
            margin-bottom: 0.4em;
        }

        table {
            width: 100%;
            border-collapse: collapse;
            margin: 18px 0 10px 0;
            font-size: 1.05em;
            overflow-x: auto;
            display: block;
        }

        th, td {
            padding: 10px 12px;
            border: 1px solid #e0e6ed;
            text-align: left;
            white-space: nowrap;
        }

        th {
            background: #eaf3fa;
            color: #0072C6;
            font-weight: 600;
        }

        tr:nth-child(even) {
            background: #f7fafd;
        }

        tr:nth-child(odd) {
            background: #fff;
        }

        tr:hover {
            background: #f1f7ff;
        }

        .ctr-uplift-pos {
            color: #28a745;
            font-weight: bold;
        }

        .ctr-uplift-neg {
            color: #d9534f;
            font-weight: bold;
        }

        .bold-keyword {
            font-weight: bold;
            background: #fffbe6;
        }

        .footer {
            margin-top: 40px;
            padding: 18px 0 0 0;
            text-align: center;
            color: #888;
            font-size: 1em;
        }

        .toggle-button {
            background-color: #0072C6;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
            cursor: pointer;
            margin-bottom: 10px;
        }

        .toggle-button:hover {
            background-color: #005fa3;
        }

        .keyword-toggle-container {
            display: flex;
            justify-content: center;
            margin-top: 30px;
            margin-bottom: 10px;
        }

        .full-table {
            display: none;
        }

        .tag-opportunity {
            background-color: #f0f4f8;
            border-radius: 16px;
            padding: 4px 10px;
            font-size: 0.9em;
            display: inline-block;
            font-weight: 500;
            color: #444;
        }

        .tag-opportunity.neutral {
            background-color: #e0e0e0;
            color: #666;
        }
    </style>
    <script>
        function toggleTable() {
            var fullTable = document.getElementById('full-keyword-table');
            var button = document.getElementById('toggle-button');
            if (fullTable.style.display === 'none') {
                fullTable.style.display = 'table';
                button.textContent = 'Hide All Keywords';
            } else {
                fullTable.style.display = 'none';
                button.textContent = 'Show All Keywords';
            }
        }
    </script>
    </head>
    <body>
    <div class="container">
    <h1>📊 SEO Case Study Report</h1>
    """)

    # Problem Statement block
    html.write("""
    <div class='problem-statement'>
    <strong>What This Report Shows:</strong> This SEO analysis evaluates the effectiveness of ad campaign keywords, identifies underperforming terms, and suggests optimized replacements to increase traffic and conversions. The report uses CTR (click-through rate), impressions, and search volume to estimate potential gains, and prioritizes keywords that could benefit from strategic improvement.
    </div>
    """)

    # Start grid container
    html.write("<div class='grid-section-container'>")

    # Executive Summary block
    html.write("<div class='grid-section'>")
    try:
        total_keywords = len(case_study_df)
        valid_ctr_rows = case_study_df.dropna(subset=["original_ctr", "estimated_ctr"])
        avg_original_ctr = valid_ctr_rows["original_ctr"].mean()
        avg_estimated_ctr = valid_ctr_rows["estimated_ctr"].mean()
        avg_uplift = valid_ctr_rows["ctr_uplift"].mean()
        has_impressions = "impressions" in case_study_df.columns
        total_impressions = valid_ctr_rows["impressions"].sum() if has_impressions else None
        # Estimated Click Gain
        if has_impressions:
            click_gain = ((valid_ctr_rows["estimated_ctr"] - valid_ctr_rows["original_ctr"]) * valid_ctr_rows["impressions"] / 100).sum()
        else:
            click_gain = None

        html.write("<h2>📌 Executive Summary</h2>")
        html.write("<ul class='summary-list'>")
        html.write(f"<li><strong>Total Keywords Analyzed:</strong> {total_keywords}</li>")
        html.write(f"<li><strong>Keywords with CTR data:</strong> {len(valid_ctr_rows)}</li>")
        html.write(f"<li><strong>Average Original CTR:</strong> <span style='color:#0072C6;'>{avg_original_ctr:.2f}%</span></li>")
        html.write(f"<li><strong>Average Estimated CTR:</strong> <span style='color:#0072C6;'>{avg_estimated_ctr:.2f}%</span></li>")
        delta_color = '#28a745' if avg_uplift > 0 else '#d9534f'
        html.write(f"<li><strong>Average CTR Uplift:</strong> <span style='color:{delta_color};'>{avg_uplift:+.2f}%</span></li>")
        if total_impressions is not None:
            html.write(f"<li><strong>Total Impressions:</strong> {int(total_impressions):,}</li>")
        if click_gain is not None:
            html.write(f"<li><strong>Estimated Click Gain:</strong> <span style='color:#0072C6;'>{int(click_gain):,}</span></li>")
        html.write("</ul>")
    except Exception as e:
        html.write("<p><em>Executive summary unavailable due to data issues.</em></p>")
    html.write("</div>")

    # Pre-calculate global top keywords by opportunity score
    global_top_keywords = set(case_study_df.sort_values('opportunity_score', ascending=False)['keyword'].head(3))

    # 📌 Recommendations section
    html.write("<div class='grid-section'>")
    html.write("<h2>📌 Recommendations</h2>")
    try:
        ctr_valid = False
        if {'original_ctr', 'estimated_ctr', 'ctr_uplift'}.issubset(case_study_df.columns):
            ctr_vals = case_study_df['original_ctr']
            if ctr_vals.notna().any() and ctr_vals.nunique() > 1:
                ctr_valid = True
        if ctr_valid:
            # Top 3 keywords by opportunity score
            top3 = case_study_df.sort_values('opportunity_score', ascending=False).head(3)
            top3_keywords = top3['keyword'].tolist()
            html.write("<ul>")
            html.write(f"<li>Focus on improving campaigns using: <strong>{', '.join(top3_keywords)}</strong> — these show the strongest projected uplift.</li>")
            # Top suggestions for each
            if suggestions and 'keywords' in suggestions:
                for kw in top3_keywords:
                    alts = suggestions['keywords'].get(kw, [])
                    if alts:
                        html.write(f"<li>Test new ad copy using suggestions for <strong>{kw}</strong>: <em>{', '.join(alts)}</em></li>")
            html.write("<li>Consider building ad groups or landing pages around these high-impact keywords.</li>")
            html.write("</ul>")
        else:
            # CTR missing or simulated
            html.write("<ul>")
            html.write("<li><strong>Note:</strong> CTR data appears to be simulated or unavailable. For deeper performance insights, consider integrating live ad performance data.</li>")
            # Use search_volume if available and numeric
            if 'search_volume' in case_study_df.columns and np.issubdtype(case_study_df['search_volume'].dtype, np.number):
                top5 = case_study_df.sort_values('search_volume', ascending=False).head(5)
                top5_keywords = top5['keyword'].tolist()
                if top5_keywords:
                    html.write(f"<li>These keywords show the highest search volume and could be strong candidates to optimize: <strong>{', '.join(top5_keywords)}</strong>.</li>")
            html.write("<li>Consider refining your targeting strategy around these topics.</li>")
            html.write("</ul>")
    except Exception as e:
        html.write("<p><em>Recommendations unavailable due to data issues.</em></p>")
    html.write("</div>")

    # 📌 Keyword Suggestion Table Section
    html.write("<div class='grid-section'>")
    html.write("<h2>📝 Keyword Suggestions and Projections</h2>")
    try:
        ctr_valid = False
        if {'original_ctr', 'estimated_ctr', 'ctr_uplift'}.issubset(case_study_df.columns):
            ctr_vals = case_study_df['original_ctr']
            if ctr_vals.notna().any() and ctr_vals.nunique() > 1:
                ctr_valid = True
        if ctr_valid:
            # Top 3 keywords by opportunity score
            top3 = case_study_df.sort_values('opportunity_score', ascending=False).head(3)
            top3_keywords = top3['keyword'].tolist()
            html.write("<ul>")
            html.write(f"<li>Focus on improving campaigns using: <strong>{', '.join(top3_keywords)}</strong> — these show the strongest projected uplift.</li>")
            # Top suggestions for each
            if suggestions and 'keywords' in suggestions:
                for kw in top3_keywords:
                    alts = suggestions['keywords'].get(kw, [])
                    if alts:
                        html.write(f"<li>Test new ad copy using suggestions for <strong>{kw}</strong>: <em>{', '.join(alts)}</em></li>")
            html.write("<li>Consider building ad groups or landing pages around these high-impact keywords.</li>")
            html.write("</ul>")
        else:
            # CTR missing or simulated
            html.write("<ul>")
            html.write("<li><strong>Note:</strong> CTR data appears to be simulated or unavailable. For deeper performance insights, consider integrating live ad performance data.</li>")
            # Use search_volume if available and numeric
            if 'search_volume' in case_study_df.columns and np.issubdtype(case_study_df['search_volume'].dtype, np.number):
                top5 = case_study_df.sort_values('search_volume', ascending=False).head(5)
                top5_keywords = top5['keyword'].tolist()
                if top5_keywords:
                    html.write(f"<li>These keywords show the highest search volume and could be strong candidates to optimize: <strong>{', '.join(top5_keywords)}</strong>.</li>")
            html.write("<li>Consider refining your targeting strategy around these topics.</li>")
            html.write("</ul>")
    except Exception as e:
        html.write("<p><em>Recommendations unavailable due to data issues.</em></p>")
    html.write("</div>")

    # Legend for Opportunity Tags
    html.write("<div class='grid-section'>")
    html.write("<h2>📘 Opportunity Tag Legend</h2>")
    html.write("<ul>")
    html.write("<li>🔥 High Potential: low CTR but high search volume — great candidate to improve</li>")
    html.write("<li>🧱 Low CTR, High Impressions: shown often but underperforming — revise or exclude</li>")
    html.write("<li>💡 Niche Opportunity: not widely searched, but could convert in niche campaigns</li>")
    html.write("<li>⭐ Branded Growth: brand-related or localized — good for brand expansion</li>")
    html.write("<li>📉 Overused / Declining: poor performance across the board — consider removing</li>")
    html.write("<li>🚧 Review Needed: underperforming with no clear reason — audit relevance</li>")
    html.write("</ul>")
    html.write("</div>")

    # Close grid container
    html.write("</div>")  # Close grid-section-container

    # New section for Top 10 Best Performing Keywords
    html.write("<div class='section'>")
    html.write("<h2>📈 Top Performing Keywords (High CTR, High Volume)</h2>")

    # Select and sort top-performing keywords (keep all columns this time)
    best_performing = case_study_df[
        case_study_df['original_ctr'] >= case_study_df['original_ctr'].quantile(0.75)
    ].sort_values(['original_ctr', 'search_volume', 'impressions'], ascending=[False, False, False]).head(10)

    # Render the table
    html.write("<table>")
    html.write("<tr>" + "".join(f"<th>{col.replace('_', ' ').title()}</th>" for col in best_performing.columns) + "<th>Opportunity Type</th></tr>")
    html.write(render_table_rows(best_performing, highlight_keywords=global_top_keywords))
    html.write("</table>")
    html.write("</div>")

    # New section for High-Exposure Keywords to Improve
    html.write("<div class='section'>")
    html.write("<h2>🚧 High Visibility, Low Engagement (High Impressions, Low Conversion Keywords)</h2>")

    # Dynamically determine thresholds
    low_ctr_thresh = case_study_df['original_ctr'].quantile(0.25)  # Bottom 25%
    high_impressions_thresh = case_study_df['impressions'].quantile(0.75)  # Top 25%
    median_search_volume = case_study_df['search_volume'].median()

    # Filter based on dynamic criteria
    high_exposure_improve = case_study_df[
        (case_study_df['original_ctr'] <= low_ctr_thresh) &
        (case_study_df['impressions'] >= high_impressions_thresh) &
        (case_study_df['search_volume'] >= median_search_volume)
    ].sort_values('opportunity_score', ascending=False).head(10)

    # Fallback if not enough matches
    if len(high_exposure_improve) < 5:
        low_ctr_thresh = case_study_df['original_ctr'].quantile(0.35)
        high_impressions_thresh = case_study_df['impressions'].quantile(0.65)
        high_exposure_improve = case_study_df[
            (case_study_df['original_ctr'] <= low_ctr_thresh) &
            (case_study_df['impressions'] >= high_impressions_thresh) &
            (case_study_df['search_volume'] >= median_search_volume)
        ].sort_values('opportunity_score', ascending=False).head(10)

    # Render the table
    html.write("<table>")
    html.write("<tr>" + "".join(f"<th>{col.replace('_', ' ').title()}</th>" for col in high_exposure_improve.columns) + "<th>Opportunity Type</th></tr>")
    html.write(render_table_rows(high_exposure_improve, highlight_keywords=global_top_keywords))
    html.write("</table>")
    html.write("</div>")


    # New section for Top 10 Worst Performing Keywords
    html.write("<div class='section'>")
    html.write("<h2>🚫 Underperforming & Low Reach (Candidates for Removal)</h2>")
    # Avoid division by zero and normalize scale by using rank percentiles
    case_study_df['low_perf_score'] = (
        case_study_df['search_volume'].rank(pct=True, ascending=True) +
        case_study_df['impressions'].rank(pct=True, ascending=True) +
        case_study_df['clicks'].rank(pct=True, ascending=True) +
        case_study_df['original_ctr'].rank(pct=True, ascending=True)
    ) / 4  # average of ranks

    # Select 10 keywords with the lowest "low performance score"
    worst_performing = case_study_df.sort_values('low_perf_score', ascending=True).head(10)

    html.write("<table>")
    html.write("<tr>" + "".join(f"<th>{col.replace('_', ' ').title()}</th>" for col in worst_performing.columns) + "<th>Opportunity Type</th></tr>")
    html.write(render_table_rows(worst_performing, highlight_keywords=global_top_keywords))
    html.write("</table>")
    html.write("</div>")

    # Toggle button for full keyword table
    html.write("<div class='keyword-toggle-container'>")
    html.write("<button id='toggle-button' class='toggle-button' onclick='toggleTable()'>Show All Keywords</button>")
    html.write("</div>")

    # Full keyword table in hidden div
    html.write("<div id='full-keyword-table' class='full-table'>")
    html.write("<table>")
    html.write("<tr>" + "".join(f"<th>{col.replace('_', ' ').title()}</th>" for col in case_study_df.columns) + "<th>Opportunity Type</th></tr>")
    html.write(render_table_rows(case_study_df, highlight_keywords=global_top_keywords))
    html.write("</table>")
    html.write("</div>")

    # Add CSS for button styling
    html.write("""
    <style>
    .keyword-toggle-container {
        display: flex;
        justify-content: center;
        margin-top: 30px;
        margin-bottom: 10px;
    }
    .toggle-button {
        background-color: #0072C6;
        color: white;
        border: none;
        padding: 12px 28px;
        font-size: 1.1em;
        border-radius: 6px;
        cursor: pointer;
        transition: background-color 0.2s;
        width: 100%;
        max-width: 300px;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.1);
    }
    .toggle-button:hover {
        background-color: #005fa3;
    }
    </style>
    """)

    # Footer
    html.write("<div class='footer'>")
    html.write(f"Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} &mdash; <strong>Generated by SEO Insights Tool</strong>")
    html.write("</div>")

    html.write("</div></body></html>")
    return html.getvalue()

def render_table_rows(df, highlight_keywords=None):
    rows_html = ""
    for _, row in df.iterrows():
        rows_html += "<tr>"
        for col in df.columns:
            val = row[col]
            cell = val
            if col == 'keyword' and highlight_keywords and row['keyword'] in highlight_keywords:
                cell = f"<span class='bold-keyword'>{val}</span>"
            elif col == 'ctr_uplift':
                try:
                    cell_val = float(val)
                    if np.isnan(cell_val):
                        raise ValueError
                    if cell_val > 0:
                        cell = f"<span class='ctr-uplift-pos'>+{cell_val:.2f}%</span>"
                    elif cell_val < 0:
                        cell = f"<span class='ctr-uplift-neg'>{cell_val:.2f}%</span>"
                    else:
                        cell = f"{cell_val:.2f}%"
                except:
                    cell = '<span title="CTR uplift not available">N/A</span>'
            elif col == 'original_ctr':
                try:
                    cell = f"{float(val):.2f}%"
                except:
                    cell = '<span title="Original CTR not available">N/A</span>'

            elif col == 'estimated_ctr':
                try:
                    cell = f"{float(val):.2f}%"
                except:
                    cell = '<span title="No estimated CTR for this keyword">N/A</span>'

            elif col == 'opportunity_score':
                try:
                    cell = f"{float(val):.2f}"
                except:
                    cell = '<span title="Opportunity score not available">N/A</span>'
            elif col in ['impressions', 'search_volume']:
                try:
                    cell = f"{int(val):,}"
                except:
                    pass
            rows_html += f"<td>{cell}</td>"
        # Add Opportunity Type column
        opportunity_type = ""
        if row['original_ctr'] < 1.0 and row['search_volume'] > 30000:
            opportunity_type = "🔥 High Potential"
        elif row['original_ctr'] < 0.5 and row['impressions'] > 1200:
            opportunity_type = "🚧 Low CTR, High Impressions"
        elif row['search_volume'] < 10000 and row['original_ctr'] < 1.0:
            opportunity_type = "💡 Niche Opportunity"
        elif any(term in row['keyword'].lower() for term in ["brand", "best", "near me"]):
            opportunity_type = "⭐ Branded Growth"
        elif row['original_ctr'] < 0.5 and row['ctr_uplift'] < 0.1:
            opportunity_type = "📉 Overused / Declining"
        else:
            opportunity_type = "Neutral"
        rows_html += f"<td><span class='tag-opportunity{' neutral' if opportunity_type == 'Neutral' else ''}'>{opportunity_type}</span></td>"
        rows_html += "</tr>\n"
    return rows_html 