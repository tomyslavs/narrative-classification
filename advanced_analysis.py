"""
Advanced statistical analysis for conference paper
Includes correlation analysis, hypothesis testing, and predictive modeling
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import the main analyzer
from analyze_news_data import NewsDataAnalyzer, PORTALS

plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")


class AdvancedAnalyzer(NewsDataAnalyzer):
    """Extended analyzer with advanced statistical methods"""
    
    def analyze_article_engagement_correlation(self, save_path='results/article_engagement_correlation.png'):
        """Analyze correlation between article features and comment engagement"""
        print("\nAnalyzing article-engagement correlations...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Article Features vs Comment Engagement', fontsize=16, fontweight='bold')
        
        all_data = []
        for portal in PORTALS:
            if portal not in self.articles or portal not in self.comments:
                continue
            
            articles = self.articles[portal]
            comments = self.comments[portal]
            
            # Aggregate comment metrics per article
            article_metrics = comments.groupby('article_id').agg({
                'comment_id': 'count',
                'total_reactions': 'sum' if 'total_reactions' in comments.columns else lambda x: 0,
                'engagement_ratio': 'mean' if 'engagement_ratio' in comments.columns else lambda x: np.nan,
                'polarization': 'mean' if 'polarization' in comments.columns else lambda x: np.nan,
            }).rename(columns={'comment_id': 'n_comments'})
            
            # Merge with article features
            article_features = articles[['article_id', 'title', 'category', 'comments_count']].copy()
            article_features['title_length'] = article_features['title'].str.len()
            article_features['has_keywords'] = articles['keywords'].notna()
            
            merged = article_features.merge(article_metrics, on='article_id', how='left')
            merged['portal'] = portal
            all_data.append(merged)
        
        if not all_data:
            print("  ✗ No data available")
            return
        
        combined = pd.concat(all_data, ignore_index=True)
        
        # 1. Title length vs comments
        ax1 = axes[0, 0]
        combined_clean = combined.dropna(subset=['title_length', 'n_comments'])
        if len(combined_clean) > 0:
            ax1.scatter(combined_clean['title_length'], combined_clean['n_comments'], 
                       alpha=0.3, s=20)
            ax1.set_xlabel('Title Length (characters)')
            ax1.set_ylabel('Number of Comments')
            ax1.set_title('Title Length vs Comment Count')
            
            # Add trend line
            z = np.polyfit(combined_clean['title_length'], combined_clean['n_comments'], 1)
            p = np.poly1d(z)
            ax1.plot(combined_clean['title_length'], p(combined_clean['title_length']), 
                    "r--", alpha=0.8, linewidth=2)
            
            # Calculate correlation
            corr = combined_clean['title_length'].corr(combined_clean['n_comments'])
            ax1.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax1.transAxes,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            ax1.grid(alpha=0.3)
        
        # 2. Category vs average engagement
        ax2 = axes[0, 1]
        if 'engagement_ratio' in combined.columns:
            category_engagement = combined.groupby('category')['engagement_ratio'].mean().sort_values(ascending=False).head(10)
            bars = ax2.barh(range(len(category_engagement)), category_engagement.values, alpha=0.8)
            ax2.set_yticks(range(len(category_engagement)))
            ax2.set_yticklabels(category_engagement.index)
            ax2.set_xlabel('Average Engagement Ratio')
            ax2.set_title('Category vs Engagement Ratio')
            ax2.axvline(x=0.5, color='r', linestyle='--', alpha=0.5)
            ax2.grid(axis='x', alpha=0.3)
        
        # 3. Comments count distribution
        ax3 = axes[1, 0]
        comments_dist = combined['n_comments'].dropna()
        comments_dist = comments_dist[comments_dist > 0]
        if len(comments_dist) > 0:
            ax3.hist(comments_dist, bins=50, alpha=0.7, edgecolor='black')
            ax3.set_xlabel('Number of Comments per Article')
            ax3.set_ylabel('Frequency')
            ax3.set_title('Distribution of Comment Counts')
            ax3.set_xlim(0, min(100, comments_dist.quantile(0.95)))  # Focus on 95th percentile
            ax3.grid(alpha=0.3)
            
            # Add statistics
            median = comments_dist.median()
            mean = comments_dist.mean()
            ax3.axvline(median, color='r', linestyle='--', label=f'Median: {median:.1f}')
            ax3.axvline(mean, color='g', linestyle='--', label=f'Mean: {mean:.1f}')
            ax3.legend()
        
        # 4. Portal comparison of engagement metrics
        ax4 = axes[1, 1]
        portal_metrics = combined.groupby('portal').agg({
            'n_comments': 'mean',
            'engagement_ratio': 'mean',
            'polarization': 'mean'
        }).dropna()
        
        if len(portal_metrics) > 0:
            x = np.arange(len(portal_metrics))
            width = 0.25
            
            # Normalize for comparison
            metrics_norm = portal_metrics / portal_metrics.max() * 100
            
            ax4.bar(x - width, metrics_norm['n_comments'], width, label='Avg Comments', alpha=0.8)
            if 'engagement_ratio' in metrics_norm.columns:
                ax4.bar(x, metrics_norm['engagement_ratio'], width, label='Engagement Ratio', alpha=0.8)
            if 'polarization' in metrics_norm.columns:
                ax4.bar(x + width, metrics_norm['polarization'], width, label='Polarization', alpha=0.8)
            
            ax4.set_xlabel('Portal')
            ax4.set_ylabel('Normalized Value')
            ax4.set_title('Portal Comparison: Engagement Metrics')
            ax4.set_xticks(x)
            ax4.set_xticklabels(portal_metrics.index, rotation=45, ha='right')
            ax4.legend()
            ax4.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved to {save_path}")
        plt.close()
    
    def test_portal_differences(self, save_path='results/statistical_tests.txt'):
        """Perform statistical tests for differences between portals"""
        print("\nPerforming statistical tests...")
        
        results = []
        
        # Combine all comments
        all_comments = []
        for portal in PORTALS:
            if portal not in self.comments:
                continue
            comments = self.comments[portal].copy()
            comments['portal'] = portal
            all_comments.append(comments)
        
        if not all_comments:
            print("  ✗ No data available")
            return
        
        combined = pd.concat(all_comments, ignore_index=True)
        
        # Test 1: Engagement ratio differences
        if 'engagement_ratio' in combined.columns:
            portals_with_data = []
            engagement_data = []
            
            for portal in PORTALS:
                portal_data = combined[combined['portal'] == portal]['engagement_ratio'].dropna()
                if len(portal_data) > 10:  # Minimum sample size
                    portals_with_data.append(portal)
                    engagement_data.append(portal_data.values)
            
            if len(engagement_data) >= 2:
                # ANOVA test
                f_stat, p_value = stats.f_oneway(*engagement_data)
                results.append({
                    'Test': 'Engagement Ratio - ANOVA',
                    'F-statistic': f_stat,
                    'P-value': p_value,
                    'Significant': 'Yes' if p_value < 0.05 else 'No',
                    'Interpretation': f'Portals differ significantly' if p_value < 0.05 else 'No significant difference'
                })
        
        # Test 2: Comment count differences
        article_counts = []
        portal_labels = []
        for portal in PORTALS:
            if portal not in self.comments:
                continue
            counts = self.comments[portal].groupby('article_id').size()
            if len(counts) > 0:
                article_counts.append(counts.values)
                portal_labels.append(portal)
        
        if len(article_counts) >= 2:
            f_stat, p_value = stats.f_oneway(*article_counts)
            results.append({
                'Test': 'Comments per Article - ANOVA',
                'F-statistic': f_stat,
                'P-value': p_value,
                'Significant': 'Yes' if p_value < 0.05 else 'No',
                'Interpretation': f'Portals differ significantly' if p_value < 0.05 else 'No significant difference'
            })
        
        # Test 3: Reply ratio differences
        if 'is_reply' in combined.columns:
            reply_ratios = []
            portal_labels_reply = []
            for portal in PORTALS:
                if portal not in self.comments:
                    continue
                ratio = self.comments[portal]['is_reply'].mean()
                if not np.isnan(ratio):
                    reply_ratios.append(ratio)
                    portal_labels_reply.append(portal)
            
            if len(reply_ratios) >= 2:
                # Chi-square test
                reply_counts = []
                total_counts = []
                for portal in portal_labels_reply:
                    portal_comments = combined[combined['portal'] == portal]
                    reply_counts.append(portal_comments['is_reply'].sum())
                    total_counts.append(len(portal_comments))
                
                chi2, p_value = stats.chi2_contingency([reply_counts, 
                                                         [t - r for t, r in zip(total_counts, reply_counts)]])[:2]
                results.append({
                    'Test': 'Reply Ratio - Chi-square',
                    'Chi-square': chi2,
                    'P-value': p_value,
                    'Significant': 'Yes' if p_value < 0.05 else 'No',
                    'Interpretation': f'Portals differ significantly' if p_value < 0.05 else 'No significant difference'
                })
        
        # Save results
        df_results = pd.DataFrame(results)
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w') as f:
            f.write("STATISTICAL TEST RESULTS\n")
            f.write("=" * 60 + "\n\n")
            f.write(df_results.to_string(index=False))
            f.write("\n\n")
            f.write("Note: Significance level α = 0.05\n")
        
        print(f"  ✓ Saved to {save_path}")
        print("\nStatistical Test Results:")
        print(df_results.to_string(index=False))
        
        return df_results
    
    def predict_engagement(self, save_path='results/engagement_prediction.png'):
        """Simple predictive model for comment engagement"""
        print("\nBuilding engagement prediction model...")
        
        # Prepare data
        all_data = []
        for portal in PORTALS:
            if portal not in self.articles or portal not in self.comments:
                continue
            
            articles = self.articles[portal]
            comments = self.comments[portal]
            
            # Article-level features
            article_features = articles[['article_id', 'title', 'category']].copy()
            article_features['title_length'] = article_features['title'].str.len()
            article_features['title_words'] = article_features['title'].str.split().str.len()
            article_features['has_category'] = article_features['category'].notna()
            
            # Comment metrics per article
            comment_metrics = comments.groupby('article_id').agg({
                'comment_id': 'count',
                'total_reactions': 'sum' if 'total_reactions' in comments.columns else lambda x: 0
            }).rename(columns={'comment_id': 'n_comments'})
            
            merged = article_features.merge(comment_metrics, on='article_id', how='left')
            merged = merged[merged['n_comments'].notna() & (merged['n_comments'] > 0)]
            all_data.append(merged)
        
        if not all_data:
            print("  ✗ No data available")
            return
        
        combined = pd.concat(all_data, ignore_index=True)
        
        # Prepare features
        feature_cols = ['title_length', 'title_words', 'has_category']
        X = combined[feature_cols].fillna(0)
        y = combined['n_comments']
        
        # Train simple linear model
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        
        # Calculate metrics
        r2 = r2_score(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        
        # Visualize
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Comment Count Prediction Model', fontsize=14, fontweight='bold')
        
        # 1. Predicted vs Actual
        ax1 = axes[0]
        ax1.scatter(y, y_pred, alpha=0.3, s=20)
        ax1.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2, label='Perfect Prediction')
        ax1.set_xlabel('Actual Comment Count')
        ax1.set_ylabel('Predicted Comment Count')
        ax1.set_title(f'Prediction Accuracy (R² = {r2:.3f}, RMSE = {rmse:.2f})')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # 2. Feature importance
        ax2 = axes[1]
        feature_importance = pd.DataFrame({
            'Feature': feature_cols,
            'Coefficient': model.coef_
        }).sort_values('Coefficient', key=abs, ascending=False)
        
        bars = ax2.barh(feature_importance['Feature'], feature_importance['Coefficient'], alpha=0.8)
        ax2.set_xlabel('Coefficient Value')
        ax2.set_title('Feature Importance (Linear Model)')
        ax2.axvline(x=0, color='r', linestyle='--', alpha=0.5)
        ax2.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved to {save_path}")
        print(f"  Model R²: {r2:.3f}, RMSE: {rmse:.2f}")
        plt.close()
    
    def generate_correlation_matrix(self, save_path='results/correlation_matrix.png'):
        """Generate correlation matrix for key metrics"""
        print("\nGenerating correlation matrix...")
        
        # Combine all data
        all_comments = []
        for portal in PORTALS:
            if portal not in self.comments:
                continue
            comments = self.comments[portal].copy()
            all_comments.append(comments)
        
        if not all_comments:
            print("  ✗ No data available")
            return
        
        combined = pd.concat(all_comments, ignore_index=True)
        
        # Select numeric columns for correlation
        numeric_cols = []
        for col in ['total_reactions', 'positive_reactions', 'negative_reactions', 
                   'engagement_ratio', 'polarization', 'thread_depth', 'thread_size']:
            if col in combined.columns:
                numeric_cols.append(col)
        
        if len(numeric_cols) < 2:
            print("  ✗ Insufficient numeric columns")
            return
        
        # Calculate correlation
        corr_matrix = combined[numeric_cols].corr()
        
        # Plot
        plt.figure(figsize=(10, 8))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
        plt.title('Correlation Matrix of Engagement Metrics', fontsize=14, fontweight='bold')
        plt.tight_layout()
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved to {save_path}")
        plt.close()


if __name__ == "__main__":
    # Initialize advanced analyzer
    analyzer = AdvancedAnalyzer(data_dir='.')
    
    # Load data
    analyzer.load_data()
    analyzer.compute_basic_stats()
    analyzer.compute_engagement_metrics()
    analyzer.compute_thread_metrics()
    
    # Run advanced analyses
    print("\n" + "=" * 60)
    print("ADVANCED ANALYSES")
    print("=" * 60)
    
    analyzer.analyze_article_engagement_correlation()
    analyzer.test_portal_differences()
    analyzer.predict_engagement()
    analyzer.generate_correlation_matrix()
    
    print("\n" + "=" * 60)
    print("ADVANCED ANALYSIS COMPLETE!")
    print("=" * 60)
