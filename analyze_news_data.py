"""
Analysis script for Lithuanian News Portals Dataset
Generates results and graphs for conference paper on cognitive modeling of news engagement
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

# Portal names
PORTALS = ['respublika', 'delfi', 'tv3', 'alfa', '15min']

# Reaction columns mapping by portal
REACTION_COLS = {
    'respublika': ['reactions_like', 'reactions_dislike'],
    'delfi': ['reactions_like', 'reactions_dislike', 'reactions_laugh', 
              'reactions_love', 'reactions_sad', 'reactions_angry', 'reactions_surprised'],
    'tv3': ['reactions_like', 'reactions_love', 'reactions_care', 
            'reactions_laugh', 'reactions_surprised', 'reactions_sad', 'reactions_angry'],
    'alfa': ['reactions_like', 'reactions_dislike'],
    '15min': ['reactions_like', 'reactions_dislike']
}

FZ = 16 # font size

class NewsDataAnalyzer:
    """Main analyzer class for news portal data"""
    
    def __init__(self, data_dir='.'):
        self.data_dir = Path(data_dir)
        self.articles = {}
        self.comments = {}
        self.portal_stats = {}
        
    def load_data(self):
        """Load all portal data"""
        print("Loading data from all portals...")
        for portal in PORTALS:
            try:
                articles_path = self.data_dir / f"{portal}_articles.csv"
                comments_path = self.data_dir / f"{portal}_comments.csv"
                
                if articles_path.exists():
                    df_articles = pd.read_csv(articles_path, low_memory=False)
                    df_articles['portal'] = portal
                    # Parse datetime
                    df_articles['published_at'] = pd.to_datetime(
                        df_articles['published_at'], errors='coerce'
                    )
                    self.articles[portal] = df_articles
                    print(f"  ✓ Loaded {len(df_articles)} articles from {portal}")
                
                if comments_path.exists():
                    df_comments = pd.read_csv(comments_path, low_memory=False)
                    df_comments['portal'] = portal
                    # Parse datetime
                    df_comments['created_at'] = pd.to_datetime(
                        df_comments['created_at'], errors='coerce'
                    )
                    self.comments[portal] = df_comments
                    print(f"  ✓ Loaded {len(df_comments)} comments from {portal}")
            except Exception as e:
                print(f"  ✗ Error loading {portal}: {e}")
        
        print(f"\nTotal articles: {sum(len(a) for a in self.articles.values())}")
        print(f"Total comments: {sum(len(c) for c in self.comments.values())}")
    
    def compute_basic_stats(self):
        """Compute basic statistics per portal"""
        print("\nComputing basic statistics...")
        stats = {}
        
        for portal in PORTALS:
            if portal not in self.articles or portal not in self.comments:
                continue
                
            articles = self.articles[portal]
            comments = self.comments[portal]
            
            stats[portal] = {
                'n_articles': len(articles),
                'n_comments': len(comments),
                'avg_comments_per_article': comments.groupby('article_id').size().mean(),
                'n_articles_with_comments': comments['article_id'].nunique(),
                'n_replies': comments['is_reply'].sum() if 'is_reply' in comments.columns else 0,
                'reply_ratio': comments['is_reply'].mean() if 'is_reply' in comments.columns else 0,
                'n_unique_users': comments['username'].nunique(),
                'avg_comments_per_user': len(comments) / comments['username'].nunique() if comments['username'].nunique() > 0 else 0,
            }
            
            # Category distribution
            if 'category' in articles.columns:
                stats[portal]['top_categories'] = articles['category'].value_counts().head(5).to_dict()
            
            print(f"  {portal}: {stats[portal]['n_articles']} articles, "
                  f"{stats[portal]['n_comments']} comments, "
                  f"{stats[portal]['avg_comments_per_article']:.1f} avg comments/article")
        
        self.portal_stats = stats
        return stats
    
    def compute_engagement_metrics(self):
        """Compute engagement metrics (likes, dislikes, reactions)"""
        print("\nComputing engagement metrics...")
        
        for portal in PORTALS:
            if portal not in self.comments:
                continue
            
            comments = self.comments[portal]
            reaction_cols = [col for col in REACTION_COLS.get(portal, []) 
                            if col in comments.columns]
            
            if not reaction_cols:
                continue
            
            # Total reactions
            comments['total_reactions'] = comments[reaction_cols].sum(axis=1)
            
            # Positive vs negative reactions
            if 'reactions_like' in comments.columns:
                comments['positive_reactions'] = comments.get('reactions_like', 0) + \
                                                comments.get('reactions_love', 0) + \
                                                comments.get('reactions_care', 0)
            
            if 'reactions_dislike' in comments.columns:
                comments['negative_reactions'] = comments['reactions_dislike']
            elif 'reactions_angry' in comments.columns:
                comments['negative_reactions'] = comments.get('reactions_angry', 0) + \
                                               comments.get('reactions_sad', 0)
            else:
                comments['negative_reactions'] = 0
            
            # Engagement ratio (positive / (positive + negative))
            total_engagement = comments['positive_reactions'] + comments['negative_reactions']
            comments['engagement_ratio'] = np.where(
                total_engagement > 0,
                comments['positive_reactions'] / total_engagement,
                np.nan
            )
            
            # Polarization score (high when likes/dislikes are imbalanced)
            if 'reactions_like' in comments.columns and 'reactions_dislike' in comments.columns:
                total = comments['reactions_like'] + comments['reactions_dislike']
                comments['polarization'] = np.where(
                    total > 0,
                    np.abs(comments['reactions_like'] - comments['reactions_dislike']) / total,
                    np.nan
                )
            
            self.comments[portal] = comments
    
    def compute_thread_metrics(self):
        """Compute thread structure metrics"""
        print("\nComputing thread structure metrics...")
        
        for portal in PORTALS:
            if portal not in self.comments:
                continue
            
            comments = self.comments[portal]
            
            if 'is_reply' not in comments.columns:
                continue
            
            # Thread depth (for each article)
            def compute_thread_depth(article_comments):
                if 'parent_ref' not in article_comments.columns:
                    return pd.Series([0] * len(article_comments), index=article_comments.index)
                
                depth = pd.Series(0, index=article_comments.index)
                replies = article_comments[article_comments['is_reply'] == True]
                
                def get_depth(comment_id, visited=None):
                    if visited is None:
                        visited = set()
                    if comment_id in visited:
                        return 0
                    visited.add(comment_id)
                    
                    parent = article_comments[article_comments['comment_id'] == comment_id]
                    if len(parent) == 0 or parent['parent_ref'].isna().all():
                        return 0
                    
                    parent_ref = parent['parent_ref'].iloc[0]
                    if pd.isna(parent_ref):
                        return 0
                    
                    return 1 + get_depth(parent_ref, visited)
                
                for idx in replies.index:
                    depth[idx] = get_depth(comments.loc[idx, 'comment_id'])
                
                return depth
            
            comments['thread_depth'] = comments.groupby('article_id').apply(
                lambda x: compute_thread_depth(x)
            ).reset_index(level=0, drop=True)
            
            # Thread size (number of comments in thread)
            def get_thread_size(comment_id, article_comments):
                if 'parent_ref' not in article_comments.columns:
                    return 1
                
                # Count all replies to this comment
                direct_replies = article_comments[
                    (article_comments['parent_ref'] == comment_id) |
                    (article_comments['comment_id'] == comment_id)
                ]
                return len(direct_replies)
            
            comments['thread_size'] = comments.groupby('article_id').apply(
                lambda x: pd.Series([get_thread_size(cid, x) for cid in x['comment_id']], 
                                   index=x.index)
            ).reset_index(level=0, drop=True)
            
            self.comments[portal] = comments
    
    def plot_portal_comparison(self, save_path='results/portal_comparison.png'):
        """Compare basic statistics across portals"""
        print("\nGenerating portal comparison plot...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Cross-Portal Comparison of News Engagement', fontsize=16, fontweight='bold')
        
        # 1. Articles and Comments count
        ax1 = axes[0, 0]
        portals = list(self.portal_stats.keys())
        n_articles = [self.portal_stats[p]['n_articles'] for p in portals]
        n_comments = [self.portal_stats[p]['n_comments'] for p in portals]
        
        x = np.arange(len(portals))
        width = 0.35
        ax1.bar(x - width/2, n_articles, width, label='Articles', alpha=0.8)
        ax1.bar(x + width/2, n_comments, width, label='Comments', alpha=0.8)
        ax1.set_xlabel('Portal', fontsize=FZ)
        ax1.set_ylabel('Count', fontsize=FZ)
        ax1.set_title('Articles and Comments Count by Portal', fontsize=FZ)
        ax1.set_yscale('log') # Log scale for Y-axis | comment this for linear scale
        ax1.set_xticks(x)
        ax1.set_xticklabels(portals, rotation=45, ha='right', fontsize=FZ)
        ax1.tick_params(axis='y', labelsize=FZ)  # y-axis tick labels
        ax1.legend(fontsize=FZ)
        ax1.grid(axis='y', alpha=0.3)
        
        # 2. Average comments per article
        ax2 = axes[0, 1]
        avg_comments = [self.portal_stats[p]['avg_comments_per_article'] for p in portals]
        bars = ax2.bar(portals, avg_comments, alpha=0.8, color='coral')
        ax2.set_xlabel('Portal', fontsize=FZ)
        ax2.set_ylabel('Average Comments per Article', fontsize=FZ)
        ax2.set_title('Engagement Level by Portal', fontsize=FZ)
        ax2.set_ylim(0, 72) # Sets range from 0 to 100
        ax2.set_xticklabels(portals, rotation=45, ha='right', fontsize=FZ)
        ax2.tick_params(axis='y', labelsize=FZ)  # y-axis tick labels
        ax2.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=FZ)
        
        # 3. Reply ratio
        ax3 = axes[1, 0]
        reply_ratios = [self.portal_stats[p]['reply_ratio'] * 100 for p in portals]
        bars = ax3.bar(portals, reply_ratios, alpha=0.8, color='lightblue')
        ax3.set_xlabel('Portal', fontsize=FZ)
        ax3.set_ylabel('Reply Ratio (%)', fontsize=FZ)
        ax3.set_title('Threading Activity by Portal', fontsize=FZ)
        ax3.set_ylim(0, 43)
        ax3.set_xticklabels(portals, rotation=45, ha='right', fontsize=FZ)
        ax3.tick_params(axis='y', labelsize=FZ)  # y-axis tick labels
        ax3.grid(axis='y', alpha=0.3)
        
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=FZ)
        
        # 4. Unique users
        ax4 = axes[1, 1]
        n_users = [self.portal_stats[p]['n_unique_users'] for p in portals]
        bars = ax4.bar(portals, n_users, alpha=0.8, color='lightgreen')
        ax4.set_xlabel('Portal', fontsize=FZ)
        ax4.set_ylabel('Number of Unique Users', fontsize=FZ)
        ax4.set_title('User Base Size by Portal', fontsize=FZ)
        ax4.set_ylim(0, 80000)
        ax4.set_xticklabels(portals, rotation=45, ha='right', fontsize=FZ)
        ax4.tick_params(axis='y', labelsize=FZ)  # y-axis tick labels
        ax4.grid(axis='y', alpha=0.3)
        
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom', fontsize=FZ)
        
        plt.tight_layout()
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved to {save_path}")
        plt.close()
    
    def plot_engagement_distribution(self, save_path='results/engagement_distribution.png'):
        """Plot engagement metrics distribution"""
        print("\nGenerating engagement distribution plot...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Engagement Patterns Across Portals', fontsize=16, fontweight='bold')
        
        # Combine all comments
        all_comments = pd.concat([self.comments[p] for p in PORTALS 
                                 if p in self.comments], ignore_index=True)
        
        # 1. Total reactions distribution
        ax1 = axes[0, 0]
        for portal in PORTALS:
            if portal not in self.comments:
                continue
            comments = self.comments[portal]
            if 'total_reactions' in comments.columns:
                reactions = comments['total_reactions'].dropna()
                reactions = reactions[reactions > 0]  # Only comments with reactions
                if len(reactions) > 0:
                    ax1.hist(reactions, bins=50, alpha=0.5, label=portal, density=True)
        ax1.set_xlabel('Total Reactions')
        ax1.set_ylabel('Density')
        ax1.set_title('Distribution of Total Reactions')
        ax1.set_xlim(0, 50)  # Focus on lower range
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # 2. Engagement ratio by portal
        ax2 = axes[0, 1]
        engagement_data = []
        portal_labels = []
        for portal in PORTALS:
            if portal not in self.comments:
                continue
            comments = self.comments[portal]
            if 'engagement_ratio' in comments.columns:
                ratios = comments['engagement_ratio'].dropna()
                engagement_data.append(ratios.values)
                portal_labels.append(portal)
        
        if engagement_data:
            bp = ax2.boxplot(engagement_data, labels=portal_labels, patch_artist=True)
            for patch in bp['boxes']:
                patch.set_facecolor('lightblue')
                patch.set_alpha(0.7)
            ax2.set_ylabel('Engagement Ratio (Positive / Total)')
            ax2.set_title('Engagement Ratio Distribution by Portal')
            ax2.set_xticklabels(portal_labels, rotation=45, ha='right')
            ax2.grid(axis='y', alpha=0.3)
            ax2.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Neutral')
            ax2.legend()
        
        # 3. Polarization score
        ax3 = axes[1, 0]
        polarization_data = []
        portal_labels_pol = []
        for portal in PORTALS:
            if portal not in self.comments:
                continue
            comments = self.comments[portal]
            if 'polarization' in comments.columns:
                pol = comments['polarization'].dropna()
                polarization_data.append(pol.values)
                portal_labels_pol.append(portal)
        
        if polarization_data:
            bp = ax3.boxplot(polarization_data, labels=portal_labels_pol, patch_artist=True)
            for patch in bp['boxes']:
                patch.set_facecolor('coral')
                patch.set_alpha(0.7)
            ax3.set_ylabel('Polarization Score')
            ax3.set_title('Comment Polarization by Portal')
            ax3.set_xticklabels(portal_labels_pol, rotation=45, ha='right')
            ax3.grid(axis='y', alpha=0.3)
        
        # 4. Thread depth distribution
        ax4 = axes[1, 1]
        for portal in PORTALS:
            if portal not in self.comments:
                continue
            comments = self.comments[portal]
            if 'thread_depth' in comments.columns:
                depth = comments['thread_depth'].dropna()
                depth = depth[depth > 0]  # Only replies
                if len(depth) > 0:
                    ax4.hist(depth, bins=20, alpha=0.5, label=portal, density=True)
        ax4.set_xlabel('Thread Depth')
        ax4.set_ylabel('Density')
        ax4.set_title('Distribution of Thread Depths')
        ax4.legend()
        ax4.grid(alpha=0.3)
        
        plt.tight_layout()
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved to {save_path}")
        plt.close()
    
    def plot_category_analysis(self, save_path='results/category_analysis.png'):
        """Analyze engagement by article category"""
        print("\nGenerating category analysis plot...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle('Engagement Patterns by Article Category', fontsize=16, fontweight='bold')
        
        # Combine articles and comments
        all_data = []
        for portal in PORTALS:
            if portal not in self.articles or portal not in self.comments:
                continue
            
            articles = self.articles[portal]
            comments = self.comments[portal]
            
            # Merge comments with articles
            merged = comments.merge(
                articles[['article_id', 'category']], 
                on='article_id', 
                how='left'
            )
            merged['portal'] = portal
            all_data.append(merged)
        
        if not all_data:
            print("  ✗ No data available for category analysis")
            return
        
        combined = pd.concat(all_data, ignore_index=True)
        
        # 1. Comments per category
        ax1 = axes[0, 0]
        category_counts = combined['category'].value_counts().head(10)
        bars = ax1.barh(range(len(category_counts)), category_counts.values, alpha=0.8)
        ax1.set_yticks(range(len(category_counts)))
        ax1.set_yticklabels(category_counts.index)
        ax1.set_xlabel('Number of Comments')
        ax1.set_title('Top 10 Categories by Comment Volume')
        ax1.grid(axis='x', alpha=0.3)
        
        # 2. Average engagement ratio by category
        ax2 = axes[0, 1]
        if 'engagement_ratio' in combined.columns:
            category_engagement = combined.groupby('category')['engagement_ratio'].mean().sort_values(ascending=False).head(10)
            bars = ax2.barh(range(len(category_engagement)), category_engagement.values, alpha=0.8, color='green')
            ax2.set_yticks(range(len(category_engagement)))
            ax2.set_yticklabels(category_engagement.index)
            ax2.set_xlabel('Average Engagement Ratio')
            ax2.set_title('Top 10 Categories by Positive Engagement')
            ax2.axvline(x=0.5, color='r', linestyle='--', alpha=0.5)
            ax2.grid(axis='x', alpha=0.3)
        
        # 3. Average polarization by category
        ax3 = axes[1, 0]
        if 'polarization' in combined.columns:
            category_polarization = combined.groupby('category')['polarization'].mean().sort_values(ascending=False).head(10)
            bars = ax3.barh(range(len(category_polarization)), category_polarization.values, alpha=0.8, color='red')
            ax3.set_yticks(range(len(category_polarization)))
            ax3.set_yticklabels(category_polarization.index)
            ax3.set_xlabel('Average Polarization Score')
            ax3.set_title('Top 10 Most Polarized Categories')
            ax3.grid(axis='x', alpha=0.3)
        
        # 4. Category distribution by portal
        ax4 = axes[1, 1]
        category_portal = pd.crosstab(combined['category'], combined['portal']).head(10)
        category_portal.plot(kind='bar', ax=ax4, stacked=True, alpha=0.8)
        ax4.set_xlabel('Category')
        ax4.set_ylabel('Number of Comments')
        ax4.set_title('Category Distribution Across Portals (Top 10)')
        ax4.legend(title='Portal', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax4.set_xticklabels(ax4.get_xticklabels(), rotation=45, ha='right')
        ax4.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved to {save_path}")
        plt.close()
    
    def plot_temporal_analysis(self, save_path='results/temporal_analysis.png'):
        """Analyze temporal patterns"""
        print("\nGenerating temporal analysis plot...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle('Temporal Patterns in News Engagement', fontsize=16, fontweight='bold')
        
        # Combine all comments
        all_comments = []
        for portal in PORTALS:
            if portal not in self.comments:
                continue
            comments = self.comments[portal].copy()
            if 'created_at' in comments.columns:
                comments['date'] = comments['created_at'].dt.date
                comments['hour'] = comments['created_at'].dt.hour
                comments['day_of_week'] = comments['created_at'].dt.day_name()
                all_comments.append(comments)
        
        if not all_comments:
            print("  ✗ No temporal data available")
            return
        
        combined = pd.concat(all_comments, ignore_index=True)
        
        # 1. Comments over time
        ax1 = axes[0, 0]
        daily_counts = combined.groupby('date').size()
        ax1.plot(daily_counts.index, daily_counts.values, linewidth=2, alpha=0.7)
        ax1.set_xlabel('Date', fontsize=FZ)
        ax1.set_ylabel('Number of Comments', fontsize=FZ)
        ax1.set_title('Comment Volume Over Time', fontsize=FZ)
        ax1.grid(alpha=0.3)
        ax1.tick_params(axis='x', rotation=45, labelsize=FZ)
        ax1.tick_params(axis='y', labelsize=FZ)  # y-axis tick labels
        
        # 2. Comments by hour of day
        ax2 = axes[0, 1]
        hourly_counts = combined.groupby('hour').size()
        ax2.bar(hourly_counts.index, hourly_counts.values, alpha=0.8, color='steelblue')
        ax2.set_xlabel('Hour of Day', fontsize=FZ)
        ax2.set_ylabel('Number of Comments', fontsize=FZ)
        ax2.set_title('Comment Activity by Hour', fontsize=FZ)
        ax2.tick_params(axis='x', labelsize=FZ)  # x-axis tick labels
        ax2.set_xticks(range(0, 24, 2))
        ax2.tick_params(axis='y', labelsize=FZ)  # y-axis tick labels
        ax2.grid(axis='y', alpha=0.3)
        
        # 3. Comments by day of week
        ax3 = axes[1, 0]
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_counts = combined['day_of_week'].value_counts().reindex(day_order, fill_value=0)
        bars = ax3.bar(range(len(day_counts)), day_counts.values, alpha=0.8, color='coral')
        ax3.set_xticks(range(len(day_counts)))
        ax3.set_xticklabels(day_counts.index, rotation=45, ha='right', fontsize=FZ)
        ax3.set_ylabel('Number of Comments', fontsize=FZ)
        ax3.set_title('Comment Activity by Day of Week', fontsize=FZ)
        ax3.tick_params(axis='y', labelsize=FZ)  # y-axis tick labels
        ax3.grid(axis='y', alpha=0.3)
        
        # 4. Engagement ratio over time
        ax4 = axes[1, 1]
        if 'engagement_ratio' in combined.columns:
            daily_engagement = combined.groupby('date')['engagement_ratio'].mean()
            ax4.plot(daily_engagement.index, daily_engagement.values, linewidth=2, alpha=0.7, color='green')
            ax4.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Neutral')
            ax4.set_xlabel('Date', fontsize=FZ)
            ax4.set_ylabel('Average Engagement Ratio', fontsize=FZ)
            ax4.set_title('Engagement Ratio Over Time', fontsize=FZ)
            ax4.legend(fontsize=FZ)
            ax4.grid(alpha=0.3)
            ax4.tick_params(axis='x', rotation=45, labelsize=FZ)
            ax4.tick_params(axis='y', labelsize=FZ)  # y-axis tick labels
        
        plt.tight_layout()
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved to {save_path}")
        plt.close()
    
    def plot_thread_structure(self, save_path='results/thread_structure.png'):
        """Analyze comment thread structures"""
        print("\nGenerating thread structure plot...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Comment Thread Structure Analysis', fontsize=16, fontweight='bold')
        
        # Combine all comments
        all_comments = []
        for portal in PORTALS:
            if portal not in self.comments:
                continue
            comments = self.comments[portal].copy()
            if 'is_reply' in comments.columns:
                all_comments.append(comments)
        
        if not all_comments:
            print("  ✗ No thread data available")
            return
        
        combined = pd.concat(all_comments, ignore_index=True)
        
        # 1. Reply ratio by portal
        ax1 = axes[0, 0]
        reply_ratios = combined.groupby('portal')['is_reply'].mean() * 100
        bars = ax1.bar(reply_ratios.index, reply_ratios.values, alpha=0.8, color='lightblue')
        ax1.set_xlabel('Portal')
        ax1.set_ylabel('Reply Ratio (%)')
        ax1.set_title('Percentage of Replies by Portal')
        ax1.set_xticklabels(reply_ratios.index, rotation=45, ha='right')
        ax1.grid(axis='y', alpha=0.3)
        
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom')
        
        # 2. Thread depth distribution
        ax2 = axes[0, 1]
        if 'thread_depth' in combined.columns:
            depth_dist = combined[combined['thread_depth'] > 0]['thread_depth'].value_counts().sort_index()
            ax2.bar(depth_dist.index[:20], depth_dist.values[:20], alpha=0.8, color='coral')
            ax2.set_xlabel('Thread Depth')
            ax2.set_ylabel('Number of Comments')
            ax2.set_title('Distribution of Thread Depths')
            ax2.grid(axis='y', alpha=0.3)
        
        # 3. Thread size distribution
        ax3 = axes[1, 0]
        if 'thread_size' in combined.columns:
            size_dist = combined['thread_size'].value_counts().sort_index().head(20)
            ax3.bar(size_dist.index, size_dist.values, alpha=0.8, color='lightgreen')
            ax3.set_xlabel('Thread Size')
            ax3.set_ylabel('Number of Threads')
            ax3.set_title('Distribution of Thread Sizes')
            ax3.grid(axis='y', alpha=0.3)
        
        # 4. Average thread metrics by portal
        ax4 = axes[1, 1]
        portal_metrics = []
        portal_names = []
        metric_labels = []
        
        for portal in PORTALS:
            if portal not in self.comments:
                continue
            comments = self.comments[portal]
            if 'thread_depth' in comments.columns and 'thread_size' in comments.columns:
                portal_names.append(portal)
                portal_metrics.append([
                    comments['thread_depth'].mean(),
                    comments['thread_size'].mean(),
                    comments['is_reply'].mean() * 100 if 'is_reply' in comments.columns else 0
                ])
        
        if portal_metrics:
            x = np.arange(len(portal_names))
            width = 0.25
            metrics_array = np.array(portal_metrics)
            
            # Normalize for visualization
            metrics_norm = metrics_array / metrics_array.max(axis=0) * 100
            
            ax4.bar(x - width, metrics_norm[:, 0], width, label='Avg Depth', alpha=0.8)
            ax4.bar(x, metrics_norm[:, 1], width, label='Avg Size', alpha=0.8)
            ax4.bar(x + width, metrics_norm[:, 2], width, label='Reply %', alpha=0.8)
            
            ax4.set_xlabel('Portal')
            ax4.set_ylabel('Normalized Metric Value')
            ax4.set_title('Thread Metrics Comparison')
            ax4.set_xticks(x)
            ax4.set_xticklabels(portal_names, rotation=45, ha='right')
            ax4.legend()
            ax4.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved to {save_path}")
        plt.close()
    
    def generate_summary_table(self, save_path='results/summary_statistics.csv'):
        """Generate summary statistics table"""
        print("\nGenerating summary statistics table...")
        
        summary_data = []
        for portal in PORTALS:
            if portal not in self.portal_stats:
                continue
            
            stats = self.portal_stats[portal]
            row = {
                'Portal': portal,
                'Articles': stats['n_articles'],
                'Comments': stats['n_comments'],
                'Avg Comments/Article': f"{stats['avg_comments_per_article']:.2f}",
                'Articles with Comments': stats['n_articles_with_comments'],
                'Replies': stats['n_replies'],
                'Reply Ratio (%)': f"{stats['reply_ratio'] * 100:.2f}",
                'Unique Users': stats['n_unique_users'],
                'Avg Comments/User': f"{stats['avg_comments_per_user']:.2f}",
            }
            summary_data.append(row)
        
        df_summary = pd.DataFrame(summary_data)
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        df_summary.to_csv(save_path, index=False)
        print(f"  ✓ Saved to {save_path}")
        print("\nSummary Statistics:")
        print(df_summary.to_string(index=False))
        
        return df_summary
    
    def run_full_analysis(self):
        """Run complete analysis pipeline"""
        print("=" * 60)
        print("LITHUANIAN NEWS PORTALS ANALYSIS")
        print("=" * 60)
        
        # Load data
        self.load_data()
        
        # Compute statistics
        self.compute_basic_stats()
        self.compute_engagement_metrics()
        self.compute_thread_metrics()
        
        # Generate visualizations
        print("\n" + "=" * 60)
        print("GENERATING VISUALIZATIONS")
        print("=" * 60)
        
        self.plot_portal_comparison()
        self.plot_engagement_distribution()
        self.plot_category_analysis()
        self.plot_temporal_analysis()
        self.plot_thread_structure()
        
        # Generate summary
        self.generate_summary_table()
        
        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETE!")
        print("=" * 60)
        print("\nResults saved to 'results/' directory:")
        print("  - portal_comparison.png")
        print("  - engagement_distribution.png")
        print("  - category_analysis.png")
        print("  - temporal_analysis.png")
        print("  - thread_structure.png")
        print("  - summary_statistics.csv")


if __name__ == "__main__":
    # Initialize analyzer
    analyzer = NewsDataAnalyzer(data_dir='.')
    
    # Run full analysis
    analyzer.run_full_analysis()
