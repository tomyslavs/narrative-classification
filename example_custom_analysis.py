"""
Example script showing how to use the analyzer classes for custom analysis
Modify this script for your specific research questions
"""

from analyze_news_data import NewsDataAnalyzer
from advanced_analysis import AdvancedAnalyzer
import pandas as pd
import matplotlib.pyplot as plt

def example_custom_analysis():
    """Example of custom analysis workflow"""
    
    # Initialize analyzer
    analyzer = NewsDataAnalyzer(data_dir='.')
    
    # Load data
    print("Loading data...")
    analyzer.load_data()
    
    # Compute metrics
    print("Computing metrics...")
    analyzer.compute_basic_stats()
    analyzer.compute_engagement_metrics()
    analyzer.compute_thread_metrics()
    
    # Example 1: Compare specific portals
    print("\nExample 1: Comparing specific portals...")
    portals_to_compare = ['respublika', 'delfi']
    
    for portal in portals_to_compare:
        if portal in analyzer.portal_stats:
            stats = analyzer.portal_stats[portal]
            print(f"\n{portal.upper()}:")
            print(f"  Articles: {stats['n_articles']}")
            print(f"  Comments: {stats['n_comments']}")
            print(f"  Avg comments/article: {stats['avg_comments_per_article']:.2f}")
    
    # Example 2: Analyze specific category
    print("\nExample 2: Analyzing 'politics' category...")
    
    all_articles = pd.concat([analyzer.articles[p] for p in analyzer.articles.keys()], 
                             ignore_index=True)
    all_comments = pd.concat([analyzer.comments[p] for p in analyzer.comments.keys()], 
                            ignore_index=True)
    
    # Merge to get category info
    merged = all_comments.merge(
        all_articles[['article_id', 'category']], 
        on='article_id', 
        how='left'
    )
    
    politics_comments = merged[merged['category'] == 'politika']
    if len(politics_comments) > 0:
        print(f"  Found {len(politics_comments)} comments in politics category")
        if 'engagement_ratio' in politics_comments.columns:
            avg_engagement = politics_comments['engagement_ratio'].mean()
            print(f"  Average engagement ratio: {avg_engagement:.3f}")
        if 'polarization' in politics_comments.columns:
            avg_polarization = politics_comments['polarization'].mean()
            print(f"  Average polarization: {avg_polarization:.3f}")
    
    # Example 3: Custom visualization
    print("\nExample 3: Creating custom visualization...")
    
    # Compare engagement ratios across portals
    fig, ax = plt.subplots(figsize=(10, 6))
    
    portal_names = []
    engagement_means = []
    
    for portal in analyzer.comments.keys():
        comments = analyzer.comments[portal]
        if 'engagement_ratio' in comments.columns:
            ratios = comments['engagement_ratio'].dropna()
            if len(ratios) > 0:
                portal_names.append(portal)
                engagement_means.append(ratios.mean())
    
    if portal_names:
        bars = ax.bar(portal_names, engagement_means, alpha=0.7, color='steelblue')
        ax.set_xlabel('Portal')
        ax.set_ylabel('Average Engagement Ratio')
        ax.set_title('Custom: Average Engagement Ratio by Portal')
        ax.set_ylim(0, 1)
        ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Neutral')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom')
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('results/custom_engagement_comparison.png', dpi=150, bbox_inches='tight')
        print("  ✓ Saved custom figure to results/custom_engagement_comparison.png")
        plt.close()
    
    # Example 4: Export specific data subset
    print("\nExample 4: Exporting data subset...")
    
    # Export highly polarized comments
    if 'polarization' in all_comments.columns:
        high_polarization = all_comments[
            (all_comments['polarization'] > 0.8) & 
            (all_comments['polarization'].notna())
        ]
        
        if len(high_polarization) > 0:
            export_cols = ['portal', 'article_id', 'comment', 'polarization', 
                          'engagement_ratio', 'total_reactions']
            export_cols = [col for col in export_cols if col in high_polarization.columns]
            
            high_polarization[export_cols].to_csv(
                'results/high_polarization_comments.csv', 
                index=False
            )
            print(f"  ✓ Exported {len(high_polarization)} highly polarized comments")
    
    print("\n" + "="*60)
    print("Custom analysis complete!")
    print("="*60)
    print("\nModify this script to answer your specific research questions.")

if __name__ == "__main__":
    example_custom_analysis()
