"""
Quick data exploration script
Use this to check data quality and get overview before running full analysis
"""

import pandas as pd
from pathlib import Path

PORTALS = ['respublika', 'delfi', 'tv3', 'alfa', '15min']

def explore_portal(portal_name, data_dir='.'):
    """Quick exploration of a single portal's data"""
    data_dir = Path(data_dir)
    
    print(f"\n{'='*60}")
    print(f"Exploring {portal_name.upper()}")
    print(f"{'='*60}")
    
    # Load articles
    articles_path = data_dir / f"{portal_name}_articles.csv"
    if articles_path.exists():
        print(f"\nArticles file: {articles_path}")
        articles = pd.read_csv(articles_path, nrows=5)  # Just peek
        print(f"  Columns: {list(articles.columns)}")
        print(f"  Sample row:")
        for col in articles.columns:
            val = articles[col].iloc[0]
            if isinstance(val, str) and len(str(val)) > 100:
                val = str(val)[:100] + "..."
            print(f"    {col}: {val}")
        
        # Get full stats
        articles_full = pd.read_csv(articles_path)
        print(f"\n  Total articles: {len(articles_full)}")
        if 'category' in articles_full.columns:
            print(f"  Categories: {articles_full['category'].nunique()}")
            print(f"  Top categories:")
            for cat, count in articles_full['category'].value_counts().head(5).items():
                print(f"    - {cat}: {count}")
    else:
        print(f"  ✗ Articles file not found: {articles_path}")
    
    # Load comments
    comments_path = data_dir / f"{portal_name}_comments.csv"
    if comments_path.exists():
        print(f"\nComments file: {comments_path}")
        comments = pd.read_csv(comments_path, nrows=5)  # Just peek
        print(f"  Columns: {list(comments.columns)}")
        print(f"  Sample row:")
        for col in comments.columns:
            val = comments[col].iloc[0]
            if isinstance(val, str) and len(str(val)) > 100:
                val = str(val)[:100] + "..."
            print(f"    {col}: {val}")
        
        # Get full stats
        comments_full = pd.read_csv(comments_path)
        print(f"\n  Total comments: {len(comments_full)}")
        print(f"  Unique articles: {comments_full['article_id'].nunique()}")
        print(f"  Unique users: {comments_full['username'].nunique()}")
        
        if 'is_reply' in comments_full.columns:
            replies = comments_full['is_reply'].sum()
            print(f"  Replies: {replies} ({replies/len(comments_full)*100:.1f}%)")
        
        # Check reaction columns
        reaction_cols = [col for col in comments_full.columns if 'reaction' in col.lower()]
        if reaction_cols:
            print(f"  Reaction columns: {reaction_cols}")
            for col in reaction_cols[:5]:  # Show first 5
                total = comments_full[col].sum()
                print(f"    {col}: {total} total reactions")
    else:
        print(f"  ✗ Comments file not found: {comments_path}")

def main():
    print("="*60)
    print("DATA EXPLORATION - LITHUANIAN NEWS PORTALS")
    print("="*60)
    
    print("\nChecking available data files...")
    data_dir = Path('.')
    
    available_portals = []
    for portal in PORTALS:
        articles_path = data_dir / f"{portal}_articles.csv"
        comments_path = data_dir / f"{portal}_comments.csv"
        
        if articles_path.exists() or comments_path.exists():
            available_portals.append(portal)
            print(f"  ✓ {portal}: ", end="")
            if articles_path.exists():
                print(f"articles ({articles_path.stat().st_size / 1024 / 1024:.1f} MB)", end=", ")
            if comments_path.exists():
                print(f"comments ({comments_path.stat().st_size / 1024 / 1024:.1f} MB)")
    
    if not available_portals:
        print("\n  ✗ No data files found!")
        print("\nExpected files:")
        for portal in PORTALS:
            print(f"    - {portal}_articles.csv")
            print(f"    - {portal}_comments.csv")
        return
    
    print(f"\nFound data for {len(available_portals)} portal(s)")
    
    # Explore each portal
    for portal in available_portals:
        explore_portal(portal)
    
    print("\n" + "="*60)
    print("EXPLORATION COMPLETE")
    print("="*60)
    print("\nNext steps:")
    print("  1. Run 'python analyze_news_data.py' for basic analysis")
    print("  2. Run 'python run_all_analysis.py' for complete analysis")

if __name__ == "__main__":
    main()
