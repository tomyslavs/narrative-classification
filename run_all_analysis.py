"""
Master script to run all analyses for conference paper
Generates all figures and statistics
"""

from analyze_news_data import NewsDataAnalyzer
from advanced_analysis import AdvancedAnalyzer
import sys

def main():
    print("=" * 70)
    print("LITHUANIAN NEWS PORTALS - COMPLETE ANALYSIS PIPELINE")
    print("=" * 70)
    print("\nThis script will:")
    print("  1. Load data from all 5 portals")
    print("  2. Compute basic statistics and metrics")
    print("  3. Generate standard visualizations")
    print("  4. Run advanced statistical analyses")
    print("  5. Create summary reports")
    print("\n" + "=" * 70 + "\n")
    
    # Initialize analyzer
    print("Initializing analyzer...")
    DATA_DIR = 'LT_datasets'
    analyzer = NewsDataAnalyzer(data_dir=DATA_DIR)
    advanced = AdvancedAnalyzer(data_dir=DATA_DIR)
    
    # Run basic analysis
    print("\n" + "=" * 70)
    print("PHASE 1: BASIC ANALYSIS")
    print("=" * 70)
    analyzer.run_full_analysis()
    
    # Run advanced analysis
    print("\n" + "=" * 70)
    print("PHASE 2: ADVANCED STATISTICAL ANALYSIS")
    print("=" * 70)
    
    # Load data for advanced analyzer
    advanced.load_data()
    advanced.compute_basic_stats()
    advanced.compute_engagement_metrics()
    advanced.compute_thread_metrics()
    
    # Run advanced analyses
    advanced.analyze_article_engagement_correlation()
    advanced.test_portal_differences()
    advanced.predict_engagement()
    advanced.generate_correlation_matrix()
    
    # Final summary
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE!")
    print("=" * 70)
    print("\nAll results have been saved to the 'results/' directory:")
    print("\nBasic Analysis:")
    print("  ✓ portal_comparison.png")
    print("  ✓ engagement_distribution.png")
    print("  ✓ category_analysis.png")
    print("  ✓ temporal_analysis.png")
    print("  ✓ thread_structure.png")
    print("  ✓ summary_statistics.csv")
    print("\nAdvanced Analysis:")
    print("  ✓ article_engagement_correlation.png")
    print("  ✓ statistical_tests.txt")
    print("  ✓ engagement_prediction.png")
    print("  ✓ correlation_matrix.png")
    print("\n" + "=" * 70)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
