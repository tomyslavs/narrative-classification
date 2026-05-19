# News Portals Data Analysis

This script generates comprehensive analysis and visualizations for the Lithuanian news portals dataset, suitable for conference paper presentation.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Full Analysis

Run the complete analysis pipeline:

```bash
python analyze_news_data.py
```

This will:
1. Load all portal data (respublika, delfi, tv3, alfa, 15min)
2. Compute basic statistics
3. Calculate engagement metrics (likes, dislikes, polarization)
4. Analyze thread structures
5. Generate all visualizations
6. Create summary statistics table

### Quick Start

Run all analyses at once:

```bash
python run_all_analysis.py
```

This will generate all figures and statistics in one go.

## Output Files

All results are saved to the `results/` directory:

### Basic Analysis (`analyze_news_data.py`)
- **portal_comparison.png** - Cross-portal comparison of basic statistics
- **engagement_distribution.png** - Engagement patterns and polarization analysis
- **category_analysis.png** - Engagement by article category
- **temporal_analysis.png** - Time-based patterns (hourly, daily, weekly)
- **thread_structure.png** - Comment thread structure analysis
- **summary_statistics.csv** - Summary table with key metrics

### Advanced Analysis (`advanced_analysis.py`)
- **article_engagement_correlation.png** - Correlation between article features and engagement
- **statistical_tests.txt** - Results of hypothesis tests (ANOVA, Chi-square)
- **engagement_prediction.png** - Predictive model for comment counts
- **correlation_matrix.png** - Correlation matrix of engagement metrics

## Running Individual Analyses

### Basic Analysis Only

```bash
python analyze_news_data.py
```

### Advanced Statistical Analysis Only

```bash
python advanced_analysis.py
```

Note: Advanced analysis requires basic analysis to be run first (or run `run_all_analysis.py`).

## Custom Analysis

You can also use the analyzer classes programmatically:

```python
from analyze_news_data import NewsDataAnalyzer
from advanced_analysis import AdvancedAnalyzer

# Basic analysis
analyzer = NewsDataAnalyzer(data_dir='.')
analyzer.load_data()
analyzer.compute_basic_stats()
analyzer.compute_engagement_metrics()
analyzer.compute_thread_metrics()
analyzer.plot_portal_comparison()

# Advanced analysis
advanced = AdvancedAnalyzer(data_dir='.')
advanced.load_data()
advanced.compute_basic_stats()
advanced.compute_engagement_metrics()
advanced.test_portal_differences()
```

## Key Metrics Computed

### Basic Statistics
- Number of articles and comments per portal
- Average comments per article
- Reply ratios
- Unique user counts

### Engagement Metrics
- **Total Reactions**: Sum of all reaction types
- **Engagement Ratio**: Positive reactions / (Positive + Negative reactions)
- **Polarization Score**: Measure of like/dislike imbalance (0 = balanced, 1 = highly polarized)

### Thread Metrics
- **Thread Depth**: Maximum nesting level of replies
- **Thread Size**: Number of comments in a thread
- **Reply Ratio**: Percentage of comments that are replies

## Figures for Conference Paper

All figures are generated at 300 DPI resolution suitable for publication. The style follows academic publication standards with:
- Clear labels and legends
- Consistent color schemes
- Professional formatting
- High-resolution output

## Notes

- The script handles missing data gracefully
- Different portals may have different reaction types (see REACTION_COLS mapping)
- Thread depth calculation may take time for large datasets
- All datetime parsing handles various formats automatically
