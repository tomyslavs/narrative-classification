# Analysis Scripts Summary

This directory contains Python scripts for analyzing Lithuanian news portals data for conference paper presentation.

## Scripts Overview

### 1. `explore_data.py` - Data Exploration
**Purpose**: Quick data quality check and overview

**Usage**:
```bash
python explore_data.py
```

**What it does**:
- Checks which portal data files are available
- Shows file sizes
- Displays sample rows and column names
- Provides basic statistics (row counts, unique values)
- Helps verify data before running full analysis

**When to use**: First step - verify your data is loaded correctly

---

### 2. `analyze_news_data.py` - Basic Analysis
**Purpose**: Core analysis and visualization pipeline

**Usage**:
```bash
python analyze_news_data.py
```

**What it does**:
- Loads all portal data
- Computes basic statistics (articles, comments, users)
- Calculates engagement metrics (likes, dislikes, polarization)
- Analyzes thread structures (depth, size, reply ratios)
- Generates 5 publication-quality figures:
  - Portal comparison charts
  - Engagement distribution plots
  - Category analysis
  - Temporal patterns
  - Thread structure analysis
- Creates summary statistics CSV

**Output**: `results/` directory with PNG figures and CSV tables

---

### 3. `advanced_analysis.py` - Statistical Analysis
**Purpose**: Advanced statistical tests and predictive modeling

**Usage**:
```bash
python advanced_analysis.py
```

**What it does**:
- Correlation analysis between article features and engagement
- Statistical hypothesis testing (ANOVA, Chi-square)
- Simple predictive model for comment counts
- Correlation matrix of engagement metrics

**Requirements**: Must run basic analysis first (or use `run_all_analysis.py`)

**Output**: Additional figures and statistical test results

---

### 4. `run_all_analysis.py` - Master Script
**Purpose**: Run everything in correct order

**Usage**:
```bash
python run_all_analysis.py
```

**What it does**:
- Runs basic analysis pipeline
- Runs advanced statistical analysis
- Generates all figures and reports
- Provides summary of all outputs

**When to use**: Recommended for generating all results at once

---

## Workflow Recommendation

1. **First time setup**:
   ```bash
   # Check data
   python explore_data.py
   
   # Run full analysis
   python run_all_analysis.py
   ```

2. **Regenerating specific figures**:
   ```bash
   # Just basic analysis
   python analyze_news_data.py
   
   # Just advanced analysis (after basic)
   python advanced_analysis.py
   ```

3. **Custom analysis**:
   ```python
   from analyze_news_data import NewsDataAnalyzer
   
   analyzer = NewsDataAnalyzer(data_dir='.')
   analyzer.load_data()
   # ... customize as needed
   ```

---

## Output Structure

```
results/
├── portal_comparison.png              # Basic portal stats
├── engagement_distribution.png        # Engagement patterns
├── category_analysis.png              # Category breakdown
├── temporal_analysis.png              # Time-based patterns
├── thread_structure.png               # Thread analysis
├── summary_statistics.csv             # Summary table
├── article_engagement_correlation.png # Feature correlations
├── statistical_tests.txt              # Test results
├── engagement_prediction.png          # Prediction model
└── correlation_matrix.png            # Correlation heatmap
```

---

## Key Metrics Computed

### Engagement Metrics
- **Total Reactions**: Sum of all reaction types
- **Engagement Ratio**: Positive / (Positive + Negative) reactions
- **Polarization Score**: Measure of like/dislike imbalance (0-1 scale)

### Thread Metrics
- **Thread Depth**: Maximum nesting level
- **Thread Size**: Number of comments in thread
- **Reply Ratio**: Percentage of comments that are replies

### Portal Statistics
- Articles count
- Comments count
- Average comments per article
- Unique users
- Category distribution

---

## Requirements

Install dependencies:
```bash
pip install -r requirements.txt
```

Required packages:
- pandas >= 1.5.0
- numpy >= 1.23.0
- matplotlib >= 3.6.0
- seaborn >= 0.12.0
- scipy (for advanced analysis)
- scikit-learn (for advanced analysis)

---

## Notes

- All figures are generated at 300 DPI for publication quality
- Scripts handle missing data gracefully
- Different portals may have different reaction types
- Thread depth calculation may take time for large datasets
- Datetime parsing handles various formats automatically

---

## Troubleshooting

**No data files found**:
- Ensure CSV files are in the current directory
- Check file names match: `{portal}_articles.csv` and `{portal}_comments.csv`

**Memory errors**:
- Large datasets may require more RAM
- Consider processing portals individually

**Missing columns**:
- Scripts handle missing columns gracefully
- Some portals may have different column structures

**Slow performance**:
- Thread depth calculation is the slowest operation
- Consider processing smaller subsets for testing

---

## For Conference Paper

All figures are publication-ready with:
- High resolution (300 DPI)
- Clear labels and legends
- Consistent styling
- Professional formatting

Use figures directly in your paper or modify styling as needed for journal requirements.
