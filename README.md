# Academic Paper Corpus Analysis System

A comprehensive system for analyzing large collections of academic papers using AI to extract structured insights about research domains, techniques, innovations, and trends.

## Overview

This system processes 300+ academic papers in parallel and extracts:
- **Research domains and topics**
- **Techniques and methodologies used**
- **Claimed innovations and contributions**
- **Publication trends over time**
- **Technical complexity and impact scores**
- **Network relationships between concepts**

## Files Created

### Core Analysis Files
- **`paper_analysis_schema.json`** - JSON schema defining the structure of extracted data
- **`corpus_analyzer.py`** - Main script that processes papers in parallel using gptme
- **`corpus_visualizer.py`** - Creates charts, networks, and interactive visualizations

### This Guide
- **`README_Corpus_Analysis.md`** - This comprehensive usage guide

## Prerequisites

### 1. gptme Installation
```bash
# Install gptme if you haven't already
pipx install gptme

# Verify installation
gptme --version
```

### 2. Python Dependencies
```bash
# Essential dependencies (required)
pip install pandas matplotlib seaborn wordcloud

# Optional but recommended for advanced features
pip install plotly networkx PyPDF2 pdfplumber openpyxl

# Or install everything at once
pip install pandas matplotlib seaborn wordcloud plotly networkx PyPDF2 pdfplumber openpyxl
```

### 3. API Access
Make sure you have access to a good LLM provider:
- **Anthropic (recommended)**: Set `ANTHROPIC_API_KEY` environment variable
- **OpenAI**: Set `OPENAI_API_KEY` environment variable
- **Local models**: Configure gptme for local model serving

## Quick Start

### Step 1: Prepare Your Papers
Organize your 300 papers in a single directory:

papers/
├── paper1.pdf
├── paper2.pdf
├── subdirectory/
│   ├── paper3.pdf
│   └── paper4.txt
└── ...

**Supported formats:**
- PDF files (`.pdf`)
- Plain text files (`.txt`)
- Markdown files (`.md`)

### Step 2: Run the Analysis
```bash
# Basic analysis (uses Anthropic Claude by default)
python corpus_analyzer.py papers/

# With custom output directory
python corpus_analyzer.py papers/ -o my_analysis_results

# Using OpenAI instead
python corpus_analyzer.py papers/ -m openai/gpt-4o

# With more parallel workers (faster but more expensive)
python corpus_analyzer.py papers/ -w 8

# Resume interrupted analysis
python corpus_analyzer.py papers/ --resume
```

### Step 3: Generate Visualizations
```bash
# Create all visualizations and reports
python corpus_visualizer.py my_analysis_results

# With custom output directory  
python corpus_visualizer.py my_analysis_results -o visualizations
```

### Step 4: View Results
Open the generated HTML report:
```bash
# Open the main report in your browser
open visualizations/report.html
```

## Detailed Usage

### Analysis Phase

The `corpus_analyzer.py` script processes your papers in parallel:

```bash
# Full command with all options
python corpus_analyzer.py papers_directory \
  --output results_directory \
  --model anthropic/claude-sonnet-4-20250514 \
  --workers 4 \
  --resume
```

**Parameters:**
- `papers_directory`: Path to folder containing your papers
- `--output, -o`: Where to save results (default: `corpus_analysis_output`)
- `--model, -m`: LLM model to use (default: `anthropic/claude-sonnet-4-20250514`)
- `--workers, -w`: Number of parallel workers (default: 4)
- `--resume`: Skip already processed papers (useful for interrupted runs)

**What it does:**
1. Scans directory for PDF/text files
2. Extracts text from each paper
3. Sends each paper to gptme for analysis using the JSON schema
4. Saves individual results and generates summary statistics
5. Creates processing logs and error reports

**Expected time:** 
- ~1-2 minutes per paper with Claude Sonnet
- 300 papers ≈ 5-10 hours total (with 4 parallel workers)
- Cost: ~$50-100 for 300 papers with Claude

### Visualization Phase

The `corpus_visualizer.py` script creates comprehensive visualizations:

```bash
python corpus_visualizer.py results_directory -o visualizations
```

**Generated outputs:**
- Interactive HTML dashboard
- Static charts (PNG format)
- Word clouds of techniques/innovations
- Network diagrams showing relationships
- Data exports (CSV, Excel)
- Comprehensive HTML report

## Understanding the Outputs

### Directory Structure

results_directory/
├── corpus_summary.json          # Overall statistics and processing info
├── corpus_report.md            # Human-readable markdown report
├── processing.log              # Detailed processing logs
└── individual_results/         # JSON analysis for each paper
    ├── paper1.json
    ├── paper2.json
    └── ...

visualizations/
├── report.html                 # Main comprehensive report
├── dashboard.html             # Interactive dashboard
├── domain_analysis.png        # Research domain charts
├── techniques_wordcloud.png   # Word cloud of techniques
├── top_techniques.png         # Bar chart of popular techniques
├── innovation_network.png     # Network diagram
├── temporal_trends.png        # Time-series analysis
├── complexity_analysis.png    # Complexity/innovation scoring
├── corpus_data.csv           # Raw data export
├── corpus_analysis.xlsx      # Excel workbook with summaries
└── network_stats.json        # Network analysis statistics

### Key Files Explained

#### `corpus_summary.json`
Contains high-level statistics about your corpus:

```json
{
  "processing_summary": {
    "total_papers": 300,
    "successful": 285,
    "failed": 15,
    "success_rate": 0.95
  },
  "corpus_statistics": {
    "domains": {
      "Machine Learning": 45,
      "Computer Vision": 38,
      "Natural Language Processing": 32,
      ...
    },
    "techniques": {
      "Deep Learning": 67,
      "Convolutional Neural Networks": 34,
      "Transformer Architecture": 28,
      ...
    }
  }
}
```

#### Individual Analysis Files
Each paper gets a JSON file with structured analysis:

```json
{
  "metadata": {
    "title": "Novel Approach to Image Classification",
    "publication_year": 2023
  },
  "content_analysis": {
    "main_topic": "Computer Vision",
    "research_domain": "Image Classification"
  },
  "methodology": {
    "research_type": "experimental",
    "techniques_used": ["CNN", "Transfer Learning", "Data Augmentation"]
  },
  "innovations": {
    "novel_contributions": ["New architecture design", "Improved accuracy"],
    "novelty_level": "substantial"
  },
  "relevance_scores": {
    "innovation_score": 4,
    "technical_complexity": 3,
    "practical_impact": 4
  }
}
```

## Troubleshooting

### Common Issues

#### 1. PDF Text Extraction Fails
```bash
# Install better PDF processing
pip install pdfplumber PyPDF2

# Or manually convert PDFs to text
# Use tools like pdftotext, Adobe Acrobat, etc.
```

#### 2. gptme Not Found
```bash
# Make sure gptme is in your PATH
which gptme

# If not found, reinstall
pipx install gptme
```

#### 3. API Rate Limits
```bash
# Reduce parallel workers
python corpus_analyzer.py papers/ -w 2

# Or add delays between requests by modifying the script
```

#### 4. Out of Memory
```bash
# Reduce workers and process in smaller batches
python corpus_analyzer.py papers/ -w 1

# Or process subsets of papers
python corpus_analyzer.py papers/batch1/ -o results_batch1
python corpus_analyzer.py papers/batch2/ -o results_batch2
```

#### 5. Analysis Interrupted
```bash
# Always use --resume to continue where you left off
python corpus_analyzer.py papers/ --resume

# Check processing.log for details
tail -f corpus_analysis_output/processing.log
```

### Validation

#### Check Analysis Quality
```bash
# Look at a few individual results
cat results_directory/individual_results/paper1.json | jq .

# Check processing success rate
grep -c "Completed:" results_directory/processing.log
grep -c "Failed:" results_directory/processing.log
```

#### Verify Output Completeness
```bash
# Count processed vs source papers
ls papers/ | wc -l
ls results_directory/individual_results/ | wc -l
```

## Advanced Usage

### Custom Analysis Schema

You can modify `paper_analysis_schema.json` to capture different information:

```json
{
  "paper_analysis_schema": {
    "properties": {
      "your_custom_field": {
        "type": "array",
        "items": {"type": "string"}
      }
    }
  }
}
```

### Batch Processing

For very large corpora, process in batches:

```bash
#!/bin/bash
# Process in batches of 50 papers each

mkdir -p batches
find papers/ -name "*.pdf" | split -l 50 - batches/batch_

for batch_file in batches/batch_*; do
    batch_dir="batch_$(basename $batch_file)"
    mkdir -p "$batch_dir"
    
    while read paper; do
        cp "$paper" "$batch_dir/"
    done < "$batch_file"
    
    python corpus_analyzer.py "$batch_dir" -o "results_$batch_dir"
done
```

### Custom Models

Use different models for different types of analysis:

```bash
# Use GPT-4 for high-quality analysis
python corpus_analyzer.py papers/ -m openai/gpt-4

# Use local models for privacy
python corpus_analyzer.py papers/ -m local/llama-3.1-70b

# Use OpenRouter for model variety
python corpus_analyzer.py papers/ -m openrouter/anthropic/claude-3.5-sonnet
```

### Integration with Existing Workflows

#### Export to Statistical Software
```python
import pandas as pd

# Load the CSV data
df = pd.read_csv('visualizations/corpus_data.csv')

# Export for R analysis
df.to_csv('corpus_for_R.csv', index=False)

# Export for SPSS
df.to_excel('corpus_for_SPSS.xlsx', index=False)
```

#### API Integration
```python
import json

# Load summary for integration with other tools
with open('results_directory/corpus_summary.json') as f:
    summary = json.load(f)

# Extract specific metrics
total_papers = summary['processing_summary']['total_papers']
top_domain = list(summary['corpus_statistics']['domains'].keys())[0]
```

## Tips for Best Results

### 1. Paper Quality
- Ensure PDFs are text-searchable (not just images)
- Remove duplicate papers before analysis
- Use consistent file naming

### 2. Cost Optimization
```bash
# Start with a small sample to test
python corpus_analyzer.py papers_sample/ -o test_results

# Use cheaper models for initial exploration
python corpus_analyzer.py papers/ -m openai/gpt-4o-mini

# Process in batches to monitor costs
```

### 3. Quality Control
```bash
# Review failed papers
grep "Failed:" corpus_analysis_output/processing.log

# Manually check a few random results
shuf -n 5 corpus_analysis_output/individual_results/*.json | head -1 | xargs cat | jq .
```

### 4. Result Validation
- Cross-check a sample of results manually
- Look for consistent patterns in the data
- Verify domain classifications make sense
- Check that innovation scores seem reasonable

## Example Workflow

Here's a complete example workflow for analyzing 300 papers:

```bash
# 1. Setup
mkdir my_corpus_analysis
cd my_corpus_analysis

# Download the analysis tools (these files should be in your directory)
# - paper_analysis_schema.json
# - corpus_analyzer.py  
# - corpus_visualizer.py

# 2. Organize papers
mkdir papers
# Copy your 300 papers into papers/ directory

# 3. Install dependencies
pip install pandas matplotlib seaborn wordcloud plotly networkx PyPDF2 pdfplumber openpyxl

# 4. Run analysis (this will take several hours)
python corpus_analyzer.py papers/ -o analysis_results -w 4

# 5. Monitor progress
tail -f analysis_results/processing.log

# 6. Generate visualizations
python corpus_visualizer.py analysis_results -o visualizations

# 7. View results
open visualizations/report.html
```

## Expected Timeline

- **Setup**: 15-30 minutes
- **Analysis**: 5-10 hours for 300 papers (depends on model and workers)
- **Visualization**: 5-10 minutes  
- **Review**: Variable, but plan several hours to explore results

## Getting Help

If you encounter issues:

1. Check the `processing.log` file for detailed error messages
2. Try with a smaller sample first (5-10 papers)
3. Verify your API keys are set correctly
4. Check that gptme is working: `gptme --version`
5. Make sure all dependencies are installed

## Next Steps

After running the analysis, you can:

1. **Export data** for statistical analysis in R, SPSS, etc.
2. **Create custom visualizations** using the exported CSV data
3. **Filter and re-analyze** specific subsets of papers
4. **Compare with other corpora** using the same schema
5. **Build interactive web applications** using the JSON data

The system is designed to be flexible and extensible - you can modify the schema, add new analysis types, or integrate with other research tools as needed.
