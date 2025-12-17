# Financial Information Extraction with LangExtract

Complete code examples for the blog post: **"LangExtract: Production LLM-Powered Information Extraction with Source Grounding"**

## Overview

This repository contains production-ready code for extracting structured financial information from SEC filings and earnings reports using LangExtract.

## Features

- ✅ **Source Grounding**: Every extraction mapped to exact text position
- ✅ **SEC EDGAR Integration**: Fetch 10-K/10-Q filings automatically
- ✅ **Multi-Format Export**: JSONL, CSV, JSON, Markdown reports
- ✅ **Interactive Visualization**: HTML interface for exploring extractions
- ✅ **Production Ready**: Error handling, caching, rate limiting

## Installation

### Prerequisites

- Python 3.10 or higher
- LangExtract API key (Gemini, OpenAI, or local Ollama)

### Setup

```bash
# Clone or copy these files
git clone https://github.com/gsantopaolo/langextract.git

cd langextract

# Install dependencies
pip install -r requirements.txt

# create a .env file, see .env.example

# source the .env
```

## Quick Start

### Example 1: Basic Financial Extraction

Extract financial information from sample text:

```bash
python financial_extraction.py
```

This will:
- Extract company info, financial metrics, risk factors
- Generate interactive HTML visualization
- Save results to JSONL format

### Example 2: Fetch and Process SEC Filing

Fetch a real SEC filing and extract structured data:

```bash
python end_to_end_example.py --ticker NVDA --filing-type 10-K
```

This will:
1. Fetch NVIDIA's latest 10-K from SEC EDGAR
2. Extract structured financial information
3. Export to multiple formats (CSV, JSON, Markdown)
4. Generate interactive visualization

### Example 3: Different Company and Filing Type

```bash
# Microsoft 10-K
python end_to_end_example.py --ticker MSFT --filing-type 10-K

# Apple 10-Q (quarterly report)
python end_to_end_example.py --ticker AAPL --filing-type 10-Q --year 2024
```

## Code Structure

### `financial_extraction.py`

Core extraction logic demonstrating:
- Prompt engineering for financial documents
- Few-shot examples for accurate extraction
- Entity types: company_info, financial_metric, risk_factor, business_segment
- Analysis and visualization functions

**Key Functions:**
- `create_financial_extraction_prompt()`: Structured prompt for LLM
- `create_financial_examples()`: High-quality few-shot examples
- `extract_from_text()`: Main extraction function
- `analyze_extractions()`: Statistical analysis of results
- `save_and_visualize()`: Export and visualization

### `sec_filing_fetcher.py`

SEC EDGAR integration:
- Fetch filings by ticker and type
- HTML parsing and text extraction
- Caching for faster repeated runs
- Rate limiting to respect SEC guidelines

**Key Classes:**
- `SECFilingFetcher`: Main fetcher class with methods for company lookup and filing retrieval

### `end_to_end_example.py`

Complete production pipeline:
- Fetch → Extract → Analyze → Export
- Multiple output formats
- Summary report generation
- CLI interface for easy usage

**Exports:**
- JSONL: Raw extraction data
- CSV: Financial metrics table
- JSON: Risk factors with categories
- Markdown: Human-readable summary report
- HTML: Interactive visualization

## Output Examples

After running the pipeline, you'll get:

```
output/
├── NVDA_10-K_extraction.jsonl
├── NVDA_10-K_extraction_visualization.html
├── NVDA_10-K_extraction_metrics.csv
├── NVDA_10-K_extraction_risks.json
├── NVDA_10-K_extraction_segments.csv
├── NVDA_10-K_extraction_summary.md
└── NVDA_10-K_extraction_full.csv
```

## Extracted Entity Types

### `company_info`
- Company name, ticker symbol, CIK number
- Sector and industry information

**Attributes:**
- `info_type`: "company_name", "ticker", "cik", "sector"

### `financial_metric`
- Revenue, operating income, net income, EBITDA, EPS
- Growth rates, margins, ratios

**Attributes:**
- `metric_name`: Name of the metric
- `value`: Numerical value
- `unit`: Currency or percentage
- `time_period`: Reporting period
- `segment`: Business segment (optional)

### `risk_factor`
- Market risks, operational risks, regulatory risks
- Geographic exposures, competitive threats

**Attributes:**
- `risk_category`: Type of risk
- `geographic_exposure`: Affected regions (optional)

### `business_segment`
- Product lines, geographic regions
- Service categories

**Attributes:**
- `segment_type`: "product_line", "geography", "service"

### `time_period`
- Fiscal years, quarters, reporting dates

**Attributes:**
- `period_type`: "annual", "quarterly", "monthly"

### `metric_change`
- Year-over-year growth, sequential changes
- Percentage changes and comparisons

**Attributes:**
- `change_type`: Type of comparison
- `percentage`: Change percentage
- `comparison_period`: Base period

## Usage Tips

### API Key Management

**Option 1: Environment Variable**
```bash
export LANGEXTRACT_API_KEY="your-key"
```

**Option 2: .env File**
```bash
echo "LANGEXTRACT_API_KEY=your-key" > .env
```

**Option 3: Pass Directly** (not recommended for production)
```python
result = extract_from_text(text, api_key="your-key")
```

### Model Selection

> **⚠️ Note:** Model names and availability change frequently. The models listed below were current as of December 2024. Before running, check the latest available models at [Google AI Studio](https://aistudio.google.com) or the [Gemini API docs](https://ai.google.dev/gemini-api/docs/models/gemini). Experimental/preview models may be removed or renamed, and free tier quotas can change.

**Gemini Models** (Recommended)
```bash
python end_to_end_example.py --model gemini-3-flash-preview  # Fast, latest (preview)
python end_to_end_example.py --model gemini-3-pro-preview    # Higher quality (preview)
python end_to_end_example.py --model gemini-2.5-flash        # Stable fallback
```

**OpenAI Models**
```bash
# Install OpenAI support first
pip install langextract[openai]

python end_to_end_example.py --model gpt-4o
```

**Local Models with Ollama**
```bash
# Install and start Ollama first
ollama pull gemma2:2b
ollama serve

python end_to_end_example.py --model gemma2:2b
```

### Performance Tuning

For faster processing of long documents:

```python
result = extract_from_text(
    text=long_document,
    extraction_passes=1,      # Reduce from 2 to 1 for speed
    max_workers=20,           # Increase for more parallelism
    max_char_buffer=3000,     # Larger chunks = faster but less accurate
)
```

For higher accuracy:

```python
result = extract_from_text(
    text=long_document,
    extraction_passes=3,      # Multiple passes for better recall
    max_workers=10,           # Moderate parallelism
    max_char_buffer=1000,     # Smaller chunks = more accurate
)
```

## Advanced Usage

### Custom Entity Types

Extend the extraction schema by modifying `create_financial_extraction_prompt()`:

```python
prompt = textwrap.dedent("""\
    Extract financial information including:
    - company_info: Company details
    - financial_metric: Numerical metrics
    - risk_factor: Risk disclosures
    - business_segment: Business units
    - executive_mention: C-suite names and titles  # NEW
    - product_mention: Product names and launches  # NEW
    
    ... (rest of prompt)
""")
```

Then add corresponding examples in `create_financial_examples()`.

### Filtering Extractions

Post-process to filter specific entity types:

```python
# Get only financial metrics
metrics = [
    e for e in result.extractions
    if e.extraction_class == "financial_metric"
]

# Get revenue metrics only
revenue_metrics = [
    e for e in result.extractions
    if e.extraction_class == "financial_metric"
    and e.attributes.get("metric_name") == "revenue"
]

# Get risks from specific category
regulatory_risks = [
    e for e in result.extractions
    if e.extraction_class == "risk_factor"
    and e.attributes.get("risk_category") == "regulatory"
]
```

### Batch Processing Multiple Filings

Process multiple companies or time periods:

```python
from end_to_end_example import main as process_filing

tickers = ["NVDA", "MSFT", "AAPL", "GOOGL"]
filing_types = ["10-K", "10-Q"]

for ticker in tickers:
    for filing_type in filing_types:
        print(f"\nProcessing {ticker} {filing_type}...")
        # Run extraction pipeline
        # ... (call extraction functions)
```

## Troubleshooting

### Common Issues

**1. API Key Not Found**
```
❌ Error: LANGEXTRACT_API_KEY not set in environment
```
**Solution:** Set the environment variable or create a .env file

**2. SEC Filing Not Found**
```
No 10-K filings found for TICKER
```
**Solution:** Check ticker symbol or try a different year

**3. Rate Limiting**
```
Too Many Requests (429)
```
**Solution:** Add delays between requests or reduce `max_workers`

**4. Out of Memory**
```
MemoryError: Unable to allocate...
```
**Solution:** Process document in smaller chunks or reduce `max_char_buffer`

## Performance Benchmarks

Tested on NVIDIA 10-K (130 pages, ~150k characters):

| Model | Time | Cost | Extractions | Quality |
|-------|------|------|-------------|---------|
| gemini-3-flash-preview | 40s | $0.03 | 295 | Excellent |
| gemini-3-pro-preview | 75s | $0.08 | 310 | Superior |
| gemini-2.5-flash | 45s | $0.03 | 287 | Excellent |
| gpt-4o | 90s | $0.25 | 295 | Excellent |
| gemma2:2b (local) | 180s | Free | 251 | Good |

*Times include fetching, extraction, and export. Costs are approximate.*

## Citation

If you use this code in your research or production systems:

```bibtex
@software{langextract2025,
  title = {LangExtract: LLM-powered structured information extraction},
  author = {Goel, Akshay},
  year = {2025},
  url = {https://github.com/google/langextract},
  doi = {10.5281/zenodo.17015089}
}
```

## License

This code is provided as examples for the LangExtract blog post and is subject to Apache 2.0 License.

## Support

For issues with:
- **LangExtract library**: [GitHub Issues](https://github.com/google/langextract/issues)
- **These examples**: Open an issue or discussion on the blog post
- **SEC EDGAR API**: [SEC Developer Resources](https://www.sec.gov/os/accessing-edgar-data)

## Additional Resources

- [LangExtract Documentation](https://github.com/google/langextract)
- [SEC EDGAR API Guide](https://www.sec.gov/edgar/searchedgar/accessing-edgar-data.htm)
- [Gemini API Documentation](https://ai.google.dev/docs)
- [Blog Post: Production LLM-Powered Information Extraction](#)
