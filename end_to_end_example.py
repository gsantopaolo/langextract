"""
End-to-End Financial Extraction Pipeline

This script demonstrates a complete production workflow:
1. Fetch SEC filing from EDGAR
2. Extract structured financial information with LangExtract
3. Analyze and export results to multiple formats
4. Generate interactive visualizations

Usage:
    python end_to_end_example.py --ticker NVDA --filing-type 10-K
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List
import pandas as pd

import langextract as lx
from sec_filing_fetcher import SECFilingFetcher
from financial_extraction import (
    create_financial_extraction_prompt,
    create_financial_examples,
    extract_from_text,
    analyze_extractions,
    save_and_visualize,
)


def export_to_dataframe(result: lx.data.AnnotatedDocument) -> pd.DataFrame:
    """
    Convert extractions to a pandas DataFrame for analysis.
    
    Args:
        result: AnnotatedDocument with extractions
        
    Returns:
        DataFrame with extraction data
    """
    data = []
    for extraction in result.extractions:
        row = {
            "extraction_class": extraction.extraction_class,
            "extraction_text": extraction.extraction_text,
            "start_pos": extraction.char_interval.start_pos if extraction.char_interval else None,
            "end_pos": extraction.char_interval.end_pos if extraction.char_interval else None,
        }
        
        # Add attributes as separate columns
        if extraction.attributes:
            for key, value in extraction.attributes.items():
                row[f"attr_{key}"] = value
        
        data.append(row)
    
    return pd.DataFrame(data)


def export_financial_metrics(result: lx.data.AnnotatedDocument, output_file: str) -> None:
    """
    Export financial metrics to a structured CSV file.
    
    This creates a clean table of metrics suitable for financial analysis tools.
    
    Args:
        result: AnnotatedDocument with extractions
        output_file: Path to output CSV file
    """
    # Filter for financial metrics
    metrics = [
        e for e in result.extractions
        if e.extraction_class == "financial_metric"
    ]
    
    metrics_data = []
    for metric in metrics:
        metrics_data.append({
            "metric_name": metric.attributes.get("metric_name", "unknown"),
            "value": metric.attributes.get("value", ""),
            "unit": metric.attributes.get("unit", ""),
            "time_period": metric.attributes.get("time_period", ""),
            "segment": metric.attributes.get("segment", ""),
            "text": metric.extraction_text,
        })
    
    df = pd.DataFrame(metrics_data)
    df.to_csv(output_file, index=False)
    print(f"✓ Exported {len(metrics_data)} financial metrics to: {output_file}")


def export_risk_factors(result: lx.data.AnnotatedDocument, output_file: str) -> None:
    """
    Export risk factors to a structured JSON file.
    
    Args:
        result: AnnotatedDocument with extractions
        output_file: Path to output JSON file
    """
    risks = [
        e for e in result.extractions
        if e.extraction_class == "risk_factor"
    ]
    
    risk_data = []
    for risk in risks:
        risk_data.append({
            "category": risk.attributes.get("risk_category", "unknown"),
            "description": risk.extraction_text,
            "geographic_exposure": risk.attributes.get("geographic_exposure", ""),
            "position": {
                "start": risk.char_interval.start_pos if risk.char_interval else None,
                "end": risk.char_interval.end_pos if risk.char_interval else None,
            }
        })
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(risk_data, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Exported {len(risk_data)} risk factors to: {output_file}")


def export_business_segments(result: lx.data.AnnotatedDocument, output_file: str) -> None:
    """
    Export business segment information with associated metrics.
    
    Args:
        result: AnnotatedDocument with extractions
        output_file: Path to output CSV file
    """
    segments = [
        e for e in result.extractions
        if e.extraction_class == "business_segment"
    ]
    
    segment_data = []
    for segment in segments:
        segment_data.append({
            "segment_name": segment.extraction_text,
            "segment_type": segment.attributes.get("segment_type", ""),
        })
    
    # Also get metrics associated with segments
    metrics = [
        e for e in result.extractions
        if e.extraction_class == "financial_metric" and e.attributes.get("segment")
    ]
    
    for metric in metrics:
        segment_data.append({
            "segment_name": metric.attributes.get("segment", ""),
            "segment_type": "metric",
            "metric_name": metric.attributes.get("metric_name", ""),
            "value": metric.attributes.get("value", ""),
            "unit": metric.attributes.get("unit", ""),
        })
    
    df = pd.DataFrame(segment_data)
    df.to_csv(output_file, index=False)
    print(f"✓ Exported {len(segment_data)} business segment entries to: {output_file}")


def generate_summary_report(
    result: lx.data.AnnotatedDocument,
    ticker: str,
    filing_type: str,
    output_file: str
) -> None:
    """
    Generate a human-readable summary report in Markdown format.
    
    Args:
        result: AnnotatedDocument with extractions
        ticker: Stock ticker
        filing_type: Type of filing (10-K, 10-Q)
        output_file: Path to output markdown file
    """
    from collections import Counter
    
    entity_counts = Counter(e.extraction_class for e in result.extractions)
    
    # Extract key information
    company_info = [e for e in result.extractions if e.extraction_class == "company_info"]
    metrics = [e for e in result.extractions if e.extraction_class == "financial_metric"]
    risks = [e for e in result.extractions if e.extraction_class == "risk_factor"]
    segments = [e for e in result.extractions if e.extraction_class == "business_segment"]
    
    # Build markdown report
    lines = [
        f"# Financial Analysis Report: {ticker} {filing_type}",
        "",
        "## Executive Summary",
        "",
        f"- **Total Extractions**: {len(result.extractions)}",
        f"- **Financial Metrics**: {len(metrics)}",
        f"- **Risk Factors**: {len(risks)}",
        f"- **Business Segments**: {len(segments)}",
        f"- **Document Length**: {len(result.text):,} characters",
        "",
        "## Company Information",
        ""
    ]
    
    for info in company_info[:5]:
        info_type = info.attributes.get("info_type", "unknown")
        lines.append(f"- **{info_type}**: {info.extraction_text}")
    
    lines.extend([
        "",
        "## Key Financial Metrics",
        "",
        "| Metric | Value | Period |",
        "|--------|-------|--------|"
    ])
    
    for metric in metrics[:10]:
        name = metric.attributes.get("metric_name", "unknown")
        value = metric.attributes.get("value", "")
        unit = metric.attributes.get("unit", "")
        period = metric.attributes.get("time_period", "")
        lines.append(f"| {name} | {value} {unit} | {period} |")
    
    lines.extend([
        "",
        "## Business Segments",
        ""
    ])
    
    for segment in segments[:5]:
        segment_type = segment.attributes.get("segment_type", "unknown")
        lines.append(f"- **{segment.extraction_text}** ({segment_type})")
    
    lines.extend([
        "",
        "## Risk Factor Categories",
        ""
    ])
    
    risk_categories = Counter(
        r.attributes.get("risk_category", "unknown") for r in risks
    )
    for category, count in risk_categories.most_common():
        lines.append(f"- **{category}**: {count} risks identified")
    
    lines.extend([
        "",
        "## Sample Risk Factors",
        ""
    ])
    
    for risk in risks[:3]:
        category = risk.attributes.get("risk_category", "unknown")
        text = risk.extraction_text[:150] + "..." if len(risk.extraction_text) > 150 else risk.extraction_text
        lines.append(f"- **[{category}]** {text}")
    
    lines.extend([
        "",
        "---",
        "",
        f"*Report generated by LangExtract financial extraction pipeline*",
    ])
    
    report_text = "\n".join(lines)
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(report_text)
    
    print(f"✓ Generated summary report: {output_file}")


def main():
    """
    Main execution function with CLI argument parsing.
    """
    parser = argparse.ArgumentParser(
        description="End-to-end financial information extraction from SEC filings"
    )
    parser.add_argument(
        "--ticker",
        type=str,
        default="NVDA",
        help="Stock ticker symbol (default: NVDA)"
    )
    parser.add_argument(
        "--filing-type",
        type=str,
        default="10-K",
        choices=["10-K", "10-Q"],
        help="Type of SEC filing (default: 10-K)"
    )
    parser.add_argument(
        "--year",
        type=str,
        default=None,
        help="Filter by year (e.g., 2024)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-3-flash-preview",
        help="LLM model ID (default: gemini-3-flash-preview)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output",
        help="Output directory for results (default: ./output)"
    )
    parser.add_argument(
        "--skip-fetch",
        action="store_true",
        help="Skip fetching and use cached filing"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Check for API key
    api_key = os.getenv("LANGEXTRACT_API_KEY")
    if not api_key:
        print("❌ Error: LANGEXTRACT_API_KEY not set in environment")
        print("   Set it with: export LANGEXTRACT_API_KEY='your-key-here'")
        sys.exit(1)
    
    print("="*80)
    print(f"Financial Extraction Pipeline: {args.ticker} {args.filing_type}")
    print("="*80 + "\n")
    
    # Step 1: Fetch SEC filing
    filing_text = ""
    
    if not args.skip_fetch:
        print("Step 1: Fetching SEC Filing")
        print("-" * 40)
        
        fetcher = SECFilingFetcher(cache_dir=str(output_dir / "sec_cache"))
        filing_text = fetcher.get_filing(
            ticker=args.ticker,
            filing_type=args.filing_type,
            year=args.year
        )
        
        if not filing_text:
            print("❌ Failed to fetch filing")
            sys.exit(1)
        
        print(f"✓ Retrieved filing: {len(filing_text):,} characters\n")
    else:
        # Try to load from cache
        cache_file = output_dir / "sec_cache" / f"{args.ticker}_{args.filing_type}.txt"
        if cache_file.exists():
            filing_text = cache_file.read_text(encoding="utf-8")
            print(f"Using cached filing: {cache_file}\n")
        else:
            print(f"❌ Cached file not found: {cache_file}")
            sys.exit(1)
    
    # Step 2: Extract structured information
    print("Step 2: Extracting Structured Information")
    print("-" * 40)
    
    result = extract_from_text(
        text=filing_text,
        model_id=args.model,
        api_key=api_key,
        extraction_passes=2,
        max_workers=10,
    )
    
    # Step 3: Analyze results
    print("\nStep 3: Analyzing Results")
    print("-" * 40)
    analyze_extractions(result)
    
    # Step 4: Save and export
    print("\nStep 4: Exporting Results")
    print("-" * 40)
    
    base_name = f"{args.ticker}_{args.filing_type}_extraction"
    
    # Save JSONL and HTML visualization
    save_and_visualize(result, output_dir=str(output_dir), base_name=base_name)
    
    # Export to various formats
    export_financial_metrics(
        result,
        str(output_dir / f"{base_name}_metrics.csv")
    )
    
    export_risk_factors(
        result,
        str(output_dir / f"{base_name}_risks.json")
    )
    
    export_business_segments(
        result,
        str(output_dir / f"{base_name}_segments.csv")
    )
    
    generate_summary_report(
        result,
        ticker=args.ticker,
        filing_type=args.filing_type,
        output_file=str(output_dir / f"{base_name}_summary.md")
    )
    
    # Export full DataFrame
    df = export_to_dataframe(result)
    df_path = output_dir / f"{base_name}_full.csv"
    df.to_csv(df_path, index=False)
    print(f"✓ Exported full extraction data to: {df_path}")
    
    print("\n" + "="*80)
    print("Pipeline Complete!")
    print("="*80)
    print(f"\nResults saved to: {output_dir}/")
    print("\nGenerated files:")
    print(f"  - {base_name}.jsonl (structured extraction data)")
    print(f"  - {base_name}_visualization.html (interactive visualization)")
    print(f"  - {base_name}_metrics.csv (financial metrics table)")
    print(f"  - {base_name}_risks.json (risk factors)")
    print(f"  - {base_name}_segments.csv (business segments)")
    print(f"  - {base_name}_summary.md (markdown report)")
    print(f"  - {base_name}_full.csv (complete extraction dataframe)")


if __name__ == "__main__":
    main()
