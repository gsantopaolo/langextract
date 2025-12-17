"""
Financial Information Extraction using LangExtract

This script demonstrates how to extract structured financial data from
SEC 10-K/10-Q filings using LangExtract with source grounding.

Key features:
- Extract company info, financial metrics, risk factors, business segments
- Source grounding: every extraction is mapped to exact text location
- Interactive HTML visualization of extractions
- Support for multiple LLM backends (Gemini, OpenAI, local Ollama)
"""

import os
import textwrap
from typing import List, Optional
import langextract as lx


def create_financial_extraction_prompt() -> str:
    """
    Create a comprehensive prompt for financial document extraction.
    
    The prompt guides the LLM to extract structured financial information
    while maintaining exact text correspondence.
    """
    return textwrap.dedent("""\
        Extract financial information from SEC filings and earnings reports.
        
        Extract the following entity types in order of appearance:
        - company_info: Company name, ticker symbol, CIK number
        - financial_metric: Revenue, operating income, net income, EBITDA, EPS, etc.
        - risk_factor: Risk disclosures and concerns
        - business_segment: Business units, product lines, geographic regions
        - time_period: Fiscal years, quarters, reporting periods
        - metric_change: Year-over-year changes, growth rates, comparisons
        
        Critical rules:
        1. Use EXACT text from the document for extraction_text
        2. Do NOT paraphrase or summarize
        3. Do NOT overlap text spans
        4. Extract entities in order of appearance
        5. Provide meaningful attributes for context
        
        For financial_metric entities, include these attributes:
        - metric_name: The specific metric (e.g., "revenue", "net_income")
        - value: The numerical value
        - unit: Currency or percentage (e.g., "USD_millions", "percent")
        - time_period: The reporting period
        
        For company_info entities, include:
        - info_type: "company_name", "ticker", "cik", "sector"
        
        For risk_factor entities, include:
        - risk_category: Type of risk (e.g., "market", "operational", "regulatory")
    """)


def create_financial_examples() -> List[lx.data.ExampleData]:
    """
    Create few-shot examples to guide the extraction model.
    
    High-quality examples are crucial for accurate extraction.
    These examples demonstrate the expected format and level of detail.
    """
    examples = [
        lx.data.ExampleData(
            text=textwrap.dedent("""\
                NVIDIA Corporation (NASDAQ: NVDA) reported fiscal year 2024 results.
                Revenue for fiscal 2024 was $60.9 billion, up 126% from $27.0 billion
                in fiscal 2023. Operating income increased to $33.0 billion compared
                to $4.2 billion in the prior year. Net income reached $29.8 billion."""),
            extractions=[
                lx.data.Extraction(
                    extraction_class="company_info",
                    extraction_text="NVIDIA Corporation",
                    attributes={
                        "info_type": "company_name"
                    }
                ),
                lx.data.Extraction(
                    extraction_class="company_info",
                    extraction_text="NASDAQ: NVDA",
                    attributes={
                        "info_type": "ticker"
                    }
                ),
                lx.data.Extraction(
                    extraction_class="time_period",
                    extraction_text="fiscal year 2024",
                    attributes={
                        "period_type": "annual"
                    }
                ),
                lx.data.Extraction(
                    extraction_class="financial_metric",
                    extraction_text="Revenue for fiscal 2024 was $60.9 billion",
                    attributes={
                        "metric_name": "revenue",
                        "value": "60.9",
                        "unit": "USD_billions",
                        "time_period": "fiscal_2024"
                    }
                ),
                lx.data.Extraction(
                    extraction_class="metric_change",
                    extraction_text="up 126% from $27.0 billion in fiscal 2023",
                    attributes={
                        "change_type": "year_over_year_growth",
                        "percentage": "126",
                        "comparison_period": "fiscal_2023"
                    }
                ),
                lx.data.Extraction(
                    extraction_class="financial_metric",
                    extraction_text="Operating income increased to $33.0 billion",
                    attributes={
                        "metric_name": "operating_income",
                        "value": "33.0",
                        "unit": "USD_billions",
                        "time_period": "fiscal_2024"
                    }
                ),
                lx.data.Extraction(
                    extraction_class="financial_metric",
                    extraction_text="$4.2 billion in the prior year",
                    attributes={
                        "metric_name": "operating_income",
                        "value": "4.2",
                        "unit": "USD_billions",
                        "time_period": "fiscal_2023"
                    }
                ),
                lx.data.Extraction(
                    extraction_class="financial_metric",
                    extraction_text="Net income reached $29.8 billion",
                    attributes={
                        "metric_name": "net_income",
                        "value": "29.8",
                        "unit": "USD_billions",
                        "time_period": "fiscal_2024"
                    }
                ),
            ]
        ),
        lx.data.ExampleData(
            text=textwrap.dedent("""\
                The Data Center segment generated $47.5 billion in revenue, representing
                78% of total revenue. Gaming revenue was $10.4 billion. The company faces
                risks related to export controls impacting sales to China."""),
            extractions=[
                lx.data.Extraction(
                    extraction_class="business_segment",
                    extraction_text="Data Center segment",
                    attributes={
                        "segment_type": "product_line"
                    }
                ),
                lx.data.Extraction(
                    extraction_class="financial_metric",
                    extraction_text="$47.5 billion in revenue",
                    attributes={
                        "metric_name": "segment_revenue",
                        "value": "47.5",
                        "unit": "USD_billions",
                        "segment": "data_center"
                    }
                ),
                lx.data.Extraction(
                    extraction_class="financial_metric",
                    extraction_text="78% of total revenue",
                    attributes={
                        "metric_name": "revenue_percentage",
                        "value": "78",
                        "unit": "percent",
                        "segment": "data_center"
                    }
                ),
                lx.data.Extraction(
                    extraction_class="business_segment",
                    extraction_text="Gaming",
                    attributes={
                        "segment_type": "product_line"
                    }
                ),
                lx.data.Extraction(
                    extraction_class="financial_metric",
                    extraction_text="Gaming revenue was $10.4 billion",
                    attributes={
                        "metric_name": "segment_revenue",
                        "value": "10.4",
                        "unit": "USD_billions",
                        "segment": "gaming"
                    }
                ),
                lx.data.Extraction(
                    extraction_class="risk_factor",
                    extraction_text="risks related to export controls impacting sales to China",
                    attributes={
                        "risk_category": "regulatory",
                        "geographic_exposure": "china"
                    }
                ),
            ]
        ),
    ]
    
    return examples


def extract_from_text(
    text: str,
    model_id: str = "gemini-3-flash-preview",
    api_key: Optional[str] = None,
    extraction_passes: int = 2,
    max_workers: int = 10,
) -> lx.data.AnnotatedDocument:
    """
    Extract financial information from text using LangExtract.
    
    Args:
        text: Input text (can be raw text or URL)
        model_id: LLM model to use (gemini-3-flash-preview, gpt-4o, gemma2:2b)
        api_key: API key (if not set in environment)
        extraction_passes: Number of extraction passes for higher recall
        max_workers: Parallel workers for chunked processing
        
    Returns:
        AnnotatedDocument with extracted entities and source positions
    """
    prompt = create_financial_extraction_prompt()
    examples = create_financial_examples()
    
    print(f"Extracting financial information using {model_id}...")
    print(f"Text length: {len(text):,} characters")
    
    result = lx.extract(
        text_or_documents=text,
        prompt_description=prompt,
        examples=examples,
        model_id=model_id,
        api_key=api_key,
        extraction_passes=extraction_passes,
        max_workers=max_workers,
        max_char_buffer=2000,  # Chunk size for long documents
    )
    
    print(f"\n✓ Extracted {len(result.extractions)} entities")
    return result


def analyze_extractions(result: lx.data.AnnotatedDocument) -> None:
    """
    Analyze and display extraction statistics.
    
    This function provides insights into what was extracted,
    showing entity type breakdown and key metrics found.
    """
    from collections import Counter
    
    # Count by entity type
    entity_counts = Counter(e.extraction_class for e in result.extractions)
    
    print("\n" + "="*60)
    print("EXTRACTION SUMMARY")
    print("="*60)
    
    print(f"\nTotal entities: {len(result.extractions)}")
    print(f"Document length: {len(result.text):,} characters")
    
    print("\nEntity Type Breakdown:")
    print("-" * 40)
    for entity_type, count in entity_counts.most_common():
        percentage = (count / len(result.extractions)) * 100
        print(f"  {entity_type}: {count} ({percentage:.1f}%)")
    
    # Show sample financial metrics
    print("\nSample Financial Metrics:")
    print("-" * 40)
    metrics = [e for e in result.extractions if e.extraction_class == "financial_metric"]
    for metric in metrics[:5]:
        metric_name = metric.attributes.get("metric_name", "unknown")
        value = metric.attributes.get("value", "N/A")
        unit = metric.attributes.get("unit", "")
        print(f"  {metric_name}: {value} {unit}")
        print(f"    → \"{metric.extraction_text[:60]}...\"")
    
    # Show risk factors
    risks = [e for e in result.extractions if e.extraction_class == "risk_factor"]
    if risks:
        print(f"\nRisk Factors Found: {len(risks)}")
        print("-" * 40)
        for risk in risks[:3]:
            category = risk.attributes.get("risk_category", "unknown")
            print(f"  [{category}] \"{risk.extraction_text[:60]}...\"")


def save_and_visualize(
    result: lx.data.AnnotatedDocument,
    output_dir: str = ".",
    base_name: str = "financial_extraction"
) -> None:
    """
    Save extractions to JSONL and generate interactive HTML visualization.
    
    The JSONL format is portable and can be loaded into pandas, Excel, etc.
    The HTML visualization allows interactive exploration of extractions.
    """
    import os
    
    # Save to JSONL
    jsonl_path = os.path.join(output_dir, f"{base_name}.jsonl")
    lx.io.save_annotated_documents([result], output_name=jsonl_path, output_dir=".")
    print(f"\n✓ Saved extractions to: {jsonl_path}")
    
    # Generate HTML visualization
    html_content = lx.visualize(jsonl_path)
    html_path = os.path.join(output_dir, f"{base_name}_visualization.html")
    
    with open(html_path, "w", encoding="utf-8") as f:
        if hasattr(html_content, 'data'):
            f.write(html_content.data)  # Jupyter/Colab format
        else:
            f.write(html_content)
    
    print(f"✓ Generated visualization: {html_path}")
    print(f"\n→ Open {html_path} in your browser to explore extractions")


def main():
    """
    Main execution function with example usage.
    """
    # Example financial text (shortened NVIDIA earnings excerpt)
    sample_text = textwrap.dedent("""\
        NVIDIA Corporation (NASDAQ: NVDA) today reported revenue for the fourth quarter
        ended January 28, 2024, of $22.1 billion, up 22% from the previous quarter and
        up 265% from a year ago.
        
        For fiscal 2024, revenue was $60.9 billion, up 126% from the prior year.
        GAAP earnings per diluted share for the quarter were $4.93, up 33% from the 
        previous quarter and up 486% from a year ago. Non-GAAP earnings per diluted
        share were $5.16, up 28% from the previous quarter and up 486% from a year ago.
        
        "Accelerated computing and generative AI have hit the tipping point," said
        Jensen Huang, founder and CEO of NVIDIA. "Demand is surging worldwide across
        companies, industries and nations."
        
        The company's Data Center revenue reached $18.4 billion, up 27% from the
        previous quarter and up 409% from a year ago. Gaming revenue was $2.9 billion,
        up 56% from the previous quarter and up 56% from a year ago.
        
        NVIDIA's outlook for the first quarter of fiscal 2025 is revenue of $24.0
        billion, plus or minus 2%. GAAP and non-GAAP gross margins are expected to
        be 76.3% and 77.0%, respectively, plus or minus 50 basis points.
        
        Risk Factors: The company faces risks related to supply chain constraints,
        geopolitical tensions affecting sales to China, and increased competition
        in the AI accelerator market.
    """)
    
    # Get API key from environment or .env file
    api_key = os.getenv("LANGEXTRACT_API_KEY")
    if not api_key:
        print("⚠️  Warning: LANGEXTRACT_API_KEY not set in environment")
        print("   Set it with: export LANGEXTRACT_API_KEY='your-key-here'")
        print("   Or add to .env file")
        return
    
    # Run extraction
    result = extract_from_text(
        text=sample_text,
        model_id="gemini-3-flash-preview",
        api_key=api_key,
        extraction_passes=2,
        max_workers=5,
    )
    
    # Analyze results
    analyze_extractions(result)
    
    # Save and visualize
    save_and_visualize(result, output_dir=".", base_name="nvidia_earnings_extraction")
    
    print("\n" + "="*60)
    print("Extraction complete!")
    print("="*60)


if __name__ == "__main__":
    main()
