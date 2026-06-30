"""Command-line entry point for the Airbnb pipeline learning project."""

from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv

from .analytics import build_listing_summary, build_neighbourhood_summary, save_analytics_outputs
from .clean import clean_listings, clean_reviews
from .enrich import enrich_reviews
from .ingest import load_listings, load_reviews, merge_data
from .llm import get_openai_client

# Loading .env at startup keeps the CLI beginner-friendly.
load_dotenv()


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the pipeline."""
    parser = argparse.ArgumentParser(description="Run the Airbnb + LLM enrichment pipeline.")
    parser.add_argument("--sample", type=int, default=None, help="Optional number of reviews to enrich.")
    parser.add_argument(
        "--listings-path",
        type=Path,
        default=Path("data/listings.csv"),
        help="Path to the listings CSV.",
    )
    parser.add_argument(
        "--reviews-path",
        type=Path,
        default=Path("data/reviews.csv"),
        help="Path to the reviews CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Directory where Parquet outputs will be saved.",
    )
    return parser.parse_args()


def run_pipeline(listings_path: Path, reviews_path: Path, output_dir: Path, sample: int | None = None) -> None:
    """Execute the full ingest -> clean -> enrich -> analytics workflow."""
    print("🏠 Starting Airbnb pipeline with LLM enrichment...")

    print("📥 Loading source data...")
    listings_raw = load_listings(listings_path)
    reviews_raw = load_reviews(reviews_path)
    print(f"   • Listings loaded: {len(listings_raw)} rows")
    print(f"   • Reviews loaded: {len(reviews_raw)} rows")

    print("🧼 Cleaning data...")
    listings_clean = clean_listings(listings_raw)
    reviews_clean = clean_reviews(reviews_raw)
    print(f"   • Clean listings: {len(listings_clean)} rows")
    print(f"   • Clean reviews: {len(reviews_clean)} rows")

    # This early merge is a quick sanity check that the two sources align before we
    # spend time or money on enrichment.
    merged_preview = merge_data(listings_clean, reviews_clean)
    print(f"🔗 Joined dataset preview shape: {merged_preview.shape}")

    print("🤖 Preparing LLM client...")
    client = get_openai_client()
    if client is None:
        print("   • No OPENAI_API_KEY found, so a local fallback will simulate structured outputs.")
    else:
        print("   • OpenAI client ready. Enrichment will use JSON mode.")

    reviews_to_process: int = sample if sample is not None else len(reviews_clean)
    print(f"✨ Enriching up to {reviews_to_process} reviews...")
    enriched_reviews = enrich_reviews(reviews_clean, client=client, sample=sample)
    print(f"   • Enriched reviews: {len(enriched_reviews)} rows")
    print(f"   • LLM errors captured: {enriched_reviews['llm_error'].notna().sum()}")

    print("📊 Building analytical summaries...")
    listing_summary = build_listing_summary(listings_clean, enriched_reviews)
    neighbourhood_summary = build_neighbourhood_summary(listing_summary)
    print(f"   • Listing summary rows: {len(listing_summary)}")
    print(f"   • Neighbourhood summary rows: {len(neighbourhood_summary)}")

    print("💾 Saving outputs...")
    output_dir.mkdir(parents=True, exist_ok=True)
    enriched_reviews_path = output_dir / "enriched_reviews.parquet"
    enriched_reviews.to_parquet(enriched_reviews_path, index=False)
    listing_path, neighbourhood_path = save_analytics_outputs(
        listing_summary_df=listing_summary,
        neighbourhood_summary_df=neighbourhood_summary,
        output_dir=output_dir,
    )

    print("✅ Pipeline complete!")
    print(f"   • Enriched reviews saved to: {enriched_reviews_path}")
    print(f"   • Listing summary saved to: {listing_path}")
    print(f"   • Neighbourhood summary saved to: {neighbourhood_path}")
    print("\n🌟 Top neighbourhoods by average sentiment:")
    print(neighbourhood_summary.head(5).to_string(index=False))


def main() -> None:
    """CLI wrapper so the module can be run with `python -m src.pipeline`."""
    args = parse_args()

    if args.sample is not None and args.sample <= 0:
        raise ValueError("--sample must be a positive integer when provided.")

    run_pipeline(
        listings_path=args.listings_path,
        reviews_path=args.reviews_path,
        output_dir=args.output_dir,
        sample=args.sample,
    )


if __name__ == "__main__":
    main()
