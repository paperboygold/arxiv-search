/// Example: Load Kaggle metadata and discover presets dynamically
///
/// Usage:
///   cargo run --example kaggle_search --features kaggle -- --file arxiv-metadata.json --top 20
///   cargo run --example kaggle_search --features kaggle -- --file arxiv-metadata.json --keywords "ddos,attack" --top 50
///   cargo run --example kaggle_search --features kaggle -- --file arxiv-metadata.json --categories "cs.NI,cs.CR" --top 100

#[cfg(feature = "kaggle")]
use arxiv_search_rs_mcp_core::metadata::{KaggleLoader, MetadataAnalyzer};

#[cfg(feature = "kaggle")]
use arxiv_search_rs_mcp_core::search::{QueryBuilder, SearchFilter};

fn main() {
    #[cfg(feature = "kaggle")]
    {
        let args: Vec<String> = std::env::args().collect();

        if args.len() < 2 {
            print_usage();
            return;
        }

        match args.iter().position(|a| a == "--file") {
            Some(idx) => {
                if idx + 1 < args.len() {
                    let file_path = &args[idx + 1];

                    match run_search(&args, file_path) {
                        Ok(_) => {},
                        Err(e) => eprintln!("Error: {}", e),
                    }
                } else {
                    eprintln!("Error: --file requires a path");
                }
            }
            None => {
                eprintln!("Error: --file <path> is required");
                print_usage();
            }
        }
    }

    #[cfg(not(feature = "kaggle"))]
    {
        eprintln!("Error: kaggle feature not enabled");
        eprintln!("Build with: cargo run --example kaggle_search --features kaggle");
    }
}

#[cfg(feature = "kaggle")]
fn run_search(args: &[String], file_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let sample_size = args
        .iter()
        .position(|a| a == "--sample")
        .and_then(|idx| args.get(idx + 1))
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(10000);

    println!("📚 Loading Kaggle metadata (sample: {} papers)...", sample_size);
    let papers = KaggleLoader::load_sample(file_path, sample_size)?;
    println!("✓ Loaded {} papers\n", papers.len());

    // Show category distribution
    let analyzer = MetadataAnalyzer::new(papers.clone());
    println!("📊 Available Categories (Top 10):");
    println!("─────────────────────────────────────────");
    let cat_dist = analyzer.category_distribution();
    for stat in cat_dist.iter().take(10) {
        println!(
            "  {} ({:5.1}% | {:6} papers)",
            stat.category, stat.percentage, stat.count
        );
        println!("    Keywords: {}", stat.sample_keywords.join(", "));
    }
    println!();

    // Detect themes
    println!("🎯 Detected Themes (Dynamic Presets):");
    println!("─────────────────────────────────────────");
    let presets = analyzer.detect_themes();
    for preset in &presets {
        println!("  {}", preset.name);
        println!("    Description: {}", preset.description);
        println!("    Categories: {}", preset.categories.join(", "));
        println!("    Est. papers: {}", preset.estimated_papers);
        println!();
    }

    // User-specified search
    if let Some(idx) = args.iter().position(|a| a == "--keywords") {
        if let Some(keywords_str) = args.get(idx + 1) {
            let keywords: Vec<&str> = keywords_str.split(',').collect();
            println!("🔍 Searching for: {}", keywords.join(", "));
            println!("─────────────────────────────────────────");

            let preset = analyzer.preset_from_keywords(&keywords, None);
            let query = preset.to_search_query();
            let filter = SearchFilter::new(query);

            let results = filter.search(&papers);
            let ranked = filter.rank(results);

            let top = args
                .iter()
                .position(|a| a == "--top")
                .and_then(|idx| args.get(idx + 1))
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(20);

            println!("Found {} matching papers (showing top {})\n", ranked.len(), top);
            for (i, (paper, score)) in ranked.iter().take(top).enumerate() {
                println!("{}. [{}] {} (score: {:.2})", i + 1, paper.arxiv_id, paper.title, score);
                println!("   Categories: {}", paper.categories.join(", "));
                println!();
            }
        }
    } else if let Some(idx) = args.iter().position(|a| a == "--categories") {
        if let Some(cats_str) = args.get(idx + 1) {
            let categories: Vec<&str> = cats_str.split(',').collect();
            println!("🔍 Searching in categories: {}", categories.join(", "));
            println!("─────────────────────────────────────────");

            let mut query_builder = QueryBuilder::new();
            for cat in categories {
                query_builder = query_builder.category(cat);
            }
            let query = query_builder.min_relevance(0.3).build();
            let filter = SearchFilter::new(query);

            let results = filter.search(&papers);
            let ranked = filter.rank(results);

            let top = args
                .iter()
                .position(|a| a == "--top")
                .and_then(|idx| args.get(idx + 1))
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(20);

            println!("Found {} papers (showing top {})\n", ranked.len(), top);
            for (i, (paper, score)) in ranked.iter().take(top).enumerate() {
                println!("{}. [{}] {} (score: {:.2})", i + 1, paper.arxiv_id, paper.title, score);
                println!("   Categories: {}", paper.categories.join(", "));
                println!();
            }
        }
    } else {
        // Just show stats
        println!("\n📋 To search, use:");
        println!("  --keywords <keywords>    Search by comma-separated keywords");
        println!("  --categories <cats>      Filter by comma-separated arXiv categories");
        println!("  --top <n>                Show top N results (default: 20)");
        println!();
        println!("Examples:");
        println!(
            "  cargo run --example kaggle_search --features kaggle -- \\
        \n    --file arxiv-metadata.json --keywords 'ddos,attack detection' --top 50"
        );
        println!(
            "  cargo run --example kaggle_search --features kaggle -- \\
        \n    --file arxiv-metadata.json --categories 'cs.NI,cs.CR' --top 100"
        );
    }

    Ok(())
}

fn print_usage() {
    println!("Kaggle Metadata Search: Discover papers and dynamic presets");
    println!();
    println!("USAGE:");
    println!("  cargo run --example kaggle_search --features kaggle -- --file <path> [OPTIONS]");
    println!();
    println!("OPTIONS:");
    println!("  --file <path>           Path to Kaggle arxiv-metadata.json");
    println!("  --sample <n>            Load only first N papers (default: 10000)");
    println!("  --keywords <kw1,kw2>    Search by comma-separated keywords");
    println!("  --categories <c1,c2>    Filter by arXiv categories");
    println!("  --top <n>               Show top N results (default: 20)");
    println!();
    println!("WORKFLOW:");
    println!("  1. Download Kaggle dataset:");
    println!("     kaggle datasets download -d Cornell-University/arxiv");
    println!();
    println!("  2. Extract arxiv-metadata.json");
    println!();
    println!("  3. Run this example:");
    println!("     cargo run --example kaggle_search --features kaggle -- \\");
    println!("       --file arxiv-metadata.json");
    println!();
    println!("  4. System will:");
    println!("     - Analyze available categories");
    println!("     - Detect dynamic themes/presets");
    println!("     - Allow custom searches");
}
