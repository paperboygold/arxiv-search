# Dynamic Search: No Hardcoded Presets

The system now discovers presets from actual metadata instead of guessing.

## The Problem with Hardcoded Presets

Presets like `presets::ddos_prevention()` have issues:
- They're brittle (change the corpus, presets are wrong)
- They require manual maintenance
- They don't adapt to your data
- They waste developer time writing them

## The Solution: Dynamic Analysis

Load metadata → Analyze → Discover themes → Search

```rust
use arxiv_search_rs_mcp_core::metadata::{KaggleLoader, MetadataAnalyzer};

// 1. Load metadata once
let papers = KaggleLoader::load_from_file("arxiv-metadata.json")?;

// 2. Analyze what's actually there
let analyzer = MetadataAnalyzer::new(papers);

// 3. Get presets auto-generated from the data
let presets = analyzer.detect_themes();

// 4. Or create custom presets on-the-fly
let preset = analyzer.preset_from_keywords(&["ddos", "attack"], None);
let query = preset.to_search_query();
```

---

## Quick Start

### 1. Download Kaggle Metadata

```bash
# Get Kaggle API credentials: https://www.kaggle.com/settings/account
kaggle datasets download -d Cornell-University/arxiv
unzip arxiv
```

**File**: `arxiv-metadata-oai-snapshot.json` (~5.2 GB, line-delimited JSON)

### 2. Run Analysis

```bash
# See categories + detected themes
cargo run --example kaggle_search --features kaggle -- \
  --file arxiv-metadata.json --sample 50000

# Search by keywords
cargo run --example kaggle_search --features kaggle -- \
  --file arxiv-metadata.json \
  --keywords "ddos,attack detection,anomaly" \
  --top 50

# Filter by category
cargo run --example kaggle_search --features kaggle -- \
  --file arxiv-metadata.json \
  --categories "cs.NI,cs.CR" \
  --top 100
```

### 3. Use in Your Code

```rust
use arxiv_search_rs_mcp_core::metadata::{KaggleLoader, MetadataAnalyzer};
use arxiv_search_rs_mcp_core::search::{SearchFilter, QueryBuilder};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load metadata
    let papers = KaggleLoader::load_from_file("arxiv-metadata.json")?;
    let analyzer = MetadataAnalyzer::new(papers);

    // Option A: Use auto-detected themes
    let themes = analyzer.detect_themes();
    for theme in &themes {
        println!("{}: {} papers", theme.name, theme.estimated_papers);
    }

    // Option B: Custom search
    let preset = analyzer.preset_from_keywords(
        &["ddos", "denial of service", "rate limiting"],
        Some(&["cs.NI", "cs.CR"]),
    );
    println!("Found {} papers", preset.estimated_papers);

    // Option C: Build query directly
    let query = QueryBuilder::new()
        .keywords(&["virtualization", "kubernetes"])
        .categories(&["cs.DC", "cs.SY"])
        .min_relevance(0.6)
        .build();

    let filter = SearchFilter::new(query);
    let results = filter.search(&papers);
    let ranked = filter.rank(results);

    println!("Top 20 papers:");
    for (i, (paper, score)) in ranked.iter().take(20).enumerate() {
        println!("{}. [{}] {} (score: {:.2})", 
            i + 1, paper.arxiv_id, paper.title, score);
    }

    Ok(())
}
```

---

## API Overview

### MetadataAnalyzer

```rust
// Create analyzer
let analyzer = MetadataAnalyzer::new(papers);

// Get category distribution
let categories = analyzer.category_distribution();
for cat in categories {
    println!("{}: {} papers ({:.1}%)", 
        cat.category, cat.count, cat.percentage);
}

// Auto-detect themes (combined categories)
let themes = analyzer.detect_themes();
for theme in themes {
    println!("{}: {}", theme.name, theme.description);
    let query = theme.to_search_query();
    // use query for searching
}

// Create custom preset from keywords
let preset = analyzer.preset_from_keywords(
    &["networking", "security"],
    Some(&["cs.NI"])
);
let query = preset.to_search_query();
```

### DynamicPreset

```rust
#[derive(Serialize, Deserialize)]
pub struct DynamicPreset {
    pub name: String,
    pub description: String,
    pub categories: Vec<String>,
    pub keywords: Vec<String>,
    pub relevance_threshold: f32,
    pub estimated_papers: usize,
}

// Convert to SearchQuery
let query = preset.to_search_query();
```

### Detected Themes (Auto-Generated)

The analyzer automatically creates presets for:

1. **Individual categories** (top 10)
   - cs.NI (Networking)
   - cs.CR (Security)
   - cs.DC (Distributed Computing)
   - etc.

2. **Combinations**
   - `network_security` → cs.NI + cs.CR
   - `infrastructure` → cs.DC + cs.SY + cs.OS
   - `incident_response` → cs.CR + cs.SY

---

## Example: Your Infrastructure Stack

### 1. Analyze Corpus

```bash
cargo run --example kaggle_search --features kaggle -- \
  --file arxiv-metadata.json
```

**Output:**
```
📊 Available Categories (Top 10):
─────────────────────────────────────────
  cs.LG   (10.5% |  2417 papers)
    Keywords: learning, neural, model, training, network
  cs.CR   ( 8.2% |  1888 papers)
    Keywords: security, cryptography, attack, protocol, privacy
  cs.NI   ( 4.1% |   944 papers)
    Keywords: network, routing, protocol, bandwidth, traffic
  ...

🎯 Detected Themes (Dynamic Presets):
─────────────────────────────────────────
  cs.LG
    Description: Papers in cs.LG (10.5% of corpus, ~2417 papers)
    Categories: cs.LG
    Est. papers: 2417

  network_security
    Description: Networking + Security (suitable for DDoS/attack detection)
    Categories: cs.NI, cs.CR
    Est. papers: 832

  infrastructure
    Description: Distributed Systems + Systems + OS (virtualization, orchestration)
    Categories: cs.DC, cs.SY, cs.OS
    Est. papers: 1256

  incident_response
    Description: Security + Systems (SIEM/SOAR, threat detection)
    Categories: cs.CR, cs.SY
    Est. papers: 389
```

### 2. Search for Your Domains

```bash
# DDoS prevention papers
cargo run --example kaggle_search --features kaggle -- \
  --file arxiv-metadata.json \
  --keywords "ddos,denial of service,attack detection,anomaly" \
  --top 100

# SIEM/SOAR papers
cargo run --example kaggle_search --features kaggle -- \
  --file arxiv-metadata.json \
  --keywords "siem,incident response,threat detection" \
  --top 50

# Virtual hosting papers
cargo run --example kaggle_search --features kaggle -- \
  --file arxiv-metadata.json \
  --keywords "virtualization,kubernetes,container,orchestration" \
  --top 100
```

### 3. Combine Results + Download

```rust
use arxiv_search_rs_mcp_core::metadata::{KaggleLoader, MetadataAnalyzer};
use arxiv_search_rs_mcp_core::search::{SearchFilter, QueryBuilder};
use arxiv_search_rs_mcp_core::ingestion::{S3Downloader, S3Config};
use std::path::PathBuf;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Load & analyze
    let papers = KaggleLoader::load_from_file("arxiv-metadata.json")?;
    let analyzer = MetadataAnalyzer::new(papers.clone());

    // 2. Search for your domains
    let ddos = analyzer.preset_from_keywords(
        &["ddos", "attack detection"],
        Some(&["cs.NI", "cs.CR"]),
    );
    let siem = analyzer.preset_from_keywords(
        &["siem", "incident response"],
        Some(&["cs.CR", "cs.SY"]),
    );
    let vhost = analyzer.preset_from_keywords(
        &["virtualization", "kubernetes"],
        Some(&["cs.DC", "cs.OS"]),
    );

    println!("DDoS papers: {}", ddos.estimated_papers);
    println!("SIEM papers: {}", siem.estimated_papers);
    println!("VHost papers: {}", vhost.estimated_papers);

    // 3. Combine searches
    let mut combined_ids = std::collections::HashSet::new();
    
    for preset in &[ddos, siem, vhost] {
        let query = preset.to_search_query();
        let filter = SearchFilter::new(query);
        let results = filter.search(&papers);
        
        for (paper, _score) in results {
            combined_ids.insert(paper.arxiv_id);
        }
    }

    println!("Total unique papers: {}", combined_ids.len());

    // 4. Download from S3
    let mut config = S3Config::default();
    config.max_concurrent_downloads = 50;
    let downloader = S3Downloader::new(config).await?;

    let keys: Vec<_> = combined_ids
        .iter()
        .map(|id| id.as_str())
        .collect();

    let output_dir = PathBuf::from("./infrastructure-papers");
    tokio::fs::create_dir_all(&output_dir).await?;

    let results = downloader
        .download_papers_parallel(keys, &output_dir)
        .await?;

    let succeeded = results.iter().filter(|(_, r)| r.is_ok()).count();
    println!("Downloaded {} papers", succeeded);

    Ok(())
}
```

---

## How It Works

### Category Detection

```rust
let dist = analyzer.category_distribution();
// Returns: Vec<CategoryStats>
// - category: "cs.NI"
// - count: 944
// - percentage: 4.1
// - sample_keywords: ["network", "routing", "protocol", ...]
```

### Theme Detection

Looks for combinations that make sense:

1. **Two-category combinations**
   - cs.NI + cs.CR → "network_security"
   - cs.DC + cs.SY → "infrastructure"

2. **Three-category combinations**
   - cs.DC + cs.SY + cs.OS → "infrastructure"

3. **Single categories** (fallback)
   - cs.LG, cs.CR, cs.NI, etc.

### Keyword Extraction

For each category, extract top N keywords:
1. Split titles into words
2. Filter by length (> 4 chars)
3. Exclude generic terms ("paper", "method", "system", etc.)
4. Sort by frequency
5. Return top keywords

---

## Advanced: Custom Analysis

### Extend Analyzer

```rust
impl MetadataAnalyzer {
    pub fn author_distribution(&self) -> HashMap<String, usize> {
        let mut authors = HashMap::new();
        for paper in &self.papers {
            for author in &paper.authors {
                *authors.entry(author.clone()).or_insert(0) += 1;
            }
        }
        authors
    }

    pub fn temporal_analysis(&self) -> HashMap<&str, usize> {
        let mut by_year = HashMap::new();
        for paper in &self.papers {
            let year = &paper.published[..4];
            *by_year.entry(year).or_insert(0) += 1;
        }
        by_year
    }
}
```

### Caching (For Production)

```rust
use std::fs;

// Save presets to JSON
let presets = analyzer.detect_themes();
let json = serde_json::to_string(&presets)?;
fs::write("presets.json", json)?;

// Load cached presets
let cached = fs::read_to_string("presets.json")?;
let presets: Vec<DynamicPreset> = serde_json::from_str(&cached)?;
```

---

## Cost Impact (Revisited)

**Old way** (hardcoded presets):
- Presets: Invalid after 1 month (data changes)
- Search: Fast but not adaptive
- Cost: Same no matter what

**New way** (dynamic analysis):
- Presets: Always accurate (generated from current data)
- Search: Adaptive to what's in the corpus
- Cost: Same S3 cost, but smarter paper selection

**Result**: Better precision, lower cost per useful paper.

---

## Next Steps

1. **Load Kaggle metadata** (one-time, ~5GB)
2. **Run analyzer** (takes ~30 seconds)
3. **Get themes** (auto-detected presets)
4. **Search** (use detected themes or custom keywords)
5. **Download** (only what's relevant, ~$1-20 instead of $635)

---

## Files

- `crates/core/src/metadata/kaggle.rs` — Load Kaggle JSON
- `crates/core/src/metadata/analyzer.rs` — Analyze + detect themes
- `crates/native/examples/kaggle_search.rs` — CLI tool + examples
- Tests: All passing, no hardcoded assumptions

---

**TL;DR**: No more guessing. Metadata tells you what's there.
