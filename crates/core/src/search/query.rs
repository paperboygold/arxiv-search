use serde::{Deserialize, Serialize};

/// arXiv category codes (reference)
/// Full list: https://arxiv.org/category_taxonomy
pub mod categories {
    // Computer Science
    pub const CS_AI: &str = "cs.AI"; // Artificial Intelligence
    pub const CS_CC: &str = "cs.CC"; // Computational Complexity
    pub const CS_CR: &str = "cs.CR"; // Cryptography and Security
    pub const CS_CV: &str = "cs.CV"; // Computer Vision
    pub const CS_CY: &str = "cs.CY"; // Computers and Society
    pub const CS_DB: &str = "cs.DB"; // Databases
    pub const CS_DC: &str = "cs.DC"; // Distributed, Parallel, Cluster Computing
    pub const CS_DL: &str = "cs.DL"; // Digital Libraries
    pub const CS_DM: &str = "cs.DM"; // Discrete Mathematics
    pub const CS_DS: &str = "cs.DS"; // Data Structures and Algorithms
    pub const CS_ET: &str = "cs.ET"; // Emerging Technologies
    pub const CS_GL: &str = "cs.GL"; // General Literature
    pub const CS_GR: &str = "cs.GR"; // Graphics
    pub const CS_GT: &str = "cs.GT"; // Computer Science and Game Theory
    pub const CS_HC: &str = "cs.HC"; // Human-Computer Interaction
    pub const CS_IR: &str = "cs.IR"; // Information Retrieval
    pub const CS_IT: &str = "cs.IT"; // Information Theory
    pub const CS_LG: &str = "cs.LG"; // Machine Learning
    pub const CS_LO: &str = "cs.LO"; // Logic in Computer Science
    pub const CS_MA: &str = "cs.MA"; // Multiagent Systems
    pub const CS_MM: &str = "cs.MM"; // Multimedia
    pub const CS_MS: &str = "cs.MS"; // Mathematical Software
    pub const CS_NA: &str = "cs.NA"; // Numerical Analysis
    pub const CS_NE: &str = "cs.NE"; // Neural and Evolutionary Computing
    pub const CS_NI: &str = "cs.NI"; // Networking and Internet Architecture
    pub const CS_OH: &str = "cs.OH"; // Other Computer Science
    pub const CS_OS: &str = "cs.OS"; // Operating Systems
    pub const CS_PL: &str = "cs.PL"; // Programming Languages
    pub const CS_RO: &str = "cs.RO"; // Robotics
    pub const CS_SC: &str = "cs.SC"; // Symbolic Computation
    pub const CS_SD: &str = "cs.SD"; // Sound
    pub const CS_SE: &str = "cs.SE"; // Software Engineering
    pub const CS_SY: &str = "cs.SY"; // Systems and Control
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchQuery {
    pub keywords: Vec<String>,
    pub categories: Vec<String>,
    pub exclude_keywords: Vec<String>,
    pub date_range: Option<(String, String)>, // YYYY-MM format
    pub min_relevance: f32,                    // 0.0-1.0
}

impl Default for SearchQuery {
    fn default() -> Self {
        Self {
            keywords: vec![],
            categories: vec![],
            exclude_keywords: vec![],
            date_range: None,
            min_relevance: 0.5,
        }
    }
}

pub struct QueryBuilder {
    query: SearchQuery,
}

impl QueryBuilder {
    pub fn new() -> Self {
        Self {
            query: SearchQuery::default(),
        }
    }

    /// Add keywords (title/abstract search)
    pub fn keyword(mut self, keyword: &str) -> Self {
        self.query.keywords.push(keyword.to_lowercase());
        self
    }

    /// Add multiple keywords at once
    pub fn keywords(mut self, keywords: &[&str]) -> Self {
        self.query
            .keywords
            .extend(keywords.iter().map(|k| k.to_lowercase()));
        self
    }

    /// Exclude papers matching these keywords
    pub fn exclude(mut self, keyword: &str) -> Self {
        self.query.exclude_keywords.push(keyword.to_lowercase());
        self
    }

    /// Filter by arXiv category
    pub fn category(mut self, category: &str) -> Self {
        self.query.categories.push(category.to_string());
        self
    }

    /// Filter by multiple categories (OR logic)
    pub fn categories(mut self, cats: &[&str]) -> Self {
        self.query
            .categories
            .extend(cats.iter().map(|c| c.to_string()));
        self
    }

    /// Set date range (YYYY-MM format)
    pub fn since(mut self, date: &str) -> Self {
        let until = self.query.date_range.as_ref().map(|(_, u)| u.clone());
        self.query.date_range = Some((date.to_string(), until.unwrap_or_else(|| "2025-12".to_string())));
        self
    }

    /// Set minimum relevance score (0.0-1.0)
    pub fn min_relevance(mut self, score: f32) -> Self {
        self.query.min_relevance = score.clamp(0.0, 1.0);
        self
    }

    pub fn build(self) -> SearchQuery {
        self.query
    }
}

impl Default for QueryBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// Presets are dynamically generated from metadata using MetadataAnalyzer.
// See: crate::metadata::analyzer::MetadataAnalyzer::detect_themes()
//
// To get presets for your domain:
// 1. Load Kaggle metadata: let papers = KaggleLoader::load_from_file("arxiv-metadata.json")?;
// 2. Analyze: let analyzer = MetadataAnalyzer::new(papers);
// 3. Get themes: let presets = analyzer.detect_themes();
// 4. Or create custom: let preset = analyzer.preset_from_keywords(&["ddos", "detection"], None);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_query_builder() {
        let query = QueryBuilder::new()
            .keywords(&["ddos", "security"])
            .category(categories::CS_CR)
            .min_relevance(0.7)
            .build();

        assert_eq!(query.keywords.len(), 2);
        assert_eq!(query.categories.len(), 1);
        assert_eq!(query.min_relevance, 0.7);
    }

    #[test]
    fn test_exclude_keywords() {
        let query = QueryBuilder::new()
            .keywords(&["security"])
            .exclude("machine learning")
            .build();

        assert_eq!(query.exclude_keywords.len(), 1);
    }

    #[test]
    fn test_date_range() {
        let query = QueryBuilder::new()
            .keywords(&["security"])
            .since("2024-01")
            .build();

        assert!(query.date_range.is_some());
    }
}
