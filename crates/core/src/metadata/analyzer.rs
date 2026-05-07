use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use crate::search::query::SearchQuery;
use crate::search::filter::PaperMetadata;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CategoryStats {
    pub category: String,
    pub count: usize,
    pub percentage: f32,
    pub sample_keywords: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DynamicPreset {
    pub name: String,
    pub description: String,
    pub categories: Vec<String>,
    pub keywords: Vec<String>,
    pub relevance_threshold: f32,
    pub estimated_papers: usize,
}

impl DynamicPreset {
    pub fn to_search_query(&self) -> SearchQuery {
        SearchQuery {
            keywords: self.keywords.clone(),
            categories: self.categories.clone(),
            exclude_keywords: vec![],
            date_range: None,
            min_relevance: self.relevance_threshold,
        }
    }
}

pub struct MetadataAnalyzer {
    papers: Vec<PaperMetadata>,
}

impl MetadataAnalyzer {
    pub fn new(papers: Vec<PaperMetadata>) -> Self {
        Self { papers }
    }

    /// Get all available categories and their frequencies
    pub fn category_distribution(&self) -> Vec<CategoryStats> {
        let mut cat_counts: HashMap<String, usize> = HashMap::new();

        for paper in &self.papers {
            for cat in &paper.categories {
                *cat_counts.entry(cat.clone()).or_insert(0) += 1;
            }
        }

        let total = self.papers.len() as f32;
        let mut stats: Vec<_> = cat_counts
            .into_iter()
            .map(|(cat, count)| {
                let percentage = (count as f32 / total) * 100.0;
                let sample_keywords = self.keywords_for_category(&cat, 5);
                CategoryStats {
                    category: cat,
                    count,
                    percentage,
                    sample_keywords,
                }
            })
            .collect();

        stats.sort_by(|a, b| b.count.cmp(&a.count));
        stats
    }

    /// Get most common keywords in a category
    fn keywords_for_category(&self, category: &str, count: usize) -> Vec<String> {
        let mut keyword_freq: HashMap<String, usize> = HashMap::new();

        for paper in &self.papers {
            if !paper.categories.iter().any(|c| c.starts_with(category)) {
                continue;
            }

            // Extract keywords from title (words > 4 chars, non-generic)
            for word in paper.title.split_whitespace() {
                let word = word.to_lowercase();
                if word.len() > 4 && !is_common_word(&word) {
                    *keyword_freq.entry(word).or_insert(0) += 1;
                }
            }
        }

        let mut keywords: Vec<_> = keyword_freq.into_iter().collect();
        keywords.sort_by(|a, b| b.1.cmp(&a.1));
        keywords
            .iter()
            .take(count)
            .map(|(k, _)| k.clone())
            .collect()
    }

    /// Detect domain themes based on keywords and categories
    pub fn detect_themes(&self) -> Vec<DynamicPreset> {
        let cat_stats = self.category_distribution();
        let mut presets = Vec::new();

        // Create presets for major categories
        for stat in cat_stats.iter().take(10) {
            let preset = DynamicPreset {
                name: stat.category.clone(),
                description: format!(
                    "Papers in {} ({:.1}% of corpus, ~{} papers)",
                    stat.category, stat.percentage, stat.count
                ),
                categories: vec![stat.category.clone()],
                keywords: stat.sample_keywords.clone(),
                relevance_threshold: 0.5,
                estimated_papers: stat.count,
            };
            presets.push(preset);
        }

        // Create combined presets for common domain combinations
        if self.has_categories(&["cs.NI", "cs.CR"]) {
            presets.push(DynamicPreset {
                name: "network_security".to_string(),
                description: "Networking + Security (suitable for DDoS/attack detection)".to_string(),
                categories: vec!["cs.NI".to_string(), "cs.CR".to_string()],
                keywords: self.combined_keywords(&["cs.NI", "cs.CR"], 8),
                relevance_threshold: 0.6,
                estimated_papers: self.count_papers_in_categories(&["cs.NI", "cs.CR"]),
            });
        }

        if self.has_categories(&["cs.DC", "cs.SY", "cs.OS"]) {
            presets.push(DynamicPreset {
                name: "infrastructure".to_string(),
                description: "Distributed Systems + Systems + OS (virtualization, orchestration)".to_string(),
                categories: vec!["cs.DC".to_string(), "cs.SY".to_string(), "cs.OS".to_string()],
                keywords: self.combined_keywords(&["cs.DC", "cs.SY", "cs.OS"], 8),
                relevance_threshold: 0.5,
                estimated_papers: self.count_papers_in_categories(&["cs.DC", "cs.SY", "cs.OS"]),
            });
        }

        if self.has_categories(&["cs.CR", "cs.SY"]) {
            presets.push(DynamicPreset {
                name: "incident_response".to_string(),
                description: "Security + Systems (SIEM/SOAR, threat detection)".to_string(),
                categories: vec!["cs.CR".to_string(), "cs.SY".to_string()],
                keywords: self.combined_keywords(&["cs.CR", "cs.SY"], 8),
                relevance_threshold: 0.6,
                estimated_papers: self.count_papers_in_categories(&["cs.CR", "cs.SY"]),
            });
        }

        presets
    }

    /// Build a dynamic preset from search terms
    pub fn preset_from_keywords(
        &self,
        keywords: &[&str],
        categories: Option<&[&str]>,
    ) -> DynamicPreset {
        let keywords_lower: Vec<String> = keywords.iter().map(|k| k.to_lowercase()).collect();

        // Find papers matching keywords
        let matching = self
            .papers
            .iter()
            .filter(|p| {
                let text = format!("{} {}", p.title.to_lowercase(), p.abstract_text.to_lowercase());
                keywords_lower
                    .iter()
                    .any(|k| text.contains(k))
            })
            .count();

        let cat_filters = categories
            .map(|cats| cats.iter().map(|c| c.to_string()).collect())
            .unwrap_or_default();

        DynamicPreset {
            name: keywords.join("_").replace(' ', "_"),
            description: format!("Dynamic preset for: {}", keywords.join(", ")),
            categories: cat_filters,
            keywords: keywords_lower,
            relevance_threshold: 0.5,
            estimated_papers: matching,
        }
    }

    // Helper functions
    fn has_categories(&self, categories: &[&str]) -> bool {
        let available: Vec<_> = self
            .papers
            .iter()
            .flat_map(|p| p.categories.iter())
            .collect();

        categories.iter().all(|cat| available.iter().any(|c| c == cat))
    }

    fn count_papers_in_categories(&self, categories: &[&str]) -> usize {
        self.papers
            .iter()
            .filter(|p| {
                p.categories
                    .iter()
                    .any(|c| categories.iter().any(|cat| c.starts_with(cat)))
            })
            .count()
    }

    fn combined_keywords(&self, categories: &[&str], limit: usize) -> Vec<String> {
        let mut keyword_freq: HashMap<String, usize> = HashMap::new();

        for paper in &self.papers {
            if !paper
                .categories
                .iter()
                .any(|c| categories.iter().any(|cat| c.starts_with(cat)))
            {
                continue;
            }

            for word in paper.title.split_whitespace() {
                let word = word.to_lowercase();
                if word.len() > 4 && !is_common_word(&word) {
                    *keyword_freq.entry(word).or_insert(0) += 1;
                }
            }
        }

        let mut keywords: Vec<_> = keyword_freq.into_iter().collect();
        keywords.sort_by(|a, b| b.1.cmp(&a.1));
        keywords
            .iter()
            .take(limit)
            .map(|(k, _)| k.clone())
            .collect()
    }
}

/// Common English words to exclude from keyword extraction
fn is_common_word(word: &str) -> bool {
    matches!(
        word,
        "paper" | "method" | "approach" | "system" | "study" | "analysis" | "model" | "data"
            | "using" | "based" | "novel" | "efficient" | "fast" | "learning" | "network"
            | "algorithm" | "framework" | "architecture" | "performance" | "optimization"
            | "implementation" | "evaluation" | "results" | "research" | "propose" | "new"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dynamic_preset_creation() {
        let papers = vec![
            PaperMetadata {
                arxiv_id: "2401.00001v1".to_string(),
                title: "DDoS Detection Using Machine Learning".to_string(),
                authors: vec!["Alice".to_string()],
                abstract_text: "A novel approach to detecting distributed denial of service attacks"
                    .to_string(),
                categories: vec!["cs.NI".to_string(), "cs.CR".to_string()],
                published: "2024-01-01".to_string(),
                updated: "2024-01-01".to_string(),
                pdf_url: "https://...".to_string(),
                s3_key: None,
            },
        ];

        let analyzer = MetadataAnalyzer::new(papers);
        let preset = analyzer.preset_from_keywords(&["ddos", "detection"], None);

        assert_eq!(preset.keywords.len(), 2);
        assert!(preset.estimated_papers > 0);
    }

    #[test]
    fn test_category_distribution() {
        let papers = vec![
            PaperMetadata {
                arxiv_id: "2401.00001v1".to_string(),
                title: "Title".to_string(),
                authors: vec![],
                abstract_text: "Abstract".to_string(),
                categories: vec!["cs.NI".to_string()],
                published: "2024-01-01".to_string(),
                updated: "2024-01-01".to_string(),
                pdf_url: "https://...".to_string(),
                s3_key: None,
            },
            PaperMetadata {
                arxiv_id: "2401.00002v1".to_string(),
                title: "Title2".to_string(),
                authors: vec![],
                abstract_text: "Abstract2".to_string(),
                categories: vec!["cs.NI".to_string(), "cs.CR".to_string()],
                published: "2024-01-01".to_string(),
                updated: "2024-01-01".to_string(),
                pdf_url: "https://...".to_string(),
                s3_key: None,
            },
        ];

        let analyzer = MetadataAnalyzer::new(papers);
        let dist = analyzer.category_distribution();

        assert!(dist.iter().any(|s| s.category == "cs.NI"));
        assert!(dist.iter().any(|s| s.category == "cs.CR"));
    }
}
