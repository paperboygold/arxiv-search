use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use crate::search::filter::PaperMetadata;

/// Kaggle arXiv metadata JSON format
/// Source: https://www.kaggle.com/datasets/Cornell-University/arxiv
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KagglePaperRecord {
    pub id: String,
    pub submitter: Option<String>,
    pub authors: String,
    pub title: String,
    pub comments: Option<String>,
    pub journal_ref: Option<String>,
    pub doi: Option<String>,
    #[serde(rename = "abstract")]
    pub abstract_text: String,
    pub categories: String,
    pub versions: Vec<KaggleVersion>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KaggleVersion {
    pub version: String,
    pub created: String,
}

impl KagglePaperRecord {
    /// Convert Kaggle format to our internal PaperMetadata
    pub fn to_paper_metadata(&self) -> PaperMetadata {
        let authors = self.authors.split(", ").map(|s| s.to_string()).collect();
        let categories = self
            .categories
            .split(' ')
            .map(|s| s.to_string())
            .collect();

        // Get first version date
        let published = self
            .versions
            .first()
            .map(|v| v.created.clone())
            .unwrap_or_else(|| "1900-01-01".to_string());

        let updated = self
            .versions
            .last()
            .map(|v| v.created.clone())
            .unwrap_or_else(|| published.clone());

        let s3_key = Some(PaperMetadata::s3_key_from_arxiv_id(&self.id));

        PaperMetadata {
            arxiv_id: self.id.clone(),
            title: self.title.clone(),
            authors,
            abstract_text: self.abstract_text.clone(),
            categories,
            published,
            updated,
            pdf_url: format!("https://arxiv.org/pdf/{}.pdf", self.id),
            s3_key,
        }
    }
}

pub struct KaggleLoader;

impl KaggleLoader {
    /// Load arXiv metadata from Kaggle JSON file (line-delimited)
    ///
    /// Kaggle dataset: https://www.kaggle.com/datasets/Cornell-University/arxiv
    /// File: arxiv-metadata-oai-snapshot.json
    ///
    /// Each line is a JSON object, so we parse line by line
    pub fn load_from_file<P: AsRef<Path>>(
        path: P,
    ) -> Result<Vec<PaperMetadata>, Box<dyn std::error::Error>> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let mut papers = Vec::new();

        for (line_num, line) in reader.lines().enumerate() {
            let line = line?;
            match serde_json::from_str::<KagglePaperRecord>(&line) {
                Ok(record) => papers.push(record.to_paper_metadata()),
                Err(e) => eprintln!("Warning: Failed to parse line {}: {}", line_num + 1, e),
            }
        }

        Ok(papers)
    }

    /// Load a sample from Kaggle JSON (for testing/preview)
    pub fn load_sample<P: AsRef<Path>>(
        path: P,
        max_papers: usize,
    ) -> Result<Vec<PaperMetadata>, Box<dyn std::error::Error>> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let mut papers = Vec::new();

        for line in reader.lines().take(max_papers) {
            let line = line?;
            match serde_json::from_str::<KagglePaperRecord>(&line) {
                Ok(record) => papers.push(record.to_paper_metadata()),
                Err(e) => eprintln!("Warning: Failed to parse record: {}", e),
            }
        }

        Ok(papers)
    }

    /// List available categories in the dataset
    pub fn analyze_categories<P: AsRef<Path>>(
        path: P,
        sample_size: usize,
    ) -> Result<std::collections::HashMap<String, usize>, Box<dyn std::error::Error>> {
        let papers = Self::load_sample(path, sample_size)?;
        let mut categories = std::collections::HashMap::new();

        for paper in papers {
            for cat in paper.categories {
                *categories.entry(cat).or_insert(0) += 1;
            }
        }

        Ok(categories)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kaggle_record_conversion() {
        let record = KagglePaperRecord {
            id: "2401.00001v1".to_string(),
            submitter: Some("John Doe".to_string()),
            authors: "Alice, Bob, Charlie".to_string(),
            title: "A Novel Approach to DDoS Detection".to_string(),
            comments: None,
            journal_ref: None,
            doi: None,
            abstract_text: "This paper presents a new method...".to_string(),
            categories: "cs.NI cs.CR cs.SY".to_string(),
            versions: vec![KaggleVersion {
                version: "v1".to_string(),
                created: "2024-01-01T12:00:00Z".to_string(),
            }],
        };

        let paper = record.to_paper_metadata();
        assert_eq!(paper.arxiv_id, "2401.00001v1");
        assert_eq!(paper.authors.len(), 3);
        assert_eq!(paper.categories.len(), 3);
    }

    #[test]
    fn test_s3_key_generation() {
        let key = PaperMetadata::s3_key_from_arxiv_id("2401.00001v1");
        assert_eq!(key, "pdf/2401/2401.00001v1.pdf");
    }
}
