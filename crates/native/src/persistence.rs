use anyhow::{Context, Result};
use directories::ProjectDirs;
use std::path::PathBuf;
use tokio::fs;

#[derive(Debug, Clone)]
pub struct ArxivCache {
    cache_dir: PathBuf,
}

impl ArxivCache {
    pub async fn new() -> Result<Self> {
        // Use standard OS cache directory to avoid littering the workspace
        let cache_dir = if let Some(proj_dirs) = ProjectDirs::from("org", "arxiv-search", "mcp") {
            proj_dirs.cache_dir().to_path_buf()
        } else {
            // Fallback to temp dir if standard paths are unavailable
            std::env::temp_dir().join("arxiv-search-mcp")
        };

        if !cache_dir.exists() {
            fs::create_dir_all(&cache_dir)
                .await
                .with_context(|| format!("Failed to create cache directory at {}", cache_dir.display()))?;
        }

        Ok(Self { cache_dir })
    }

    pub async fn get_html(&self, paper_id: &str) -> Result<Option<String>> {
        let path = self.cache_dir.join(format!("{paper_id}.html"));
        if path.exists() {
            let content = fs::read_to_string(path).await?;
            return Ok(Some(content));
        }
        Ok(None)
    }

    pub async fn set_html(&self, paper_id: &str, content: &str) -> Result<()> {
        let path = self.cache_dir.join(format!("{paper_id}.html"));
        fs::write(path, content).await?;
        Ok(())
    }

    pub async fn get_pdf(&self, paper_id: &str) -> Result<Option<Vec<u8>>> {
        let path = self.cache_dir.join(format!("{paper_id}.pdf"));
        if path.exists() {
            let content = fs::read(path).await?;
            return Ok(Some(content));
        }
        Ok(None)
    }

    pub async fn set_pdf(&self, paper_id: &str, content: &[u8]) -> Result<()> {
        let path = self.cache_dir.join(format!("{paper_id}.pdf"));
        fs::write(path, content).await?;
        Ok(())
    }
}

#[cfg(test)]
#[expect(clippy::panic_in_result_fn)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_cache_hit() -> Result<()> {
        let temp = tempdir()?;
        let cache_dir = temp.path().join(".arxiv_cache");
        fs::create_dir_all(&cache_dir).await?;

        let cache = ArxivCache {
            cache_dir: cache_dir.clone(),
        };

        let paper_id = "1234.5678";
        let html_content = "<html><body>Test</body></html>";

        // Initial state: cache miss
        assert!(cache.get_html(paper_id).await?.is_none());

        // Set cache
        cache.set_html(paper_id, html_content).await?;

        // Cache hit
        let retrieved = cache.get_html(paper_id).await?;
        assert_eq!(retrieved, Some(html_content.to_string()));

        Ok(())
    }

    #[tokio::test]
    async fn test_pdf_cache_hit() -> Result<()> {
        let temp = tempdir()?;
        let cache_dir = temp.path().join(".arxiv_cache");
        fs::create_dir_all(&cache_dir).await?;

        let cache = ArxivCache {
            cache_dir: cache_dir.clone(),
        };

        let paper_id = "1234.5678";
        let pdf_content = vec![0xDE, 0xAD, 0xBE, 0xEF];

        // Initial state: cache miss
        assert!(cache.get_pdf(paper_id).await?.is_none());

        // Set cache
        cache.set_pdf(paper_id, &pdf_content).await?;

        // Cache hit
        let retrieved = cache.get_pdf(paper_id).await?;
        assert_eq!(retrieved, Some(pdf_content));

        Ok(())
    }
}
