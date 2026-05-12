use async_trait::async_trait;

/// A trait for enforcing rate limits on API requests.
#[async_trait]
pub trait RateLimiter: Send + Sync {
    /// Wait until the rate limit allows the next request.
    async fn wait(&self);
}

/// A no-op rate limiter for environments where rate limiting is handled elsewhere
/// or not possible.
pub struct NoopRateLimiter;

#[async_trait]
impl RateLimiter for NoopRateLimiter {
    async fn wait(&self) {}
}
