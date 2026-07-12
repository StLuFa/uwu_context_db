use crate::{
    EmbeddingVector, ExecutionContext, ExecutionGate, ExecutionKind, ExecutionRequest,
    ExecutionResponse, JsonSchema, LlmClient, LlmError, LlmOpts, LlmStream,
};
use async_trait::async_trait;
use moka::future::Cache;
use std::sync::Arc;

pub struct PolicyEnforcedLlmClient {
    inner: Arc<dyn LlmClient>,
    gate: Arc<dyn ExecutionGate>,
    context: Arc<dyn Fn(&str) -> ExecutionContext + Send + Sync>,
    cache: Cache<String, String>,
}
impl PolicyEnforcedLlmClient {
    pub fn new(
        inner: Arc<dyn LlmClient>,
        gate: Arc<dyn ExecutionGate>,
        context: impl Fn(&str) -> ExecutionContext + Send + Sync + 'static,
    ) -> Self {
        Self {
            inner,
            gate,
            context: Arc::new(context),
            cache: Cache::new(10_000),
        }
    }
    fn preflight(
        &self,
        op: &str,
        input: &str,
    ) -> Result<(ExecutionContext, crate::PolicyDecision), LlmError> {
        let mut c = (self.context)(op);
        c.kind = ExecutionKind::Llm;
        let d = self
            .gate
            .preflight(&c, ExecutionRequest::new(op, input))
            .map_err(policy_error)?;
        if !d.allowed {
            return Err(LlmError::Provider(format!(
                "policy denied: {:?}",
                d.selected_rule
            )));
        }
        Ok((c, d))
    }
    async fn text(
        &self,
        op: &str,
        input: &str,
        opts: &LlmOpts,
        json: Option<&JsonSchema>,
    ) -> Result<String, LlmError> {
        let (c, d) = self.preflight(op, input)?;
        let key = format!(
            "{}\0{}\0{:?}\0{}",
            d.policy_version, op, opts, d.request.content
        );
        if let Some(v) = self.cache.get(&key).await {
            return Ok(v);
        }
        let raw = match json {
            Some(s) => {
                self.inner
                    .complete_json(&d.request.content, s, opts)
                    .await?
            }
            None => self.inner.complete(&d.request.content, opts).await?,
        };
        let out = self
            .gate
            .postflight(&c, &d, ExecutionResponse::new(raw))
            .map_err(policy_error)?
            .content;
        self.cache.insert(key, out.clone()).await;
        Ok(out)
    }
}
fn policy_error(e: crate::PolicyError) -> LlmError {
    LlmError::Provider(format!("policy: {e}"))
}
#[async_trait]
impl LlmClient for PolicyEnforcedLlmClient {
    async fn complete(&self, p: &str, o: &LlmOpts) -> Result<String, LlmError> {
        self.text("complete", p, o, None).await
    }
    async fn complete_json(
        &self,
        p: &str,
        s: &JsonSchema,
        o: &LlmOpts,
    ) -> Result<String, LlmError> {
        self.text("complete_json", p, o, Some(s)).await
    }
    async fn embed(&self, text: &str) -> Result<EmbeddingVector, LlmError> {
        let (c, d) = self.preflight("embed", text)?;
        let out = self.inner.embed(&d.request.content).await?;
        self.gate
            .postflight(&c, &d, ExecutionResponse::new("embedding"))
            .map_err(policy_error)?;
        Ok(out)
    }
    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<EmbeddingVector>, LlmError> {
        let mut rewritten = Vec::with_capacity(texts.len());
        let mut checks = Vec::new();
        for t in texts {
            let pair = self.preflight("embed_batch", t)?;
            rewritten.push(pair.1.request.content.clone());
            checks.push(pair);
        }
        let out = self.inner.embed_batch(&rewritten).await?;
        for (c, d) in checks {
            self.gate
                .postflight(&c, &d, ExecutionResponse::new("embedding"))
                .map_err(policy_error)?;
        }
        Ok(out)
    }
    async fn stream_complete(
        &self,
        p: &str,
        o: &LlmOpts,
    ) -> Result<Box<dyn LlmStream + Send>, LlmError> {
        let text = self.text("stream_complete", p, o, None).await?;
        Ok(Box::new(OneShot(Some(text))))
    }
    async fn batch_complete(&self, ps: &[String], o: &LlmOpts) -> Result<Vec<String>, LlmError> {
        let mut out = Vec::with_capacity(ps.len());
        for p in ps {
            out.push(self.text("batch_complete", p, o, None).await?)
        }
        Ok(out)
    }
    async fn speculative_complete(&self, p: &str, o: &LlmOpts) -> Result<String, LlmError> {
        self.text("speculative_complete", p, o, None).await
    }
}
struct OneShot(Option<String>);
#[async_trait]
impl LlmStream for OneShot {
    async fn next_chunk(&mut self) -> Option<Result<String, LlmError>> {
        self.0.take().map(Ok)
    }
}
