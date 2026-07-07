use std::sync::Arc;

use agent_context_db_retrieve::{
    BuiltinIntentPolicyProvider, CompiledIntentPolicy, IntentPolicyPack, IntentPolicyProvider,
    LayeredIntentPolicyProvider, RetrieveContext, RuleBasedIntentAnalyzer,
};
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn bench_compile_default_policy(c: &mut Criterion) {
    let pack = IntentPolicyPack::default_builtin().expect("default policy parses");
    c.bench_function("intent_compile_default_policy", |b| {
        b.iter(|| CompiledIntentPolicy::compile(black_box(pack.clone())).expect("policy compiles"));
    });
}

fn bench_decide_default_policy(c: &mut Criterion) {
    let analyzer = RuleBasedIntentAnalyzer::new("u1", "a1");
    let ctx = RetrieveContext {
        user_id: Some("u1".into()),
        agent_id: Some("a1".into()),
        ..Default::default()
    };
    c.bench_function("intent_decide_default_policy", |b| {
        b.iter(|| analyzer.decide(black_box("when did that migration happen?"), black_box(&ctx)));
    });
}

fn bench_reload_layered_provider(c: &mut Criterion) {
    let provider = Arc::new(LayeredIntentPolicyProvider::new(vec![Arc::new(
        BuiltinIntentPolicyProvider,
    )]));
    let analyzer = RuleBasedIntentAnalyzer::new("u1", "a1").with_policy_provider(provider);
    c.bench_function("intent_reload_layered_provider", |b| {
        b.iter(|| analyzer.reload_from_provider());
    });
}

fn benches(c: &mut Criterion) {
    bench_compile_default_policy(c);
    bench_decide_default_policy(c);
    bench_reload_layered_provider(c);
}

criterion_group!(intent_policy, benches);
criterion_main!(intent_policy);
