//! `--edges`: suggest edges between related memories, and list stored edges
//! that no longer show in the graph.
//!
//! A report, like `--consolidate` and `--stale`. It writes nothing. Each line
//! ends in the tool call to make, and a person or the model reading the report
//! decides.
//!
//! Suggestions come from the same chunk-to-chunk distance `--consolidate` uses,
//! in the band above it. Below `consolidate::DEFAULT_THRESHOLD` two memories
//! are candidates to merge, not to link. Above it, up to the cutoff here, they
//! share a topic without saying the same thing -- which is the band that makes
//! a bad merge and a good `relates_to`. The distance can only propose
//! `relates_to`: it measures closeness, not direction, so `depends_on` or
//! `refines` stay a reader's call.
//!
//! Housekeeping is also report-only (see the eviction decision behind
//! `--stale`). The read path already hides every edge listed there, so leaving
//! them costs nothing but rows.

use std::collections::HashMap;

use anyhow::Result;

use crate::graph::Dropped;
use crate::{consolidate, db, Scope};

/// Default upper cutoff for "these share a topic". Measured, see the README.
pub const DEFAULT_THRESHOLD: f64 = 0.30;

/// Pairs closer than this are merge candidates; `--consolidate` owns them.
const MERGE_BAND: f64 = consolidate::DEFAULT_THRESHOLD;

pub async fn run(database_url: &str, scope: &Scope, threshold: f64, limit: i64) -> Result<()> {
    let pool = db::connect(database_url).await?;
    let graph = db::load_graph(&pool, &scope.client_id).await?;

    suggest(&pool, scope, &graph, threshold, limit).await?;
    housekeeping(&pool, scope, &graph).await?;

    println!("\nNothing was written.");
    Ok(())
}

async fn suggest(
    pool: &sqlx::PgPool,
    scope: &Scope,
    graph: &crate::graph::Graph,
    threshold: f64,
    limit: i64,
) -> Result<()> {
    if threshold <= MERGE_BAND {
        println!(
            "no suggestions: --threshold {threshold:.2} is inside the merge band (<= {MERGE_BAND:.2}); \
             use --consolidate for those."
        );
        return Ok(());
    }

    let pairs = db::near_pairs(pool, scope, threshold).await?;
    let ids: Vec<i64> = {
        let mut v: Vec<i64> = pairs.iter().flat_map(|p| [p.a, p.b]).collect();
        v.sort_unstable();
        v.dedup();
        v
    };
    let briefs: HashMap<i64, db::Brief> = db::briefs(pool, scope, &ids)
        .await?
        .into_iter()
        .map(|b| (b.id, b))
        .collect();

    let candidates: Vec<&db::NearPair> = pairs
        .iter()
        .filter(|p| p.dist > MERGE_BAND)
        // A compaction summary is near everything its session touched; linking
        // it would bury the real edges.
        .filter(|p| {
            [p.a, p.b].iter().all(|id| {
                briefs
                    .get(id)
                    .is_some_and(|b| b.kind != db::SESSION_SUMMARY)
            })
        })
        .filter(|p| !graph.linked(p.a, p.b))
        .collect();

    println!(
        "{} suggested edge(s) in {} (chunks between {MERGE_BAND:.2} and {threshold:.2} cosine, \
         not already linked, session summaries left out)",
        candidates.len(),
        scope.namespace
    );

    let shown = candidates.len().min(limit.max(1) as usize);
    for p in candidates.iter().take(shown) {
        let (a, b) = (&briefs[&p.a], &briefs[&p.b]);
        println!("\n  {:.2}  [id={}] ({}) {}", p.dist, a.id, a.kind, a.title);
        println!("        [id={}] ({}) {}", b.id, b.kind, b.title);
        println!(
            "        link: context_link src_id={} dst_id={} rel=relates_to  \
             (or depends_on / refines / contradicts, if that is what they are)",
            a.id, b.id
        );
    }
    if candidates.len() > shown {
        println!(
            "\n  {} more not shown; raise the limit with --edges <N>.",
            candidates.len() - shown
        );
    }
    println!(
        "\nRead both memories before linking. A close pair is a shared topic, not a proven \
         relation, and a pair that says the same thing belongs in --consolidate instead."
    );
    Ok(())
}

async fn housekeeping(pool: &sqlx::PgPool, scope: &Scope, graph: &crate::graph::Graph) -> Result<()> {
    let ours = db::edge_ids_in_namespace(pool, scope).await?;
    let dropped: Vec<_> = graph
        .dropped()
        .iter()
        .filter(|(e, _)| ours.contains(&e.id))
        .collect();

    println!("\n{} stored edge(s) in {} no longer show in the graph", dropped.len(), scope.namespace);
    if dropped.is_empty() {
        return Ok(());
    }

    for (e, why) in &dropped {
        let (reason, advice) = match why {
            Dropped::Hidden => (
                "an end is forgotten or expired".to_string(),
                "comes back if that memory is restored; forget only if it is gone for good",
            ),
            Dropped::SelfEdge { head } => (
                format!("both ends merged into id={head}"),
                "comes back if a --restore --detach splits the merge",
            ),
            Dropped::Duplicate { of } => (
                format!("same as edge={of} after a merge"),
                "safe to forget",
            ),
        };
        println!(
            "\n  edge={} ({} {} {}): {reason}\n        {advice}\n        forget: context_forget \
             edge_id={} reason=\"{reason}\"",
            e.id,
            e.src,
            e.rel.as_str(),
            e.dst,
            e.id
        );
    }
    Ok(())
}
