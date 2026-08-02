//! `--consolidate`: find clusters of overlapping memories, so several can be
//! replaced by one merged memory.
//!
//! A store that is only ever appended to accumulates near-duplicates: the same
//! constraint learned twice, a decision recorded once when it was made and again
//! when it was questioned, four session summaries covering one week of work.
//! Nothing is wrong with any of them individually, and together they push the
//! real answer down the ranking behind three paraphrases of itself.
//!
//! This mode reports; it does not merge. Two reasons, and the second is the
//! binding one. Merging is the same judgement `--stale` refuses to automate --
//! two memories that a distance metric calls duplicates are routinely a rule and
//! its exception. And nothing in this stack can write the merged text: the
//! embedding server turns text into vectors and cannot produce a sentence. The
//! only thing here that can summarise is the model reading this report, which is
//! why the output ends in the exact tool call to make.
//!
//! Like the other operator modes, it fails loudly rather than swallowing errors.

use std::collections::HashMap;

use anyhow::Result;

use crate::{db, Scope};

/// Default cosine-distance cutoff for "these overlap".
///
/// Measured with bge-m3 on this project's own notes: a paraphrase of a stored
/// note sits near 0.33 and an unrelated note near 0.66, which is what
/// `CTXDB_MAX_DISTANCE` 0.55 is drawn from. Redundancy is a stricter claim than
/// relevance -- 0.55 would group everything this project has ever written about
/// chunking -- so this sits well inside the paraphrase band and errs towards
/// reporting too few clusters. A cluster that is missed costs nothing; a bad
/// merge destroys the distinction between two memories.
///
/// Measured against this project's own 45-memory corpus, where the collapse is
/// steep and one-sided: 0.22 groups 3 memories, 0.26 groups 7, 0.28 groups 10,
/// 0.35 groups 13 of them into a single cluster. Past roughly 0.25 the chaining
/// stops finding duplicates and starts finding the topic.
pub const DEFAULT_THRESHOLD: f64 = 0.22;

/// Cluster size past which the grouping is more likely an artefact of the
/// cutoff than a real pile of duplicates. Not a limit -- the cluster is still
/// printed, with a warning.
const CHAINING_SUSPECT: usize = 6;

/// Group pairs into connected components.
///
/// Transitive on purpose: if A overlaps B and B overlaps C, all three are one
/// merge decision even when A and C are far apart on their own. Chains are how
/// a topic actually accretes -- each new memory written against the one before
/// it -- and reporting the pairs separately would hide that they are one topic.
fn clusters(pairs: &[(i64, i64)]) -> Vec<Vec<i64>> {
    let mut parent: HashMap<i64, i64> = HashMap::new();

    fn find(parent: &mut HashMap<i64, i64>, x: i64) -> i64 {
        let mut root = x;
        while let Some(&p) = parent.get(&root) {
            if p == root {
                break;
            }
            root = p;
        }
        // Path compression, so a long chain does not re-walk on every lookup.
        let mut cur = x;
        while let Some(&p) = parent.get(&cur) {
            if p == root {
                break;
            }
            parent.insert(cur, root);
            cur = p;
        }
        root
    }

    for &(a, b) in pairs {
        parent.entry(a).or_insert(a);
        parent.entry(b).or_insert(b);
        let (ra, rb) = (find(&mut parent, a), find(&mut parent, b));
        if ra != rb {
            parent.insert(ra.max(rb), ra.min(rb));
        }
    }

    let mut groups: HashMap<i64, Vec<i64>> = HashMap::new();
    let members: Vec<i64> = parent.keys().copied().collect();
    for id in members {
        let root = find(&mut parent, id);
        groups.entry(root).or_default().push(id);
    }

    // Sorted throughout: the report is read by a human comparing two runs, and
    // HashMap order would reshuffle identical output between them.
    let mut out: Vec<Vec<i64>> = groups
        .into_values()
        .map(|mut g| {
            g.sort_unstable();
            g
        })
        .collect();
    out.sort_by_key(|g| g[0]);
    out
}

pub async fn run(database_url: &str, scope: &Scope, threshold: f64, limit: i64) -> Result<()> {
    let pool = db::connect(database_url).await?;
    let pairs = db::near_pairs(&pool, scope, threshold).await?;

    if pairs.is_empty() {
        println!(
            "no overlapping memories in {} within {threshold:.2}. Raise the cutoff with \
             --threshold to look wider.",
            scope.namespace
        );
        return Ok(());
    }

    // The closest pair inside a cluster, kept so the report can say how tight
    // the grouping is. A cluster held together at 0.05 is near-certainly one
    // memory written twice; one held at 0.21 needs reading before merging.
    let mut closest: HashMap<i64, f64> = HashMap::new();
    let flat: Vec<(i64, i64)> = pairs.iter().map(|p| (p.a, p.b)).collect();
    let groups = clusters(&flat);

    let mut group_of: HashMap<i64, usize> = HashMap::new();
    for (i, g) in groups.iter().enumerate() {
        for &id in g {
            group_of.insert(id, i);
        }
    }
    for p in &pairs {
        if let Some(&g) = group_of.get(&p.a) {
            let e = closest.entry(g as i64).or_insert(f64::MAX);
            *e = e.min(p.dist);
        }
    }

    let shown = groups.len().min(limit.max(1) as usize);
    println!(
        "{} cluster(s) of overlapping memories in {} (chunks within {threshold:.2} cosine)",
        groups.len(),
        scope.namespace
    );

    let mut total_chars = 0i64;
    for (i, g) in groups.iter().take(shown).enumerate() {
        let briefs = db::briefs(&pool, scope, g).await?;
        let chars: i64 = briefs.iter().map(|b| i64::from(b.body_chars)).sum();
        total_chars += chars;
        let tightest = closest.get(&(i as i64)).copied().unwrap_or(f64::NAN);

        println!(
            "\ncluster {} -- {} memories, {chars} chars, closest pair {tightest:.2}",
            i + 1,
            briefs.len()
        );
        for b in &briefs {
            println!(
                "  [id={}] {}({}) {} | {} chars | {}",
                b.id,
                if b.pinned { "PINNED " } else { "" },
                b.kind,
                b.title,
                b.body_chars,
                b.created_at.format("%Y-%m-%d")
            );
        }
        // Transitive grouping is what makes a chain one decision, and it is also
        // how a slightly loose cutoff turns a whole topic into one cluster. The
        // report says so rather than presenting a 13-way merge as a finding.
        if briefs.len() >= CHAINING_SUSPECT {
            println!(
                "  large cluster: probably chained through a shared topic rather than duplicated. \
                 Lower --threshold before merging this."
            );
        }
        // The ids in the form they go into the tool call, because retyping them
        // by hand is where a merge retires the wrong memory.
        println!(
            "  merge: read these with context_get, then context_save the merged text with \
             supersedes={:?}",
            g
        );
    }

    if groups.len() > shown {
        println!(
            "\n{} more cluster(s) not shown; raise the limit with --consolidate <N>.",
            groups.len() - shown
        );
    }

    println!(
        "\nNothing was written. A merged memory supersedes the whole cluster in one save, so the \
         originals stay on disk and --history can bring any of them back. Merge only what says the \
         same thing -- a rule and its exception look identical to a distance metric.\n\
         {total_chars} chars sit in the clusters above."
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_chain_is_one_cluster() {
        // A-B and B-C: A and C may be far apart, but the three are one topic and
        // one merge decision.
        let g = clusters(&[(1, 2), (2, 3)]);
        assert_eq!(g, vec![vec![1, 2, 3]]);
    }

    #[test]
    fn separate_pairs_stay_separate() {
        let g = clusters(&[(5, 6), (1, 2)]);
        assert_eq!(g, vec![vec![1, 2], vec![5, 6]]);
    }

    #[test]
    fn merging_two_existing_clusters_keeps_every_member() {
        // The bridging pair arrives last, which is the case a naive
        // insert-into-the-first-match grouping gets wrong.
        let g = clusters(&[(1, 2), (3, 4), (2, 3)]);
        assert_eq!(g, vec![vec![1, 2, 3, 4]]);
    }

    #[test]
    fn output_is_ordered_the_same_way_every_run() {
        // Same edges, different order in: two runs of --consolidate on an
        // unchanged corpus must be diffable.
        let a = clusters(&[(9, 4), (4, 7), (1, 2)]);
        let b = clusters(&[(1, 2), (4, 7), (9, 4)]);
        assert_eq!(a, b);
        assert_eq!(a, vec![vec![1, 2], vec![4, 7, 9]]);
    }

    #[test]
    fn no_pairs_means_no_clusters() {
        assert!(clusters(&[]).is_empty());
    }
}
