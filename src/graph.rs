//! The memory graph: typed edges between memories, and how to walk them.
//!
//! Traversal runs here, in Rust, over the client's whole edge list and
//! supersede map, rather than as a recursive CTE. Every hop has to follow
//! `superseded_by` from each end of an edge to the live head of its chain, and
//! that walk nested inside a recursive CTE is SQL nobody can read or test. At
//! this corpus's size (hundreds of rows) loading the lot is two cheap queries;
//! it is the first thing to revisit at tens of thousands.
//!
//! Edges are never rewritten when an end is superseded. Resolution happens at
//! read time, so a merge moves every edge of its originals onto the merged
//! memory without touching a row, and `--restore --detach` moves them back.

use std::collections::{HashMap, HashSet, VecDeque};

use serde::{Deserialize, Serialize};

/// Deepest `context_related` will walk, whatever the caller asks for.
pub const MAX_DEPTH: usize = 3;

/// Most memories one `context_related` call returns. The caller is told when
/// this cuts the result.
pub const MAX_RESULTS: usize = 25;

/// Longest supersede chain followed before giving up. Chains are acyclic by
/// construction (a save only retires rows older than itself), so this only
/// bounds the damage of a hand-edited database.
pub const MAX_CHAIN: usize = 64;

// The relation types. Checked here, not by a SQL CHECK, so a new type needs no
// migration -- add a variant, and `as_str`/`parse` below. The doc line below is
// the whole description the tool schema shows the model; keep it short.

/// An edge reads as "src <rel> dst".
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Rel {
    // src is related to dst. Symmetric.
    RelatesTo,
    // src is only true while dst is true.
    DependsOn,
    // src and dst disagree; neither replaces the other. Symmetric.
    Contradicts,
    // src adds detail to dst without replacing it.
    Refines,
}

impl Rel {
    pub fn as_str(self) -> &'static str {
        match self {
            Rel::RelatesTo => "relates_to",
            Rel::DependsOn => "depends_on",
            Rel::Contradicts => "contradicts",
            Rel::Refines => "refines",
        }
    }

    pub fn parse(s: &str) -> Option<Rel> {
        match s {
            "relates_to" => Some(Rel::RelatesTo),
            "depends_on" => Some(Rel::DependsOn),
            "contradicts" => Some(Rel::Contradicts),
            "refines" => Some(Rel::Refines),
            _ => None,
        }
    }

    pub fn symmetric(self) -> bool {
        matches!(self, Rel::RelatesTo | Rel::Contradicts)
    }

    /// The order an edge is stored in. A symmetric edge has one stored order
    /// only, smaller id first; without it the unique index would accept both
    /// `relates_to A-B` and `relates_to B-A`.
    pub fn stored_order(self, src: i64, dst: i64) -> (i64, i64) {
        if self.symmetric() && src > dst {
            (dst, src)
        } else {
            (src, dst)
        }
    }
}

/// One edge as stored.
#[derive(Debug, Clone)]
pub struct Edge {
    pub id: i64,
    pub src: i64,
    pub dst: i64,
    pub rel: Rel,
    pub note: Option<String>,
}

/// An edge as seen from one memory, after both ends are resolved to live heads.
#[derive(Debug, Clone, PartialEq)]
pub struct Link {
    pub edge_id: i64,
    pub rel: Rel,
    /// The memory at the other end (a live head).
    pub other: i64,
    /// True when this memory is the edge's src: "this <rel> other".
    pub outgoing: bool,
    /// The ids the edge was written against. They differ from the heads when an
    /// end was superseded since, and the caller is shown where it came from.
    pub stored_src: i64,
    pub stored_dst: i64,
    pub note: Option<String>,
}

impl Link {
    /// The edge as a sentence over live ids, from the point of view of `node`.
    pub fn sentence(&self, node: i64) -> String {
        let (a, b) = if self.outgoing {
            (node, self.other)
        } else {
            (self.other, node)
        };
        format!("{a} {} {b}", self.rel.as_str())
    }

    /// Set when either end was superseded since the edge was written.
    pub fn moved_from(&self, node: i64) -> Option<(i64, i64)> {
        let (src, dst) = if self.outgoing {
            (node, self.other)
        } else {
            (self.other, node)
        };
        (src != self.stored_src || dst != self.stored_dst).then_some((self.stored_src, self.stored_dst))
    }
}

/// One memory reached by a walk.
#[derive(Debug, Clone, PartialEq)]
pub struct Hop {
    pub id: i64,
    pub hop: usize,
    /// The memory it was reached from.
    pub via: i64,
    pub link: Link,
}

#[derive(Debug, Default)]
pub struct Related {
    pub hits: Vec<Hop>,
    /// True when `MAX_RESULTS` cut the walk short.
    pub truncated: bool,
}

/// The client's graph with every edge already resolved onto live heads.
#[derive(Debug, Default)]
pub struct Graph {
    superseded_by: HashMap<i64, i64>,
    /// Rows that are neither forgotten nor expired. A superseded row can be in
    /// here; whether it is the head of its chain is `superseded_by`'s business.
    visible: HashSet<i64>,
    adj: HashMap<i64, Vec<Link>>,
}

impl Graph {
    /// `edges` are the live (not forgotten) edges; `superseded_by` and `visible`
    /// describe the client's memory rows.
    pub fn build(
        mut edges: Vec<Edge>,
        superseded_by: HashMap<i64, i64>,
        visible: HashSet<i64>,
    ) -> Graph {
        let mut g = Graph {
            superseded_by,
            visible,
            adj: HashMap::new(),
        };

        // Oldest edge wins a duplicate, so the id a caller sees is stable from
        // one call to the next.
        edges.sort_by_key(|e| e.id);
        let mut seen: HashSet<(i64, i64, Rel)> = HashSet::new();

        for e in edges {
            // An end that resolves to nothing live hides the edge entirely, so a
            // walk never passes through a forgotten or expired memory.
            let (Some(s), Some(d)) = (g.resolve(e.src), g.resolve(e.dst)) else {
                continue;
            };
            // A merge can put both ends on the same memory.
            if s == d {
                continue;
            }
            // A merge can also turn two edges into one.
            let key = match e.rel.symmetric() {
                true => (s.min(d), s.max(d), e.rel),
                false => (s, d, e.rel),
            };
            if !seen.insert(key) {
                continue;
            }

            let link = |other, outgoing| Link {
                edge_id: e.id,
                rel: e.rel,
                other,
                outgoing,
                stored_src: e.src,
                stored_dst: e.dst,
                note: e.note.clone(),
            };
            g.adj.entry(s).or_default().push(link(d, true));
            g.adj.entry(d).or_default().push(link(s, false));
        }
        g
    }

    /// The live head of `id`'s supersede chain, or None when that head is
    /// forgotten, expired, or not one of this client's rows.
    pub fn resolve(&self, id: i64) -> Option<i64> {
        let mut cur = id;
        for _ in 0..MAX_CHAIN {
            match self.superseded_by.get(&cur) {
                Some(next) => cur = *next,
                None => return self.visible.contains(&cur).then_some(cur),
            }
        }
        None
    }

    /// Direct edges of a live memory. Empty for a superseded or hidden one: its
    /// edges show on the head of its chain instead.
    pub fn links(&self, id: i64) -> &[Link] {
        match self.resolve(id) {
            Some(h) if h == id => self.adj.get(&id).map(Vec::as_slice).unwrap_or(&[]),
            _ => &[],
        }
    }

    /// Breadth-first walk from `start`, `depth` hops at most (capped at
    /// `MAX_DEPTH`), following only `rel` when given. Each memory appears once,
    /// at the hop where it was first reached.
    pub fn related(&self, start: i64, depth: usize, rel: Option<Rel>) -> Related {
        let depth = depth.clamp(1, MAX_DEPTH);
        let mut out = Related::default();

        let Some(start) = self.resolve(start) else {
            return out;
        };

        let mut visited: HashSet<i64> = HashSet::from([start]);
        let mut queue: VecDeque<(i64, usize)> = VecDeque::from([(start, 0)]);

        while let Some((node, d)) = queue.pop_front() {
            if d >= depth {
                continue;
            }
            for link in self.adj.get(&node).into_iter().flatten() {
                if rel.is_some_and(|r| r != link.rel) || !visited.insert(link.other) {
                    continue;
                }
                if out.hits.len() == MAX_RESULTS {
                    out.truncated = true;
                    return out;
                }
                out.hits.push(Hop {
                    id: link.other,
                    hop: d + 1,
                    via: node,
                    link: link.clone(),
                });
                queue.push_back((link.other, d + 1));
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn edge(id: i64, src: i64, dst: i64, rel: Rel) -> Edge {
        Edge {
            id,
            src,
            dst,
            rel,
            note: None,
        }
    }

    fn graph(edges: Vec<Edge>, superseded: &[(i64, i64)], visible: &[i64]) -> Graph {
        Graph::build(
            edges,
            superseded.iter().copied().collect(),
            visible.iter().copied().collect(),
        )
    }

    fn ids(r: &Related) -> Vec<(i64, usize)> {
        r.hits.iter().map(|h| (h.id, h.hop)).collect()
    }

    #[test]
    fn rel_round_trips_through_its_stored_name() {
        for r in [Rel::RelatesTo, Rel::DependsOn, Rel::Contradicts, Rel::Refines] {
            assert_eq!(Rel::parse(r.as_str()), Some(r));
            // The serde name is what the tool schema shows; it must be the
            // stored name too, or a value the model sends would not round-trip.
            assert_eq!(serde_json::to_string(&r).unwrap(), format!("\"{}\"", r.as_str()));
        }
        assert_eq!(Rel::parse("caused_by"), None);
    }

    #[test]
    fn a_symmetric_edge_has_one_stored_order() {
        assert_eq!(Rel::RelatesTo.stored_order(9, 3), (3, 9));
        assert_eq!(Rel::Contradicts.stored_order(3, 9), (3, 9));
        // Direction is the meaning of an asymmetric edge, so it is kept.
        assert_eq!(Rel::DependsOn.stored_order(9, 3), (9, 3));
    }

    #[test]
    fn a_walk_does_not_loop() {
        let g = graph(
            vec![
                edge(1, 1, 2, Rel::DependsOn),
                edge(2, 2, 3, Rel::DependsOn),
                edge(3, 3, 1, Rel::DependsOn),
            ],
            &[],
            &[1, 2, 3],
        );
        let r = g.related(1, 3, None);
        assert_eq!(ids(&r), vec![(2, 1), (3, 1)]);
        assert!(!r.truncated);
    }

    #[test]
    fn depth_is_capped_on_the_server() {
        // A chain of 10: asking for 99 hops still stops at MAX_DEPTH.
        let edges = (1..10).map(|i| edge(i, i, i + 1, Rel::RelatesTo)).collect();
        let g = graph(edges, &[], &(1..=10).collect::<Vec<_>>());
        let r = g.related(1, 99, None);
        assert_eq!(ids(&r), vec![(2, 1), (3, 2), (4, 3)]);
        assert_eq!(ids(&g.related(1, 1, None)), vec![(2, 1)]);
    }

    #[test]
    fn a_walk_does_not_pass_through_a_hidden_memory() {
        // 2 is forgotten (not visible). 3 is reachable only through it.
        let g = graph(
            vec![edge(1, 1, 2, Rel::RelatesTo), edge(2, 2, 3, Rel::RelatesTo)],
            &[],
            &[1, 3],
        );
        assert!(g.related(1, 3, None).hits.is_empty());
    }

    #[test]
    fn the_rel_filter_applies_at_every_hop() {
        let g = graph(
            vec![
                edge(1, 1, 2, Rel::DependsOn),
                edge(2, 2, 3, Rel::RelatesTo),
                edge(3, 2, 4, Rel::DependsOn),
            ],
            &[],
            &[1, 2, 3, 4],
        );
        assert_eq!(ids(&g.related(1, 3, Some(Rel::DependsOn))), vec![(2, 1), (4, 2)]);
    }

    #[test]
    fn an_edge_follows_its_end_to_the_live_head() {
        // Edge written against 2; 2 was later superseded by 5.
        let g = graph(vec![edge(1, 1, 2, Rel::DependsOn)], &[(2, 5)], &[1, 2, 5]);
        let r = g.related(1, 1, None);
        assert_eq!(ids(&r), vec![(5, 1)]);
        let link = &r.hits[0].link;
        assert_eq!(link.sentence(1), "1 depends_on 5");
        assert_eq!(link.moved_from(1), Some((1, 2)));

        // The superseded row itself shows nothing; its edges live on the head.
        assert!(g.links(2).is_empty());
        assert_eq!(g.links(5).len(), 1);
        assert_eq!(g.links(5)[0].sentence(5), "1 depends_on 5");
    }

    #[test]
    fn an_edge_whose_chain_ends_hidden_is_gone() {
        // 2 -> 5, and 5 was forgotten.
        let g = graph(vec![edge(1, 1, 2, Rel::RelatesTo)], &[(2, 5)], &[1, 2]);
        assert!(g.links(1).is_empty());
    }

    #[test]
    fn a_detach_moves_the_edge_back() {
        // Same rows as above after `--restore --detach`: 2 is no longer
        // superseded, so the edge resolves to it again with no rewrite.
        let before = graph(vec![edge(1, 1, 2, Rel::RelatesTo)], &[(2, 5)], &[1, 2, 5]);
        assert_eq!(ids(&before.related(1, 1, None)), vec![(5, 1)]);
        let after = graph(vec![edge(1, 1, 2, Rel::RelatesTo)], &[], &[1, 2, 5]);
        assert_eq!(ids(&after.related(1, 1, None)), vec![(2, 1)]);
    }

    #[test]
    fn a_merge_collapses_duplicate_edges_and_drops_self_edges() {
        // 1 and 2 both relate to 3, and 1 relates to 2. Merge 1 and 2 into 9.
        let g = graph(
            vec![
                edge(10, 1, 3, Rel::RelatesTo),
                edge(11, 2, 3, Rel::RelatesTo),
                edge(12, 1, 2, Rel::RelatesTo),
                // Written in the other stored order for the merged pair.
                edge(13, 3, 2, Rel::RelatesTo),
            ],
            &[(1, 9), (2, 9)],
            &[1, 2, 3, 9],
        );
        let links = g.links(9);
        assert_eq!(links.len(), 1, "one edge to 3, and no 9-9 edge");
        assert_eq!(links[0].other, 3);
        // The oldest edge is the one kept.
        assert_eq!(links[0].edge_id, 10);
    }

    #[test]
    fn opposite_directions_of_an_asymmetric_edge_are_both_kept() {
        let g = graph(
            vec![edge(1, 1, 2, Rel::DependsOn), edge(2, 2, 1, Rel::DependsOn)],
            &[],
            &[1, 2],
        );
        assert_eq!(g.links(1).len(), 2);
    }

    #[test]
    fn a_big_neighbourhood_is_cut_and_says_so() {
        let n = MAX_RESULTS as i64 + 5;
        let edges = (0..n).map(|i| edge(i, 1000, i, Rel::RelatesTo)).collect();
        let mut visible: Vec<i64> = (0..n).collect();
        visible.push(1000);
        let g = graph(edges, &[], &visible);
        let r = g.related(1000, 1, None);
        assert_eq!(r.hits.len(), MAX_RESULTS);
        assert!(r.truncated);
    }

    #[test]
    fn a_hidden_start_returns_nothing() {
        let g = graph(vec![edge(1, 1, 2, Rel::RelatesTo)], &[], &[2]);
        assert!(g.related(1, 1, None).hits.is_empty());
    }
}
