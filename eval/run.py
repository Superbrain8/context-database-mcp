"""Retrieval evaluation for context-db: context_search with and without `expand`.

Drives a real server binary over stdio MCP, the same path an agent uses, so
the numbers include chunking, both rankers, fusion and the distance cutoff.

A question set is a JSON file:

    {
      "client_id": "claude-code",
      "namespace": "some-project",
      "questions": [
        {"q": "why do banded streamlines print mid air", "relevant": [249, 250]}
      ]
    }

`relevant` lists every memory a good answer needs. Sets hold real memory ids
and titles from real projects, so they live in eval/sets/, which git ignores.

Usage:
    python eval/run.py SET.json [--bin PATH] [--limit 8] [--slots 0 ...]

Several --slots values run one expanded pass each against the same baseline.
"""

import argparse
import json
import os
import re
import subprocess
import sys

DEFAULT_BIN = os.path.join(os.path.dirname(__file__), "..", "target", "debug", "context-database-mcp.exe"
                           if os.name == "nt" else "context-database-mcp")


class Server:
    def __init__(self, binary, env):
        self.p = subprocess.Popen([os.path.abspath(binary)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  env=env, text=True, encoding="utf-8")
        self.n = 0
        self.rpc("initialize", {"protocolVersion": "2025-11-25", "capabilities": {},
                                "clientInfo": {"name": "eval", "version": "0"}})
        self.send({"jsonrpc": "2.0", "method": "notifications/initialized"})

    def send(self, msg):
        self.p.stdin.write(json.dumps(msg) + "\n")
        self.p.stdin.flush()

    def rpc(self, method, params):
        self.n += 1
        self.send({"jsonrpc": "2.0", "id": self.n, "method": method, "params": params})
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError("server exited; run it by hand to see why")
            r = json.loads(line)
            if r.get("id") == self.n:
                if "error" in r:
                    raise RuntimeError(r["error"])
                return r["result"]

    def search(self, query, limit, expand):
        res = self.rpc("tools/call", {"name": "context_search",
                                      "arguments": {"query": query, "limit": limit, "expand": expand}})
        text = res["content"][0]["text"]
        return [int(m) for m in re.findall(r"^\[id=(\d+)\]", text, re.M)]

    def close(self):
        self.p.stdin.close()
        self.p.wait(timeout=10)


def score(ids, relevant):
    """Recall at the returned depth, and the reciprocal rank of the first hit."""
    rel = set(relevant)
    found = [i for i in ids if i in rel]
    recall = len(found) / len(rel)
    rr = next((1.0 / (k + 1) for k, i in enumerate(ids) if i in rel), 0.0)
    return recall, rr


def run(binary, spec, limit, expand, slots=0):
    env = dict(os.environ, CTXDB_CLIENT_ID=spec["client_id"], CTXDB_NAMESPACE=spec["namespace"],
               CTXDB_LOG="error", CTXDB_EXPAND_SLOTS=str(slots))
    srv = Server(binary, env)
    try:
        return [srv.search(q["q"], limit, expand) for q in spec["questions"]]
    finally:
        srv.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("set")
    ap.add_argument("--bin", default=DEFAULT_BIN)
    ap.add_argument("--limit", type=int, default=8)
    ap.add_argument("--slots", type=int, action="append")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    slots = a.slots or [0]

    with open(a.set, encoding="utf-8") as f:
        spec = json.load(f)
    qs = spec["questions"]

    passes = [("baseline", run(a.bin, spec, a.limit, False))]
    for n in slots:
        passes.append((f"expand {n} slot", run(a.bin, spec, a.limit, True, n)))

    print(f"{spec['namespace']}: {len(qs)} questions, limit {a.limit}\n")
    print(f"{'pass':<14} {'recall':>7} {'MRR':>6} {'full':>5}  (full = every relevant memory returned)")
    base = passes[0][1]
    for name, results in passes:
        scores = [score(ids, q["relevant"]) for ids, q in zip(results, qs)]
        recall = sum(s[0] for s in scores) / len(qs)
        mrr = sum(s[1] for s in scores) / len(qs)
        full = sum(1 for s in scores if s[0] == 1.0)
        print(f"{name:<14} {recall:>7.3f} {mrr:>6.3f} {full:>5}")

    if a.verbose or len(passes) > 1:
        print("\nper question (recall baseline -> each expanded pass):")
        for i, q in enumerate(qs):
            row = [score(p[1][i], q["relevant"])[0] for p in passes]
            mark = "" if len(set(row)) == 1 else ("  +" if row[-1] > row[0] else "  -")
            missing = sorted(set(q["relevant"]) - set(passes[-1][1][i]))
            print(f"  {' -> '.join(f'{r:.2f}' for r in row)}{mark}  {q['q']}"
                  + (f"  (missing {missing})" if missing and a.verbose else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
