//! Startup must survive dependencies that are not up yet.
//!
//! The server is spawned once per MCP client and never respawned, so a process
//! that exits during startup costs the whole session its memory tools. Docker
//! Desktop brings Postgres and the embedder up tens of seconds after login, and
//! the embedder then needs seconds more to load its model: sessions started in
//! that window used to die with "embedding server unreachable".

use std::io::{BufRead, BufReader, Write};
use std::process::{Command, Stdio};
use std::time::Duration;

#[test]
fn an_unreachable_embedder_does_not_kill_the_server() {
    let mut child = Command::new(env!("CARGO_BIN_EXE_context-database-mcp"))
        // Port 1: nothing listens there, so every connection is refused.
        .env("CTXDB_EMBED_URL", "http://127.0.0.1:1")
        .env("CTXDB_DATABASE_URL", "postgres://nobody@127.0.0.1:1/none")
        .env("CTXDB_CLIENT_ID", "test")
        .env("CTXDB_NAMESPACE", "test")
        .env("CTXDB_LOG", "error")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawning the server");

    let mut stdin = child.stdin.take().unwrap();
    let stdout = child.stdout.take().unwrap();
    for msg in [
        r#"{"jsonrpc":"2.0","id":0,"method":"initialize","params":{"protocolVersion":"2025-11-25","capabilities":{},"clientInfo":{"name":"test","version":"0"}}}"#,
        r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#,
        r#"{"jsonrpc":"2.0","id":1,"method":"tools/list"}"#,
    ] {
        // A write can fail if the server already exited; the read below then
        // reports that as the real failure.
        let _ = writeln!(stdin, "{msg}");
    }

    // Read on a thread so a hung server fails the test instead of hanging it.
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        for line in BufReader::new(stdout).lines().map_while(Result::ok) {
            if tx.send(line).is_err() {
                break;
            }
        }
    });
    let mut tools = None;
    while let Ok(line) = rx.recv_timeout(Duration::from_secs(20)) {
        let v: serde_json::Value = serde_json::from_str(&line).unwrap();
        if v["id"] == 1 {
            tools = Some(v);
            break;
        }
    }
    let _ = child.kill();
    let _ = child.wait();

    let tools = tools.expect("the server exited or never answered tools/list");
    let names: Vec<&str> = tools["result"]["tools"]
        .as_array()
        .unwrap_or_else(|| panic!("tools/list failed: {tools}"))
        .iter()
        .filter_map(|t| t["name"].as_str())
        .collect();
    assert!(names.contains(&"context_search"), "{names:?}");
}
