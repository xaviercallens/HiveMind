// Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
//! Integration tests: run the built binary on the problem library and on malformed input.
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};

fn bin() -> &'static str {
    env!("CARGO_BIN_EXE_hivemind-solver")
}

fn run_cli(args: &[&str], stdin: &[u8]) -> (i32, String, String) {
    let mut child = Command::new(bin())
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn hivemind-solver");
    {
        let mut si = child.stdin.take().unwrap();
        // ignore EPIPE if the child exits before reading everything
        let _ = si.write_all(stdin);
    }
    let out = child.wait_with_output().unwrap();
    (
        out.status.code().unwrap_or(-1),
        String::from_utf8(out.stdout).unwrap(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

fn problem_specs() -> Vec<(String, Vec<u8>, serde_json::Value)> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../problems");
    let mut v = Vec::new();
    for e in std::fs::read_dir(&dir).expect("problems dir") {
        let p = e.unwrap().path();
        if p.extension().and_then(|s| s.to_str()) != Some("json") {
            continue;
        }
        let doc: serde_json::Value = serde_json::from_slice(&std::fs::read(&p).unwrap()).unwrap();
        let spec = doc["spec"].clone();
        let bytes = serde_json::to_vec(&spec).unwrap();
        v.push((p.file_stem().unwrap().to_string_lossy().into_owned(), bytes, doc));
    }
    v.sort_by(|a, b| a.0.cmp(&b.0));
    v
}

#[test]
fn library_has_at_least_twelve_problems() {
    assert!(problem_specs().len() >= 12);
}

#[test]
fn every_library_problem_runs_ok_with_exit_zero() {
    for (name, bytes, doc) in problem_specs() {
        let (code, out, _err) = run_cli(&["run"], &bytes);
        assert_eq!(code, 0, "{name}");
        let v: serde_json::Value = serde_json::from_str(out.trim_end()).expect("stdout is pure JSON");
        assert_eq!(v["status"], "ok", "{name}: {v}");
        assert_eq!(v["schema"], "hivemind-solver-result/1");
        assert_eq!(v["id"], doc["id"]);
        assert_eq!(v["t"], doc["spec"]["t_out"], "{name}");
        assert_eq!(v["y"].as_array().unwrap().len(), doc["reference"]["y"].as_array().unwrap().len());
        assert_eq!(v["solver"]["rev"], hivemind_solver::SOLVER_REV);
        assert_eq!(v["input_sha256"], hivemind_solver::sha256_hex(&bytes));
    }
}

#[test]
fn repeated_process_runs_are_byte_identical() {
    let (_, bytes, _) = problem_specs().into_iter().find(|p| p.0 == "robertson").unwrap();
    let (_, first, _) = run_cli(&["run"], &bytes);
    for _ in 0..9 {
        let (_, again, _) = run_cli(&["run"], &bytes);
        assert_eq!(first, again);
    }
}

#[test]
fn malformed_input_is_a_clean_error_with_exit_zero() {
    for input in [&b"not json"[..], b"", b"{}", br#"{"schema":"hivemind-solver-spec/1","id":"x","bogus":1}"#] {
        let (code, out, _) = run_cli(&["run"], input);
        assert_eq!(code, 0);
        let v: serde_json::Value = serde_json::from_str(out.trim_end()).unwrap();
        assert_eq!(v["status"], "error");
        assert!(v["error"].as_str().unwrap().len() > 3);
    }
}

#[test]
fn oversized_input_is_rejected_cleanly() {
    let big = vec![b' '; 100 * 1024];
    let (code, out, _) = run_cli(&["run"], &big);
    assert_eq!(code, 0);
    let v: serde_json::Value = serde_json::from_str(out.trim_end()).unwrap();
    assert_eq!(v["status"], "error");
    assert!(v["error"].as_str().unwrap().contains("exceeds"));
}

#[test]
fn version_and_families_subcommands() {
    let (code, out, _) = run_cli(&["version"], b"");
    assert_eq!(code, 0);
    let v: serde_json::Value = serde_json::from_str(out.trim_end()).unwrap();
    assert_eq!(v["solver_rev"], hivemind_solver::SOLVER_REV);
    assert_eq!(v["crate_version"], env!("CARGO_PKG_VERSION"));
    let (code, out, _) = run_cli(&["families"], b"");
    assert_eq!(code, 0);
    let f: serde_json::Value = serde_json::from_str(out.trim_end()).unwrap();
    assert_eq!(f["families"].as_array().unwrap().len(), 10);
}

#[test]
fn unknown_subcommand_exits_nonzero() {
    let (code, _, err) = run_cli(&["frobnicate"], b"");
    assert_eq!(code, 2);
    assert!(err.contains("usage"));
}
