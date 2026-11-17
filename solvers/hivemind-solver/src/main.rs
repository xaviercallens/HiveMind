// Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
use std::io::{Read, Write};

#[cfg(unix)]
mod divert {
    use std::os::unix::io::FromRawFd;
    extern "C" {
        fn dup(fd: i32) -> i32;
        fn dup2(old: i32, new: i32) -> i32;
    }
    /// The pinned solver still contains a few `println!` diagnostics on failure paths. They would
    /// corrupt the JSON on stdout, so for the duration of the solve fd 1 points at stderr and the
    /// real stdout is kept in a private `File` that only this program writes to.
    pub fn save_and_divert() -> Option<std::fs::File> {
        // SAFETY: plain POSIX fd duplication of the process's own stdout/stderr.
        unsafe {
            let saved = dup(1);
            if saved < 0 || dup2(2, 1) < 0 {
                return None;
            }
            Some(std::fs::File::from_raw_fd(saved))
        }
    }
}

fn emit(real_stdout: Option<std::fs::File>, text: &str) {
    match real_stdout {
        Some(mut f) => {
            let _ = f.write_all(text.as_bytes());
            let _ = f.write_all(b"\n");
        }
        None => {
            let mut so = std::io::stdout().lock();
            let _ = so.write_all(text.as_bytes());
            let _ = so.write_all(b"\n");
        }
    }
}

fn main() {
    let arg = std::env::args().nth(1).unwrap_or_default();
    match arg.as_str() {
        "run" => {
            let mut buf = Vec::new();
            // Bounded read: at most MAX_INPUT_BYTES + 1 bytes are ever buffered.
            let r = std::io::stdin()
                .lock()
                .take(hivemind_solver::MAX_INPUT_BYTES as u64 + 1)
                .read_to_end(&mut buf);
            #[cfg(unix)]
            let real = divert::save_and_divert();
            #[cfg(not(unix))]
            let real: Option<std::fs::File> = None;
            let out = match r {
                Ok(_) => hivemind_solver::run_bytes(&buf),
                Err(e) => hivemind_solver::error_result(&buf, None, &format!("cannot read stdin: {e}")),
            };
            emit(real, &out);
        }
        "version" => emit(None, &hivemind_solver::version_json()),
        "families" => emit(None, &hivemind_solver::families_json()),
        _ => {
            eprintln!("usage: hivemind-solver run < spec.json > result.json | version | families");
            std::process::exit(2);
        }
    }
}
