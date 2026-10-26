# Acceptance criteria — the HiveMind QLoRA MoE apps on a real BOINC server (built from source), written before the first run

Server: BOINC 7.24.4 built from source (`scheduler`, `feeder`, `transitioner`, `file_deleter`, `file_upload_handler`,
`script_validator`, `script_assimilator`), MariaDB 10.6 (private instance, socket only, no TCP), Apache 2.4 + mod_php (private
instance, `127.0.0.1:8081`), all run unprivileged by one user, nothing installed as a system service. Client: stock
BOINC 7.18.1, Tesla T4, **no `app_info.xml`**: the client must download and verify the code-signed app version itself.

This is the "real server" step announced as next in `docs/BOINC_REAL_CLIENT.md`. It is **not** a public production
deployment: it listens on loopback only, keeps the code-signing private key on the server (BOINC warns this is a
vulnerability for a public project), and has one volunteer host.

| ID | Criterion (all must hold for PASS) |
| --- | --- |
| C1 | An account is created through the project's PHP web stack (`create_account.php`) and the stock client attaches with the returned authenticator; its log does **not** contain "anonymous platform". |
| C2 | The real scheduler assigns the training workunit to the T4 under plan class `cuda_hivemind` (DB: `result.app_version_id` is the `hivemind_qlora_moe` version) and the client downloads the signed app and the input over HTTP. |
| C3 | The client uploads `out.json` and `out.pt` through the real `file_upload_handler` (signed upload) and reports the result. |
| C4 | BOINC's `script_validator` runs HiveMind's random-segment replay: honest result `validate_state = valid`, BOINC grants credit > 0. |
| C5 | The "lazy" volunteer workunit is `validate_state = invalid` and BOINC grants 0 credit. |
| C6 | `script_assimilator` stores the adapter, and its sha256 equals the adapter assimilated by the emulator run of the same workunit (`artifacts/boinc_core_node/e2e_report.json`, last `state_sha256`) — the result does not depend on which server drove the client. |
| C7 | A batch **inference** workunit using that adapter is validated (identical greedy token ids) and its completions equal those of the emulator run. |
| C8 | Afterwards: every listening socket of the stack is on 127.0.0.1 (MariaDB has none), no BOINC client or app process is left, and the daemons can be stopped cleanly with `bin/stop`. |

Informational: wall time per stage, credit values (BOINC's own and HiveMind's `verified_credit` shown side by side; they are different quantities).

Any failed row is reported as failed, with the log.
