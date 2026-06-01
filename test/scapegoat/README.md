# Scapegoat load scripts (macOS / MLX worker)

Real, reproducible pressure generators for the **manual** signal scenarios
(`memory`, `thermal`, `jitter`). Run the matching script on the **scapegoat
worker** (the Mac running MLX, e.g. `qwen-strong`) so that node's telemetry moves,
then run the scenario. With the signal off (round-robin) vs on (`<signal>=1`) the
scheduler routes differently to that node — that's the difference each test reports.

| Script | Scenario | Signal it moves | Privilege |
|---|---|---|---|
| `mac_memory_load.py` | `memory` | `mw` (unified-memory pressure) | none |
| `mac_thermal_load.sh` | `thermal` | `theta_w` (NSProcessInfo.thermalState) | none |
| `mac_jitter.sh` | `jitter` | `j_w` (coordinator-measured RTT) | **sudo** |

## Workflow (per scenario)

1. **On the Mac**, copy the script over and start the load for the scenario you're running:
   ```bash
   python3 mac_memory_load.py 12          # memory (≈60–80% of free RAM)
   ./mac_thermal_load.sh                   # thermal (run a few min before)
   sudo ./mac_jitter.sh start 200          # jitter (wait ~12s, then run)
   ```
2. **Verify the signal actually moved** on the coordinator before testing:
   ```bash
   curl -s http://<COORD>:8080/v1/workers | python3 -m json.tool | grep -A1 qwen-strong
   ```
   You want `mw` / `theta_w` / `jw` elevated on the scapegoat node.
3. **Run the scenario** (it just runs — no need to tell it which node you loaded):
   ```bash
   python test/run_tests.py --coordinator http://<COORD>:8080 --scenarios <memory|thermal|jitter> --verbose
   ```
   It prints **each worker's share `round_robin → signal`** (and the delta) plus a
   distribution plot — the node you loaded should **drop** under the signal.
4. **Stop the load** afterwards:
   - memory / thermal: `Ctrl-C`
   - jitter: `sudo ./mac_jitter.sh stop`  ← always run this to restore networking

## Notes
- The test **doesn't need to know which node you loaded** — it prints every worker's
  `round_robin → signal` share, so you just read off the node you stressed.
- **Thermal** needs *sustained* load — Macs cool fast, so run `mac_thermal_load.sh` for a few minutes and let the test's inference traffic add GPU heat too. Even "fair" state gives `theta_w=0.35`, which is enough.
- **Jitter** uses macOS `dnctl`/`pfctl` (dummynet); syntax can vary by macOS version. If `start` errors, adjust per `man pfctl`. Always `stop` when done.
- **Memory** uses random bytes so macOS memory compression can't reclaim them — that's why it's real pressure, not just allocated-but-free.
