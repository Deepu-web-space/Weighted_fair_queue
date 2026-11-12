| ID | Decision | Choice made | Why | Setting key |
|---|---|---|---|---|
| A | Queue capacity | 32 packets | To be justified from project design | queue_capacity |
| B | Warm-up | 1000 steps | To be justified from experimental design | Check config |
| C | Random streams | One per queue | Independent reproducible traffic streams | seed |
| D | Weight normalization | Sum equals 1 | Allocates total bandwidth | None |
| E | Episode termination | Done remains False | Simulation is continuous | None |
| F | State normalization | To be confirmed | Must match paper | None |