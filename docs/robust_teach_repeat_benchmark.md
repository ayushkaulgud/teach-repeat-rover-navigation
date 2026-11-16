# Robust teach-and-repeat benchmark

## Purpose

The original comparisons use exact simulated pose and a static scene.  Those
experiments remain useful clean-condition baselines, but they do not reproduce
the main uncertainties of a repeat traversal.  This benchmark adds two
separate evidence layers:

1. naturally complex route segments mined from the recorded Freiburg log;
2. controlled repeat-time simulations using the same taught route and static
   laser scene.

Never describe the second layer as recorded field performance.  The output
JSON records which inputs are recorded and which are simulated.

## Route selection

Run:

```powershell
python src/route_miner.py --top 5
```

For a downloaded corrected CARMEN log, prefer:

```powershell
python src/route_miner.py --log data/freiburg_campus/corrected.log.gz --top 5
```

The miner scans 25–55 m windows and ranks them using only route geometry and
local-map context: tortuosity, total heading change, meaningful turn count,
obstacle exposure, and minimum clearance.  Planner outcomes are not used.  It
then selects the highest-scoring non-overlapping windows.  Outputs are written
to `results/route_mining/`:

- `candidates.csv`: every feasible ranked candidate;
- `selected_routes.json`: fixed selection manifest and map assumptions;
- `route_N.csv`: route coordinates used by later benchmarks.

Do not manually replace a selected route after viewing planner results.  If
selection criteria change, record the change and regenerate all methods.

## Repeat disturbances

Run a quick validation:

```powershell
python src/repeat_disturbance_benchmark.py --runs 3 --routes 1
```

Run the intended comparison:

```powershell
python src/repeat_disturbance_benchmark.py --runs 20 --routes 3
```

Five paired methods are evaluated:

- raw waypoint replay;
- cubic-spline-smoothed waypoint replay;
- online local A* replanning;
- online classical DMP;
- online obstacle-aware DMP.

The scenario suite contains clean conditions, localization noise with drift and
dropout, a repeat-time obstacle, reduced traction, and a combined condition.
The Freiburg taught route and static laser scene are recorded; all repeat rover
motion and disturbances are simulated.  Identical seeds are shared across
methods for each route/scenario/run tuple.

Outputs are written to `results/repeat_disturbances/`:

- `trials.csv`: status, clearance, tracking, steering, and replanning data;
- `summary.json`: completion intervals and per-scenario method medians.

## Interpretation

The primary outcome is safe completion.  Smoothness or tracking advantages are
secondary and should only be compared among successful trials.  Report:

- completion with Wilson 95% confidence intervals;
- collisions and planning failures;
- minimum clearance;
- cross-track RMSE and 95th-percentile error;
- integrated steering effort and RMS turn rate;
- online replans;
- route-shape error.

Planning time should still be measured by the PyBullet scripts.  This
sensitivity benchmark is intentionally deterministic and does not claim that a
kinematic runtime equals physical rover compute time.

An academically defensible DMP result would be comparable or higher safe
completion with lower steering effort or route-shape error as disturbance
severity increases.  A clean-condition waypoint win is expected and should not
be hidden.  If DMP does not outperform strong waypoint and replanning baselines,
report that result rather than tuning scenarios after inspection.

## Corrected-log limitation

The repository currently includes the Freiburg raw log.  `route_miner.py`
accepts a corrected log through `--log`, but does not silently download or
substitute one.  Consequently, output metadata contains the exact source path.
For long routes, corrected poses or locally consistent submaps should be used
before treating minimum-clearance values as strong evidence.
