# Teach & Repeat Rover Navigation

Project update for October 7, 2026. The route and obstacles now come from **recorded outdoor robot data**. The four repeat runs are **simulations** in PyBullet; they are not field trials.

## Presentation

- [Editable Google Slides](https://docs.google.com/presentation/d/1gNvwtoaDBl7mD5Ic5DboO7ct6j_jSnlH-fsdeZ0XbuQ/edit)
- [PowerPoint export](docs/Teach_and_Repeat_Project_Update_Oct_7_2026.pptx)
- [PDF preview](docs/Teach_and_Repeat_Project_Update_Oct_7_2026.pdf)

The deck starts with the method, dataset, and simulation software, then shows an actual PyBullet render, route plots, metrics, a recorded-scan ICP test, and limitations. Its cream/teal layout follows the team's original class deck.

## Data and experiment

The [Freiburg Campus log](https://dspace.mit.edu/entities/publication/de754aaf-ddfc-479c-a193-dc77c7005816) was recorded outdoors by Cyrill Stachniss and Giorgio Grisetti on July 14, 2004 with a Pioneer 2AT and SICK LMS291 laser. MIT lists it as CC0. The raw CARMEN log in [`data/freiburg_campus/raw.log.gz`](data/freiburg_campus/raw.log.gz) contains 15,683 front-laser scans and recorded poses. [Dataset details](data/freiburg_campus/README.md) include its source and checksum.

[`src/recorded_demo.py`](src/recorded_demo.py) uses scans 300–549 as a 29.59 m taught route, and nearby laser returns to make a 0.25 m obstacle grid. Four planners start from the same 15 offset poses. [PyBullet](https://github.com/bulletphysics/bullet3) drives a Husky model at 120 Hz through static geometry reconstructed from those returns. This is a surrogate vehicle: the recording robot was a Pioneer 2AT. Navigation uses exact simulated pose; the ICP check is separate.

| Method | Completed | Rejected | Collided | Median plan time |
| --- | ---: | ---: | ---: | ---: |
| Direct waypoints | 15/15 | 0/15 | 0/15 | 0.12 ms |
| Local A* | 15/15 | 0/15 | 0/15 | 1.59 ms |
| Classical DMP | 0/15 | 15/15 | 0/15 | 7.73 ms |
| Obstacle-aware DMP | 15/15 | 0/15 | 0/15 | 10.15 ms |

The classical DMP's candidate paths failed the 0.15 m map-clearance rule and were rejected before execution. The other planners completed all starts in this **one** route segment. There were no Bullet obstacle contacts. Completion means ending within 0.65 m of the logged goal; it does not imply centimeter-scale accuracy. Full measurements are in [`metrics.json`](results/recorded/metrics.json) and each trial is in [`trials.csv`](results/recorded/trials.csv).

A held-out recorded scan was aligned to nearby returns with ICP after an injected 0.64 m pose error. Its final displacement from **logged odometry** was 0.044 m, with 0.047 m map residual. Logged odometry is not independent surveyed ground truth.

## Reproduce

```powershell
python -m pip install -r requirements.txt
python src/recorded_demo.py
```

The run recreates the CSV, JSON, and figures in `results/recorded/`. It takes about one minute on this machine. Timing varies by computer. The scripts run headless; [`bullet_scene.png`](results/recorded/bullet_scene.png) is a TinyRenderer screenshot from the actual physics run.

## Limits

Laser returns from one short recording define a static map. Moving people and cars in the original scene are frozen into the grid. The test varies starting pose, not terrain or weather. The 0.42 m footprint radius is a path-screening proxy, and grid clearance has 0.25 m resolution. Results should not be presented as physical rover performance. Next: test additional recorded routes, integrate ICP into the navigation loop, and tune the rover model to the intended hardware.
