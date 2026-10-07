# Project scope and measured update

Goal: teach a rover one route, then compare four ways to reach the route and goal from offset starts: direct waypoints, local A* reconnection, classical discrete dynamic movement primitives (DMP), and a DMP with obstacle coupling.

The October 7 update uses the recorded Freiburg Campus 2004 raw CARMEN log as its route and laser-obstacle source. Python parses the FLASER records and builds a 0.25 m grid. PyBullet runs a Husky model on that grid at 120 Hz. The tests cover five lateral offsets and three heading offsets, with the same controller and collision geometry for every planner. A path is rejected if the grid-estimated obstacle clearance after a 0.42 m body-radius proxy is below 0.15 m.

Measurements: completion, path rejection, safety stops, physical Bullet contacts, minimum grid-estimated clearance, goal-position error, distance traveled, and planning time. Completion requires ending within 0.65 m of the recorded goal. All per-trial and summary values are checked in under `results/recorded/`.

This is a recorded-data simulation, not a replay of real repeat drives or a physical rover result. Logged odometry is treated as a reference route, not verified ground truth. Navigation currently uses exact simulated pose. A separate ICP example aligns a held-out recorded scan to the map and compares the estimate with logged odometry.
