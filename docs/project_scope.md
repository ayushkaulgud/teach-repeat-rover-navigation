# Project scope

Develop and evaluate a simulated teach-and-repeat navigation system that lets a ground robot reuse a single human-demonstrated route from different starting positions and headings.

The initial prototype uses a static 10 × 7 m map, two circular obstacles, a 0.16 m rover radius, and a 0.12 m path safety margin. It compares four trajectory variants from the team's proposal and paper introduction: direct waypoint tracking, local A* reconnection, classical discrete DMP, and obstacle-aware DMP. It generates 60 starting poses from four x positions, five y positions, and three headings. A shared bounded unicycle controller executes each accepted path.

Metrics from the original slideshow: collision-free completion, collisions, route rejection, minimum obstacle clearance, goal-position error, and computation time. The simulator also records runtime safety stops and route distance. Candidate paths that fail the clearance check are rejected; execution stops when the clearance margin is crossed.

ICP is demonstrated on a synthetic 55-point scan. The next phase is to integrate periodic ICP correction with noisy odometry during navigation and test additional map and obstacle configurations.
