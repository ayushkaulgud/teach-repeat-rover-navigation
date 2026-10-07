# Freiburg Campus raw robot log

Recorded by Cyrill Stachniss and Giorgio Grisetti on July 14, 2004 at the University of Freiburg campus. The robot was a Pioneer 2AT with a SICK LMS291-S05 front laser. The log is CARMEN format and contains `ODOM`, `FLASER`, and `PARAM` records. The [MIT CSAIL archive](https://dspace.mit.edu/entities/publication/de754aaf-ddfc-479c-a193-dc77c7005816) lists the dataset under CC0 1.0; the [StachnissLab dataset index](https://www.ipb.uni-bonn.de/datasets/) credits both recorders.

Raw download: `http://www2.informatik.uni-freiburg.de/~stachnis/datasets/datasets/freiburg-campus/fr-campus-20040714.carmen.log.gz`

Local file: `raw.log.gz` (9,206,989 bytes)

SHA-256: `669b9a94f609e473fed0b49c9b9561e4f501fd57196dab81b528f11dd54322f9`

The class benchmark parses laser records 250–569 for its obstacle map and poses 300–549 for its taught route. These indices and the map rule are fixed in `src/recorded_demo.py`.
