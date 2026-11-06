# Protocol frozen before the first test evaluation

Written after design runs on calibration partitions only, and before this method's score was computed on any test row.

What had already been seen: the repo's published test metrics for the older methods, read while studying the repo. The test positions are therefore not untouched in the strict sense. The new method's own test results had not been computed.

Calibration labels were used for design choices that apply to every dataset. Nothing was tuned per dataset, and test labels played no part.

`frozen_config.json` holds the configuration and endpoints. `frozen_manifest.json` records the SHA-256 of that file and of the code, so any later edit can be seen.

Rules for what follows:

1. Any change after the first test run is listed in the report, with both the before and after numbers.
2. Datasets with fewer than 5 test positives are descriptive only.
3. Every variant run is reported.
