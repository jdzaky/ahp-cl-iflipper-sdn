# Fair ground truth for SDN controller selection: data and code

Data and code that accompany the paper *Reducing Topology Bias in AHP-Based Labels for SDN Controller Selection Using Confident Learning and iFlipper* (J. D. Dzaky).

The package contains the SDN controller measurements (Ryu, ONOS, Floodlight), the AHP labels (Dataset A), the pipeline that cleans and repairs these labels with Confident Learning and iFlipper (Dataset B, the ground truth), and the scripts that produce the accuracy, fairness and feature-importance results of the paper.

- Paper: citation to be added after publication

## Repository layout

```
.
├── README.md
├── LICENSE              (MIT, source code)
├── LICENSE-DATA         (CC BY 4.0, data and result files)
├── requirements.txt
└── sdn-all-data/
    ├── code/
    │   ├── 01_simulate_and_label_ahp.py          data collection (Mininet + Docker) and AHP labelling
    │   ├── 02_confident_learning_iflipper.py     Confident Learning + iFlipper (Dataset A -> Dataset B)
    │   ├── 03_cv_f1_paired_ttest.py              5-fold weighted F1, paired t-test, CI, Cohen's d, Wilcoxon
    │   ├── 04_fairness_metrics.py                SPD, DI, EOD, Individual Fairness of Random Forest
    │   ├── 05_feature_importance_mdi.py          MDI feature importance, rank of the "nodes" feature
    │   
    │   
    ├── data/
    │   ├── dataset_A_ahp.csv                     1,050 samples with AHP labels (Dataset A)
    │   ├── dataset_clean_after_cl.csv            833 samples left after Confident Learning
    │   └── dataset_B_ground_truth.csv            833 samples after iFlipper (Dataset B, ground truth)
    └── results/
        ├── 02_cleaning_console.txt               console output of step 2
        ├── 02_pipeline_summary.json              sample counts of step 2
        ├── 03_cv_f1_paired_ttest_console.txt     console output of step 3
        ├── 03_statistical_validation_results.json
        ├── 04_fairness_report.txt
        ├── 05_feature_importance_console.txt
        
```

The `*_console.txt` files are the original console captures of the runs reported in the paper. Files that start with `legacy_` or `extra_` are not used for the paper results.

## Data

`dataset_A_ahp.csv` has one row per controller and scenario run (3 controllers x 175 scenarios x 2 runs = 1,050 rows). `dataset_B_ground_truth.csv` has the same columns plus `iflipper_applied`.

| Column | Description |
|---|---|
| `sample_id` | Counter inside one collection batch. It is **not unique** in the combined file (values 1 to 165, each repeated up to 10 times). |
| `iteration` | Always 1. The second run of each scenario was collected as a separate batch. |
| `timestamp` | Time of the measurement (ISO 8601). |
| `controller_name` | `ryu`, `onos` or `floodlight`. |
| `scenario_name` | Scenario identifier, for example `Star_8n_5M`. |
| `nodes` | Number of nodes of the topology. |
| `topology_type` | `star`, `linear`, `tree` (privileged group) or `ring`, `mesh` (disadvantaged group). |
| `bandwidth_demand` | Link bandwidth of the scenario, for example `5M`. |
| `throughput_mbps` | TCP throughput measured with iperf3 (Mbps). |
| `latency_ms` | Mean of 15 ping round trips (ms). |
| `packet_loss_percent` | UDP packet loss measured with iperf3 at the scenario bandwidth (%). |
| `jitter_ms` | Ping mean deviation divided by the square root of 2 (ms). |
| `cpu_usage_percent` | 95th percentile of the controller container CPU usage (%). |
| `memory_usage_mb` | 95th percentile of the controller container memory usage (MB). |
| `flow_setup_time_ms` | Time to install a test flow and pass one ping (ms). |
| `score_reliability`, `score_compliance`, `score_efficiency`, `score_stability` | Normalised AHP criterion scores (0 to 1). |
| `ahp_score` | Final AHP score, 0.724 x compliance + 0.193 x efficiency + 0.083 x stability. |
| `best_controller_label` | Winning controller of the scenario run. This is the target label. |
| `score_gap` | Difference between the best and the second AHP score of the scenario run. |
| `resource_method` | How CPU and memory were obtained (`hybrid_p95`, or `NONE` when no sample was collected). |
| `connectivity_status` | `CONTROLLER`, or `STANDALONE` if the controller gave no connectivity and the switches fell back to standalone mode. |
| `iflipper_applied` | Only in Dataset B. `True` if iFlipper changed the label of this row (225 rows). |

Dataset A is the concatenation of 10 collection batches (5 topology families x 2 runs). Dataset B is an ordered subset of Dataset A, so rows can be matched with the key `(scenario_name, controller_name, timestamp)`.

## Requirements

- Python 3.9 or newer and the packages in `requirements.txt` (steps 2 to 5).
- Step 1 additionally needs Linux with root access, Docker, Mininet (with its Python API), Open vSwitch, iperf3 and arping. The paper used Ubuntu 24.04 LTS on an Intel Core i5-1035G1 with 8 GB RAM, OpenFlow 1.3, and one Docker container per controller limited to 2 CPU cores and 2 GB RAM. The controller images are `osrg/ryu:latest`, `latarc/floodlight:latest` and `latarc/onos:latest`.

```bash
python3 -m venv .venv --system-site-packages   # --system-site-packages is only needed for step 1 (Mininet Python API)
source .venv/bin/activate
pip install -r requirements.txt
```

## How to run

All scripts find the data folders from their own location, so they can be started from any directory.

```bash
cd sdn-all-data/code

# Step 1 (optional, long running, root): collect and label one topology family, one run
sudo ../../.venv/bin/python 01_simulate_and_label_ahp.py --topology star --run 1
#   topology: star | linear | tree | ring | mesh   (run each family twice, --run 1 and --run 2)
#   output:   sdn-all-data/data/raw/sdn_dataset_ahp_<topology>_run<run>.csv

# Step 2: Dataset A -> cleaned data -> Dataset B (about 2 minutes)
python3 02_confident_learning_iflipper.py

# Steps 3 to 5: evaluation
python3 03_cv_f1_paired_ttest.py
python3 04_fairness_metrics.py
python3 05_feature_importance_mdi.py
```

Steps 2 to 4 write their outputs into `sdn-all-data/data/` and `sdn-all-data/results/` and overwrite the released files with the same names. Run them in a copy of the repository if you want to keep the released files. Step 5 prints its tables to the console.

## Map from the paper to the files

The numbering follows the submitted manuscript.

| Paper item | Produced by | File |
|---|---|---|
| Table 2, scenario list (175 scenarios) | `SCENARIO_GRIDS` in step 1 | `data/dataset_A_ahp.csv` (column `scenario_name`) |
| Table 3, AHP weights | `AHPCalculator.weights` in step 1 | `data/dataset_A_ahp.csv` |
| Table 4, model settings | `MODELS` in steps 3 and 4 | |
| Table 5, label distribution of Dataset A | crosstab of `topology_type` and `best_controller_label` | `data/dataset_A_ahp.csv` |
| Table 6, noise removed and labels flipped per topology | step 2 | `data/dataset_A_ahp.csv`, `data/dataset_clean_after_cl.csv`, `data/dataset_B_ground_truth.csv` |
| Table 7, label transitions of iFlipper | step 2 | `data/dataset_clean_after_cl.csv` vs `data/dataset_B_ground_truth.csv` |
| Section 4.2 and 4.3 numbers (217 noisy, 225 flipped, error 1,118 to 224, 2,792 pairs) | step 2 | `results/02_pipeline_summary.json`, `results/02_cleaning_console.txt` |
| Table 8, F1 per fold, mean and standard deviation | step 3 | `results/03_statistical_validation_results.json`, `results/03_cv_f1_paired_ttest_console.txt` |
| Table 9, paired t-test, 95% CI, Cohen's d, Wilcoxon | step 3 | `results/03_statistical_validation_results.json` |
| Table 10, accuracy and fairness (SPD, DI, EOD, IF) | step 4 | `results/04_fairness_report.txt` |
| Table 11, feature importance and rank of `nodes` | step 5 | `results/05_feature_importance_console.txt` |
| Figure 4, AHP score gap of noisy and clean samples | `score_gap` column and the rows removed in step 2 | `data/dataset_A_ahp.csv`, `data/dataset_clean_after_cl.csv` |

Tables 5 to 7 and Figure 4 are computed directly from the CSV files with pandas. A script for them is not part of this release yet.

## Notes on reproducibility

- **Row order matters.** The 5-fold `StratifiedKFold` uses `shuffle=False`, so the folds are fixed by the row order of the CSV files. Do not sort or shuffle the files.
- **Seeds.** All models and the 80/20 hold-out split of step 4 use `random_state=42`.
- **Standardisation.** Steps 3 and 4 standardise the features on the whole dataset before splitting, without refitting per fold. Step 5 uses the raw features.
- **Standard deviation.** The standard deviations printed by step 3 are sample standard deviations (`ddof=1`).
- **Fairness audit.** Step 4 uses one 80/20 hold-out split and a Random Forest with `class_weight="balanced"`. SPD, DI and EOD are one-vs-rest per controller and reported as the worst case over the three classes. DI uses no smoothing (`di_epsilon=0`).
- **Individual Fairness.** IF is computed as 1 minus the weighted mean of `|h(x_i) - h(x_j)|` over the 5 nearest neighbours, where `h` is the label-encoded prediction (`floodlight=0`, `onos=1`, `ryu=2`). The value therefore depends on the class encoding.
- **Controller images** use the `latest` tag and their digests were not recorded, so a new collection run can differ from the released data.
- **Console messages** of some scripts are partly in Indonesian because they are the original run messages.

## License

Source code is released under the MIT License (`LICENSE`). The data and result files are released under CC BY 4.0 (`LICENSE-DATA`).

## How to cite

Please cite the paper (details to be added after publication).
