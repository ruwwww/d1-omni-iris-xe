# D1-Omni Image Classifier Evaluation Report

- Generated: 2026-10-22T04:36:56.309137+00:00
- Manifest: `C:\Users\darlis\d1-omni-iris-xe\dataset_manifest.json`
- Samples: 38
- Positive class: `art`

## Metrics

| Metric | Value |
| --- | ---: |
| Accuracy | 78.95% |
| Precision | 73.91% |
| Recall | 89.47% |
| F1 Score | 80.95% |
| Average Latency | 1865.59 ms |
| P95 Latency | 2171.33 ms |
| Min / Max Latency | 1348.04 / 4499.83 ms |

## Confusion Matrix

| | Predicted Art | Predicted Not Art |
| --- | ---: | ---: |
| **Actual Art** | 17 (TP) | 2 (FN) |
| **Actual Not Art** | 6 (FP) | 13 (TN) |

## Misclassifications

| ID | Actual | Predicted | Confidence | Latency (ms) | Path |
| --- | --- | --- | ---: | ---: | --- |
| art_003 | art | not_art | 0.5473 | 1348.04 | `C:\Users\darlis\Downloads\anya\__anya_mouthwashing_drawn_by_l1m3candy__3cf3204f9377480d3afb21d229cc631a.jpg` |
| art_007 | art | not_art | 0.9706 | 1366.14 | `C:\Users\darlis\Downloads\anya\__anya_mouthwashing_drawn_by_rottedchun__36f02fba2035d1719f2c304fced13c6a.png` |
| not_art_002 | not_art | art | 0.6974 | 1885.97 | `C:\Users\darlis\Pictures\Screenshots\Screenshot 2025-06-20 134611.png` |
| not_art_008 | not_art | art | 0.7601 | 1567.74 | `C:\Users\darlis\Pictures\Screenshots\Screenshot 2025-06-20 092606.png` |
| not_art_009 | not_art | art | 0.8774 | 1544.68 | `C:\Users\darlis\Pictures\Screenshots\Screenshot 2026-09-24 195859.png` |
| not_art_012 | not_art | art | 0.7988 | 1758.85 | `C:\Users\darlis\Pictures\Screenshots\Screenshot 2026-06-17 082327.png` |
| not_art_013 | not_art | art | 0.6465 | 1860.66 | `C:\Users\darlis\Pictures\Screenshots\Screenshot 2026-04-05 211305.png` |
| not_art_015 | not_art | art | 0.4381 | 1823.39 | `C:\Users\darlis\Pictures\Screenshots\Screenshot 2025-03-10 072840.png` |
