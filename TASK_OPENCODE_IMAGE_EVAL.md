# Task for OpenCode Agent (Herdr Collaboration): Image Classification Dataset & Evaluator

## Objective
Collaborate with Antigravity to build and evaluate the D1-Omni Image Classifier that classifies whether an image is an illustration art or not.

## Your Sub-Tasks:
1. **Dataset Sampler (`dataset_sampler.py`)**:
   - Source positive samples (Illustration Art): `C:\Users\darlis\Downloads\anya` (all valid image files: .jpg, .png, .jpeg, .webp).
   - Source negative samples (Not Art / Screenshots): `C:\Users\darlis\Pictures\Screenshots` (sample 19 diverse screenshots to create a balanced evaluation set).
   - Verify that images open cleanly using `PIL.Image`.
   - Output a JSON manifest `dataset_manifest.json` containing:
     ```json
     [
       {
         "id": "art_001",
         "path": "C:\\Users\\darlis\\Downloads\\anya\\...",
         "label": "art",
         "is_art": true,
         "width": 1200,
         "height": 900
       },
       ...
     ]
     ```
   - Provide a command-line interface: `python dataset_sampler.py --output dataset_manifest.json --sample-negative-count 19`

2. **Automated Evaluation Runner (`eval_runner.py`)**:
   - Imports `from d1_image_classifier import D1ImageClassifier`
   - Loads `dataset_manifest.json`
   - Runs predictions on every sample
   - Computes evaluation metrics:
     - True Positives (TP), False Positives (FP), True Negatives (TN), False Negatives (FN)
     - Accuracy, Precision, Recall, F1 Score
     - Average Latency (ms) per image
   - Outputs a clean Markdown report `EVALUATION_REPORT.md` and detailed predictions in `eval_results.json`.

Please implement `dataset_sampler.py` and run it now to create `dataset_manifest.json`.
Then implement `eval_runner.py` so it is ready to benchmark once the classifier is loaded.
Notify when complete or write progress.
