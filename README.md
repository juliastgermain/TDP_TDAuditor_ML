# TDP_TDAuditor_ML



# Mass Spectrometry Hit Prediction & Analysis

This repository contains scripts and tools for preprocessing, matching, modeling, and analyzing mass spectrometry hit data (e.g., TopFD, FlashDeconv, PRO/PSPDXtract). All code is designed for reproducibility, ease of extension, and cross-dataset compatibility as used in my thesis research.

---

## Contents

**1. File Annotation and Preprocessing**
- [`match_and_annotate_hits.py`](match_and_annotate_hits.py):  
    Annotates TDAuditor reports with "hit" labels by matching scan numbers and file names against TopPIC/FlashDeconv/Published hit reports. Also merges instrument metadata.  
    **Outputs:** TSV files with new "hit" and "Instrument" columns for downstream modeling.

**2. Machine Learning Models**
- [`train_logit_model.py`](train_logit_model.py):  
    Flexible pipeline for logistic regression modeling to predict "hit" outcomes. Includes one-hot encoding, SMOTE balancing, group-aware cross-validation, feature engineering, and prediction output.  
    **Usage:**  
    ```
    python train_logit_model.py <input_data_file.tsv>
    ```
    Outputs test set results and writes `<input_data_file>_predictions.tsv` with predicted probabilities/labels.

**3. Graphs and Visualization**
- [`plot_model_graphs.py`](plot_model_graphs.py):  
    Standardized plots for model calibration, feature weights, and Venn overlap between predicted and actual hits.
    **Usage:**  
    ```
    python plot_model_graphs.py <predictions_file.tsv>
    ```
    For Venn plots or custom analyses, modify/extend the function calls as needed.

---

## Typical Usage

1. **Annotate hits and add instrument info:**  
   ```
   python match_and_annotate_hits.py
   ```

2. **Train model & predict:**  
   ```
   python train_logit_model.py tdauditor_flashdeconv_with_instrument.tsv
   ```

3. **Generate analysis plots:**  
   ```
   python plot_model_graphs.py tdauditor_flashdeconv_with_instrument_predictions.tsv
   ```

---

## Requirements

- Python 3.7+
- pandas, numpy, scikit-learn, imbalanced-learn, matplotlib, seaborn, matplotlib-venn

```
pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn matplotlib-venn
```

---

## Notes

- Code is modular: simply supply the correct input filename and the pipeline will process TopFD, FlashDeconv, or PRO datasets with no code modifications.
- Make sure all file paths in the scripts are updated to match your file locations if running outside this repository.
- For detailed logic, refer to docstrings in each script.

Detailed pipeline incorporating instruments, deconvolution engines, and final ML training dataset
![Detailed pipeline incorporating instruments, deconvolution engines, and final ML
training dataset](images/Thesis_workflow.png)
