# Liver Transplant Outcomes Prediction

A machine learning analysis for predicting 30-day hospital readmission and post-transplant complications in liver transplant patients.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## Overview

This project applies machine learning techniques to predict adverse outcomes following liver transplantation using de-identified patient registry data. The analysis compares multiple classification algorithms and identifies key risk factors associated with poor outcomes.

### Key Findings

| Prediction Task | Best Model | AUC-ROC |
|----------------|------------|---------|
| 30-Day Readmission | Logistic Regression | 0.52 |
| Post-Transplant Complications | Random Forest | 0.73 |

**Top Risk Factors for Complications:**
1. Recipient intubation status at transplant
2. Hypertension
3. Smoking history (within 1 year)
4. Elevated liver enzymes (AST, ALT)
5. Cold ischemia time

## Project Structure

```
liver-transplant-prediction/
│
├── README.md                           # Project documentation
├── LICENSE                             # MIT License
├── requirements.txt                    # Python dependencies
│
├── notebooks/
│   └── liver_transplant_analysis.ipynb # Main analysis notebook
│
├── src/
│   └── liver_transplant_prediction.py  # Clean script version
│
├── reports/
│   └── Liver_Transplant_Analysis_Report.docx  # Full analysis report
│
├── figures/
│   ├── roc_curves.png                  # ROC curve comparison
│   ├── feature_importance.png          # Feature importance plot
│   ├── confusion_matrix.png            # Best model confusion matrix
│   ├── model_comparison.png            # All metrics comparison
│   └── class_distribution.png          # Class balance visualization
│
└── results/
    ├── model_results_summary.csv       # Model performance metrics
    ├── feature_importances.csv         # Readmission feature rankings
    └── composite_feature_importances.csv # Complication feature rankings
```

## Methods

### Data
- **Source:** Liver Outcome Monitoring Registry (de-identified)
- **Patients:** ~2,400 liver transplant recipients
- **Features:** 40-52 clinical variables including donor characteristics, recipient demographics, lab values, and transplant details

### Preprocessing
- Missing value imputation using K-Nearest Neighbors (KNN)
- Label encoding for categorical variables
- Feature standardization for distance-based algorithms

### Class Imbalance Handling
- Class weighting (penalizing minority class misclassification)
- Random undersampling for comparison

### Models Evaluated
| Model | Description |
|-------|-------------|
| Logistic Regression | Linear baseline with L2 regularization |
| Random Forest | Ensemble of 100 decision trees |
| Gradient Boosting | Sequential ensemble with 100 estimators |
| K-Nearest Neighbors | Instance-based with k=5, distance-weighted |

### Evaluation Metrics
- **AUC-ROC:** Primary metric for imbalanced classification
- **Precision/Recall/F1:** Secondary metrics
- **5-Fold Stratified Cross-Validation:** For robust estimation

## Results

### Readmission Prediction
Readmission prediction proved challenging (AUC ~0.52), likely because:
- Post-discharge factors (medication adherence, social support) not captured
- Readmission influenced by events occurring after discharge
- Consistent with published literature on readmission prediction difficulty

### Complication Prediction
Complication prediction showed meaningful performance (AUC 0.73):
- Clinical status at transplant directly affects immediate outcomes
- Risk factors are captured in available data
- Model can identify high-risk patients for targeted intervention

## Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/liver-transplant-prediction.git
cd liver-transplant-prediction

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

```python
# Run the full analysis
python src/liver_transplant_prediction.py

# Or import functions for custom analysis
from src.liver_transplant_prediction import (
    load_and_preprocess_data,
    train_and_evaluate_models,
    get_feature_importance
)
```

## Requirements

- Python 3.8+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

See `requirements.txt` for complete dependencies.

## Clinical Implications

This model could support clinical decision-making by:
- Identifying high-risk patients before discharge
- Enabling targeted interventions (extended monitoring, follow-up scheduling)
- Informing resource allocation for post-transplant care
- Highlighting modifiable risk factors (smoking, hypertension management)

## Limitations

- Single-center data may limit generalizability
- External validation needed before clinical deployment
- Post-discharge factors not captured in dataset
- Class imbalance affects model performance despite mitigation

## References

- Feng S, et al. (2006). Characteristics Associated with Liver Graft Failure: The Concept of a Donor Risk Index. *American Journal of Transplantation*, 6:783-790.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

**Rohini Vishwanathan**

## Acknowledgments

- Data provided by Liver Outcome Monitoring Registry
- Analysis conducted during internship at Purgo AI
- Claude (Anthropic) assisted with code development and documentation
