# BGPO: Boundary-Guided Composite Pool Oversampling

BGPO is a pool-based oversampling method for binary imbalanced classification.
It first constructs a composite synthetic candidate pool using four generators:
SMOTE, ADASYN, MWMOTE, and G-SMOTE. Instead of adding the whole pool or drawing
samples randomly, BGPO selects the synthetic candidates that are most informative
with respect to the decision boundary.

The implementation follows a simple `fit_resample(X, y)` interface.

## Method summary

BGPO separates oversampling into two stages:

1. **Composite pool construction**  
   Candidate synthetic minority samples are generated using SMOTE, ADASYN,
   MWMOTE, and G-SMOTE, then merged into a single pool.

2. **Boundary-guided selection**  
   For each candidate sample, BGPO computes its average distance to the nearest
   minority samples and its average distance to the nearest majority samples in
   the original training set. Candidates are preferred when they are:

   - similarly close to both classes, indicating proximity to the decision boundary;
   - not far away from both classes, avoiding isolated or uninformative regions.

The selected candidates are appended to the original training set.

## Installation

```bash
pip install -r requirements.txt
```

For local development:

```bash
pip install -e .
```

## Quick start

```python
from bgpo import BGPO
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split

X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=10,
    n_redundant=2,
    weights=[0.9, 0.1],
    random_state=42,
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

sampler = BGPO(augmentation_rate=1.0, random_state=42)
X_res, y_res = sampler.fit_resample(X_train, y_train)

clf = RandomForestClassifier(random_state=42)
clf.fit(X_res, y_res)
y_pred = clf.predict(X_test)

print("Balanced Accuracy:", balanced_accuracy_score(y_test, y_pred))
```

## Repository structure

```text
BGPO/
├── bgpo/
│   ├── __init__.py
│   └── bgpo.py
├── examples/
│   └── example_usage.py
├── README.md
├── requirements.txt
├── pyproject.toml
├── LICENSE
└── CITATION.cff
```

## Citation

If you use this code in your research, please cite:

```text
Kaya, E., Korkmaz, S., & Şahman, M. A.
Boundary-Guided Sample Selection from a Composite Synthetic Pool for Imbalanced Classification.
Manuscript under review.
```

## License

This project is released under the MIT License.
