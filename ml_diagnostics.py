# ml_diagnostics.py
# ML model diagnostics: bootstrap AUC CI, Brier score, calibration curve

import numpy as np
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

def run_ml_diagnostics(y_test, y_prob, model_name="Random Forest"):
    np.random.seed(42)
    aucs = []
    for _ in range(1000):
        idx = np.random.randint(0, len(y_test), len(y_test))
        if len(np.unique(np.array(y_test)[idx])) < 2:
            continue
        aucs.append(roc_auc_score(np.array(y_test)[idx], y_prob[idx]))

    auc_mean = roc_auc_score(y_test, y_prob)
    auc_lo   = np.percentile(aucs, 2.5)
    auc_hi   = np.percentile(aucs, 97.5)
    print(f"{model_name} AUC: {auc_mean:.3f} (95% CI {auc_lo:.3f}-{auc_hi:.3f})")

    brier = brier_score_loss(y_test, y_prob)
    print(f"Brier Score: {brier:.3f}")

    prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10)
    print("\nCalibration (predicted -> observed):")
    for pt, pp in zip(prob_pred, prob_true):
        print(f"   {pt:.2f}   ->   {pp:.2f}")

    plt.figure(figsize=(8, 6))
    plt.plot(prob_pred, prob_true, "s-", color="#2E86AB", label=model_name)
    plt.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives (Observed Rate)")
    plt.title(f"Calibration Curve — {model_name}\nER Visit Prediction", fontweight="bold")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("01_data/processed/rf_calibration_curve.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Calibration curve saved.")
