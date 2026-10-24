"""Train the intent classifier, evaluate it, and save the model + report.

Usage:  python train_eval.py
"""
import json
from pathlib import Path

from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from chatbot.intent_classifier import IntentClassifier, build_pipeline
from chatbot.knowledge_base import KnowledgeBase
from chatbot.preprocessing import preprocess

REPORT = Path("reports")


def main() -> None:
    kb = KnowledgeBase()
    X, y = kb.training_data()
    Xp = [preprocess(t) for t in X]
    print(f"{len(X)} training phrases, {len(set(y))} intents")

    X_tr, X_te, y_tr, y_te = train_test_split(Xp, y, test_size=0.25, stratify=y, random_state=42)
    model = build_pipeline().fit(X_tr, y_tr)
    pred = model.predict(X_te)
    acc = accuracy_score(y_te, pred)
    report = classification_report(y_te, pred, zero_division=0)
    print(f"\nHold-out accuracy: {acc:.3f}\n{report}")

    cv = cross_val_score(build_pipeline(), Xp, y, cv=StratifiedKFold(5, shuffle=True, random_state=42))
    print(f"5-fold CV accuracy: {cv.mean():.3f} (+/- {cv.std():.3f})")

    REPORT.mkdir(exist_ok=True)
    (REPORT / "metrics.json").write_text(json.dumps(
        {"holdout_accuracy": round(acc, 4), "cv_mean": round(float(cv.mean()), 4),
         "cv_std": round(float(cv.std()), 4), "n_samples": len(X), "n_intents": len(set(y))}, indent=2))
    (REPORT / "classification_report.txt").write_text(report)

    IntentClassifier().train(X, y).save()  # final model on all data
    print("\nModel saved to models/intent_model.joblib")


if __name__ == "__main__":
    main()
