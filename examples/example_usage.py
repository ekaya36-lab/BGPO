from bgpo import BGPO
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split


def main():
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=10,
        n_redundant=2,
        weights=[0.9, 0.1],
        class_sep=1.0,
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

    print("Original class counts:", dict(zip(*__import__('numpy').unique(y_train, return_counts=True))))
    print("Resampled class counts:", dict(zip(*__import__('numpy').unique(y_res, return_counts=True))))
    print("F1:", f1_score(y_test, y_pred))
    print("Balanced Accuracy:", balanced_accuracy_score(y_test, y_pred))


if __name__ == "__main__":
    main()
