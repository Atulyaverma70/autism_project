# model.py
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

features_folder = "features"
n_mfcc = 20  # must match mfcc_extract.py and UI

def label_from_filename(fname):
    name = fname.lower()
    if "aut_" in name or "autistic" in name:
        return 1
    if "non_" in name or "split-" in name or "control" in name or "typical" in name:
        return 0
    # default: raise so you can fix file names
    raise ValueError(f"Filename {fname} doesn't match expected label patterns. Rename or adjust label_from_filename.")

def load_features_and_labels(folder):
    files = [f for f in os.listdir(folder) if f.endswith('.npy')]
    X_list, y_list = [], []
    for f in files:
        try:
            arr = np.load(os.path.join(folder, f))
            arr = np.asarray(arr).reshape(-1)  # ensure 1D length n_mfcc
            if arr.shape[0] != n_mfcc:
                print(f"Skipping {f}: expected {n_mfcc} MFCCs, got {arr.shape[0]}")
                continue
            label = label_from_filename(f)
            X_list.append(arr)
            y_list.append(label)
        except Exception as e:
            print("Error loading", f, e)
    if not X_list:
        raise RuntimeError("No valid feature files loaded. Run mfcc_extract and check filenames.")
    return np.vstack(X_list), np.array(y_list)

def train_and_save():
    X, y = load_features_and_labels(features_folder)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)

    classifiers = {
        "rf": RandomForestClassifier(n_estimators=100, random_state=42),
        "svm": SVC(kernel='rbf', probability=True, C=1, gamma='scale', random_state=42),
        "nb": GaussianNB(),
        "ann": MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42),
    }

    for key, clf in classifiers.items():
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", clf)
        ])
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"{key.upper()} accuracy:", acc)
        print(classification_report(y_test, preds, digits=4))
        outpath = f"{key}.pkl"
        joblib.dump(pipeline, outpath)
        print("Saved:", outpath)

if __name__ == "__main__":
    train_and_save()
