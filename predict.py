import joblib
import numpy as np

CLASS_NAMES = {1: "PE", 2: "PP", 3: "PET", 4: "PS", 5: "PVC"}

model = joblib.load('nir_pipeline.pkl')
svm = model['svm']
pca = model['pca']
dark_ref = model['dark_ref']
feat_min = model['feat_min']
feat_max = model['feat_max']


def savgol_fpga(X):
    kernel = np.array([-22, 88, 124, 88, -22], dtype=np.float32) / 256.0
    out = np.zeros_like(X, dtype=np.float32)
    for i in range(X.shape[0]):
        out[i] = np.convolve(X[i], kernel, mode="same")
    return out


def snv(X):
    mu = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True)
    std[std == 0] = 1.0
    return (X - mu) / std


def first_derivative(X):
    return np.diff(X, n=1, axis=1)


def build_features(X_snv):
    deriv = first_derivative(X_snv)
    deriv = snv(deriv)
    return np.hstack([X_snv, deriv])


def quantise_features(X_float, feat_min, feat_max):
    X = X_float.astype(np.float32)
    den = feat_max - feat_min
    den = np.where(den == 0, 1.0, den)
    Xq = np.round(((X - feat_min) / den) * 255) - 128
    Xq = np.clip(Xq, -128, 127).astype(np.float32)
    return Xq


def classify(raw_spectrum_224):
    """
    raw_spectrum_224 : list or 1D array of 224 raw NIR channel values
    Returns: (class_id, class_name)
    """
    X_raw = np.array(raw_spectrum_224, dtype=np.float32).reshape(1, -1)

    if X_raw.shape[1] != 224:
        raise ValueError(f"Expected 224 channels, got {X_raw.shape[1]}")

    X = X_raw - dark_ref
    X = savgol_fpga(X)
    X_snv = snv(X)
    X_feat = build_features(X_snv)
    X_pca = pca.transform(X_feat)
    X_q = quantise_features(X_pca, feat_min, feat_max)

    pred = int(svm.predict(X_q)[0])
    name = CLASS_NAMES.get(pred, "UNKNOWN")
    return pred, name


if __name__ == "__main__":
    dummy = np.random.rand(224) * 1000
    cls_id, cls_name = classify(dummy)
    print(f"Test prediction: class_id={cls_id}, class_name={cls_name}")