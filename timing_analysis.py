import pandas as pd
import numpy as np
import time
import joblib

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


def classify_timed(raw_spectrum_224):
    """Returns (class_id, class_name, stage_timings_dict)"""
    timings = {}
    X_raw = np.array(raw_spectrum_224, dtype=np.float32).reshape(1, -1)

    t0 = time.perf_counter()
    X = X_raw - dark_ref
    timings['dark_correction'] = time.perf_counter() - t0

    t0 = time.perf_counter()
    X = savgol_fpga(X)
    timings['sg_smoothing'] = time.perf_counter() - t0

    t0 = time.perf_counter()
    X_snv = snv(X)
    timings['snv_normalize'] = time.perf_counter() - t0

    t0 = time.perf_counter()
    X_feat = build_features(X_snv)
    timings['feature_engineering'] = time.perf_counter() - t0

    t0 = time.perf_counter()
    X_pca = pca.transform(X_feat)
    timings['pca_transform'] = time.perf_counter() - t0

    t0 = time.perf_counter()
    X_q = quantise_features(X_pca, feat_min, feat_max)
    timings['quantization'] = time.perf_counter() - t0

    t0 = time.perf_counter()
    pred = int(svm.predict(X_q)[0])
    timings['svm_predict'] = time.perf_counter() - t0

    name = CLASS_NAMES.get(pred, "UNKNOWN")
    return pred, name, timings


if __name__ == "__main__":
    df = pd.read_csv('sample_validation.csv')
    X_raw = df.iloc[:, 3:].values.astype(np.float32)

    # Warm-up run (first call always slower due to JIT/cache warming)
    _ = classify_timed(X_raw[0])

    N_RUNS = 200
    all_timings = {k: [] for k in
                   ['dark_correction','sg_smoothing','snv_normalize',
                    'feature_engineering','pca_transform',
                    'quantization','svm_predict']}
    total_times = []

    for i in range(N_RUNS):
        sample = X_raw[i % len(X_raw)]
        t0 = time.perf_counter()
        pred, name, timings = classify_timed(sample)
        total = time.perf_counter() - t0
        total_times.append(total * 1000)  # ms
        for k, v in timings.items():
            all_timings[k].append(v * 1000)  # ms

    print(f"\n{'='*60}")
    print(f"INFERENCE TIMING ANALYSIS  (N={N_RUNS} runs)")
    print(f"{'='*60}\n")

    print(f"{'STAGE':<25} {'MEAN':>8} {'MIN':>8} {'MAX':>8} {'P95':>8}  (ms)")
    print("-" * 60)
    for stage, times in all_timings.items():
        arr = np.array(times)
        print(f"{stage:<25} {arr.mean():>8.3f} {arr.min():>8.3f} "
              f"{arr.max():>8.3f} {np.percentile(arr,95):>8.3f}")

    arr = np.array(total_times)
    print("-" * 60)
    print(f"{'TOTAL (end-to-end)':<25} {arr.mean():>8.3f} {arr.min():>8.3f} "
          f"{arr.max():>8.3f} {np.percentile(arr,95):>8.3f}")

    print(f"\nThroughput: {1000/arr.mean():.1f} samples/sec")
    print(f"Worst case (P95): {np.percentile(arr,95):.2f} ms")
    print(f"Worst case (max): {arr.max():.2f} ms")