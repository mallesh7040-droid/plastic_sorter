import pandas as pd
import numpy as np
import time
from predict import classify

def run_validation(csv_path, label):
    print(f"\n{'='*50}")
    print(f"{label}")
    print(f"{'='*50}")

    df = pd.read_csv(csv_path)
    y_true = df['class_id'].values
    X_raw = df.iloc[:, 3:].values.astype(np.float32)

    print(f"Total samples: {len(df)}")

    start = time.time()
    correct = 0
    per_class_correct = {1:0, 2:0, 3:0, 4:0, 5:0}
    per_class_total   = {1:0, 2:0, 3:0, 4:0, 5:0}

    for i in range(len(df)):
        pred_id, _ = classify(X_raw[i])
        true_id = y_true[i]
        per_class_total[true_id] += 1
        if pred_id == true_id:
            correct += 1
            per_class_correct[true_id] += 1

        if (i+1) % 10000 == 0:
            print(f"  Processed {i+1}/{len(df)}...")

    elapsed = time.time() - start

    print(f"\nOverall Accuracy : {correct}/{len(df)} = {100*correct/len(df):.2f}%")
    print(f"Time taken        : {elapsed:.1f}s  ({elapsed/len(df)*1000:.2f} ms/sample)")
    print(f"\nPer-class accuracy:")
    names = {1:"PE", 2:"PP", 3:"PET", 4:"PS", 5:"PVC"}
    for cls in sorted(per_class_total.keys()):
        tot = per_class_total[cls]
        cor = per_class_correct[cls]
        pct = 100*cor/tot if tot > 0 else 0
        print(f"  {names[cls]:>4} (class {cls}): {cor:>5}/{tot:<5} = {pct:.2f}%")

    return correct, len(df)


if __name__ == "__main__":
    run_validation("SpectrumData_2021Y-testSetLowNoise.csv", "LOW NOISE TEST SET")
    run_validation("SpectrumData_2021Y-testSetHighNoise.csv", "HIGH NOISE TEST SET")
