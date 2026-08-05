import pandas as pd
import numpy as np
from scipy.optimize import nnls
import os

def main():
    # Build Feature Dictionary
    feature_dict = {}
    for f in ['circuit_features_master.csv', 'deep_validation_results.csv']:
        if os.path.exists(f):
            df = pd.read_csv(f)
            for _, row in df.iterrows():
                m = row.get("model_type", row.get("model"))
                s = int(row.get("parameter_size", row.get("size", 0)))
                if "total_assignments" in df.columns and not pd.isna(row["total_assignments"]):
                    feature_dict[(m, s)] = {
                        "domain_size": row.get("domain_size", 0),
                        "total_assignments": row["total_assignments"],
                        "lookup_span": row.get("lookup_span", 0),
                        "total_const_size": row.get("total_const_size", 0)
                    }

    # Gather all benchmarks
    all_files = ['circuit_features_master.csv', 'deep_validation_results.csv',
                 'high_res_rigorous_results.csv', 'transformer_block_results.csv']

    dataset_rows = []
    for f in all_files:
        if not os.path.exists(f): continue
        df = pd.read_csv(f)
        time_col = next((c for c in ["actual_proving_time", "proving_time", "time"] if c in df.columns), None)
        if not time_col: continue

        for _, row in df.iterrows():
            m = row.get("model_type", row.get("model", ""))
            s = int(row.get("parameter_size", row.get("size", 0)))
            t = row[time_col]

            if pd.isna(t): continue
            if (m, s) in feature_dict:
                feat = feature_dict[(m, s)]
                dataset_rows.append({
                    "model_type": str(m),
                    "size": s,
                    "actual_proving_time": t,
                    "domain_size": feat["domain_size"],
                    "total_assignments": feat["total_assignments"],
                    "lookup_span": feat["lookup_span"],
                    "total_const_size": feat["total_const_size"]
                })

    df_all = pd.DataFrame(dataset_rows).drop_duplicates(subset=["model_type", "size", "actual_proving_time"])

    # Split by Architecture
    df_cnn = df_all[df_all["model_type"].str.contains("CNN", case=False, na=False)]
    df_trans = df_all[df_all["model_type"].str.contains("Transformer", case=False, na=False)]

    print("=" * 80)
    print(" LEAVE-ONE-ARCHITECTURE-OUT (LOAO) ZERO-SHOT PREDICTION ")
    print("=" * 80)

    features = ["domain_size", "total_assignments", "lookup_span", "total_const_size"]

    # EXPERIMENT A: Train on CNN, Predict Transformer
    print("\n--- EXPERIMENT A: Train on CNNs -> Predict Transformers ---")
    w_cnn, _ = nnls(df_cnn[features].values, df_cnn["actual_proving_time"].values)

    preds_trans = df_trans[features].values @ w_cnn
    errors_trans = np.abs(df_trans["actual_proving_time"].values - preds_trans) / df_trans["actual_proving_time"].values * 100

    print(f"Trained on {len(df_cnn)} CNNs. Tested on {len(df_trans)} Transformers.")
    print(f"--> Zero-Shot Prediction Error (MAPE): {np.mean(errors_trans):.2f}%\n")

    # EXPERIMENT B: Train on Transformer, Predict CNN
    print("--- EXPERIMENT B: Train on Transformers -> Predict CNNs ---")
    w_trans, _ = nnls(df_trans[features].values, df_trans["actual_proving_time"].values)

    preds_cnn = df_cnn[features].values @ w_trans
    errors_cnn = np.abs(df_cnn["actual_proving_time"].values - preds_cnn) / df_cnn["actual_proving_time"].values * 100

    print(f"Trained on {len(df_trans)} Transformers. Tested on {len(df_cnn)} CNNs.")
    print(f"--> Zero-Shot Prediction Error (MAPE): {np.mean(errors_cnn):.2f}%")
    print("=" * 80)

if __name__ == "__main__":
    main()
