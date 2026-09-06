"""Comprehensive audit of the REAL IO-VNBD dataset.
Uses encoding='latin-1' to properly handle (m/s²) and degree symbols in headers.
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd

PROJECT_ROOT = Path("D:/dead_reckoning")
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "IO-VNBD"
OUTPUT_DIR = PROJECT_ROOT / "results" / "io_vnbd"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 75)
    print("AUDITING REAL IO-VNBD DATASET (WITH LATIN-1 ENCODING)")
    print("=" * 75)

    all_csvs = list(DATA_DIR.rglob("*.csv"))
    print(f"Total CSV files found: {len(all_csvs)}")

    inventory = []
    sampling_records = []
    sync_records = []

    s_files = {}
    v_files = {}

    for f in all_csvs:
        name = f.name
        rel_path = str(f.relative_to(PROJECT_ROOT)).replace("\\", "/")
        parent_dir = f.parent.name
        size_b = f.stat().st_size

        if name.startswith("S-") or name.startswith("s-"):
            rec_type = "Smartphone"
            sensor_type = "AndroSensor (IMU + GPS)"
            base_key = name[2:-4].lower() # strip S- and .csv
            s_files[base_key] = f
        elif name.startswith("V-") or name.startswith("v-"):
            rec_type = "Vehicle"
            sensor_type = "Vehicle CAN/ECU (Dynamics + GPS)"
            base_key = name[2:-4].lower() # strip V- and .csv
            v_files[base_key] = f
        else:
            rec_type = "Other CSV"
            sensor_type = "Unknown"

        # Read CSV with latin-1
        try:
            df = pd.read_csv(f, encoding="latin-1")
            row_count = len(df)
            col_count = len(df.columns)
            cols = df.columns.tolist()
        except Exception as e:
            print(f"Error reading {f.name}: {e}")
            continue

        inventory.append({
            "file_path": rel_path,
            "file_size": size_b,
            "file_type": ".csv",
            "parent_sequence": parent_dir,
            "recording_type": rec_type,
            "estimated_sensor_type": sensor_type,
            "row_count": row_count,
            "column_count": col_count,
        })

        # If smartphone, analyze sampling rate and timestamps
        if rec_type == "Smartphone":
            # Look for time column
            time_col = None
            for c in cols:
                if "time" in c.lower() or "date" in c.lower():
                    time_col = c
                    break

            if time_col:
                time_series = pd.to_numeric(df[time_col], errors="coerce").dropna().to_numpy()
                if len(time_series) > 5:
                    dt = np.diff(time_series)
                    dt_pos = dt[dt > 0]
                    if len(dt_pos) > 0:
                        med_dt = float(np.median(dt_pos))
                        mean_dt = float(np.mean(dt_pos))
                        min_dt = float(np.min(dt_pos))
                        max_dt = float(np.max(dt_pos))

                        # Determine units: if median dt ~ 100, it's ms (10 Hz). If median dt ~ 0.1, it's seconds (10 Hz).
                        if 80.0 <= med_dt <= 120.0:
                            unit = "milliseconds"
                            freq_hz = 1000.0 / med_dt
                            dt_sec = med_dt / 1000.0
                        elif 0.08 <= med_dt <= 0.12:
                            unit = "seconds"
                            freq_hz = 1.0 / med_dt
                            dt_sec = med_dt
                        elif 80000.0 <= med_dt <= 120000.0:
                            unit = "microseconds"
                            freq_hz = 1000000.0 / med_dt
                            dt_sec = med_dt / 1000000.0
                        elif 80000000.0 <= med_dt <= 120000000.0:
                            unit = "nanoseconds"
                            freq_hz = 1000000000.0 / med_dt
                            dt_sec = med_dt / 1000000000.0
                        else:
                            unit = "custom"
                            freq_hz = 1.0 / med_dt if med_dt > 0 else 0.0
                            dt_sec = med_dt

                        sampling_records.append({
                            "sequence": name,
                            "time_column": time_col,
                            "timestamp_unit": unit,
                            "median_dt": med_dt,
                            "mean_dt": mean_dt,
                            "min_dt": min_dt,
                            "max_dt": max_dt,
                            "estimated_sampling_frequency_hz": round(freq_hz, 2),
                            "number_of_samples": len(time_series),
                            "duration_seconds": round(len(time_series) * dt_sec, 2),
                        })

    # Save dataset_inventory.csv
    df_inv = pd.DataFrame(inventory)
    df_inv.to_csv(OUTPUT_DIR / "dataset_inventory.csv", index=False)
    print(f"\nSaved results/io_vnbd/dataset_inventory.csv ({len(df_inv)} rows)")

    # Save sampling_rate_audit.csv
    df_samp = pd.DataFrame(sampling_records)
    df_samp.to_csv(OUTPUT_DIR / "sampling_rate_audit.csv", index=False)
    print(f"Saved results/io_vnbd/sampling_rate_audit.csv ({len(df_samp)} rows)")

    # Analyze synchronization between S-* and V-* pairs
    print(f"\nAnalyzing synchronization between {len(s_files)} smartphone and {len(v_files)} vehicle files...")
    matched_keys = sorted(list(set(s_files.keys()).intersection(set(v_files.keys()))))
    print(f"Found {len(matched_keys)} perfectly matching sequence pairs!")

    for k in matched_keys:
        s_p = s_files[k]
        v_p = v_files[k]

        try:
            df_s = pd.read_csv(s_p, encoding="latin-1")
            df_v = pd.read_csv(v_p, encoding="latin-1")
            sync_records.append({
                "sequence_key": k,
                "smartphone_file": s_p.name,
                "vehicle_file": v_p.name,
                "category": s_p.parent.name,
                "smartphone_rows": len(df_s),
                "vehicle_rows": len(df_v),
                "row_difference": abs(len(df_s) - len(df_v)),
                "smartphone_cols": len(df_s.columns),
                "vehicle_cols": len(df_v.columns),
                "smartphone_duration_s": round(len(df_s) * 0.1, 1),
                "vehicle_duration_s": round(len(df_v) * 0.1, 1),
            })
        except Exception as e:
            print(f"Error syncing {k}: {e}")

    df_sync = pd.DataFrame(sync_records)
    df_sync.to_csv(OUTPUT_DIR / "synchronization_audit.csv", index=False)
    print(f"Saved results/io_vnbd/synchronization_audit.csv ({len(df_sync)} pairs)")

    # Print summary insights
    print("\n" + "=" * 75)
    print("AUDIT SUMMARY")
    print("=" * 75)
    print("Sampling Frequency Stats (Smartphone IMU):")
    if not df_samp.empty:
        print(df_samp["estimated_sampling_frequency_hz"].describe())
        print("Timestamp Units:")
        print(df_samp["timestamp_unit"].value_counts())

    print(f"\nTotal synchronized sequences: {len(df_sync)}")
    if not df_sync.empty:
        print(f"Total recorded driving time: {df_sync['smartphone_duration_s'].sum() / 3600.0:.2f} hours")
        print(f"Median sequence duration: {df_sync['smartphone_duration_s'].median():.1f} seconds")
        print(f"Min / Max sequence duration: {df_sync['smartphone_duration_s'].min():.1f}s / {df_sync['smartphone_duration_s'].max():.1f}s")


if __name__ == "__main__":
    main()
