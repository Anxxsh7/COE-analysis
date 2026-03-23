import pandas as pd
import numpy as np
import statsmodels.api as sm
from pathlib import Path
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.outliers_influence import variance_inflation_factor

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)


def parse_singstat_wide(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, header=None)
    labels = raw[0].astype(str)
    ds_idx = labels[labels.str.strip().eq("Data Series")].index[0]

    date_labels = raw.iloc[ds_idx, 1:]
    valid_cols = date_labels[date_labels.notna()].index.tolist()

    records = []
    for row_idx in range(ds_idx + 1, len(raw)):
        series_name = raw.iloc[row_idx, 0]
        if pd.isna(series_name):
            continue
        values = raw.iloc[row_idx, valid_cols]
        for col_idx, value in values.items():
            date_label = str(raw.iloc[ds_idx, col_idx]).strip()
            if date_label and date_label.lower() != "nan":
                records.append((str(series_name).strip(), date_label, value))

    df = pd.DataFrame(records, columns=["series", "period", "value"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"])

    df["month"] = pd.to_datetime(df["period"], format="%Y %b", errors="coerce")
    df = df.dropna(subset=["month"]) 
    return df[["series", "month", "value"]]


def adf_summary(series: pd.Series, name: str) -> dict:
    clean = series.dropna()
    if len(clean) < 20:
        return {"series": name, "adf_stat": np.nan, "p_value": np.nan, "n_obs": len(clean)}
    stat, p, *_ = adfuller(clean)
    return {"series": name, "adf_stat": stat, "p_value": p, "n_obs": len(clean)}


def main() -> None:
    coe = pd.read_csv(ROOT / "COEBiddingResultsPrices.csv")
    coe["month"] = pd.to_datetime(coe["month"])
    for col in ["premium", "quota", "bids_received", "bids_success"]:
        coe[col] = pd.to_numeric(coe[col], errors="coerce")

    coe = coe[coe["vehicle_class"].isin(["Category A", "Category B", "Category E"])].dropna(
        subset=["premium", "quota", "bids_received", "bids_success"]
    )
    coe["success_rate"] = coe["bids_success"] / coe["bids_received"]

    cpi_long = parse_singstat_wide(ROOT / "ConsumerPriceIndex.csv")
    rsi_long = parse_singstat_wide(ROOT / "RetailSalesIndex.csv")

    cpi_extract = (
        cpi_long[cpi_long["series"].isin(["Transport", "Private Transport", "Motor Cars"])]
        .pivot(index="month", columns="series", values="value")
        .reset_index()
    )

    rsi_extract = (
        rsi_long[rsi_long["series"].isin(["Motor Vehicles, Parts & Accessories", "Petrol Service Stations"])]
        .pivot(index="month", columns="series", values="value")
        .reset_index()
    )

    panel = (
        coe.merge(cpi_extract, on="month", how="left")
        .merge(rsi_extract, on="month", how="left")
        .sort_values(["vehicle_class", "month"])
        .reset_index(drop=True)
    )

    panel = panel.rename(
        columns={
            "Transport": "cpi_transport",
            "Private Transport": "cpi_private_transport",
            "Motor Cars": "cpi_motor_cars",
            "Motor Vehicles, Parts & Accessories": "rsi_motor_vehicles",
            "Petrol Service Stations": "rsi_petrol",
        }
    )

    model_df = panel.dropna(subset=["premium", "quota", "bids_received", "cpi_private_transport", "rsi_motor_vehicles"]).copy()

    model_df["log_premium"] = np.log(model_df["premium"])
    model_df["log_quota"] = np.log(model_df["quota"])
    model_df["log_bids"] = np.log(model_df["bids_received"])
    model_df["log_cpi_pt"] = np.log(model_df["cpi_private_transport"])
    model_df["log_rsi_mv"] = np.log(model_df["rsi_motor_vehicles"])

    class_dummies = pd.get_dummies(model_df["vehicle_class"], drop_first=True, prefix="class").astype(float)

    x = pd.concat(
        [
            model_df[["log_quota", "log_bids", "log_cpi_pt", "log_rsi_mv"]].astype(float),
            class_dummies,
        ],
        axis=1,
    )
    y = model_df["log_premium"].astype(float)

    x_const = sm.add_constant(x)
    model = sm.OLS(y, x_const).fit(cov_type="HC3")

    vif_rows = []
    x_vif = x.copy()
    for idx, col in enumerate(x_vif.columns):
        vif_rows.append({"feature": col, "vif": variance_inflation_factor(x_vif.values, idx)})
    vif_df = pd.DataFrame(vif_rows).sort_values("vif", ascending=False)

    adf_rows = [
        adf_summary(model_df["log_premium"], "log_premium"),
        adf_summary(model_df["log_quota"], "log_quota"),
        adf_summary(model_df["log_bids"], "log_bids"),
        adf_summary(model_df["log_cpi_pt"], "log_cpi_private_transport"),
        adf_summary(model_df["log_rsi_mv"], "log_rsi_motor_vehicles"),
    ]
    adf_df = pd.DataFrame(adf_rows)

    diff_df = model_df.sort_values(["vehicle_class", "month"]).copy()
    for col in ["log_premium", "log_quota", "log_bids", "log_cpi_pt", "log_rsi_mv"]:
        diff_df[f"d_{col}"] = diff_df.groupby("vehicle_class")[col].diff()
    diff_df = diff_df.dropna(subset=["d_log_premium", "d_log_quota", "d_log_bids", "d_log_cpi_pt", "d_log_rsi_mv"])

    dx = diff_df[["d_log_quota", "d_log_bids", "d_log_cpi_pt", "d_log_rsi_mv"]].astype(float)
    d_class = pd.get_dummies(diff_df["vehicle_class"], drop_first=True, prefix="class").astype(float)
    dx = pd.concat([dx, d_class], axis=1)
    dy = diff_df["d_log_premium"].astype(float)

    diff_model = sm.OLS(dy, sm.add_constant(dx)).fit(cov_type="HC3")

    panel.to_csv(REPORTS / "stage2_monthly_panel.csv", index=False)
    adf_df.to_csv(REPORTS / "stage2_adf_results.csv", index=False)
    vif_df.to_csv(REPORTS / "stage2_vif_results.csv", index=False)

    coef = pd.DataFrame(
        {
            "variable": model.params.index,
            "coef": model.params.values,
            "p_value": model.pvalues.values,
        }
    )
    coef.to_csv(REPORTS / "stage2_model_coefficients.csv", index=False)

    coef_diff = pd.DataFrame(
        {
            "variable": diff_model.params.index,
            "coef": diff_model.params.values,
            "p_value": diff_model.pvalues.values,
        }
    )
    coef_diff.to_csv(REPORTS / "stage2_diff_model_coefficients.csv", index=False)

    summary_lines = [
        "Stage 2 Extended Model Summary",
        f"Observations (level model): {len(model_df)}",
        f"R-squared (level model): {model.rsquared:.4f}",
        f"Adj. R-squared (level model): {model.rsquared_adj:.4f}",
        "",
        "Key coefficients (HC3 robust SE):",
    ]
    for key in ["log_quota", "log_bids", "log_cpi_pt", "log_rsi_mv"]:
        if key in model.params.index:
            summary_lines.append(
                f"- {key}: coef={model.params[key]:.4f}, p={model.pvalues[key]:.3e}"
            )

    summary_lines += [
        "",
        f"Observations (differenced model): {len(diff_df)}",
        f"R-squared (differenced model): {diff_model.rsquared:.4f}",
        "",
        "Key differenced coefficients (HC3 robust SE):",
    ]
    for key in ["d_log_quota", "d_log_bids", "d_log_cpi_pt", "d_log_rsi_mv"]:
        if key in diff_model.params.index:
            summary_lines.append(
                f"- {key}: coef={diff_model.params[key]:.4f}, p={diff_model.pvalues[key]:.3e}"
            )

    summary_lines += [
        "",
        "Outputs saved:",
        "- reports/stage2_monthly_panel.csv",
        "- reports/stage2_adf_results.csv",
        "- reports/stage2_vif_results.csv",
        "- reports/stage2_model_coefficients.csv",
        "- reports/stage2_diff_model_coefficients.csv",
    ]

    out_path = REPORTS / "stage2_model_summary.txt"
    out_path.write_text("\n".join(summary_lines), encoding="utf-8")

    print(out_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
