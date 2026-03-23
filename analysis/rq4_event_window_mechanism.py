from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)

ELECTION_DATES = {
    2011: pd.Timestamp("2011-05-07"),
    2015: pd.Timestamp("2015-09-11"),
    2020: pd.Timestamp("2020-07-10"),
    2025: pd.Timestamp("2025-05-03"),
}


def month_diff(date_series: pd.Series, ref_date: pd.Timestamp) -> pd.Series:
    return (date_series.dt.year - ref_date.year) * 12 + (date_series.dt.month - ref_date.month)


def build_panel() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "COEBiddingResultsPrices.csv")
    df["month"] = pd.to_datetime(df["month"])
    for col in ["premium", "quota", "bids_received"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[df["vehicle_class"].isin(["Category A", "Category B"])].dropna(subset=["premium", "quota", "bids_received"])

    monthly = (
        df.groupby(["vehicle_class", "month"], as_index=False)
        .agg(
            premium=("premium", "mean"),
            quota=("quota", "mean"),
            bids=("bids_received", "mean"),
        )
        .sort_values(["vehicle_class", "month"])
    )

    rows = []
    for year, edate in ELECTION_DATES.items():
        for vclass in ["Category A", "Category B"]:
            sub = monthly[monthly["vehicle_class"] == vclass].copy()
            sub["event_month"] = month_diff(sub["month"], edate)
            window = sub[(sub["event_month"] >= -12) & (sub["event_month"] <= 3)].copy()
            if window.empty:
                continue
            window["election_year"] = year
            window["event_id"] = f"{vclass}_{year}"
            rows.append(window)

    panel = pd.concat(rows, ignore_index=True)
    panel["log_premium"] = np.log(panel["premium"])
    panel["log_quota"] = np.log(panel["quota"])

    panel = panel.sort_values(["event_id", "event_month"]) 
    panel["dlog_premium"] = panel.groupby("event_id")["log_premium"].diff()
    panel["dlog_quota"] = panel.groupby("event_id")["log_quota"].diff()
    panel["dlog_bids"] = panel.groupby("event_id")["bids"].apply(lambda s: np.log(s).diff()).reset_index(level=0, drop=True)

    return panel


def main() -> None:
    panel = build_panel()
    pre = panel[(panel["event_month"] >= -12) & (panel["event_month"] <= 0)].copy()

    # 1) Is quota increasing as election approaches?
    quota_model = smf.ols("log_quota ~ event_month + C(event_id)", data=pre).fit(cov_type="HC3")

    # 2) Do monthly quota changes reduce monthly premium changes near election?
    flow = pre.dropna(subset=["dlog_premium", "dlog_quota", "dlog_bids"]).copy()
    price_model = smf.ols(
        "dlog_premium ~ dlog_quota + dlog_bids + C(event_id)",
        data=flow,
    ).fit(cov_type="HC3")

    # 3) election-proximity effect conditional on quota/bids changes
    prox_model = smf.ols(
        "dlog_premium ~ event_month + dlog_quota + dlog_bids + C(event_id)",
        data=flow,
    ).fit(cov_type="HC3")

    out = {
        "n_obs_pre": int(len(pre)),
        "n_obs_flow": int(len(flow)),
        "quota_event_month_coef": float(quota_model.params.get("event_month", np.nan)),
        "quota_event_month_p": float(quota_model.pvalues.get("event_month", np.nan)),
        "price_on_dquota_coef": float(price_model.params.get("dlog_quota", np.nan)),
        "price_on_dquota_p": float(price_model.pvalues.get("dlog_quota", np.nan)),
        "prox_event_month_coef": float(prox_model.params.get("event_month", np.nan)),
        "prox_event_month_p": float(prox_model.pvalues.get("event_month", np.nan)),
        "prox_dquota_coef": float(prox_model.params.get("dlog_quota", np.nan)),
        "prox_dquota_p": float(prox_model.pvalues.get("dlog_quota", np.nan)),
    }

    pd.DataFrame([out]).to_csv(REPORTS / "rq4_event_window_mechanism_results.csv", index=False)

    lines = [
        "RQ4 Article-Style Mechanism Check",
        "Hypothesis: As elections approach, quota rises, and higher quota lowers prices.",
        f"n_obs_pre={out['n_obs_pre']}, n_obs_flow={out['n_obs_flow']}",
        "",
        f"Quota trend near election (log_quota ~ event_month + event FE): coef={out['quota_event_month_coef']:.6f}, p={out['quota_event_month_p']:.6g}",
        f"Price response to quota change (dlog_price ~ dlog_quota + dlog_bids + event FE): coef={out['price_on_dquota_coef']:.6f}, p={out['price_on_dquota_p']:.6g}",
        f"Proximity effect net of controls (dlog_price ~ event_month + dlog_quota + dlog_bids + event FE): event_month coef={out['prox_event_month_coef']:.6f}, p={out['prox_event_month_p']:.6g}",
        "",
    ]

    supports = (
        out["quota_event_month_p"] < 0.05 and out["quota_event_month_coef"] > 0 and
        out["price_on_dquota_p"] < 0.05 and out["price_on_dquota_coef"] < 0
    )

    if supports:
        lines.append("Conclusion: Evidence supports election-approach quota increase linked to lower prices.")
    else:
        lines.append("Conclusion: Evidence does NOT show a consistent election-approach quota increase causing lower prices.")

    (REPORTS / "rq4_event_window_mechanism_summary.txt").write_text("\n".join(lines), encoding="utf-8")

    print("Saved:")
    print("- reports/rq4_event_window_mechanism_results.csv")
    print("- reports/rq4_event_window_mechanism_summary.txt")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
