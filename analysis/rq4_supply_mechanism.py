from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)

ELECTION_YEARS = {2011, 2015, 2020, 2025}


def build_panel() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "COEBiddingResultsPrices.csv")
    df["month"] = pd.to_datetime(df["month"])
    for col in ["premium", "quota", "bids_received"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[df["vehicle_class"].isin(["Category A", "Category B", "Category E"])].dropna(
        subset=["premium", "quota", "bids_received"]
    )

    df["year"] = df["month"].dt.year

    # class-year aggregation to avoid mixing bid frequency noise
    yearly = (
        df.groupby(["vehicle_class", "year"], as_index=False)
        .agg(
            premium=("premium", "mean"),
            quota=("quota", "mean"),
            bids=("bids_received", "mean"),
        )
        .sort_values(["vehicle_class", "year"])
    )

    # within-class annual changes
    for col in ["premium", "quota", "bids"]:
        yearly[f"log_{col}"] = np.log(yearly[col])
        yearly[f"dlog_{col}"] = yearly.groupby("vehicle_class")[f"log_{col}"].diff()

    yearly["election_year"] = yearly["year"].isin(ELECTION_YEARS).astype(int)

    panel = yearly.dropna(subset=["dlog_premium", "dlog_quota", "dlog_bids"]).copy()
    return panel


def run_models(panel: pd.DataFrame):
    # Path A: election -> quota change
    m_a = smf.ols("dlog_quota ~ election_year + C(vehicle_class)", data=panel).fit(cov_type="HC3")

    # Path B (without election): quota change -> price change
    m_b = smf.ols("dlog_premium ~ dlog_quota + C(vehicle_class)", data=panel).fit(cov_type="HC3")

    # Path B (with election + demand control): does quota still predict and election remain?
    m_b2 = smf.ols(
        "dlog_premium ~ dlog_quota + dlog_bids + election_year + C(vehicle_class)",
        data=panel,
    ).fit(cov_type="HC3")

    return m_a, m_b, m_b2


def save_outputs(panel: pd.DataFrame, m_a, m_b, m_b2):
    # Mechanism summary table
    res = {
        "n_obs": len(panel),
        "pathA_election_to_quota_coef": float(m_a.params.get("election_year", np.nan)),
        "pathA_election_to_quota_p": float(m_a.pvalues.get("election_year", np.nan)),
        "pathB_quota_to_price_coef": float(m_b.params.get("dlog_quota", np.nan)),
        "pathB_quota_to_price_p": float(m_b.pvalues.get("dlog_quota", np.nan)),
        "pathB2_quota_to_price_coef": float(m_b2.params.get("dlog_quota", np.nan)),
        "pathB2_quota_to_price_p": float(m_b2.pvalues.get("dlog_quota", np.nan)),
        "pathB2_election_direct_coef": float(m_b2.params.get("election_year", np.nan)),
        "pathB2_election_direct_p": float(m_b2.pvalues.get("election_year", np.nan)),
        "r2_pathA": float(m_a.rsquared),
        "r2_pathB": float(m_b.rsquared),
        "r2_pathB2": float(m_b2.rsquared),
    }

    pd.DataFrame([res]).to_csv(REPORTS / "rq4_supply_mechanism_results.csv", index=False)

    # Save model coefficient tables
    def coef_table(model, name):
        ci = model.conf_int()
        return pd.DataFrame(
            {
                "model": name,
                "term": model.params.index,
                "coef": model.params.values,
                "p_value": model.pvalues.values,
                "ci_low_95": ci.iloc[:, 0].values,
                "ci_high_95": ci.iloc[:, 1].values,
            }
        )

    coef_all = pd.concat(
        [
            coef_table(m_a, "pathA_dlog_quota_on_election"),
            coef_table(m_b, "pathB_dlog_price_on_dlog_quota"),
            coef_table(m_b2, "pathB2_dlog_price_on_quota_bids_election"),
        ],
        ignore_index=True,
    )
    coef_all.to_csv(REPORTS / "rq4_supply_mechanism_coefficients.csv", index=False)

    # Human summary
    lines = [
        "RQ4 Supply Mechanism Test",
        "Question: Does election year increase supply (quota), which then lowers COE prices?",
        "",
        f"Sample: {len(panel)} class-year observations (A/B/E, annual changes)",
        "",
        "Path A (Election -> Quota change):",
        f"  coef={res['pathA_election_to_quota_coef']:.6f}, p={res['pathA_election_to_quota_p']:.6g}",
        "",
        "Path B (Quota change -> Price change):",
        f"  coef={res['pathB_quota_to_price_coef']:.6f}, p={res['pathB_quota_to_price_p']:.6g}",
        "",
        "Path B2 (with demand control + election):",
        f"  dlog_quota coef={res['pathB2_quota_to_price_coef']:.6f}, p={res['pathB2_quota_to_price_p']:.6g}",
        f"  election direct coef={res['pathB2_election_direct_coef']:.6f}, p={res['pathB2_election_direct_p']:.6g}",
        "",
    ]

    # logic-based conclusion
    if res["pathA_election_to_quota_p"] < 0.05 and res["pathA_election_to_quota_coef"] > 0 and res["pathB_quota_to_price_p"] < 0.05 and res["pathB_quota_to_price_coef"] < 0:
        lines.append("Conclusion: Data supports the mechanism election-year -> higher quota -> lower prices.")
    else:
        lines.append("Conclusion: Data does not provide strong support for a consistent election-year -> higher quota -> lower prices mechanism.")

    (REPORTS / "rq4_supply_mechanism_summary.txt").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    panel = build_panel()
    m_a, m_b, m_b2 = run_models(panel)
    save_outputs(panel, m_a, m_b, m_b2)
    print("Saved:")
    print("- reports/rq4_supply_mechanism_results.csv")
    print("- reports/rq4_supply_mechanism_coefficients.csv")
    print("- reports/rq4_supply_mechanism_summary.txt")
