from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
PLOTS = ROOT / "plots"
REPORTS.mkdir(exist_ok=True)
PLOTS.mkdir(exist_ok=True)

# Singapore GE dates (main elections used in project)
ELECTION_DATES = {
    2011: pd.Timestamp("2011-05-07"),
    2015: pd.Timestamp("2015-09-11"),
    2020: pd.Timestamp("2020-07-10"),
    2025: pd.Timestamp("2025-05-03"),
}


def month_diff(date_series: pd.Series, ref_date: pd.Timestamp) -> pd.Series:
    return (date_series.dt.year - ref_date.year) * 12 + (date_series.dt.month - ref_date.month)


def load_monthly_ab() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "COEBiddingResultsPrices.csv")
    df["month"] = pd.to_datetime(df["month"])  # already month-level string
    df["premium"] = pd.to_numeric(df["premium"], errors="coerce")
    df = df[df["vehicle_class"].isin(["Category A", "Category B"])].dropna(subset=["premium"])

    # Convert bid-level to monthly average to match article-style monthly narrative
    monthly = (
        df.groupby(["vehicle_class", "month"], as_index=False)
        .agg(premium=("premium", "mean"))
        .sort_values(["vehicle_class", "month"])
    )
    return monthly


def build_event_panel(monthly: pd.DataFrame, pre_months: int = 12, post_months: int = 3) -> pd.DataFrame:
    rows = []
    for election_year, election_date in ELECTION_DATES.items():
        for vclass in ["Category A", "Category B"]:
            sub = monthly[monthly["vehicle_class"] == vclass].copy()
            sub["event_month"] = month_diff(sub["month"], election_date)
            window = sub[(sub["event_month"] >= -pre_months) & (sub["event_month"] <= post_months)].copy()
            if window.empty:
                continue
            window["election_year"] = election_year
            window["event_id"] = f"{vclass}_{election_year}"
            rows.append(window)

    panel = pd.concat(rows, ignore_index=True)
    panel["log_premium"] = np.log(panel["premium"])
    return panel


def event_metrics(panel: pd.DataFrame) -> pd.DataFrame:
    out = []
    for event_id, g in panel.groupby("event_id"):
        g = g.sort_values("event_month")

        pre = g[(g["event_month"] >= -12) & (g["event_month"] <= -1)]
        if pre.empty:
            continue
        peak_pre = pre["premium"].max()

        # election month point (0), if missing then nearest pre-election month
        if (g["event_month"] == 0).any():
            election_price = g.loc[g["event_month"] == 0, "premium"].iloc[0]
        else:
            election_price = g.loc[g["event_month"] < 0, "premium"].iloc[-1]

        drop_from_peak_pct = (election_price / peak_pre - 1) * 100
        pre_start = g.loc[g["event_month"] == -12, "premium"]
        if len(pre_start) == 0:
            pre_start = pre["premium"].iloc[0]
        else:
            pre_start = pre_start.iloc[0]
        change_12m_pct = (election_price / pre_start - 1) * 100

        # near vs early window means
        near = g[(g["event_month"] >= -3) & (g["event_month"] <= 0)]["premium"].mean()
        early = g[(g["event_month"] >= -12) & (g["event_month"] <= -9)]["premium"].mean()
        near_vs_early_pct = (near / early - 1) * 100

        out.append(
            {
                "event_id": event_id,
                "vehicle_class": g["vehicle_class"].iloc[0],
                "election_year": int(g["election_year"].iloc[0]),
                "peak_pre_price": peak_pre,
                "election_month_price": election_price,
                "drop_from_peak_pct": drop_from_peak_pct,
                "change_12m_to_election_pct": change_12m_pct,
                "near_vs_early_pct": near_vs_early_pct,
            }
        )

    return pd.DataFrame(out).sort_values(["vehicle_class", "election_year"])


def run_tests(panel: pd.DataFrame, metrics: pd.DataFrame) -> dict:
    results = {}

    # pooled pre-election slope test by class (article-style: did prices come down approaching election?)
    pre = panel[(panel["event_month"] >= -12) & (panel["event_month"] <= 0)].copy()

    for vclass in ["Category A", "Category B"]:
        d = pre[pre["vehicle_class"] == vclass].copy()
        # Event fixed effects so we compare within each election window
        model = smf.ols("log_premium ~ event_month + C(election_year)", data=d).fit(cov_type="HC3")
        results[f"{vclass}_slope_coef"] = float(model.params.get("event_month", np.nan))
        results[f"{vclass}_slope_p"] = float(model.pvalues.get("event_month", np.nan))

    # One-sample test of drop-from-peak across elections (by class)
    for vclass in ["Category A", "Category B"]:
        s = metrics.loc[metrics["vehicle_class"] == vclass, "drop_from_peak_pct"].dropna()
        if len(s) >= 2:
            tstat, pval = stats.ttest_1samp(s, popmean=0.0)
            results[f"{vclass}_drop_mean_pct"] = float(s.mean())
            results[f"{vclass}_drop_t_p"] = float(pval)
            results[f"{vclass}_n_events"] = int(len(s))
        else:
            results[f"{vclass}_drop_mean_pct"] = np.nan
            results[f"{vclass}_drop_t_p"] = np.nan
            results[f"{vclass}_n_events"] = int(len(s))

    return results


def make_plot(panel: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)

    for ax, vclass in zip(axes, ["Category A", "Category B"]):
        d = panel[(panel["vehicle_class"] == vclass) & (panel["event_month"] >= -12) & (panel["event_month"] <= 3)]
        for year, g in d.groupby("election_year"):
            g = g.sort_values("event_month")
            ax.plot(g["event_month"], g["premium"], marker="o", label=str(year), alpha=0.8)
        ax.axvline(0, color="black", linestyle="--", linewidth=1)
        ax.set_title(f"{vclass}: COE around election month")
        ax.set_xlabel("Months relative to election (0 = election month)")
        ax.set_ylabel("Monthly mean premium (SGD)")
        ax.legend(title="Election year")
        ax.grid(alpha=0.25)

    plt.tight_layout()
    plt.savefig(PLOTS / "rq4_event_window_article_style.png", dpi=150)
    plt.close()


def main() -> None:
    monthly = load_monthly_ab()
    panel = build_event_panel(monthly, pre_months=12, post_months=3)
    metrics = event_metrics(panel)
    test_res = run_tests(panel, metrics)

    metrics.to_csv(REPORTS / "rq4_event_window_metrics.csv", index=False)
    pd.DataFrame([test_res]).to_csv(REPORTS / "rq4_event_window_tests.csv", index=False)
    make_plot(panel)

    lines = [
        "RQ4 Event-Window (Article-Style) Test",
        "Method: Cat A/B monthly COE around each election, window -12 to +3 months.",
        "Primary metric: drop from pre-election peak to election month.",
        "Primary test: pre-election slope (event_month coefficient) with election fixed effects.",
        "",
    ]

    for vclass in ["Category A", "Category B"]:
        lines.append(
            f"{vclass}: slope={test_res[f'{vclass}_slope_coef']:.6f}, p={test_res[f'{vclass}_slope_p']:.6g}; "
            f"mean drop-from-peak={test_res[f'{vclass}_drop_mean_pct']:.3f}% (p={test_res[f'{vclass}_drop_t_p']:.6g}, n={test_res[f'{vclass}_n_events']})"
        )

    (REPORTS / "rq4_event_window_summary.txt").write_text("\n".join(lines), encoding="utf-8")

    print("Saved:")
    print("- reports/rq4_event_window_metrics.csv")
    print("- reports/rq4_event_window_tests.csv")
    print("- reports/rq4_event_window_summary.txt")
    print("- plots/rq4_event_window_article_style.png")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
