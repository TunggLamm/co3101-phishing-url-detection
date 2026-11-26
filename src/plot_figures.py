"""Generate the 4 report figures (300 DPI PNG) from ``results/metrics.json``."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_METRICS_PATH = PROJECT_DIR / "results" / "metrics.json"
DEFAULT_FIGURES_DIR = PROJECT_DIR / "results" / "figures"

C_RAND, C_ETLD = "#4C78A8", "#E45756"
C_RULE, C_LR, C_GB = "#2A9D8F", "#F28E2B", "#2B5B84"
C_POS, C_NEG, C_FP, C_FN = "#D64550", "#3B7EA1", "#E67E22", "#C0392B"
HALO = {"boxstyle": "square,pad=0.05", "facecolor": "white", "edgecolor": "none", "alpha": 0.78}


def _configure_style() -> None:
    plt.rcParams.update({
        "font.sans-serif": ["Segoe UI", "Arial", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": "#E5E5E5",
        "grid.linestyle": "--",
        "grid.linewidth": 0.7,
        "axes.titlesize": 11.2,
        "axes.titleweight": "bold",
        "axes.labelsize": 10.2,
        "savefig.dpi": 300,
    })


def _save(fig: plt.Figure, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def _lookup(metrics: dict, model: str, split: str, rate_key: str, field: str) -> tuple[float, float]:
    for r in metrics["baselines"]:
        if r["model"] == model and r["split"] == split and r["rate_key"] == rate_key:
            return float(r[field][0]), float(r[field][1])
    raise KeyError((model, split, rate_key, field))


def plot_fig1_random_vs_etld1(metrics: dict, figures_dir: Path) -> Path:
    models = ["One-Line Rule", "Logistic Regression", "Gradient Boosting"]
    labels = ["Luật một dòng\n(https://www.)", "Logistic Regression\n(10 từ vựng)", "Gradient Boosting\n(10 từ vựng)"]
    panels = [
        ("original", "F1-score", "A. F1-score — Tỉ lệ gốc (~42.8% Phishing)"),
        ("original", "PR-AUC", "B. PR-AUC — Tỉ lệ gốc (~42.8% Phishing)"),
        ("1.00%", "F1-score", "C. F1-score — Tỉ lệ thực tế (1.0% Phishing)"),
        ("1.00%", "PR-AUC", "D. PR-AUC — Tỉ lệ thực tế (1.0% Phishing)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.6), sharey=True)
    x, w = np.arange(len(models)), 0.34

    for ax, (rk, field, title) in zip(axes.flat, panels):
        rv = [_lookup(metrics, m, "Random Split", rk, field) for m in models]
        ev = [_lookup(metrics, m, "eTLD+1 Split", rk, field) for m in models]
        rm, rhw = [v[0] for v in rv], [v[1] for v in rv]
        em, ehw = [v[0] for v in ev], [v[1] for v in ev]

        b1 = ax.bar(x - w / 2, rm, w, yerr=rhw, capsize=3, color=C_RAND, edgecolor="white", label="Random Split (đối chứng)")
        b2 = ax.bar(x + w / 2, em, w, yerr=ehw, capsize=3, color=C_ETLD, edgecolor="white", label="eTLD+1 Split (tách tên miền)")
        ax.axhline(0.99, color="#555555", ls=":", lw=1.2, zorder=1, label="Mốc 0.99 thường công bố")

        for bars, ms, hws, col in [(b1, rm, rhw, "#26466D"), (b2, em, ehw, "#9E2A2B")]:
            for bar, v, hw in zip(bars, ms, hws):
                ax.text(bar.get_x() + bar.get_width() / 2, v + hw + 0.015, f"{v:.4f}", ha="center", va="bottom", fontsize=8.4, fontweight="bold", color=col, bbox=HALO)
        for i in range(len(models)):
            top = min(max(rm[i] + rhw[i], em[i] + ehw[i]) + 0.072, 1.09)
            ax.text(x[i], top, f"Δ = {em[i] - rm[i]:+.4f}", ha="center", va="bottom", fontsize=8.1, color="#444444",
                    bbox={"boxstyle": "round,pad=0.16", "facecolor": "#F7F7F7", "edgecolor": "#DDDDDD", "lw": 0.6})

        ax.set(title=title, xticks=x, xticklabels=labels, ylim=(0.0, 1.17), yticks=np.linspace(0, 1, 6))

    axes[0, 0].set_ylabel("Giá trị chỉ số")
    axes[1, 0].set_ylabel("Giá trị chỉ số")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.965))
    k = len(metrics["meta"]["baseline_seeds"])
    fig.suptitle(f"So sánh hiệu năng: Random Split vs eTLD+1 Split (K = {k} seeds, Mean ± 95% CI)", fontsize=13.2, fontweight="bold", y=1.01)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return _save(fig, figures_dir / "fig1_random_vs_etld1.png")


def plot_fig2_realistic_positive_rates(metrics: dict, figures_dir: Path) -> Path:
    rks = ["original", "10.00%", "5.00%", "1.00%"]
    rlabels = ["42.8%\n(Gốc)", "10.0%", "5.0%", "1.0%\n(Thực tế)"]
    styles = [
        ("One-Line Rule", "Luật một dòng (https://www.)", C_RULE, "s", "--"),
        ("Gradient Boosting", "Gradient Boosting (10 từ vựng)", C_GB, "o", "-"),
        ("Logistic Regression", "Logistic Regression (10 từ vựng)", C_LR, "^", "-"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14.8, 5.2), sharey=True)
    x = np.arange(len(rks))

    for ax, (field, title) in zip(axes, [("Precision", "A. Precision (Độ chính xác dương)"), ("F1-score", "B. F1-score"), ("PR-AUC", "C. PR-AUC (Average Precision)")]):
        ax.axhline(0.99, color="#666666", ls=":", lw=1.1, label="Mốc 0.99")
        r_vals = [_lookup(metrics, "One-Line Rule", "eTLD+1 Split", rk, field)[0] for rk in rks]
        g_vals = [_lookup(metrics, "Gradient Boosting", "eTLD+1 Split", rk, field)[0] for rk in rks]

        for name, label, col, mk, ls in styles:
            pairs = [_lookup(metrics, name, "eTLD+1 Split", rk, field) for rk in rks]
            ms, hws = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])
            ax.errorbar(x, ms, yerr=hws, color=col, marker=mk, ls=ls, lw=2.1, ms=6.2, capsize=3, label=label)
            ax.fill_between(x, ms - hws, np.minimum(ms + hws, 1.0), color=col, alpha=0.10)
            for i, yi in enumerate(ms):
                above = (name == "One-Line Rule" and r_vals[i] >= g_vals[i]) or (name == "Gradient Boosting" and g_vals[i] > r_vals[i])
                ax.annotate(f"{yi:.3f}", (x[i], yi), xytext=(0, 10 if above else -14), textcoords="offset points",
                            ha="center", va="bottom" if above else "top", fontsize=8.3, fontweight="bold", color=col, bbox=HALO)

        ax.set(title=title, xticks=x, xticklabels=rlabels, xlabel="Tỉ lệ mẫu Phishing (giữ 100% mẫu dương)", ylim=(-0.04, 1.13), yticks=np.linspace(0, 1, 6))

    axes[0].set_ylabel("Giá trị chỉ số (Mean ± 95% CI)")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.96))
    k = len(metrics["meta"]["baseline_seeds"])
    fig.suptitle(f"Độ tụt Precision, F1-score và PR-AUC theo tỉ lệ dương thực tế (eTLD+1 Split, K = {k} seeds)", fontsize=12.8, fontweight="bold", y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    return _save(fig, figures_dir / "fig2_realistic_positive_rates.png")


def plot_fig3_feature_ablation(metrics: dict, figures_dir: Path) -> Path:
    abl_lr, abl_gb = metrics["ablation"]["Logistic Regression"], metrics["ablation"]["Gradient Boosting"]
    leak = sorted(metrics["single_leakage"], key=lambda r: r["f1"][0])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.8, 5.8), gridspec_kw={"width_ratios": [1.35, 1.0]})

    stages = [f"Full Numeric\n({abl_gb[0]['cols']} cột)", f"Bỏ 4 cột rò rỉ\n({abl_gb[1]['cols']} cột)",
              f"Bỏ HTML\n({abl_gb[2]['cols']} cột từ vựng)", f"Từ vựng cốt lõi\n({abl_gb[3]['cols']} cột)"]
    x = np.arange(len(stages))
    ax1.axhline(0.99, color="#666666", ls=":", lw=1.1, label="Mốc 0.99")

    curves = [
        (abl_gb, "f1_orig", "GB — F1 (Gốc 42.8%)", C_GB, "o", "-", {1: (-18, 10), 2: (0, 10), 3: (0, 10)}),
        (abl_lr, "f1_orig", "LR — F1 (Gốc 42.8%)", C_LR, "s", "-", {1: (18, 10), 2: (0, -9), 3: (0, 10)}),
        (abl_gb, "f1_1pct", "GB — F1 (Thực tế 1%)", C_GB, "^", "--", {1: (0, -10), 2: (0, -13), 3: (0, -14)}),
        (abl_lr, "f1_1pct", "LR — F1 (Thực tế 1%)", C_LR, "D", "--", {1: (0, -14), 2: (0, 7), 3: (0, -14)}),
    ]
    for rows, field, label, col, mk, ls, offs in curves:
        y, hw = np.array([r[field][0] for r in rows]), np.array([r[field][1] for r in rows])
        ax1.plot(x, y, drawstyle="steps-mid", color=col, ls=ls, marker=mk, ms=6.8, lw=2.0, label=label)
        if np.any(hw > 0):
            ax1.errorbar(x, y, yerr=hw, fmt="none", ecolor=col, capsize=3)
        for xi, (dx, dy) in offs.items():
            ax1.annotate(f"{y[xi]:.3f}", (xi, y[xi]), xytext=(dx, dy), textcoords="offset points",
                         ha="center", va="bottom" if dy > 0 else "top", fontsize=8.3, fontweight="bold", color=col, bbox=HALO)

    ax1.annotate("Cả 4 cấu hình ≈ 1.000", (0, 1.0), xytext=(6, 11), textcoords="offset points", ha="center", va="bottom",
                 fontsize=8.2, fontweight="bold", bbox={"boxstyle": "round,pad=0.16", "facecolor": "#F5F5F5", "edgecolor": "#CCCCCC", "lw": 0.6})
    ax1.axvspan(1.5, 2.5, color="#FFF3CD", alpha=0.45, zorder=0)
    gb_drop = abl_gb[2]["f1_1pct"][0] - abl_gb[0]["f1_1pct"][0]
    ax1.text(2.0, 0.04, f"Loại 28 cột HTML\n(GB 1% tụt {gb_drop:+.3f})", ha="center", va="bottom", fontsize=8.3, color="#856404", fontweight="bold")
    seeds = metrics["meta"]["ablation_seeds"]
    snote = f"seed={seeds[0]}" if len(seeds) == 1 else f"K={len(seeds)} seeds, Mean ± 95% CI"
    ax1.set(title=f"A. Biểu đồ bậc thang Ablation 50 → 46 → 18 → 10 (eTLD+1, {snote})",
            xticks=x, xticklabels=stages, ylabel="F1-score", ylim=(0.0, 1.13), xlim=(-0.38, 3.38))
    ax1.legend(loc="lower left", fontsize=8.3, ncol=2)

    yp, h = np.arange(len(leak)), 0.35
    f1m, prm = [r["f1"][0] for r in leak], [r["pr_auc"][0] for r in leak]
    b1 = ax2.barh(yp + h / 2, f1m, h, color=C_ETLD, edgecolor="white", label="F1-score (Tỉ lệ gốc)")
    b2 = ax2.barh(yp - h / 2, prm, h, color=C_RAND, edgecolor="white", label="PR-AUC (Tỉ lệ gốc)")
    ax2.axvline(0.99, color="#555555", ls=":", lw=1.2, label="Mốc 0.99")
    for bars, vals, col in [(b1, f1m, "#9E2A2B"), (b2, prm, "#26466D")]:
        for bar, v in zip(bars, vals):
            ax2.text(v + 0.012, bar.get_y() + bar.get_height() / 2, f"{v:.4f}", va="center", ha="left", fontsize=8.5, fontweight="bold", color=col)
    ax2.set(title="B. Kiểm toán rò rỉ đơn cột (Single-Feature Audit)", yticks=yp,
            yticklabels=[r["feature"] for r in leak], xlim=(0.0, 1.15), xlabel="Hiệu năng chỉ với 1 cột duy nhất (LR trên eTLD+1)")
    ax2.legend(loc="lower right", fontsize=8.5)

    fig.suptitle("Kiểm toán cắt giảm đặc trưng (Feature Ablation) & Bằng chứng rò rỉ nhãn trên PhiUSIIL", fontsize=13.0, fontweight="bold", y=1.01)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return _save(fig, figures_dir / "fig3_feature_ablation.png")


def plot_fig4_importance_and_errors(metrics: dict, figures_dir: Path) -> Path:
    gb = sorted(metrics["feature_importance"]["gb"], key=lambda r: r["mean"])
    lr = sorted(metrics["feature_importance"]["lr"], key=lambda r: abs(r["mean"]))
    err = metrics["errors"]
    fig, ((ax_gb, ax_lr), (ax_err, ax_tax)) = plt.subplots(2, 2, figsize=(14.8, 9.8))

    # A. GB Feature Importance
    yg = np.arange(len(gb))
    bars = ax_gb.barh(yg, [r["mean"] for r in gb], xerr=[r["hw"] for r in gb], capsize=3, color=C_GB, edgecolor="white")
    max_g = max(r["mean"] + r["hw"] for r in gb)
    for bar, r in zip(bars, gb):
        ax_gb.text(r["mean"] + r["hw"] + max_g * 0.02, bar.get_y() + bar.get_height() / 2, f"{r['share_pct']:.1f}%", va="center", fontsize=8.5, fontweight="bold", color="#1F3B57")
    ax_gb.set(title="A. Feature Importance — Gradient Boosting (eTLD+1)", yticks=yg, yticklabels=[r["feature"] for r in gb], xlim=(0, max_g * 1.22), xlabel="Độ quan trọng đặc trưng (Mean ± 95% CI, K=5)")

    # B. LR Standardized Weights
    yl = np.arange(len(lr))
    bars = ax_lr.barh(yl, [r["mean"] for r in lr], xerr=[r["hw"] for r in lr], capsize=3, color=[C_POS if r["mean"] > 0 else C_NEG for r in lr], edgecolor="white")
    ax_lr.axvline(0.0, color="#333333", lw=1.0)
    max_w = max(abs(r["mean"]) + r["hw"] for r in lr)
    for bar, r in zip(bars, lr):
        w, hw = r["mean"], r["hw"]
        x_t, ha, col = (w + hw + max_w * 0.03, "left", "#9E2A2B") if w >= 0 else (w - hw - max_w * 0.03, "right", "#26466D")
        ax_lr.text(x_t, bar.get_y() + bar.get_height() / 2, f"{w:+.2f}", va="center", ha=ha, fontsize=8.5, fontweight="bold", color=col)
    ax_lr.set(title="B. Trọng số chuẩn hóa — Logistic Regression (eTLD+1, K=5)", yticks=yl, yticklabels=[r["feature"] for r in lr], xlim=(-max_w * 1.28, max_w * 1.28), xlabel="Trọng số chuẩn hóa w (Đỏ: Phishing +, Xanh: Legitimate −)")

    # C. FP vs FN counts & rates
    cmap = {r["model"]: r for r in err["counts"]}
    order = ["One-Line Rule", "Logistic Regression", "Gradient Boosting"]
    fps, fns = [cmap[m]["fp"] for m in order], [cmap[m]["fn"] for m in order]
    xe, we = np.arange(len(order)), 0.35
    b1 = ax_err.bar(xe - we / 2, fps, we, color=C_FP, edgecolor="white", label="False Positive (Cảnh báo giả: Legit → Phish)")
    b2 = ax_err.bar(xe + we / 2, fns, we, color=C_FN, edgecolor="white", label="False Negative (Bỏ lọt: Phish → Legit)")
    max_e = max(max(fps), max(fns))
    for bars, vals, key, col in [(b1, fps, "fpr_pct", "#A0522D"), (b2, fns, "fnr_pct", "#8B1A1A")]:
        for bar, v, m in zip(bars, vals, order):
            ax_err.text(bar.get_x() + bar.get_width() / 2, v + max_e * 0.02, f"{v:,}\n({cmap[m][key]:.2f}%)", ha="center", va="bottom", fontsize=8.3, fontweight="bold", color=col)
    ax_err.set(title=f"C. Phân loại lỗi FP & FN của 3 Baseline (eTLD+1, seed={metrics['meta']['error_seed']})",
               xticks=xe, xticklabels=["Luật một dòng", "Logistic Regression\n(10 từ vựng)", "Gradient Boosting\n(10 từ vựng)"],
               ylim=(0, max_e * 1.28), ylabel="Số lượng URL dự đoán sai (tập test eTLD+1)")
    ax_err.legend(loc="upper left", fontsize=8.3)

    # D. FN traits & cross-model overlap
    ov, tr = err["overlap_pct"], err["fn_traits_pct"]
    items = [
        ("Giao thoa FP (GB ∩ LR / GB FP)", ov["gb_lr_fp"], C_FP),
        ("Giao thoa FN cả 3 mô hình (/ GB FN)", ov["all3_fn"], "#8E44AD"),
        ("Giao thoa FN (GB ∩ Luật 1 dòng / GB FN)", ov["gb_rule_fn"], C_RULE),
        ("Giao thoa FN (GB ∩ LR / GB FN)", ov["gb_lr_fn"], "#6C5CE7"),
        ("GB FN có cấu trúc gốc (LenDiff == 7)", tr["gb_len_diff_7"], C_GB),
        ("GB FN bắt đầu bằng https://www.", tr["gb_https_www"], C_GB),
        ("LR FN không chứa chữ số (0 digits)", tr["lr_zero_digits"], C_LR),
        ("GB FN không chứa chữ số (0 digits)", tr["gb_zero_digits"], C_GB),
    ]
    yd = np.arange(len(items))
    bars = ax_tax.barh(yd, [v for _, v, _ in items], color=[c for _, _, c in items], edgecolor="white", height=0.62)
    for bar, (_, v, _) in zip(bars, items):
        ax_tax.text(v + 1.5, bar.get_y() + bar.get_height() / 2, f"{v:.1f}%", va="center", fontsize=8.5, fontweight="bold")
    ax_tax.axhline(3.5, color="#AAAAAA", ls="--", lw=0.9)
    ax_tax.text(109, 1.5, "Giao thoa\nlỗi chéo", ha="right", va="center", fontsize=8.2, color="#555555", style="italic")
    ax_tax.text(109, 5.5, "Đặc điểm\nnhóm FN", ha="right", va="center", fontsize=8.2, color="#555555", style="italic")
    ax_tax.set(title="D. Đặc trưng cấu trúc nhóm FN & Giao thoa lỗi chéo mô hình", yticks=yd, yticklabels=[l for l, _, _ in items], xlim=(0, 114), xlabel="Tỷ lệ phần trăm (%)")

    fig.suptitle("Độ quan trọng 10 đặc trưng từ vựng & Phân tích lỗi FP/FN chéo mô hình trên tập kiểm tra eTLD+1", fontsize=13.2, fontweight="bold", y=1.005)
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    return _save(fig, figures_dir / "fig4_importance_and_errors.png")


def generate_all_figures(metrics_path: Path = DEFAULT_METRICS_PATH, figures_dir: Path = DEFAULT_FIGURES_DIR) -> list[Path]:
    _configure_style()
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    return [
        plot_fig1_random_vs_etld1(metrics, figures_dir),
        plot_fig2_realistic_positive_rates(metrics, figures_dir),
        plot_fig3_feature_ablation(metrics, figures_dir),
        plot_fig4_importance_and_errors(metrics, figures_dir),
    ]


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    for path in generate_all_figures():
        print(f"Saved {path.relative_to(PROJECT_DIR)}")


if __name__ == "__main__":
    main()
