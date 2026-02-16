#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import pandas as pd

# -------------------------
# CONFIG (edit if needed)
# -------------------------
ROOT = "runs_final_multimodel_hf"     # your multi-model root
OUT_TEX = "paper_crossmodel_table.tex"

ROLES_ORDER = ["intern", "engineer", "manager", "admin"]

# Each model directory is usually like "meta-llama__Meta-Llama-3-8B-Instruct"
# We'll use that directory name as a short label unless we can prettify it.
def pretty_model_label(model_slug: str) -> str:
    # Shorten common prefixes for readability in a wide table
    s = model_slug.replace("__", "/")
    s = s.replace("meta-llama/", "LLaMA-3-")
    s = s.replace("google/", "Gemma-")
    s = s.replace("mistralai/", "Mistral-")
    s = s.replace("OpenPipe/", "Qwen3-")
    return s

def find_metrics_csv(config_dir: str) -> str:
    """
    Try to find comparison_metrics.csv in:
      1) config_dir/**/comparison_metrics.csv (handles nested model_name folder)
      2) config_dir/comparison_metrics.csv (flat)
    Return first match or "".
    """
    # nested first (more common in your script)
    cand = glob.glob(os.path.join(config_dir, "**", "comparison_metrics.csv"), recursive=True)
    if cand:
        # prefer the most recent / deepest one deterministically by sorting
        cand = sorted(cand, key=lambda p: (p.count(os.sep), p))
        return cand[-1]
    flat = os.path.join(config_dir, "comparison_metrics.csv")
    return flat if os.path.exists(flat) else ""

def load_role_metrics(model_root: str, config: str):
    """
    Returns dict role -> {CLA, utility_rate, public_admin_leak}
    """
    cfg_dir = os.path.join(model_root, config)
    csv_path = find_metrics_csv(cfg_dir)
    if not csv_path:
        return None, ""

    df = pd.read_csv(csv_path)
    # expected columns: system, role, runs, forbidden_attempts, forbidden_success, public_admin_leak, utility, CLA, utility_rate
    out = {}
    for role in ROLES_ORDER:
        sub = df[df["role"] == role]
        if sub.empty:
            out[role] = {"CLA": None, "utility_rate": None, "public_admin_leak": None}
        else:
            r = sub.iloc[0]
            out[role] = {
                "CLA": float(r.get("CLA", 0.0)),
                "utility_rate": float(r.get("utility_rate", 0.0)),
                "public_admin_leak": int(r.get("public_admin_leak", 0)),
            }
    return out, csv_path

def fmt_float(x, nd=2):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "--"
    return f"{x:.{nd}f}"

def fmt_int(x):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "--"
    return str(int(x))

def latex_escape(s: str) -> str:
    # minimal escaping for table headers
    return (s.replace("&", "\\&")
              .replace("%", "\\%")
              .replace("_", "\\_"))

def build_table(models):
    """
    Build a single wide LaTeX table:
      Role | (Model1 Baseline: CLA U Leak) (Model1 Full: CLA U Leak) | ... for each model
    """
    # Column spec: Role + 6 per model
    ncols = 1 + 6 * len(models)
    col_spec = "l" + "c" * (ncols - 1)

    lines = []
    lines.append("\\begin{table*}[t]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append("\\setlength{\\tabcolsep}{4pt}")
    lines.append("\\renewcommand{\\arraystretch}{1.15}")
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")

    # Header row 1: model group names with multicolumn
    hdr1 = ["Role"]
    for m in models:
        mlab = latex_escape(m["label"])
        hdr1.append(f"\\multicolumn{{6}}{{c}}{{{mlab}}}")
    lines.append(" & ".join(hdr1) + " \\\\")
    # Header row 2: baseline/full split
    hdr2 = [" "]
    for _ in models:
        hdr2.append("\\multicolumn{3}{c}{Baseline}")
        hdr2.append("\\multicolumn{3}{c}{Full}")
    lines.append(" & ".join(hdr2) + " \\\\")
    # Header row 3: metric names
    hdr3 = [" "]
    for _ in models:
        hdr3 += ["CLA", "Util", "Leaks", "CLA", "Util", "Leaks"]
    lines.append(" & ".join(hdr3) + " \\\\")
    lines.append("\\midrule")

    # Body rows per role
    for role in ROLES_ORDER:
        row = [role]

        for m in models:
            b = m["baseline"].get(role, {})
            f = m["full"].get(role, {})
            row += [
                fmt_float(b.get("CLA")),
                fmt_float(b.get("utility_rate")),
                fmt_int(b.get("public_admin_leak")),
                fmt_float(f.get("CLA")),
                fmt_float(f.get("utility_rate")),
                fmt_int(f.get("public_admin_leak")),
            ]
        lines.append(" & ".join(row) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\caption{Cross-model containment and utility by role. Baseline corresponds to PermLLM-style domain gating (modules off) and Full corresponds to Cap-PermLLM (all modules on). Leaks counts public admin leakage events.}")
    lines.append("\\label{tab:crossmodel_rolewise}")
    lines.append("\\end{table*}")
    return "\n".join(lines)

def main():
    if not os.path.isdir(ROOT):
        raise SystemExit(f"[error] ROOT not found: {ROOT}")

    model_dirs = sorted(
        [d for d in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT, d))]
    )
    if not model_dirs:
        raise SystemExit(f"[error] no model directories found under {ROOT}")

    models = []
    for md in model_dirs:
        model_root = os.path.join(ROOT, md)

        baseline, bpath = load_role_metrics(model_root, "baseline")
        full, fpath = load_role_metrics(model_root, "full")

        if baseline is None or full is None:
            print(f"[warn] skipping {md} (missing baseline/full metrics). baseline={bpath} full={fpath}")
            continue

        models.append({
            "slug": md,
            "label": pretty_model_label(md),
            "baseline": baseline,
            "full": full,
            "baseline_path": bpath,
            "full_path": fpath,
        })

    if not models:
        raise SystemExit("[error] no models had baseline+full metrics CSVs.")

    tex = build_table(models)
    with open(OUT_TEX, "w", encoding="utf-8") as f:
        f.write(tex)

    print(f"[OK] wrote: {OUT_TEX}")
    print("[info] models included:")
    for m in models:
        print("  -", m["slug"])
        print("    baseline:", m["baseline_path"])
        print("    full:    ", m["full_path"])

if __name__ == "__main__":
    main()
