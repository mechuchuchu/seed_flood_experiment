"""
Collect all per-combo JSON result files in a directory into one sorted summary.
Usage: python aggregate_results.py <results_dir>
"""
import sys
import os
import json
import glob


def main():
    if len(sys.argv) < 2:
        print("usage: python aggregate_results.py <results_dir>")
        sys.exit(1)
    outdir = sys.argv[1]

    files = glob.glob(os.path.join(outdir, "*.json"))
    results = []
    for fp in files:
        try:
            with open(fp) as f:
                results.append(json.load(f))
        except Exception as e:
            results.append({"status": f"UNREADABLE: {e}", "test_loss": float("nan"),
                            "test_acc": 0.0, "mode": "?", "lr": None, "mu": None})

    def sort_key(r):
        tl = r.get("test_loss", float("nan"))
        return tl if (tl is not None and tl == tl) else 1e9  # NaN-safe

    results.sort(key=sort_key)

    n_ok = sum(1 for r in results if r.get("status") == "OK")
    n_diverged = sum(1 for r in results if r.get("status") == "DIVERGED")
    n_crashed = sum(1 for r in results if r.get("status", "").startswith(("ERROR", "TIMEOUT")))

    print(f"total: {len(results)}  OK: {n_ok}  DIVERGED: {n_diverged}  CRASHED/TIMEOUT: {n_crashed}")
    print(f"\n=== top 25 by test_loss ===")
    print(f"{'mode':>6} {'lr':>10} {'mu':>10} {'b1':>5} {'b2':>7} {'test_loss':>10} {'test_acc':>8}  status")
    for r in results[:25]:
        lr = r.get("lr")
        mu = r.get("mu")
        b1 = r.get("beta1", "-")
        b2 = r.get("beta2", "-")
        tl = r.get("test_loss", float("nan"))
        ta = r.get("test_acc", 0.0)
        lr_s = f"{lr:.4g}" if isinstance(lr, (int, float)) else str(lr)
        mu_s = f"{mu:.4g}" if isinstance(mu, (int, float)) else str(mu)
        b1_s = f"{b1:.2f}" if isinstance(b1, (int, float)) else str(b1)
        b2_s = f"{b2:.4f}" if isinstance(b2, (int, float)) else str(b2)
        tl_s = f"{tl:.4f}" if isinstance(tl, (int, float)) else str(tl)
        print(f"{r.get('mode','?'):>6} {lr_s:>10} {mu_s:>10} {b1_s:>5} {b2_s:>7} "
              f"{tl_s:>10} {ta:>8.3f}  {r.get('status','?')}")

    summary_path = os.path.join(outdir, "_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nfull sorted results saved to {summary_path}")


if __name__ == "__main__":
    main()
