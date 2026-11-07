"""Turns the note-taker's chart specs into images.

The AI never writes code that gets run; it only fills in a chart type and
numbers, and this module draws from a fixed set of chart types.
"""
from __future__ import annotations

import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

ACCENT, SHADE, INK = "#3b5bdb", "#a5b4fc", "#1f2937"


def _num(p: dict, key: str, default: float) -> float:
    try:
        return float(p.get(key, default))
    except (TypeError, ValueError):
        return default


def _list(p: dict, key: str) -> list:
    v = p.get(key) or []
    return list(v) if isinstance(v, (list, tuple)) else []


def render_chart(chart_type: str, params: dict | None) -> bytes | None:
    """Return PNG bytes, or None if the spec can't be drawn."""
    p = params or {}
    fig, ax = plt.subplots(figsize=(6, 3.4), dpi=150)
    try:
        if chart_type in ("normal", "t", "chi2", "uniform"):
            if chart_type == "normal":
                mu, sd = _num(p, "mu", 0), max(_num(p, "sigma", 1), 1e-6)
                dist, lo, hi = stats.norm(mu, sd), mu - 4 * sd, mu + 4 * sd
            elif chart_type == "t":
                dist, lo, hi = stats.t(max(_num(p, "df", 10), 1)), -5, 5
            elif chart_type == "chi2":
                df = max(_num(p, "df", 4), 1)
                dist, lo, hi = stats.chi2(df), 0, df + 5 * np.sqrt(2 * df)
            else:
                a, b = _num(p, "a", 0), _num(p, "b", 1)
                if b <= a:
                    b = a + 1
                dist, lo, hi = stats.uniform(a, b - a), a - (b - a) * 0.2, b + (b - a) * 0.2
            x = np.linspace(lo, hi, 500)
            ax.plot(x, dist.pdf(x), color=ACCENT, lw=2)
            if "shade_from" in p or "shade_to" in p:
                s0, s1 = _num(p, "shade_from", lo), _num(p, "shade_to", hi)
                xs = x[(x >= s0) & (x <= s1)]
                ax.fill_between(xs, dist.pdf(xs), color=SHADE, alpha=0.7,
                                label=f"P = {dist.cdf(s1) - dist.cdf(s0):.3f}")
                ax.legend(frameon=False)
            ax.set_ylim(bottom=0)
        elif chart_type in ("binomial", "poisson"):
            if chart_type == "binomial":
                n, pr = int(max(_num(p, "n", 10), 1)), min(max(_num(p, "p", 0.5), 0), 1)
                k = np.arange(0, n + 1)
                y = stats.binom.pmf(k, n, pr)
            else:
                lam = max(_num(p, "lam", 3), 0.1)
                k = np.arange(0, int(lam + 4 * np.sqrt(lam)) + 2)
                y = stats.poisson.pmf(k, lam)
            ax.bar(k, y, color=ACCENT, width=0.8)
        elif chart_type in ("scatter", "line"):
            x, y = _list(p, "x"), _list(p, "y")
            if not x or len(x) != len(y):
                return None
            if chart_type == "scatter":
                ax.scatter(x, y, color=ACCENT)
                if len(x) > 2:
                    m, c = np.polyfit(np.array(x, float), np.array(y, float), 1)
                    xx = np.linspace(min(x), max(x), 50)
                    ax.plot(xx, m * xx + c, color=INK, lw=1, ls="--", label=f"ŷ = {c:.2f} + {m:.2f}x")
                    ax.legend(frameon=False)
            else:
                ax.plot(x, y, color=ACCENT, lw=2, marker="o")
        elif chart_type == "bar":
            labels, values = _list(p, "labels"), _list(p, "values")
            if not labels or len(labels) != len(values):
                return None
            ax.bar([str(l) for l in labels], values, color=ACCENT)
        elif chart_type in ("boxplot", "histogram"):
            data = np.array(_list(p, "data"), float)
            if data.size < 2:
                return None
            if chart_type == "boxplot":
                ax.boxplot(data, vert=False, patch_artist=True,
                           boxprops=dict(facecolor=SHADE, color=ACCENT))
            else:
                ax.hist(data, bins="auto", color=ACCENT, edgecolor="white")
        else:
            return None

        ax.set_title(p.get("title", ""), fontsize=11, color=INK)
        ax.set_xlabel(p.get("xlabel", ""))
        ax.set_ylabel(p.get("ylabel", ""))
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        return buf.getvalue()
    except Exception:
        return None
    finally:
        plt.close(fig)
