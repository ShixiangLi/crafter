"""Export episode dashboards and a self-contained local analysis report."""
import html
import json
import os
import tempfile
import textwrap
from pathlib import Path

from .statistics import ACHIEVEMENTS, RESOURCES, VITALS, load_run, summarize


def _pyplot():
    # Delay plotting imports until evaluation has finished, including forked workers.
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / f"crafter-matplotlib-{os.getuid()}"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _missing(ax, message="Not recorded in this run"):
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes, color="#697586")
    ax.set_xticks([])
    ax.set_yticks([])


def _format(value, digits=1):
    return "n/a" if value is None else f"{value:.{digits}f}"


def _unlocked(episode):
    if not episode["completed"] and not episode["states"]:
        return None
    return sum(count > 0 for count in episode["achievements"].values())


def plot_episode(episode, output):
    """Six panels with environment steps and model calls on separate axes."""
    plt = _pyplot()
    fig, axes = plt.subplots(3, 2, figsize=(16, 12), layout="constrained")
    states, calls = episode["states"], episode["calls"]
    for ax in axes.flat:
        ax.grid(alpha=0.18)
        ax.set_axisbelow(True)

    for ax, names, title, ylabel in (
        (axes[0, 0], VITALS, "Survival state", "Value"),
        (axes[1, 0], RESOURCES, "Resources currently held", "Inventory count"),
    ):
        ax.set(title=title, xlabel="Environment step", ylabel=ylabel)
        if states:
            for name in names:
                values = [state["inventory"].get(name, float("nan") if name in VITALS else 0) for state in states]
                ax.step([state["step"] for state in states], values, where="post", label=name, linewidth=1.5)
            ax.legend(loc="best", ncol=len(names), fontsize=9)
            ax.set_ylim(bottom=-0.15)
            if names == VITALS:
                ax.set_ylim(-0.15, 9.5)
            else:
                maximum = max(state["inventory"].get(name, 0) for state in states for name in names)
                ax.set_ylim(-0.15, max(1, maximum) + 0.25)
                ax.yaxis.set_major_locator(plt.matplotlib.ticker.MaxNLocator(integer=True))
        else:
            _missing(ax)

    ax = axes[0, 1]
    ax.set(title="Achievement milestones", xlabel="Environment step", ylabel="Unique achievements")
    if states:
        unlocked, counts = set(), []
        for state in states:
            new = {name for name, count in state["achievements"].items() if count > 0} - unlocked
            unlocked.update(new)
            counts.append(len(unlocked))
            if new:
                label = textwrap.fill(", ".join(sorted(new)), width=34)
                ax.annotate(label, (state["step"], len(unlocked)), xytext=(4, 5),
                            textcoords="offset points", fontsize=8)
        ax.step([state["step"] for state in states], counts, where="post", color="#16845b")
        ax.set_ylim(-0.2, max(counts, default=0) + 2)
        ax.margins(x=0.12)
        ax.yaxis.set_major_locator(plt.matplotlib.ticker.MaxNLocator(integer=True))
    elif episode["achievements"]:
        names = [name for name, count in episode["achievements"].items() if count > 0]
        if names:
            ax.barh(names, [1] * len(names), color="#16845b")
            ax.set(title="Final unlocked achievements (timing unavailable)", xlabel="Unlocked", ylabel="")
        else:
            _missing(ax, "No achievements unlocked; milestone timing unavailable")
    else:
        _missing(ax)

    ax = axes[1, 1]
    requested = episode["requested_actions"]
    executed = episode["executed_actions"] or (episode["result"] or {}).get("action_frequency", {})
    names = sorted(requested.keys() | executed.keys(), key=lambda name: requested.get(name, 0), reverse=True)
    ax.set(title="Actions requested and executed", xlabel="Count")
    if names:
        positions = list(range(len(names)))
        bars = ax.barh([index - 0.18 for index in positions], [requested.get(name, 0) for name in names],
                       height=0.35, label="Requested", color="#367cc0")
        for bar in bars:
            ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                    str(int(bar.get_width())), va="center", fontsize=8)
        if executed:
            ax.barh([index + 0.18 for index in positions], [executed.get(name, 0) for name in names],
                    height=0.35, label="Executed", color="#99bfdf")
        ax.set_yticks(positions, names, fontsize=9)
        ax.invert_yaxis()
        ax.margins(x=0.12)
        ax.legend(fontsize=9)
    else:
        _missing(ax, "No environment actions recorded")

    ax = axes[2, 0]
    ax.set(title="Reported tokens per generation", xlabel="Model generation call", ylabel="Thousands of tokens")
    if calls:
        for key, label in (("input_tokens", "Input"), ("output_tokens", "Output")):
            ax.plot([call["call"] for call in calls],
                    [call[key] / 1000 if call.get(key) is not None else float("nan") for call in calls],
                    label=label, linewidth=1.3)
        ax.legend(fontsize=9)
    else:
        _missing(ax)

    ax = axes[2, 1]
    ax.set(title="Generation latency (includes retry waits)", xlabel="Model generation call", ylabel="Seconds")
    if calls:
        ax.plot([call["call"] for call in calls], [call["elapsed_seconds"] for call in calls], color="#c67723")
        failures = [call for call in calls if call["status"] == "failed"]
        if failures:
            ax.scatter([call["call"] for call in failures], [call["elapsed_seconds"] for call in failures],
                       marker="x", color="#d94b42", label="Failed generation", zorder=3)
            ax.legend(fontsize=9)
    else:
        _missing(ax)

    diamond = episode["achievements"].get("collect_diamond", 0) > 0 if _unlocked(episode) is not None else "n/a"
    minutes = episode["elapsed_seconds"] / 60 if episode["elapsed_seconds"] is not None else None
    fig.suptitle(f"{episode['id']} | seed={episode['seed']} | steps={episode['steps']} | "
                 f"diamond={diamond} | stop={episode['stop']} | time={_format(minutes)} min", fontsize=13)
    fig.supxlabel("Inventory is current holdings. Executed actions do not prove achievement success. "
                  "Tokens cover returned generations; failed attempts may incur unrecorded usage.", fontsize=9)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(output, dpi=150)
    finally:
        plt.close(fig)


def plot_overview(episodes, summary, output):
    plt = _pyplot()
    fig, axes = plt.subplots(2, 2, figsize=(16, 11), layout="constrained")
    for ax in axes.flat:
        ax.grid(alpha=0.18)
        ax.set_axisbelow(True)
    ax = axes[0, 0]
    rates = summary["achievement_success_rates"]
    ax.set(title="Achievement success rates (completed episodes)", xlabel="Success rate (%)")
    if summary["completed_episodes"]:
        names = sorted(ACHIEVEMENTS, key=lambda name: rates[name], reverse=True)
        bars = ax.barh(names, [rates[name] for name in names], color="#16845b")
        ax.bar_label(bars, fmt="%.0f%%", padding=3, fontsize=8)
        ax.tick_params(axis="y", labelsize=9)
        ax.set_xlim(0, 115)
        ax.invert_yaxis()
    else:
        _missing(ax, "No completed episodes; aggregate score unavailable")

    labels = [episode["id"].split("/")[-1].replace("default_", "") + ("*" if not episode["completed"] else "")
              for episode in episodes]
    colors = ["#367cc0" if episode["completed"] else "#b0b8c3" for episode in episodes]
    for ax, values, title, ylabel in (
        (axes[0, 1], [_unlocked(episode) for episode in episodes],
         "Achievements per episode", "Unique achievements"),
        (axes[1, 0], [episode["steps"] for episode in episodes], "Episode length", "Environment steps"),
    ):
        ax.set(title=title, ylabel=ylabel)
        if episodes:
            ax.bar(range(len(episodes)), [float("nan") if value is None else value for value in values], color=colors)
            ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=9)
            ax.set_xlim(-0.6, len(episodes) - 0.4)
            for index, value in enumerate(values):
                if value is None:
                    ax.text(index, 0, "n/a", ha="center", va="bottom", color="#697586")
        else:
            _missing(ax, "No episode records")

    ax = axes[1, 1]
    ax.set(title="Returned-generation tokens per episode", ylabel="Thousands of tokens")
    if episodes:
        totals = []
        for episode in episodes:
            result = episode["result"]
            totals.append((result["input_tokens"], result["output_tokens"]) if result else tuple(
                sum(call[key] for call in episode["calls"] if call.get(key) is not None)
                if any(call.get(key) is not None for call in episode["calls"]) else float("nan")
                for key in ("input_tokens", "output_tokens")))
        xs = list(range(len(episodes)))
        ax.bar(xs, [pair[0] / 1000 for pair in totals], color="#367cc0", label="Input")
        ax.bar(xs, [pair[1] / 1000 for pair in totals], bottom=[pair[0] / 1000 for pair in totals],
               color="#ef9839", label="Output")
        ax.set_xticks(xs, labels, rotation=45, ha="right", fontsize=9)
        ax.set_xlim(-0.6, len(episodes) - 0.4)
        for index, episode in enumerate(episodes):
            if not episode["result"] and not any(call.get("input_tokens") is not None for call in episode["calls"]):
                ax.text(index, 0, "n/a", ha="center", va="bottom", color="#697586")
        ax.legend(fontsize=9)
    else:
        _missing(ax, "No token records")
    fig.suptitle(f"Completed={summary['completed_episodes']} / expected={_format(summary['expected_episodes'], 0)} | "
                 f"Crafter score={_format(summary['crafter_score'], 2)} | "
                 f"diamond success={_format(summary['diamond_success_rate'])}%", fontsize=14)
    fig.supxlabel("* Partial episode, excluded from success rates and Crafter score. "
                  "Failed attempts may incur unrecorded tokens.", fontsize=10)
    try:
        fig.savefig(output, dpi=150)
    finally:
        plt.close(fig)


def analyze_run(run_dir):
    """Generate PNGs, summary JSON, and an HTML report without model requests."""
    run_dir = Path(run_dir).resolve()
    episodes = load_run(run_dir)
    status_path = run_dir / "status.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    summary = summarize(episodes, status.get("expected_episodes"))
    output = run_dir / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    plot_overview(episodes, summary, output / "overview.png")
    for episode in episodes:
        plot_episode(episode, output / "episodes" / (episode["id"] + ".png"))
    # The suffix avoids being mistaken for an episode by BALROG's JSON collector.
    (output / "analysis_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    rows, sections = [], []
    for episode in episodes:
        name = html.escape(episode["id"])
        image = html.escape("episodes/" + episode["id"] + ".png", quote=True)
        rows.append(f'<tr><td><a href="{image}">{name}</a></td><td>{episode["seed"]}</td>'
                    f'<td>{episode["steps"]}</td><td>{_format(_unlocked(episode), 0)}</td>'
                    f'<td>{episode["stop"]}</td><td>{_format(episode["elapsed_seconds"])}</td></tr>')
        sections.append(f'<section><h2>{name}</h2><a href="{image}"><img src="{image}" loading="lazy" alt="Episode analysis"></a></section>')
    report = f'''<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Crafter 试验分析</title>
<style>body{{font:16px system-ui,sans-serif;max-width:1500px;margin:32px auto;padding:0 20px;color:#243247;background:#f6f8fb}}
section,header{{background:white;padding:20px;margin:20px 0;border-radius:12px}}img{{width:100%;height:auto}}
table{{width:100%;border-collapse:collapse}}td,th{{padding:10px;text-align:left;border-bottom:1px solid #e1e6ed}}
a{{color:#2671b8}}.table{{overflow-x:auto}}p{{line-height:1.7}}</style>
<header><h1>Crafter 试验分析</h1><p>{html.escape(run_dir.name)}</p>
<p>已完成 {summary['completed_episodes']} / 预期 {_format(summary['expected_episodes'], 0)} 回合；
Crafter score：{_format(summary['crafter_score'], 2)}；采集钻石成功率：{_format(summary['diamond_success_rate'])}%。</p>
<p>成就成功率与分数只统计完整回合。已完成回合包含 {summary['distinct_seeds']} 个不同种子。
钻石成功指解锁 collect_diamond；动作执行次数不代表目标达成。旧试验没有保存的数据会标为未记录。</p>
<p>每次 generate 调用单独记录，ReAct 思考和 SPRING 节点都计数；耗时包含内部重试及等待。
Token 来自成功返回的响应，未包含内部失败尝试中无法取得的用量。
记录到 {summary['recorded_model_calls']} 次调用，其中 {summary['failed_model_calls']} 次最终失败。</p>
<p>分数按 <a href="https://github.com/danijar/crafter/blob/main/analysis/common.py">Crafter 官方公式</a>
exp(mean(log(1 + 各成就成功率百分数))) − 1 计算；此处为当前评估回合的分数，并不表示复现论文的训练预算与结果。</p></header>
<section><h2>试验汇总</h2><a href="overview.png"><img src="overview.png" alt="Run overview"></a>
<p><a href="analysis_summary.json">下载统计数据</a></p><div class="table"><table><thead><tr>
<th>回合</th><th>种子</th><th>步数</th><th>成就数</th><th>结束原因</th><th>耗时（秒）</th>
</tr></thead><tbody>{''.join(rows)}</tbody></table></div></section>{''.join(sections)}</html>'''
    (output / "report.html").write_text(report, encoding="utf-8")
    return output / "report.html"
