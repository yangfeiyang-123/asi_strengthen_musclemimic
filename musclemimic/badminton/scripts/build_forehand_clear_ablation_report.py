from __future__ import annotations


def render_markdown_report(rows: list[dict]) -> str:
    ranked = sorted(
        rows,
        key=lambda row: (float(row["success_rate"]), float(row["backcourt_rate"])),
        reverse=True,
    )
    lines = [
        "# ForehandClear Racket Ablation Summary",
        "",
        "| Arm | Success Rate | Backcourt Rate |",
        "| --- | ---: | ---: |",
    ]
    for row in ranked:
        lines.append(f"| {row['arm']} | {float(row['success_rate']):.3f} | {float(row['backcourt_rate']):.3f} |")
    lines.append("")
    return "\n".join(lines)
