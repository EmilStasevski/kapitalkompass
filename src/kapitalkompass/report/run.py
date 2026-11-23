"""`task report`: rebuild README tables, reports/ and the PDF from results/.

uv run python -m kapitalkompass.report.run
"""

from __future__ import annotations

from kapitalkompass.mlops.hypotheses import evaluate
from kapitalkompass.report import facts, figures, markdown, pdf


def main() -> None:
    evaluate()  # verdicts from the current results/
    f = facts.load()
    figs = figures.all_figures(f)
    written = markdown.write_all(f, figs)
    written.append(pdf.build_pdf(f, figs))
    for w in written:
        print("wrote", w)


if __name__ == "__main__":
    main()
