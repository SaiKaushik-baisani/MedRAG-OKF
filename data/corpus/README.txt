Optional: drop extra plain .md / .txt / .pdf documents here. They are loaded next to the OKF bundle in data/okf
and wrapped as draft `Document` concepts. For proper OKF concepts, put them under data/okf/<folder>/<name>.md
with at least `type:` in the frontmatter (see data/okf/conditions/*.md), or run: python cli.py okf-import <folder>
