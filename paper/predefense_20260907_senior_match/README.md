# Senior-reference matched report

Reference: the user-supplied `t2430084_doc.pdf` (two A4 pages).
Output: `output/pdf/FoodStateEdit_PreDefense_GUO_2530030_SeniorFormat_20260907.pdf`.
Earlier reports are preserved unchanged. No experiment or publication is run.

## Measured typography and page structure

- Japanese body 9.212 pt, Roman body 9.963 pt, leading 12.752 pt.
- Harano Aji Mincho Regular / Gothic Medium original outlines; Nimbus Roman
  Type 1 fonts, as in the reference. The Japanese CFF outlines are converted
  to renamed TrueType subsets with a maximum curve approximation error of
  0.5 units per em. Original fonts and OFL license are retained under `fonts/`.
- Left and right column origins: 48.905 pt and 304.724 pt.
- Title: two lines, 13.266 pt; author and supervisor block below.
- Page 1 has the same header position. Page 2 has no repeated header; neither
  page has the extra page footer used in the previous draft.
- Two vector architecture figures at the top right of page 1; a results table
  at the top left of page 2 and qualitative comparison at the top right.
- Plain numbered equations, Japanese punctuation, indented paragraphs,
  booktabs-style horizontal table rules, and a grey proposed-method row.

The unconfirmed affiliation / supervisor entries remain explicitly marked
`要確認`. The senior student's affiliation and supervisors are not copied.
Different research text necessarily results in different line breaks and figure
contents; this is not a pixel-identical copy of the original research report.

## Editable figures and evidence

`build_report.py` contains the vector figure source and Japanese text. Running it
also saves separate full-resolution vector PDFs under `figures/` for reuse.
These are intermediate figure assets; the report PDF is the final deliverable.

The framework uses actual source/control/output crops plus a clearly labelled
relative-3D proxy schematic. The constraint diagram separates 3D expectations,
image evidence, phase gates, visibility, path/contact terms and validation.
SAM3 validation and latent guidance are marked as future / unexecuted.

All experimental images are layout-only crops; no image is retouched or improved.
Day 13 numeric values are read directly from its result JSON; Day 14 values are
read directly from its recorded diagnostic. The report retains the negative
primary gate and does not turn a synthetic-mask result into a VACE result.
Source-containing crops remain for local report review; no public upload occurs.

## Build

Use the Codex bundled Python with ReportLab, Pillow, NumPy, pypdf, and fontTools
(the latter lives in `C:/Users/kaku/.cache/foodstateedit_pdf_dependencies`).
The builder locates existing MiKTeX Nimbus Roman fonts and system Arial.
Original Japanese font files are from the official Harano Aji Fonts repository:
https://github.com/trueroad/HaranoAjiFonts (SIL OFL 1.1).

After rebuilding, render and visually inspect both pages and both framework
figures. `layout_audit.json` records placement and evidence/output hashes.
