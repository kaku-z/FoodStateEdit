# FoodStateEdit two-page Japanese report, 2026-09-07

Output: `output/pdf/FoodStateEdit_PreDefense_Handout_GUO_2530030_20260907.pdf`.

This is a new revision; the 2026-09-06 builder and PDF are preserved.
The format follows the supplied `t2430084_doc.pdf`: A4, two columns, Japanese
academic headings, numbered figures/tables/equations and references. The research
content, plots, measurements and conclusions are FoodStateEdit's, not the senior
student's. Author information uses only the supplied GUO ZHENGPENG / 2530030;
unconfirmed programme and supervisor information is omitted.

`build_handout.py` is the editable source. Run it from any directory using the
bundled Python with ReportLab, Pillow and pypdf. Windows Yu Mincho / Meiryo fonts
are embedded. `report_ja.md` is the generated text companion; edit the builder
before rebuilding, because the text companion and layout audit are regenerated.

Evidence is limited to the Day 13 matched negative VACE comparison and the Day 14
synthetic-mask / RGB observer diagnostic. SAM3 is a proposed follow-up observer,
not a new validated result. The code checks the numeric table against the Day 13
manifest and reads Day 14 values directly from JSON. The contact-sheet frame
indices are checked against the frozen evaluation configuration: 0, 3, 6, 10,
15, 20. Only frames 6, 10 and 20 are displayed, using identical crop coordinates
across methods. Images are cropped for layout but never retouched.

The PDF contains locally held source-containing experiment crops. It is intended
for local report review; this task did not publish the file or change image
redistribution permissions. `layout_audit.json` records input and output hashes.
After every rebuild, render both PDF pages and inspect them before delivery.
