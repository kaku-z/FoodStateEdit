<!-- ppt-master-schema: spec-lock/v1 -->
# Execution Lock

## canvas
- viewBox: 0 0 1280 720
- format: PPT 16:9

## communication
- primary_language: ja-JP
- audience: 大学教員、指導教員、研究発表審査者
- objective: 8分間で、食べる場面を想像させる目的、局所保持の必要性、明示的な動作制御、現時点の到達点と限界を理解してもらう。
- core_message: 普通の食物写真に食べる瞬間の臨場感を加えるため、必要な操作領域だけを変え、接触から持ち上げまでを制御・検証する。

## mode
- mode: custom
- mode_references: briefing, instructional
- mode_behavior: 結論を先に示し、背景から検証までを一本道で説明し、最後に残課題を明示する。

## visual_style
- visual_style: custom
- visual_style_references: swiss-minimal, editorial
- visual_style_behavior: 白地と淡青を基調に、細い罫線、矩形、番号、余白で階層を作り、実験画像と数値を主役にする。

## colors
- background: #FFFFFF
- secondary_background: #EAF6FC
- primary: #0070C0
- accent: #E31B23
- secondary_accent: #5BC0EB
- body_text: #222222
- secondary_text: #666666
- divider: #D8DDE3

## typography
- font_family: Yu Gothic, Arial, Meiryo, sans-serif
- title_family: Yu Gothic, Arial, Meiryo, sans-serif
- body_family: Yu Gothic, Arial, Meiryo, sans-serif
- data_family: Arial, Yu Gothic, sans-serif
- annotation_family: Yu Gothic, Arial, Meiryo, sans-serif
- body: 26
- title: 42
- subtitle: 32
- annotation: 18
- footnote: 15
- display: 54

## icons
- library: none
- inventory: none

## images
- p02-cake-input: images/p02_cake_input.png | source=user | crop=no-crop
- p02-cake-output: images/p02_cake_fork_scale0p6.png | source=user | crop=no-crop
- p03-ramen-input: images/p09_ramen_input.png | source=user | crop=no-crop
- p03-ramen-qwen: images/p09_ramen_qwen_seed1.png | source=user | crop=no-crop
- p05-framework-report-exact: images/p05_framework_report_exact.png | source=user | crop=no-crop
- p09-soup-qwen-day34: images/p09_soup_qwen_day34_seed1.png | source=user | crop=no-crop
- p10-success-soup-input: images/p10_success_soup_input.png | source=user | crop=no-crop
- p10-success-soup-output: images/p10_success_soup_output.png | source=user | crop=no-crop
- p09comp-noodle: images/p09comp_noodle.png | source=user | crop=no-crop
- p09comp-soup: images/p09comp_soup.png | source=user | crop=no-crop
- p09comp-cake: images/p09comp_cake.png | source=user | crop=no-crop

## page_rhythm
- P01: anchor
- P02: breathing
- P03: anchor
- P04: anchor
- P05: dense
- P06: anchor
- P07: dense
- P08: breathing
- P09: dense
- P10: dense
- P11: anchor
- P12: dense
- P13: anchor

## pptx_structure
- mode: flat

## forbidden
- `mask`, `<style>`, `class`, external CSS, `<foreignObject>`, `textPath`, `@font-face`, `<animate*>`, `<set>`, `<script>` / event attributes, `<iframe>`
- HTML named entities in text; write typography as raw Unicode and escape XML reserved characters
- 方形勺子修复不用写了，我们实验做完之后加上就行了 (user)
