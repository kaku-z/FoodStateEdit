<!-- ppt-master-schema: design-spec/v1 -->
# FoodStateEdit 8-minute Pre-defense - Design Spec

## I. Project Information

| Item | Value |
| --- | --- |
| Project Name | FoodStateEdit: 3次元動作制御に基づく食物画像編集 |
| Canvas Format | PPT 16:9, 1280 × 720 px |
| Page Count | 13 |
| Primary Language | ja-JP |
| Target Audience | 大学教員、指導教員、研究発表審査者 |
| Communication Intent | 8分間で、食べる場面を想像させる目的、既存編集の問題、提案、現時点の検証結果、未完項目を順に説明する |
| Desired Audience Outcome | 明示的な動作制御の必要性と、相対3D制御の現時点の到達点・限界を区別して理解できる |
| Core Message / Ask / Action | 普通の食物写真に食べる瞬間の臨場感を加えるため、必要な操作領域だけを変え、接触から持ち上げまでを制御・検証する |
| Delivery Context | 修士研究の予備発表、口頭説明約8分 |
| Artifact Afterlife | 予備発表後に実験結果を追記し、正式発表用資料へ更新する |
| Reading Mode | presentation |
| Content Strategy | 体験価値と既存編集の問題を先に示し、方法、相対3DからVACEへの制御経路、単一成功例、複数食品の多手法比較、定量評価、今後の検証の順に説明する |
| Design Style | 学長資料忠実型：白地、淡青の章扉、細い罫線、青の構造色、赤の限定的強調 |
| AI Image Acquisition Path | not applicable; project experiment images only |
| Generation Mode | continuous |
| Spec Refinement | disabled |
| Speaker Notes | enabled — 日本語の口頭説明を各ページ約35–40秒で付与 |
| Custom Animations | disabled — static academic deck |
| Narration Audio | disabled — live oral presentation |
| Created Date | 2026-09-13 |

## II. Canvas Specification

| Property | Value |
| --- | --- |
| Format | PPT 16:9 |
| Dimensions | 1280 × 720 px |
| viewBox | `0 0 1280 720` |
| Margins | left/right 72 px, top 54 px, bottom 42 px |
| Content Area | 1136 × 624 px within safe margins |

## III. Visual Theme

### Theme Style

- **Mode**: custom
- **Mode References**: briefing, instructional
- **Mode Behavior**: 結論を先に示し、背景から検証までを一本道で説明し、最後に残課題を明示する。
- **Visual style**: custom
- **Visual Style References**: swiss-minimal, editorial
- **Visual Style Behavior**: 白地と淡青を基調に、細い罫線、矩形、番号、余白で階層を作る。装飾は控え、実験画像・数値・矢印を主役にする。
- **Theme**: Japanese academic laboratory briefing
- **Tone**: sober, evidence-led, transparent about limitations

### Color Scheme

| Role | HEX | Purpose |
| --- | --- | --- |
| Background | #FFFFFF | 通常ページ背景 |
| Secondary background | #EAF6FC | 章扉、淡い情報帯、比較カード |
| Primary | #0070C0 | 見出し、構造線、提案手法 |
| Accent | #E31B23 | 重要な結論、注意、未完状態 |
| Secondary accent | #5BC0EB | 補助図形、フェーズ、比較補助 |
| Body text | #222222 | 本文 |
| Secondary text | #666666 | 注釈、出典、脚注 |
| Divider | #D8DDE3 | 罫線、表境界 |

## IV. Typography System

### Font Plan

| Role | Character (Reference) | Primary | English if non-English | Fallback tail |
| --- | --- | --- | --- | --- |
| Title | Japanese Gothic, medium-bold | Yu Gothic | Arial | Meiryo, sans-serif |
| Body | Japanese Gothic, regular | Yu Gothic | Arial | Meiryo, sans-serif |
| Data | Tabular emphasis | Arial | Arial | Yu Gothic, sans-serif |
| Annotation | Compact Japanese Gothic | Yu Gothic | Arial | Meiryo, sans-serif |

- **Title stack**: Yu Gothic, Arial, Meiryo, sans-serif
- **Body stack**: Yu Gothic, Arial, Meiryo, sans-serif
- **Data stack**: Arial, Yu Gothic, sans-serif
- **Annotation stack**: Yu Gothic, Arial, Meiryo, sans-serif

### Font Size Hierarchy

| Purpose | Anchor Size (px) |
| --- | ---: |
| Body | 26 |
| Title | 42 |
| Subtitle | 32 |
| Annotation | 18 |
| Footnote | 15 |
| Display | 54 |

## V. Layout Principles

### Deck-wide Direction

- **Hierarchy direction**: 左上の短い見出しから主図または主数値へ視線を送り、下部に結論を置く。
- **Composition tendency**: 一つの主張につき一つの主図。比較は左右または横一列、工程は左から右。
- **Cross-page continuity**: 上端の細い灰色罫線、青いページ番号、下端の研究名フッターを継続する。章扉は淡青背景で変化を付ける。
- **Spacing posture**: open; 表と比較ページのみ密度を上げる。
- **Spacing anchors**: page margin 72 px, block gap 24 px, column gutter 32 px, corner radius 8 px, body leading 1.35 |

## VI. Icon Usage Specification

- **Primary bundled library**: none

| Icon Path | Suitable Scenarios |
| --- | --- |

## VIII. Image Resource List

| Filename | Dimensions | Ratio | Purpose | Type | Image pattern | Crop Policy | Acquire Via | Status | Reference | text_policy | page_role |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| images/p02_cake_input.png | 688 × 512 | 1.34 | 背景ページの入力例 | Source image | input/output pair | no-crop | user | Existing | project artifact | no overlay | evidence |
| images/p02_cake_fork_scale0p6.png | 688 × 512 | 1.34 | 背景ページのフォーク成型例 | Result image | input/output pair | no-crop | user | Existing | Day21 VACE scale 0.6 final-image gate pass | short label only | evidence |
| images/p09_ramen_input.png | 688 × 512 | 1.34 | 既存編集の問題を示す入力例 | Source image | input/output pair | no-crop | user | Existing | project artifact | short label only | evidence |
| images/p09_ramen_qwen_seed1.png | 1088 × 960 | 1.13 | 既存編集で画角・背景・器まで変わる例 | Result image | input/output pair | no-crop | user | Existing | same input, seed 1 internal comparison | short label only | evidence |
| images/p05_framework_report_exact.png | 1030 × 518 | 1.99 | 二頁報告の提案手法図 | Report figure | full-width framework | no-crop | user | Existing | exact crop of Figure 1 in the two-page report | labels retained | method overview |
| images/p10_success_soup_input.png | 736 × 592 | 1.24 | スープ同一入力比較の入力参照 | Source image | equal triptych | no-crop | user | Existing | Day10 seen-sample broth | large label above | qualitative evidence |
| images/p09comp_noodle.png | 1136 × 130 | 8.74 | ラーメンの多手法比較行 | Composite result strip | full-width row | no-crop | user | Existing | selected seed 1 internal comparison | labels embedded | qualitative evidence |
| images/p09comp_soup.png | 1136 × 130 | 8.74 | スープの多手法比較行 | Composite result strip | full-width row | no-crop | user | Existing | selected seed 1 internal comparison | labels embedded | qualitative evidence |
| images/p09comp_cake.png | 1136 × 130 | 8.74 | ケーキの多手法比較行 | Composite result strip | full-width row | no-crop | user | Existing | selected seed 1 internal comparison | labels embedded | qualitative evidence |
| images/p09_soup_qwen_day34_seed1.png | 1152 × 928 | 1.24 | 同じスープ入力のQwen出力 | Result image | equal triptych | no-crop | user | Existing | Day34 gp38 raw Qwen seed 1, SHA-256 e3498b06… | large label above | qualitative evidence |
| images/p10_success_soup_output.png | 736 × 592 | 1.24 | 同じスープ入力のFoodStateEdit出力 | Result image | equal triptych | no-crop | user | Existing | Day10 LoRA-off edited output | large label above | qualitative evidence |

## IX. Content Outline

### Part 1: 導入

#### Slide 01 - 表紙

- **Audience move**: 題目未把握 → 研究対象と発表者を把握
- **Relationships**: 研究題目、英語プロジェクト名、所属、氏名、指導教員の所属関係
- **Composition**: 白地に大きな和文題目、青い細線、右下に発表者情報。
- **Title**: 3次元動作制御に基づく食物画像編集
- **Core message**: FoodStateEditは柔軟な食物と道具の操作過程を制御する画像編集研究である。
- **Content**: FoodStateEdit / GUO ZHENGPENG / 2530030 / メディア情報学 / 主指導 柳井啓司 / 副指導 高橋裕樹

#### Slide 02 - 研究目的

- **Audience move**: 食物画像編集を単なる加工と認識 → 食べる場面を想像させる視覚体験として認識
- **Relationships**: 普通の食物写真、道具による操作状態、食べる直前の視覚的臨場感、利用場面の因果関係
- **Composition**: 左に研究目的と利用場面、右にケーキの入力と操作後の対比。
- **Title**: 研究目的：食べる瞬間を感じる食物画像
- **Core message**: 普通の食物写真へ道具による操作状態を加え、見る人が味・食感・動きを想像できる視覚表現を目指す。
- **Content**: 食べる直前の視覚的臨場感 / 食品メニュー・広告 / レシピ・食育・デジタル創作 / 味覚刺激そのものは対象外
- **Images**: images/p02_cake_input.png and images/p02_cake_fork_scale0p6.png as an input/result pair; cake is synthetic supplementary evidence and the result is VACE scale 0.6, not Qwen refinement.

#### Slide 03 - 既存画像編集の問題

- **Audience move**: 既存画像編集で十分と考える → 不要領域の再生成が原画像との不一致を生むと理解
- **Relationships**: 入力画像、編集したい局所領域、一般的な生成編集出力、不要な背景・器・構図変化、本研究の保持方針
- **Composition**: ラーメン入力と一般的な生成編集例を左右に大きく置き、下部に編集意図と実際の変化、本研究の方針を示す。
- **Title**: 既存画像編集の問題：不要な領域まで変化する
- **Core message**: 道具と食物だけを編集したい場合でも、既存の生成編集は画角、背景、容器、食物全体を再生成し、原画像との局所対応を崩すことがある。
- **Content**: 編集意図＝箸と持ち上げる麺 / 実際の変化＝画角・背景・器・麺全体 / 本研究＝必要な操作領域だけを変更し、その他は原画像のまま保持
- **Images**: images/p09_ramen_input.png and images/p09_ramen_qwen_seed1.png as a same-input, seed-1 internal comparison; the right image is evidence of global reconstruction, not a universal claim about every image editor.

#### Slide 04 - 課題と目標

- **Audience move**: 漠然と難しい → 失敗要因と評価対象を把握
- **Relationships**: 道具、接触、持ち上げ、背景保持の四条件が厳密成功の構成要件
- **Composition**: 中央に四条件の直列関係、下部に研究目標。
- **Title**: 課題：最終画像だけでは操作の正しさを保証できない
- **Core message**: 「道具がある」だけでなく、正しい接触、食物移動、非編集領域保持を同時に満たす必要がある。
- **Content**: Action Success · Photo Success · Preservation Success · Strict End-to-End Success / 目標：接触から持ち上げまでを明示的に制御する

#### Slide 05 - 関連研究と本研究の位置付け

- **Audience move**: 既存法の役割が混在 → 各方法の強みと不足を整理
- **Relationships**: 静止画編集、動画拡散、幾何編集、セグメンテーションの補完関係とFoodStateEditの統合位置
- **Composition**: 四つの研究領域を横並びにし、下段中央に本研究を置く。
- **Title**: 関連研究と位置付け
- **Core message**: 本研究は既存基盤モデルを置き換えるのではなく、柔軟物の操作制御と検証を追加する。
- **Content**: Qwen Image：写真品質 / VACE：時間連続生成 / GeoEdit系：幾何制御 / SAM：領域抽出 / Ours：相対3D代理＋段階動作＋保持評価

### Part 2: 提案手法

#### Slide 06 - 提案手法の全体像

- **Audience move**: 部品の理解 → 入力から最終画像までの一連の流れを把握
- **Relationships**: 入力画像・動作と材料・編集マスク、相対3D動作制御、透視投影と可視性、材料別制御、凍結VACE、支持領域合成、編集出力の順序
- **Composition**: 二頁報告の図1をそのまま大きく配置し、モジュール、連線、文字、証拠画像を変更しない。
- **Title**: 提案手法：相対3D制御と凍結VACE
- **Core message**: 材料別の相対3D操作条件を凍結VACEへ渡し、支持領域内だけを元画像へ合成する。
- **Content**: Input：source image・action/material・edit mask / Relative 3D action control：contact/depth・perspective/visibility・material-specific controls / Frozen diffusion：condition encoder・Wan2.2 Fun A14B・LoRA off・TTM off / Local compositing：support composite・edited output
- **Images**: images/p05_framework_report_exact.png as the exact Figure 1 extracted from the two-page report.

#### Slide 07 - 相対3D代理とVACEの接続

- **Audience move**: 提案手法の全体像を見る → 相対3D代理がモデルのどこに入り、何を制御するか説明できる
- **Relationships**: 入力と編集領域 → 相対3D代理 → 21フレーム制御とマスク → 凍結VACE → 動画と最終画像、という順序と依存関係
- **Composition**: 上部に「相対3D代理はVACE前段の制御信号生成部」と定義し、中央に五段階の水平フロー、下部に三つのVACE入力と担当する制御内容を対応付ける。
- **Title**: 相対3D代理とVACEの接続
- **Core message**: 相対3D代理は新しい生成モデルではない。接触点、相対深度、動作段階をVACEが読める制御動画とマスクへ変換するFoodStateEditの前段制御器である。
- **Content**: FoodStateEdit = relative3D controller + frozen VACE + local composition / vace_video = where and when / vace_video_mask = union of changed pixels across 21 proxy frames, followed by dilation and feathering / vace_reference_image + prompt = input content and appearance / current mask is generated from proxy trajectories rather than SAM / normalized pinhole scaffold, not reconstructed scene geometry.

#### Slide 08 - 同一入力での生成結果比較

- **Audience move**: 手法の構造を理解 → 実際の入力と各手法の出力差を視覚的に判断
- **Relationships**: 同じスープ入力に対するInput参照、Qwen Image、FoodStateEditの三者比較
- **Composition**: 三つの等幅画像を横一列に大きく配置し、各列上部に26px以上の方法名を置く。FoodStateEdit列は緑で強調する。
- **Title**: 同一入力での生成結果比較
- **Core message**: Qwen Imageは写真らしい出力を作るが画角と背景を再構成し、FoodStateEditはスプーン動作を追加しながら支持領域外を保持した。
- **Content**: Input（参照画像） / Qwen Image / FoodStateEdit（提案手法） / 同一入力・seed 1 / 目標動作の実画像GTなし / 選択済み合成1例
- **Images**: images/p10_success_soup_input.png, images/p09_soup_qwen_day34_seed1.png, images/p10_success_soup_output.png.

### Part 3: 実験

#### Slide 09 - 複数食品での多手法比較

- **Audience move**: 単一成功例を見る → 食品種類を変えたときの方法間差を横断的に確認
- **Relationships**: ラーメン、スープ、ケーキの各入力に対するInput、ChordEdit、Qwen Image、FoodStateEditの出力を同じ列順で対応付ける
- **Composition**: 既存の三つの比較ストリップを全幅で縦に並べ、埋め込み済みの大きな方法名と食品名をそのまま読める寸法で配置する。
- **Title**: 複数食品での多手法比較
- **Core message**: 同一入力でも、静止画編集、既存編集、FoodStateEditでは操作表現と入力保持の傾向が異なる。
- **Content**: Noodle / Soup / Cake × Input / ChordEdit / Qwen Image / FoodStateEdit（Ours） / selected seed 1 / no real-image GT / internal qualitative comparison only.
- **Images**: images/p09comp_noodle.png, images/p09comp_soup.png, images/p09comp_cake.png.

#### Slide 10 - 実験設定と評価項目

- **Audience move**: 方法理解 → 比較が公平か判断できる
- **Relationships**: 同一入力・seed・生成条件を固定し、五条件と四評価指標を対応付ける
- **Composition**: 左に実験条件、右に評価指標、下に対象カテゴリ。
- **Title**: 実験設定
- **Core message**: 4カテゴリ×3 seeds×5条件の同条件比較で制御要素を検証した。
- **Content**: 4 cases × 3 seeds × 5 conditions = 60 cells / native, planar, fixed relative3D, material-adaptive relative3D, rollback-related condition / 21 frames, 20 steps, seed固定 / Action, Photo, Preservation, Strict

#### Slide 11 - 定量評価表

- **Audience move**: 定性例の観察 → 同一入力に対する保持量と変更量を数値で把握
- **Relationships**: ChordEdit、Qwen Image、VACE系四条件を同じ六指標列で比較し、提案手法の行だけを緑色と★で明示する。
- **Composition**: 七列の定量表を主役とし、最下段のFoodStateEdit（提案手法）行を薄緑で全面強調する。
- **Title**: 評価結果：入力保持と編集量
- **Core message**: FoodStateEditは領域外を完全保持し、比較したVACE系条件の中で編集領域内の変化量が最大だった。
- **Content**: 各手法 n=12 / PSNR / SSIM / outside-support MAE / outside exact preservation / inside-support change MAE / 変化量は動作成功率ではない。

#### Slide 12 - 消融実験表

- **Audience move**: 総合評価を見る → 提案手法内部の各制御条件が入力保持と編集量に与える差を把握
- **Relationships**: Native VACE、2D planar proxy、fixed relative3D、material-adaptive relative3Dを同一指標列で比較
- **Composition**: 消融表だけを一頁に配置し、material-adaptive relative3D（Ours）行を薄緑と★で強調する。下部に確認済み事項と未確認事項を二つの小枠で分離する。
- **Title**: 消融実験：制御条件の定量比較
- **Core message**: Oursは領域外を保持したまま編集領域内の変化量を最大化したが、relative3Dと材質適応の動作成功率上の優位はまだ確認できていない。
- **Content**: 4 food cases × 3 seeds / n=12 per condition / PSNR / SSIM / inside-support change MAE / outside exact preservation / selected development data only.

### Part 4: 今後の検証

#### Slide 13 - 現在の課題と今後の予定

- **Audience move**: 現在の結果を完成済みと捉える → 証拠の限界、次の目標、具体的な検証を区別して理解する
- **Relationships**: 現在の課題、次の目標、三つの実施項目を別々の情報群として提示
- **Composition**: 左に現在の課題、右に目標と実施項目、下部に研究上の到達目標を置く。
- **Title**: 現在の課題と今後の予定
- **Core message**: 実画像への一般化と独立評価を追加し、複数の食品カテゴリで再現可能な改善を確認する。
- **Content**: 課題：選択済み合成例に偏る、実画像・時系列未検証、内部非盲検評価 / 目標：Action・Photo・Preservationの同時成立 / 実施：held-out実画像40件を複数seedで生成、2–3名盲評、bootstrap CI・失敗分類・材質制御の修正

## X. Speaker Notes Requirements

- **Generation**: enabled
- **Filename**: match each SVG filename under `notes/`
- **Content**: 各ページの主張、数値の読み方、限界を日本語の自然な口頭説明で記す。未完結果は推測せず、合成ケーキは補助例と明示する。
- **Total duration**: approximately 8 minutes
- **Notes style**: formal but conversational Japanese
- **Presentation purpose**: 8分間で、体験価値、既存編集の問題、提案、現時点の検証結果、未完項目を順に説明する
