# FoodStateEdit 日文中期报告

作者：GUO ZHENGPENG；学号：2530030。报告日期为2026-09-06，实验结果基准日为2026-09-05。

参考 `250930nakamizo_1.pdf` 的 A4 日文论文版式：封面、摘要、目录、五章正文、参考文献，共14页。采用明朝体正文、黑体节标题、带竖线的章首页和页眉页码。未复制参考文件的个人信息或研究内容。

- `report_ja.json`：可编辑内容与分页源文件。
- `build_report.py`：ReportLab 生成器，使用系统日文字体，不需要服务器或模型。
- `report_ja.md`：生成的可读文字稿；应修改 JSON 后重新生成，以免下一次覆盖。
- 最终 PDF：仓库 `output/pdf/FoodStateEdit_Midterm_GUO_2530030_20260906.pdf`。

生成命令（仓库根目录，需安装 ReportLab 与 Pillow）：

```powershell
python paper/midterm_20260905/build_report.py
```

报告使用仓库内已核验的配图和结果数值。生成器检查每页是否溢出并输出布局记录；最终PDF还需重新渲染检查。所属、主指导教师和指导教师尚未提供，封面明确标明待补。本文作为草稿，不声称Day 13拓扑加权条件已获得改善，也不将现有内部评审称为两人独立盲评。

文献条目已核对ControlNet官方论文页面、VACE官方论文、LoRA与Flow Matching原始论文页面。运行环境引用固定的DiffSynth-Studio commit；不是以当前主分支替代实验环境。
