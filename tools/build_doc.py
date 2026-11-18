#!/usr/bin/env python3
"""把开题报告 Markdown 构建为 DOCX / PDF（复刻原文档排版）。

排版参数来自仓库中原有 docx 的样式表（宋体正文 12pt / 黑体标题、
A4、上下左右页边距 2.5/2.5/2.5/3cm、表格 15.5cm 宽、图 14.8cm 宽）。

流程：Markdown -> HTML（图片 base64 内嵌）-> LibreOffice 导入 HTML 保存为
ODT -> ODT 转 DOCX -> 修补 DOCX 页面设置 -> DOCX 转 PDF。
之所以中间要过一层 HTML/ODT，是因为 LibreOffice 直接 md->docx 会丢失表格、
且把图片导出为外部链接；HTML 导入路径能正确生成表格、样式与内嵌图片。

用法::

    uv run python tools/build_doc.py                # 默认构建 doc/ 下的开题报告
    uv run python tools/build_doc.py --output-dir .cache/doc_build/out  # 只出中间产物

依赖：LibreOffice（soffice）、pdfinfo/pdfimages（poppler，仅用于校验）。
"""

from __future__ import annotations

import argparse
import base64
import html
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MD = REPO_ROOT / "doc" / "开题报告_基于深度学习的人脸表情识别算法研究.md"
WORK_ROOT = REPO_ROOT / ".cache" / "doc_build"
LO_PROFILE = REPO_ROOT / ".cache" / "lo_profile"

# 与原 docx 的 w:sectPr 保持一致（单位：twip，1cm ≈ 567 twip）
PAGE_SIZE = (11906, 16838)  # A4
PAGE_MARGINS = dict(top=1417, right=1417, bottom=1417, left=1701, header=851, footer=992)

# 原 docx 的样式表 / 字体表（LO 从 HTML 导入时无法还原脚注、正文行距等细节，
# 因此在导出 docx 后直接用原样式表覆盖；模板即仓库原有 docx 中提取）。
TEMPLATE_DIR = Path(__file__).resolve().parent
STYLE_TEMPLATE = TEMPLATE_DIR / "docx_style_template.xml"
FONT_TABLE_TEMPLATE = TEMPLATE_DIR / "docx_font_table_template.xml"

# 表格列宽（twip），按文档中表格出现顺序套用；前三张沿用原 docx，
# 第一张（FER2013/RAF-DB 与 CK+ 类别映射，修订版新增）按三列均分估算。
TABLE_GRIDS: list[tuple[int, ...]] = [
    (2600, 2200, 3988),  # 表 1 类别映射（新增）
    (1758, 4394, 2636),  # 七、进度安排
    (1300, 3700, 2500, 1288),  # 附录 A
    (1500, 3300, 3988),  # 附录 B
]

# 原 docx 中图片的显示尺寸（EMU）：14.8cm × 9.25cm，与技术路线图 2024×1264 同比例
IMAGE_EXTENT = (5334000, 3324225)

TABLE_PROPS = (
    '<w:tblPr><w:tblW w:w="8788" w:type="dxa"/><w:jc w:val="left"/>'
    '<w:tblInd w:w="0" w:type="dxa"/><w:tblLayout w:type="fixed"/>'
    '<w:tblCellMar><w:top w:w="60" w:type="dxa"/><w:left w:w="60" w:type="dxa"/>'
    '<w:bottom w:w="60" w:type="dxa"/><w:right w:w="60" w:type="dxa"/></w:tblCellMar>'
    '<w:tblBorders>'
    '<w:top w:val="single" w:sz="4" w:space="0" w:color="000000"/>'
    '<w:left w:val="single" w:sz="4" w:space="0" w:color="000000"/>'
    '<w:bottom w:val="single" w:sz="4" w:space="0" w:color="000000"/>'
    '<w:right w:val="single" w:sz="4" w:space="0" w:color="000000"/>'
    '<w:insideH w:val="single" w:sz="4" w:space="0" w:color="000000"/>'
    '<w:insideV w:val="single" w:sz="4" w:space="0" w:color="000000"/>'
    "</w:tblBorders></w:tblPr>"
)

CSS = """
@page {
    size: 21cm 29.7cm;
    margin: 2.5cm 2.5cm 2.5cm 3cm;
}
body {
    font-family: "SimSun", "Songti SC", serif;
    font-size: 12pt;
    line-height: 160%;
    margin: 0;
}
h1 {
    font-family: "SimHei", "Heiti SC", sans-serif;
    font-size: 20pt;
    font-weight: bold;
    text-align: center;
    line-height: 140%;
    margin: 12pt 0 10pt 0;
}
h2 {
    font-family: "SimHei", "Heiti SC", sans-serif;
    font-size: 15pt;
    font-weight: bold;
    line-height: 140%;
    margin: 14pt 0 6pt 0;
}
h3 {
    font-family: "SimHei", "Heiti SC", sans-serif;
    font-size: 13pt;
    font-weight: bold;
    line-height: 140%;
    margin: 10pt 0 4pt 0;
}
p {
    margin: 0 0 6pt 0;
    text-align: justify;
    text-indent: 0;
}
p.topic {
    font-family: "SimHei", "Heiti SC", sans-serif;
    font-size: 15pt;
    font-weight: bold;
    text-align: center;
    text-indent: 0;
    margin: 6pt 0 10pt 0;
}
p.noindent { text-indent: 0; }
p.note { font-size: 10.5pt; text-indent: 0; }
p.ref { font-size: 10.5pt; text-indent: 0; }
p.cap { font-size: 10.5pt; text-align: center; text-indent: 0; margin: 2pt 0 6pt 0; }
img { display: block; width: 14.8cm; margin: 0 auto; }
table {
    border-collapse: collapse;
    width: 15.5cm;
    margin: 6pt 0;
}
th, td {
    border: 0.5pt solid #000000;
    padding: 1.5pt 3pt;
    font-family: "SimSun", "Songti SC", serif;
    font-size: 12pt;
    text-align: left;
    vertical-align: middle;
}
th {
    font-family: "SimHei", "Heiti SC", sans-serif;
    font-weight: bold;
    background-color: #F2F2F2;
}
table p { text-indent: 0; text-align: left; margin: 0; }
"""


def _escape(text: str) -> str:
    return html.escape(text, quote=False)


def _inline(text: str) -> str:
    """转义 HTML 后处理 **加粗** 与 `行内代码`。"""
    out = _escape(text)
    out = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", out)
    out = re.sub(r"`([^`]+)`", r"<code>\1</code>", out)
    return out


def _image_data_uri(path: Path) -> tuple[str, int]:
    """读取图片并转成 base64 data URI（LibreOffice HTML 导入会内嵌它）。"""
    data = path.read_bytes()
    suffix = path.suffix.lower().lstrip(".")
    mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg"}.get(suffix, "image/png")
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}", len(data)


def _is_table_separator(line: str) -> bool:
    stripped = line.strip()
    if not stripped.startswith("|") or not stripped.endswith("|"):
        return False
    cells = [c.strip() for c in stripped.strip("|").split("|")]
    return all(re.fullmatch(r":?-{2,}:?", c) for c in cells)


def _split_table_row(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _render_table(header: list[str], rows: list[list[str]]) -> str:
    # 注意：不要使用 <thead>/<tbody>，LibreOffice 的 HTML 导入无法识别，会退化成正文段落
    parts = ["<table>"]
    parts.append("<tr>" + "".join(f"<th>{_inline(c)}</th>" for c in header) + "</tr>")
    for row in rows:
        row = (row + [""] * len(header))[: len(header)]
        parts.append("<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in row) + "</tr>")
    parts.append("</table>")
    return "\n".join(parts)


def md_to_html(md_text: str, md_dir: Path) -> tuple[str, list[str]]:
    """把开题报告 Markdown 转成带样式的 HTML 片段。返回 (html, warnings)。"""
    lines = md_text.splitlines()
    warnings: list[str] = []
    blocks: list[str] = []
    in_refs = False
    i = 0
    while i < len(lines):
        raw = lines[i].rstrip()
        line = raw.strip()
        if not line:
            i += 1
            continue

        # 标题
        m = re.match(r"^(#{1,6})\s+(.*)$", line)
        if m:
            level = min(len(m.group(1)), 3)
            text = m.group(2).strip()
            if re.search(r"参考文献", text):
                in_refs = True
            elif level <= 2:
                in_refs = False
            blocks.append(f"<h{level}>{_inline(text)}</h{level}>")
            i += 1
            continue

        # 水平线
        if re.fullmatch(r"-{3,}", line):
            i += 1
            continue

        # 表格（原 Markdown 表格缺少分隔行，因此按连续以 | 开头的行识别；
        # 若第二行是分隔行则跳过它）
        if line.startswith("|"):
            raw_rows: list[list[str]] = []
            j = i
            while j < len(lines) and lines[j].strip().startswith("|"):
                raw_rows.append(_split_table_row(lines[j].strip()))
                j += 1
            if len(raw_rows) >= 2:
                has_sep = _is_table_separator(lines[i + 1])
                header = raw_rows[0]
                body = raw_rows[2:] if has_sep else raw_rows[1:]
                blocks.append(_render_table(header, body))
                i = j
                continue

        # 图片（独占一行）
        m = re.fullmatch(r"!\[(.*?)\]\((.+?)\)", line)
        if m:
            alt, rel = m.group(1).strip(), m.group(2).strip()
            img_path = (md_dir / rel).resolve()
            if img_path.exists():
                uri, nbytes = _image_data_uri(img_path)
                blocks.append(f'<p><img src="{uri}" alt="{_escape(alt)}"></p>')
                if alt:
                    blocks.append(f'<p class="cap">{_inline(alt)}</p>')
                warnings.append(f"已内嵌图片 {rel}（{nbytes / 1024:.0f} KB）")
            else:
                warnings.append(f"警告：图片不存在，已跳过：{img_path}")
            i += 1
            continue

        # 题目行
        if re.match(r"^\*\*题目[:：]", line):
            blocks.append(f'<p class="topic">{_inline(line)}</p>')
            i += 1
            continue

        # 封面信息 / 签字栏：不缩进
        if re.match(r"^(学生姓名|专业|指导教师|签字)", line):
            blocks.append(f'<p class="noindent">{_inline(line)}</p>')
            i += 1
            continue

        # 注释行
        if re.match(r"^注[:：]", line):
            blocks.append(f'<p class="note">{_inline(line)}</p>')
            i += 1
            continue

        # 参考文献条目
        if in_refs and re.match(r"^\[\d+\]", line):
            blocks.append(f'<p class="ref">{_inline(line)}</p>')
            i += 1
            continue

        # 普通正文
        blocks.append(f"<p>{_inline(line)}</p>")
        i += 1

    doc = (
        "<!DOCTYPE html>\n<html lang=\"zh-CN\"><head><meta charset=\"utf-8\">"
        f"<title>开题报告</title><style>{CSS}</style></head><body>\n"
        + "\n".join(blocks)
        + "\n</body></html>\n"
    )
    return doc, warnings


def _run(cmd: list[str], what: str) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(f"{what} 失败（{proc.returncode}）：\n{proc.stdout}\n{proc.stderr}")
    if proc.stdout.strip():
        print(proc.stdout.strip())


def _soffice(args: list[str]) -> None:
    _run(
        [
            "soffice",
            "--headless",
            f"-env:UserInstallation=file://{LO_PROFILE}",
            *args,
        ],
        "LibreOffice",
    )


def _patch_single_table(table: str, index: int) -> str:
    table = re.sub(r"<w:tblPr>.*?</w:tblPr>", lambda _m: TABLE_PROPS, table, count=1, flags=re.S)
    grid = TABLE_GRIDS[index] if index < len(TABLE_GRIDS) else None
    if grid is not None:
        grid_xml = "<w:tblGrid>" + "".join(f'<w:gridCol w:w="{w}"/>' for w in grid) + "</w:tblGrid>"
        table = re.sub(r"<w:tblGrid>.*?</w:tblGrid>", lambda _m: grid_xml, table, count=1, flags=re.S)

    row_counter = {"i": 0}

    def patch_row(match: re.Match[str]) -> str:
        row = match.group(0)
        is_header = row_counter["i"] == 0
        style = "Style17" if is_header else "Style14"
        row = re.sub(r'<w:pStyle w:val="[^"]*"/>', f'<w:pStyle w:val="{style}"/>', row)
        if is_header:
            row = row.replace(
                f'<w:pStyle w:val="{style}"/>',
                f'<w:pStyle w:val="{style}"/>'
                '<w:shd w:val="clear" w:color="auto" w:fill="F2F2F2"/>',
            )
        row_counter["i"] += 1
        return row

    return re.sub(r"<w:tr\b.*?</w:tr>", patch_row, table, flags=re.S)


def _patch_tables(doc_xml: str) -> tuple[str, int]:
    """表格套用原文档的表属性与列宽，表头/内容单元格分别改用 Style17/Style14。"""
    out: list[str] = []
    last = 0
    count = 0
    for match in re.finditer(r"<w:tbl>.*?</w:tbl>", doc_xml, re.S):
        out.append(doc_xml[last : match.start()])
        out.append(_patch_single_table(match.group(0), count))
        count += 1
        last = match.end()
    out.append(doc_xml[last:])
    return "".join(out), count


def _patch_docx(docx_path: Path) -> dict[str, int | list[str]]:
    """修补 LibreOffice 导出的 docx：页面尺寸/边距、外部图片链接改为内嵌。"""
    with zipfile.ZipFile(docx_path) as zin:
        items = {name: zin.read(name) for name in zin.namelist()}

    doc_xml = items["word/document.xml"].decode("utf-8")
    rels_xml = items["word/_rels/document.xml.rels"].decode("utf-8")
    content_types = items["[Content_Types].xml"].decode("utf-8")
    external_images: list[str] = []

    # 1) 把外部图片链接改成内嵌
    for rel in re.findall(r"<Relationship\b[^>]*TargetMode=\"External\"[^>]*/>", rels_xml):
        m = re.search(r'Id="([^"]+)"', rel)
        if not m:
            continue
        rid = m.group(1)
        target = re.search(r'Target="([^"]+)"', rel)
        if not target:
            continue
        src = target.group(1)
        if src.startswith("file://"):
            src = src[len("file://"):]
        src_path = Path(src)
        if not src_path.exists():
            continue
        suffix = src_path.suffix.lower()
        media_name = f"word/media/image{len(external_images) + 1}{suffix}"
        items[media_name] = src_path.read_bytes()
        new_rel = re.sub(r'\s+TargetMode="External"', "", rel)
        new_rel = re.sub(r'Target="[^"]+"', f'Target="media/image{len(external_images) + 1}{suffix}"', new_rel)
        rels_xml = rels_xml.replace(rel, new_rel)
        doc_xml = doc_xml.replace(f'r:link="{rid}"', f'r:embed="{rid}"')
        external_images.append(media_name)
        if f'Extension="{suffix.lstrip(".")}"' not in content_types:
            content_types = content_types.replace(
                "</Types>",
                f'<Default Extension="{suffix.lstrip(".")}" ContentType="image/'
                f"{'png' if suffix == '.png' else 'jpeg'}\"/></Types>",
            )

    # 2) 图片显示尺寸按原文档固定（LO 会按 PNG 内嵌 DPI 放大到 25.7cm，需强制改回 14.8cm）
    doc_xml = re.sub(
        r'<wp:extent cx="\d+" cy="\d+"/>',
        f'<wp:extent cx="{IMAGE_EXTENT[0]}" cy="{IMAGE_EXTENT[1]}"/>',
        doc_xml,
    )
    doc_xml = re.sub(
        r'<a:ext cx="\d+" cy="\d+"/>',
        f'<a:ext cx="{IMAGE_EXTENT[0]}" cy="{IMAGE_EXTENT[1]}"/>',
        doc_xml,
    )

    # 3) 清除 HTML 导入产生的直接段落格式（行距/段距/缩进/bidi 等），交给原样式表控制
    doc_xml = re.sub(r"<w:spacing\b[^>]*/>", "", doc_xml)
    doc_xml = re.sub(r"<w:ind\b[^>]*/>", "", doc_xml)
    doc_xml = re.sub(r"<w:bidi\b[^>]*/>", "", doc_xml)
    doc_xml = re.sub(r"<w:numPr>.*?</w:numPr>", "", doc_xml, flags=re.S)

    # 4) 表格：套用原文档表属性/列宽，表头 Style17 + 浅灰底纹
    doc_xml, n_tables = _patch_tables(doc_xml)

    # 5) 参考文献条目：悬挂缩进（原文档为 left=420, hanging=420）
    doc_xml = doc_xml.replace(
        '<w:pStyle w:val="ref"/>',
        '<w:pStyle w:val="ref"/><w:ind w:left="420" w:hanging="420"/>',
    )

    # 6) “指导教师意见：”与签字栏保持同页，避免单独留出只有签字的空白页
    def _keep_with_next(match: re.Match[str]) -> str:
        block = match.group(0)
        if "指导教师意见" not in block:
            return block
        return block.replace(
            '<w:pStyle w:val="noindent"/>',
            '<w:pStyle w:val="noindent"/><w:keepNext/>',
            1,
        )

    doc_xml = re.sub(r"<w:p>.*?</w:p>", _keep_with_next, doc_xml, flags=re.S)

    # 7) 样式表 / 字体表直接采用原 docx 模板，复刻行距、字体、字号与段距
    if STYLE_TEMPLATE.exists():
        items["word/styles.xml"] = STYLE_TEMPLATE.read_bytes()
    if FONT_TABLE_TEMPLATE.exists():
        items["word/fontTable.xml"] = FONT_TABLE_TEMPLATE.read_bytes()

    # 8) 覆盖页面设置
    sect = (
        "<w:sectPr>"
        f'<w:pgSz w:w="{PAGE_SIZE[0]}" w:h="{PAGE_SIZE[1]}"/>'
        f'<w:pgMar w:top="{PAGE_MARGINS["top"]}" w:right="{PAGE_MARGINS["right"]}" '
        f'w:bottom="{PAGE_MARGINS["bottom"]}" w:left="{PAGE_MARGINS["left"]}" '
        f'w:header="{PAGE_MARGINS["header"]}" w:footer="{PAGE_MARGINS["footer"]}" w:gutter="0"/>'
        "</w:sectPr>"
    )
    if re.search(r"<w:sectPr\b.*?</w:sectPr>", doc_xml, re.S):
        doc_xml = re.sub(r"<w:sectPr\b.*?</w:sectPr>", sect, doc_xml, count=1, flags=re.S)
    else:
        doc_xml = doc_xml.replace("</w:body>", sect + "</w:body>")

    items["word/document.xml"] = doc_xml.encode("utf-8")
    items["word/_rels/document.xml.rels"] = rels_xml.encode("utf-8")
    items["[Content_Types].xml"] = content_types.encode("utf-8")

    with zipfile.ZipFile(docx_path, "w", zipfile.ZIP_DEFLATED) as zout:
        for name, data in items.items():
            zout.writestr(name, data)

    media = [n for n in items if n.startswith("word/media/")]
    return {"media": len(media), "external_images_fixed": external_images, "tables": n_tables}


def _pdf_info(pdf_path: Path) -> tuple[int | None, int | None]:
    pages = images = None
    try:
        proc = subprocess.run(["pdfinfo", str(pdf_path)], capture_output=True, text=True)
        m = re.search(r"^Pages:\s+(\d+)", proc.stdout, re.M)
        pages = int(m.group(1)) if m else None
    except FileNotFoundError:
        pass
    try:
        proc = subprocess.run(["pdfimages", "-list", str(pdf_path)], capture_output=True, text=True)
        images = max(0, len(proc.stdout.strip().splitlines()) - 1)
    except FileNotFoundError:
        pass
    return pages, images


def build(md_path: Path, output_dir: Path) -> int:
    md_path = md_path.resolve()
    if not md_path.exists():
        raise SystemExit(f"找不到 Markdown：{md_path}")
    md_dir = md_path.parent
    stem = md_path.stem
    work = WORK_ROOT
    work.mkdir(parents=True, exist_ok=True)
    for sub in ("html", "odt", "docx", "pdf"):
        (work / sub).mkdir(exist_ok=True)

    md_text = md_path.read_text(encoding="utf-8")
    doc_html, warnings = md_to_html(md_text, md_dir)
    html_path = work / "html" / f"{stem}.html"
    html_path.write_text(doc_html, encoding="utf-8")
    for w in warnings:
        print(f"[html] {w}")

    # 1) HTML -> ODT（直接 html->docx 会报 "no export filter"）
    _soffice(["--infilter=HTML (StarWriter)", "--convert-to", "odt", str(html_path), "--outdir", str(work / "odt")])
    odt_path = work / "odt" / f"{stem}.odt"
    if not odt_path.exists():
        raise SystemExit(f"ODT 未生成：{odt_path}")

    # 2) ODT -> DOCX
    _soffice(["--convert-to", "docx", str(odt_path), "--outdir", str(work / "docx")])
    docx_path = work / "docx" / f"{stem}.docx"
    if not docx_path.exists():
        raise SystemExit(f"DOCX 未生成：{docx_path}")

    # 3) 修补页面设置 / 图片内嵌，再转 PDF
    patch = _patch_docx(docx_path)
    _soffice(["--convert-to", "pdf", str(docx_path), "--outdir", str(work / "pdf")])
    pdf_path = work / "pdf" / f"{stem}.pdf"

    output_dir.mkdir(parents=True, exist_ok=True)
    final_docx = output_dir / f"{stem}.docx"
    final_pdf = output_dir / f"{stem}.pdf"
    shutil.copyfile(docx_path, final_docx)
    shutil.copyfile(pdf_path, final_pdf)

    pages, images = _pdf_info(final_pdf)
    print("\n构建完成：")
    print(f"  HTML: {html_path}")
    print(
        f"  DOCX: {final_docx}（表格 {patch['tables']} 个，内嵌图片 {patch['media']} 张，"
        f"修补外部链接 {len(patch['external_images_fixed'])} 个）"
    )
    print(f"  PDF : {final_pdf}（{pages} 页，位图 {images} 张）")
    if images is not None and images < 1:
        print("警告：PDF 中没有检测到位图，图片可能未成功内嵌。", file=sys.stderr)
        return 2
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--md", type=Path, default=DEFAULT_MD, help="Markdown 源文件")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="docx/pdf 输出目录（默认与 md 同目录）",
    )
    args = parser.parse_args(argv)
    output_dir = args.output_dir if args.output_dir is not None else args.md.resolve().parent
    return build(args.md, output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
