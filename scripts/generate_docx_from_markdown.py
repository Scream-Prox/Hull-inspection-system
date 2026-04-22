import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from xml.sax.saxutils import escape


def twips_from_cm(value_cm: float) -> int:
    return int(round(value_cm / 2.54 * 1440))


def paragraph_xml(
    text: str,
    *,
    font: str = "Times New Roman",
    size_pt: int = 14,
    bold: bool = False,
    italic: bool = False,
    align: str = "both",
    first_line_twips: int = twips_from_cm(1.25),
    left_twips: int = 0,
    before_twips: int = 0,
    after_twips: int = 0,
) -> str:
    safe_text = escape(text)
    rpr = [
        f'<w:rFonts w:ascii="{font}" w:hAnsi="{font}" w:cs="{font}" w:eastAsia="{font}"/>',
        f'<w:sz w:val="{size_pt * 2}"/>',
        f'<w:szCs w:val="{size_pt * 2}"/>',
    ]
    if bold:
        rpr.append("<w:b/>")
    if italic:
        rpr.append("<w:i/>")

    ppr = [
        f'<w:jc w:val="{align}"/>',
        f'<w:spacing w:before="{before_twips}" w:after="{after_twips}" w:line="360" w:lineRule="auto"/>',
        f'<w:ind w:firstLine="{first_line_twips}" w:left="{left_twips}"/>',
    ]

    return (
        "<w:p>"
        f"<w:pPr>{''.join(ppr)}</w:pPr>"
        "<w:r>"
        f"<w:rPr>{''.join(rpr)}</w:rPr>"
        f'<w:t xml:space="preserve">{safe_text}</w:t>'
        "</w:r>"
        "</w:p>"
    )


def blank_paragraph() -> str:
    return paragraph_xml("", first_line_twips=0, after_twips=0)


def build_document_body(lines: list[str]) -> str:
    body_parts: list[str] = []
    in_code_block = False
    first_title = True

    for raw_line in lines:
        line = raw_line.rstrip("\n")

        if line.startswith("```"):
            in_code_block = not in_code_block
            body_parts.append(blank_paragraph())
            continue

        if in_code_block:
            body_parts.append(
                paragraph_xml(
                    line,
                    font="Consolas",
                    size_pt=10,
                    align="left",
                    first_line_twips=0,
                    left_twips=twips_from_cm(0.8),
                )
            )
            continue

        if not line.strip():
            body_parts.append(blank_paragraph())
            continue

        if line.startswith("# "):
            text = line[2:].strip()
            if first_title:
                body_parts.append(
                    paragraph_xml(
                        text,
                        size_pt=16,
                        bold=True,
                        align="center",
                        first_line_twips=0,
                        before_twips=120,
                        after_twips=120,
                    )
                )
                first_title = False
            else:
                body_parts.append(
                    paragraph_xml(
                        text,
                        size_pt=14,
                        bold=True,
                        align="left",
                        first_line_twips=0,
                        before_twips=120,
                        after_twips=80,
                    )
                )
            continue

        if line.startswith("## "):
            body_parts.append(
                paragraph_xml(
                    line[3:].strip(),
                    size_pt=14,
                    bold=True,
                    align="left",
                    first_line_twips=0,
                    before_twips=100,
                    after_twips=60,
                )
            )
            continue

        if line.startswith("### "):
            body_parts.append(
                paragraph_xml(
                    line[4:].strip(),
                    size_pt=14,
                    bold=True,
                    align="left",
                    first_line_twips=0,
                    before_twips=80,
                    after_twips=40,
                )
            )
            continue

        if line.startswith("#### "):
            body_parts.append(
                paragraph_xml(
                    line[5:].strip(),
                    size_pt=13,
                    bold=True,
                    align="left",
                    first_line_twips=0,
                    before_twips=60,
                    after_twips=20,
                )
            )
            continue

        if line.startswith("[") and line.endswith("]"):
            body_parts.append(
                paragraph_xml(
                    line,
                    size_pt=12,
                    italic=True,
                    align="center",
                    first_line_twips=0,
                    before_twips=40,
                    after_twips=40,
                )
            )
            continue

        if line.startswith("- "):
            body_parts.append(
                paragraph_xml(
                    line,
                    size_pt=14,
                    align="both",
                    first_line_twips=0,
                    left_twips=twips_from_cm(0.7),
                )
            )
            continue

        body_parts.append(paragraph_xml(line))

    section = (
        "<w:sectPr>"
        '<w:pgSz w:w="11906" w:h="16838"/>'
        '<w:pgMar w:top="1134" w:right="850" w:bottom="1134" w:left="1701" w:header="708" w:footer="708" w:gutter="0"/>'
        "</w:sectPr>"
    )

    return "".join(body_parts) + section


def build_document_xml(body: str) -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:wpc="http://schemas.microsoft.com/office/word/2010/wordprocessingCanvas" '
        'xmlns:mc="http://schemas.openxmlformats.org/markup-compatibility/2006" '
        'xmlns:o="urn:schemas-microsoft-com:office:office" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
        'xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math" '
        'xmlns:v="urn:schemas-microsoft-com:vml" '
        'xmlns:wp14="http://schemas.microsoft.com/office/word/2010/wordprocessingDrawing" '
        'xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing" '
        'xmlns:w10="urn:schemas-microsoft-com:office:word" '
        'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
        'xmlns:w14="http://schemas.microsoft.com/office/word/2010/wordml" '
        'xmlns:wpg="http://schemas.microsoft.com/office/word/2010/wordprocessingGroup" '
        'xmlns:wpi="http://schemas.microsoft.com/office/word/2010/wordprocessingInk" '
        'xmlns:wne="http://schemas.microsoft.com/office/2006/wordml" '
        'xmlns:wps="http://schemas.microsoft.com/office/word/2010/wordprocessingShape" '
        'mc:Ignorable="w14 wp14">'
        f"<w:body>{body}</w:body>"
        "</w:document>"
    )


def content_types_xml() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/word/document.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
        '<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>'
        '<Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>'
        "</Types>"
    )


def rels_xml() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="word/document.xml"/>'
        '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" '
        'Target="docProps/core.xml"/>'
        '<Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" '
        'Target="docProps/app.xml"/>'
        "</Relationships>"
    )


def core_xml() -> str:
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" '
        'xmlns:dc="http://purl.org/dc/elements/1.1/" '
        'xmlns:dcterms="http://purl.org/dc/terms/" '
        'xmlns:dcmitype="http://purl.org/dc/dcmitype/" '
        'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">'
        "<dc:title>Курсовая работа по проекту инспекции корпуса судна</dc:title>"
        "<dc:creator>OpenAI Codex</dc:creator>"
        "<cp:lastModifiedBy>OpenAI Codex</cp:lastModifiedBy>"
        f'<dcterms:created xsi:type="dcterms:W3CDTF">{now}</dcterms:created>'
        f'<dcterms:modified xsi:type="dcterms:W3CDTF">{now}</dcterms:modified>'
        "</cp:coreProperties>"
    )


def app_xml() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" '
        'xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">'
        "<Application>Microsoft Office Word</Application>"
        "</Properties>"
    )


def main() -> int:
    if len(sys.argv) != 3:
        print("Usage: python generate_docx_from_markdown.py <source.md> <output.docx>")
        return 1

    source = Path(sys.argv[1]).resolve()
    output = Path(sys.argv[2]).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    lines = source.read_text(encoding="utf-8").splitlines()
    body = build_document_body(lines)
    document_xml = build_document_xml(body)

    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types_xml())
        zf.writestr("_rels/.rels", rels_xml())
        zf.writestr("docProps/core.xml", core_xml())
        zf.writestr("docProps/app.xml", app_xml())
        zf.writestr("word/document.xml", document_xml)

    print(f"Saved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
