"""Create the sample NovaTech knowledge base (PDF + TXT + CSV) in data/."""
import csv
from pathlib import Path

import kb_data


def write_sample_kb(data_dir="data"):
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

    out = Path(data_dir)
    out.mkdir(exist_ok=True)
    styles = getSampleStyleSheet()
    for fname, (title, sections) in kb_data.PDF_DOCS.items():
        story = [Paragraph(title, styles["Title"]), Spacer(1, 8)]
        for heading, paragraphs in sections:
            story.append(Paragraph(heading, styles["Heading2"]))
            for p in paragraphs:
                story += [Paragraph(p.replace("&", "&amp;"), styles["BodyText"]), Spacer(1, 6)]
        SimpleDocTemplate(str(out / fname), pagesize=A4, title=title).build(story)
    (out / kb_data.TXT_NAME).write_text(kb_data.TXT_DOC, encoding="utf-8")
    with open(out / kb_data.FAQ_NAME, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["category", "question", "answer"])
        w.writerows(kb_data.FAQ_ROWS)
    return sorted(p.name for p in out.iterdir())


SAMPLE_FILES = set(list(kb_data.PDF_DOCS) + [kb_data.TXT_NAME, kb_data.FAQ_NAME])

if __name__ == "__main__":
    print("\n".join(write_sample_kb()))
