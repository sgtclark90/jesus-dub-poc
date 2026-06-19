"""JESUS Film AI Dubbing — local web UI.

    python app.py        # then open the printed http://127.0.0.1:7860

Drag in a clip, pick a language, watch the progress bar, get a dubbed video +
subtitles + the human-in-the-loop key-term report.
"""
from __future__ import annotations

import gradio as gr

from src import api
from src.languages import NAMES

WHISPER_CHOICES = ["base", "small", "medium"]


def _qa_markdown(qa) -> str:
    s = qa.summary()
    head = (f"### Key-term QA — human-in-the-loop\n"
            f"- Terms checked: **{s['terms_checked']}**\n"
            f"- Auto-passed: **{s['auto_passed']}**\n"
            f"- Needs human review: **{s['needs_human_review']}**\n")
    if qa.needs_review:
        head += "\n**Flagged for a reviewer:**\n"
        for f in qa.needs_review:
            head += f"- seg {f.segment_id}: `{f.term_source}` → expected `{f.expected_target}` ({f.status})\n"
    else:
        head += "\n✅ All approved key terms rendered correctly.\n"
    return head


def run(video, target_name, whisper_model, gate, progress=gr.Progress()):
    if not video:
        raise gr.Error("Please upload a video clip first.")

    def cb(frac, msg):
        progress(frac, desc=msg)

    result = api.dub(video, target_name, whisper_model=whisper_model, gate=gate, progress=cb)

    rows = [[f"{s.start:.1f}", f"{s.end:.1f}", s.text_source, s.text_target or ""]
            for s in result["segments"]]
    qa_md = _qa_markdown(result["qa"])

    if result["gated"]:
        gr.Warning("Key terms need review — fix the glossary/translation, then uncheck the gate to dub.")
        return None, rows, qa_md, None

    return result["video"], rows, qa_md, result["srt"]


with gr.Blocks(title="JESUS Film — AI Dubbing") as demo:
    gr.Markdown(
        "# ✝ JESUS Film — AI Dubbing\n"
        "Translate and re-voice a clip into another language in minutes. "
        "A human stays in charge of every sacred word — the tool flags uncertain key "
        "terms instead of shipping them silently."
    )
    with gr.Row():
        with gr.Column(scale=1):
            in_video = gr.Video(label="Source clip")
            target = gr.Dropdown(NAMES, value="Swahili", label="Target language")
            model = gr.Dropdown(WHISPER_CHOICES, value="base",
                                label="Transcription quality (slower = better)")
            gate = gr.Checkbox(value=False, label="Stop if a key term needs review (QA gate)")
            go = gr.Button("Dub it", variant="primary")
        with gr.Column(scale=1):
            out_video = gr.Video(label="Dubbed result")
            out_srt = gr.File(label="Subtitles (.srt)")
    qa = gr.Markdown()
    table = gr.Dataframe(headers=["Start", "End", "Original", "Translation"],
                         wrap=True, label="Transcript → translation")

    go.click(run, [in_video, target, model, gate], [out_video, table, qa, out_srt])


if __name__ == "__main__":
    demo.launch(inbrowser=True, theme=gr.themes.Soft())
