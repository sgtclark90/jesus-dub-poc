"""JESUS Film AI Dubbing — local web UI.

    python app.py        # then open the printed http://127.0.0.1:7860

Tab 1 — dub one clip into one language (with progress, subtitles, QA, music/voice options).
Tab 2 — "Language Suite": dub one clip into several languages at once.
"""
from __future__ import annotations

import gradio as gr

from src import api
from src.languages import NAMES

WHISPER_CHOICES = ["base", "small", "medium"]


def _qa_markdown(qa) -> str:
    s = qa.summary()
    md = (f"### Key-term QA — human-in-the-loop\n"
          f"- Terms checked: **{s['terms_checked']}** · auto-passed: **{s['auto_passed']}** · "
          f"needs review: **{s['needs_human_review']}**\n")
    if qa.needs_review:
        md += "\n**Flagged for a reviewer:**\n"
        for f in qa.needs_review:
            md += f"- seg {f.segment_id}: `{f.term_source}` → expected `{f.expected_target}` ({f.status})\n"
    else:
        md += "\n✅ All approved key terms rendered correctly.\n"
    return md


def run_single(video, target_name, whisper_model, gate, keep_music, voice_clone,
               progress=gr.Progress()):
    if not video:
        raise gr.Error("Please upload a video clip first.")

    def cb(frac, msg):
        progress(frac, desc=msg)

    try:
        result = api.dub(video, target_name, whisper_model=whisper_model, gate=gate,
                         keep_music=keep_music, voice_clone=voice_clone, progress=cb)
    except RuntimeError as e:
        raise gr.Error(str(e))

    rows = [[f"{s.start:.1f}", f"{s.end:.1f}", s.text_source, s.text_target or ""]
            for s in result["segments"]]
    qa_md = _qa_markdown(result["qa"])
    notes = "  \n".join(f"• {n}" for n in result.get("notes", []))

    if result["gated"]:
        gr.Warning("Key terms need review — fix them, then uncheck the gate to dub.")
        return None, rows, qa_md, None, notes
    return result["video"], rows, qa_md, result["srt"], notes


def run_suite(video, target_names, whisper_model, keep_music, progress=gr.Progress()):
    if not video:
        raise gr.Error("Please upload a video clip first.")
    if not target_names:
        raise gr.Error("Pick at least one language.")

    files, lines = [], []
    n = len(target_names)
    for i, name in enumerate(target_names):
        progress(i / n, desc=f"Dubbing {name} ({i+1}/{n})…")
        r = api.dub(video, name, whisper_model=whisper_model, keep_music=keep_music)
        files += [r["video"], r["srt"]]
        lines.append(f"- **{name}** → `{r['video']}`")
    progress(1.0, desc="Done.")
    return files, "### Generated\n" + "\n".join(lines)


with gr.Blocks(title="JESUS Film — AI Dubbing") as demo:
    gr.Markdown(
        "# ✝ JESUS Film — AI Dubbing\n"
        "Translate and re-voice a clip into another language in minutes. A human stays in "
        "charge of every sacred word — the tool flags uncertain key terms, never ships them silently."
    )

    with gr.Tab("Dub a clip"):
        with gr.Row():
            with gr.Column(scale=1):
                in_video = gr.Video(label="Source clip")
                target = gr.Dropdown(NAMES, value="Swahili", label="Target language")
                model = gr.Dropdown(WHISPER_CHOICES, value="base",
                                    label="Transcription quality (slower = better)")
                keep_music = gr.Checkbox(value=False, label="Keep original music / SFX")
                voice_clone = gr.Checkbox(value=False, label="Clone original voice (needs GPU / Colab)")
                gate = gr.Checkbox(value=False, label="Stop if a key term needs review (QA gate)")
                go = gr.Button("Dub it", variant="primary")
            with gr.Column(scale=1):
                out_video = gr.Video(label="Dubbed result")
                out_srt = gr.File(label="Subtitles (.srt)")
                notes = gr.Markdown()
        qa = gr.Markdown()
        table = gr.Dataframe(headers=["Start", "End", "Original", "Translation"],
                             wrap=True, label="Transcript → translation")
        go.click(run_single, [in_video, target, model, gate, keep_music, voice_clone],
                 [out_video, table, qa, out_srt, notes])

    with gr.Tab("Language Suite"):
        gr.Markdown("Dub one clip into **several languages at once** — the 'reach every "
                    "tongue' view. Each language produces a video + subtitle file.")
        with gr.Row():
            with gr.Column(scale=1):
                suite_video = gr.Video(label="Source clip")
                suite_langs = gr.CheckboxGroup(NAMES, value=["Swahili", "Spanish", "Hindi"],
                                               label="Target languages")
                suite_model = gr.Dropdown(WHISPER_CHOICES, value="base", label="Transcription quality")
                suite_music = gr.Checkbox(value=False, label="Keep original music / SFX")
                suite_go = gr.Button("Dub the suite", variant="primary")
            with gr.Column(scale=1):
                suite_files = gr.Files(label="All dubbed videos + subtitles")
                suite_summary = gr.Markdown()
        suite_go.click(run_suite, [suite_video, suite_langs, suite_model, suite_music],
                       [suite_files, suite_summary])


if __name__ == "__main__":
    demo.launch(inbrowser=True, theme=gr.themes.Soft())
