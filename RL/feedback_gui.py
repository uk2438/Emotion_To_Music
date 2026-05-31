import argparse
import json
import mimetypes
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

from feedback import (
    FEEDBACK_LOG_CSV,
    FEEDBACK_LOG_JSONL,
    append_feedback,
    get_reward_weight_overrides_for_mood,
)
from main import (
    EXPERIMENTS_ROOT,
    MODE_PROFILES,
    TEAM_CLASSIFIER_DEFAULT_CHECKPOINT,
    analyze_mood,
    calculate_melody_metrics,
    calculate_selection_score,
    convert_to_note_names,
    export_melody_to_midi,
    generate_melody,
    save_loss_csv,
    save_reward_breakdown_csv,
    save_reward_csv,
    set_seed,
    summarize_reward_breakdowns,
    train_factorized_dqn_agent,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
GUI_ROOT = PROJECT_ROOT / "results" / "feedback_gui"
UPLOAD_ROOT = GUI_ROOT / "uploads"
GENERATION_ROOT = GUI_ROOT / "generations"


INDEX_HTML = r"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>EmotionRL Feedback</title>
  <style>
    :root {
      color-scheme: light;
      --ink: #1d2428;
      --muted: #66737b;
      --line: #d9e0e4;
      --panel: #f6f8f9;
      --accent: #2f7d68;
      --accent-2: #9b3f50;
      --bg: #fbfcfc;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    * { box-sizing: border-box; }
    body { margin: 0; background: var(--bg); color: var(--ink); }
    main { width: min(1180px, calc(100vw - 32px)); margin: 24px auto 40px; }
    header { display: flex; justify-content: space-between; gap: 16px; align-items: end; margin-bottom: 18px; }
    h1 { font-size: 24px; line-height: 1.1; margin: 0; letter-spacing: 0; }
    .status { color: var(--muted); font-size: 13px; min-height: 18px; text-align: right; }
    .layout { display: grid; grid-template-columns: 360px 1fr; gap: 18px; align-items: start; }
    section, aside { border: 1px solid var(--line); border-radius: 8px; background: white; }
    .panel { padding: 16px; }
    .stack { display: grid; gap: 14px; }
    label { display: grid; gap: 6px; font-size: 13px; color: var(--muted); }
    input, select, textarea { width: 100%; border: 1px solid var(--line); border-radius: 6px; padding: 9px 10px; font: inherit; color: var(--ink); background: white; }
    input[type="range"] { padding: 0; accent-color: var(--accent); }
    input[type="file"] { padding: 8px; }
    textarea { min-height: 72px; resize: vertical; }
    button { border: 1px solid var(--line); background: white; color: var(--ink); border-radius: 6px; padding: 9px 12px; font: inherit; cursor: pointer; }
    button.primary { background: var(--accent); color: white; border-color: var(--accent); }
    button.secondary { background: #eef4f1; border-color: #cfe0da; }
    button:disabled { opacity: .45; cursor: not-allowed; }
    .buttons { display: flex; gap: 8px; flex-wrap: wrap; }
    .image-preview { width: 100%; aspect-ratio: 4 / 3; border: 1px dashed var(--line); border-radius: 8px; object-fit: cover; background: var(--panel); }
    .result-grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; margin-top: 12px; }
    .metric { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 10px; min-height: 72px; }
    .metric b { display: block; font-size: 20px; margin-top: 5px; }
    .metric span { color: var(--muted); font-size: 12px; }
    .wide { grid-column: span 2; }
    .mood-list { display: grid; gap: 6px; margin-top: 10px; }
    .mood-row { display: grid; grid-template-columns: 90px 1fr 44px; gap: 8px; align-items: center; font-size: 13px; }
    .bar { height: 8px; background: var(--panel); border-radius: 999px; overflow: hidden; }
    .fill { height: 100%; width: 0%; background: var(--accent); }
    .feedback-grid { display: grid; gap: 14px; margin-top: 12px; }
    .rating { display: grid; grid-template-columns: 140px repeat(5, 1fr); gap: 8px; align-items: center; }
    .rating > span { color: var(--muted); font-size: 13px; }
    .rating input { display: none; }
    .rating label { display: block; text-align: center; padding: 8px 0; border: 1px solid var(--line); border-radius: 6px; color: var(--ink); cursor: pointer; }
    .rating input:checked + label { background: var(--accent); color: white; border-color: var(--accent); }
    .sequence { margin-top: 12px; padding: 10px; border-radius: 8px; background: #101719; color: #d7ece5; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; overflow-x: auto; white-space: nowrap; }
    .paths { margin-top: 10px; color: var(--muted); font-size: 12px; line-height: 1.5; }
    .hidden { display: none; }
    @media (max-width: 860px) {
      .layout { grid-template-columns: 1fr; }
      .result-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .rating { grid-template-columns: 1fr repeat(5, 40px); }
    }
  </style>
</head>
<body>
  <main>
    <header>
      <h1>EmotionRL Feedback</h1>
      <div class="status" id="status"></div>
    </header>
    <div class="layout">
      <aside class="panel">
        <form id="generateForm" class="stack">
          <img id="preview" class="image-preview" alt="">
          <label>Image
            <input id="imageInput" name="image" type="file" accept="image/*" required>
          </label>
          <label>Episodes
            <input name="episodes" type="number" min="1" max="20000" value="800">
          </label>
          <label>Melody Length
            <input name="melody_length" type="number" min="8" max="96" value="32">
          </label>
          <label>Seed
            <input name="seed" type="number" value="42">
          </label>
          <button class="primary" type="submit">Generate</button>
        </form>
      </aside>

      <section class="panel">
        <div class="buttons">
          <button id="playBtn" class="secondary" disabled>Play</button>
          <button id="stopBtn" disabled>Stop</button>
          <a id="midiLink" class="hidden" href="#" download><button type="button">MIDI</button></a>
        </div>

        <div class="result-grid">
          <div class="metric"><span>Mood</span><b id="moodValue">-</b></div>
          <div class="metric"><span>Raw Label</span><b id="rawValue">-</b></div>
          <div class="metric"><span>Selection</span><b id="selectionValue">-</b></div>
          <div class="metric"><span>Quality</span><b id="qualityValue">-</b></div>
          <div class="metric"><span>Repeat</span><b id="repeatValue">-</b></div>
          <div class="metric"><span>Short Notes</span><b id="shortValue">-</b></div>
          <div class="metric wide"><span>Reward Weights</span><b id="weightValue">base</b></div>
        </div>

        <div class="mood-list" id="moodList"></div>
        <div class="sequence" id="sequence">No generation yet.</div>

        <form id="feedbackForm" class="feedback-grid">
          <div id="ratings"></div>
          <label>Comment
            <textarea name="comment"></textarea>
          </label>
          <button class="primary" id="feedbackBtn" type="submit" disabled>Save Feedback</button>
        </form>
        <div class="paths" id="paths"></div>
      </section>
    </div>
  </main>
  <script>
    const ratingFields = [
      ["emotion_match", "Emotion Match"],
      ["naturalness", "Naturalness"],
      ["repetition_control", "Repetition"],
      ["richness", "Richness"],
      ["overall", "Overall"],
    ];
    let currentGeneration = null;
    let audioContext = null;
    let scheduled = [];

    function setStatus(text) { document.getElementById("status").textContent = text; }
    function fixed(value, digits = 2) {
      if (value === undefined || value === null || Number.isNaN(Number(value))) return "-";
      return Number(value).toFixed(digits);
    }
    function buildRatings() {
      const root = document.getElementById("ratings");
      root.innerHTML = "";
      for (const [name, label] of ratingFields) {
        const row = document.createElement("div");
        row.className = "rating";
        row.append(Object.assign(document.createElement("span"), { textContent: label }));
        for (let value = 1; value <= 5; value++) {
          const input = document.createElement("input");
          input.type = "radio";
          input.name = name;
          input.value = String(value);
          input.id = `${name}_${value}`;
          if (value === 3) input.checked = true;
          const box = document.createElement("label");
          box.htmlFor = input.id;
          box.textContent = String(value);
          row.append(input, box);
        }
        root.append(row);
      }
    }
    buildRatings();

    document.getElementById("imageInput").addEventListener("change", (event) => {
      const file = event.target.files[0];
      if (!file) return;
      document.getElementById("preview").src = URL.createObjectURL(file);
    });

    document.getElementById("generateForm").addEventListener("submit", async (event) => {
      event.preventDefault();
      stopPreview();
      setStatus("Generating...");
      document.querySelector("#generateForm button").disabled = true;
      const formData = new FormData(event.target);
      try {
        const response = await fetch("/api/generate", { method: "POST", body: formData });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || "generation failed");
        currentGeneration = data;
        renderGeneration(data);
        setStatus("Ready");
      } catch (error) {
        setStatus(error.message);
      } finally {
        document.querySelector("#generateForm button").disabled = false;
      }
    });

    document.getElementById("feedbackForm").addEventListener("submit", async (event) => {
      event.preventDefault();
      if (!currentGeneration) return;
      const formData = new FormData(event.target);
      const feedback = {};
      for (const [name] of ratingFields) feedback[name] = Number(formData.get(name));
      setStatus("Saving...");
      const response = await fetch("/api/feedback", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          generation_id: currentGeneration.generation_id,
          feedback,
          comment: formData.get("comment") || "",
        }),
      });
      const data = await response.json();
      if (response.ok) {
        setStatus(`Saved. Adjusted ${fixed(data.feedback_adjusted_selection_score)}`);
        document.getElementById("weightValue").textContent = Object.keys(data.updated_reward_overrides || {}).length;
      } else {
        setStatus(data.error || "feedback failed");
      }
    });

    function renderGeneration(data) {
      document.getElementById("moodValue").textContent = data.music_mode;
      document.getElementById("rawValue").textContent = data.raw_label;
      document.getElementById("selectionValue").textContent = fixed(data.selection_score);
      document.getElementById("qualityValue").textContent = fixed(data.metrics.melodic_quality_score);
      document.getElementById("repeatValue").textContent = fixed(data.metrics.same_adjacent_ratio);
      document.getElementById("shortValue").textContent = fixed(data.metrics.short_note_ratio);
      document.getElementById("weightValue").textContent = Object.keys(data.reward_weight_overrides || {}).length || "base";
      document.getElementById("sequence").textContent = data.note_names.join("  ");
      document.getElementById("playBtn").disabled = false;
      document.getElementById("stopBtn").disabled = false;
      document.getElementById("feedbackBtn").disabled = false;
      const link = document.getElementById("midiLink");
      link.href = data.midi_url;
      link.classList.remove("hidden");
      renderMoodScores(data.music_scores);
      document.getElementById("paths").textContent = `${data.midi_file} | ${data.feedback_log_csv}`;
    }

    function renderMoodScores(scores) {
      const root = document.getElementById("moodList");
      root.innerHTML = "";
      const entries = Object.entries(scores || {}).sort((a, b) => b[1] - a[1]);
      for (const [label, score] of entries) {
        const row = document.createElement("div");
        row.className = "mood-row";
        const name = Object.assign(document.createElement("span"), { textContent: label });
        const bar = document.createElement("div");
        bar.className = "bar";
        const fill = document.createElement("div");
        fill.className = "fill";
        fill.style.width = `${Math.max(0, Math.min(100, score * 100))}%`;
        bar.append(fill);
        const value = Object.assign(document.createElement("span"), { textContent: fixed(score, 2) });
        row.append(name, bar, value);
        root.append(row);
      }
    }

    document.getElementById("playBtn").addEventListener("click", playPreview);
    document.getElementById("stopBtn").addEventListener("click", stopPreview);

    function midiToHz(pitch) { return 440 * Math.pow(2, (pitch - 69) / 12); }
    function playPreview() {
      if (!currentGeneration) return;
      stopPreview();
      audioContext = new AudioContext();
      const start = audioContext.currentTime + 0.05;
      for (const event of currentGeneration.preview_events) {
        const osc = audioContext.createOscillator();
        const gain = audioContext.createGain();
        osc.type = "triangle";
        osc.frequency.value = midiToHz(event.pitch);
        gain.gain.setValueAtTime(0.0001, start + event.start);
        gain.gain.exponentialRampToValueAtTime(Math.max(0.03, event.velocity / 900), start + event.start + 0.02);
        gain.gain.exponentialRampToValueAtTime(0.0001, start + event.start + Math.max(0.05, event.duration));
        osc.connect(gain).connect(audioContext.destination);
        osc.start(start + event.start);
        osc.stop(start + event.start + Math.max(0.08, event.duration + 0.05));
        scheduled.push(osc);
      }
      setStatus("Playing");
    }
    function stopPreview() {
      for (const osc of scheduled) {
        try { osc.stop(); } catch (_) {}
      }
      scheduled = [];
      if (audioContext) {
        audioContext.close();
        audioContext = null;
      }
    }
  </script>
</body>
</html>
"""


def _json_response(handler, payload, status=200):
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _text_response(handler, text, content_type="text/html; charset=utf-8", status=200):
    body = text.encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _safe_int(value, default, min_value, max_value):
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return max(min_value, min(max_value, number))


def _parse_content_disposition(value):
    parts = [part.strip() for part in value.split(";")]
    disposition = parts[0].lower() if parts else ""
    params = {}
    for part in parts[1:]:
        if "=" not in part:
            continue
        key, raw_value = part.split("=", 1)
        params[key.strip().lower()] = raw_value.strip().strip('"')
    return disposition, params


def _parse_multipart_form(headers, body):
    content_type = headers.get("Content-Type", "")
    boundary_marker = "boundary="
    if boundary_marker not in content_type:
        raise ValueError("multipart boundary is missing")

    boundary = content_type.split(boundary_marker, 1)[1].split(";", 1)[0].strip().strip('"')
    boundary_bytes = ("--" + boundary).encode("utf-8")
    form = {}

    for part in body.split(boundary_bytes):
        part = part.strip()
        if not part or part == b"--":
            continue
        if part.endswith(b"--"):
            part = part[:-2].strip()
        if b"\r\n\r\n" not in part:
            continue
        raw_headers, content = part.split(b"\r\n\r\n", 1)
        part_headers = {}
        for header_line in raw_headers.decode("utf-8", errors="replace").split("\r\n"):
            if ":" not in header_line:
                continue
            key, value = header_line.split(":", 1)
            part_headers[key.strip().lower()] = value.strip()
        disposition, params = _parse_content_disposition(
            part_headers.get("content-disposition", "")
        )
        if disposition != "form-data" or "name" not in params:
            continue
        name = params["name"]
        if content.endswith(b"\r\n"):
            content = content[:-2]
        form[name] = {
            "filename": params.get("filename"),
            "content": content,
            "value": content.decode("utf-8", errors="replace"),
            "headers": part_headers,
        }

    return form


def _save_upload(field):
    UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    original = Path(field.get("filename") or "upload.jpg").name
    suffix = Path(original).suffix or ".jpg"
    upload_path = UPLOAD_ROOT / f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}{suffix}"
    with open(upload_path, "wb") as image_file:
        image_file.write(field["content"])
    return upload_path


def _build_preview_events(melody, durations, velocities):
    events = []
    start = 0.0
    for pitch, duration, velocity in zip(melody, durations, velocities):
        duration = float(duration)
        events.append({
            "pitch": int(pitch),
            "duration": duration,
            "velocity": int(velocity),
            "start": round(start, 4),
        })
        start += duration
    return events


def generate_from_image(image_path, episodes, melody_length, seed):
    set_seed(seed)
    raw_label, mode, music_scores, classifier_result = analyze_mood(
        source="teammate",
        image_path=image_path,
        classifier_checkpoint=TEAM_CLASSIFIER_DEFAULT_CHECKPOINT,
    )
    reward_overrides = get_reward_weight_overrides_for_mood(mode)
    env, agent, rewards, training_metrics = train_factorized_dqn_agent(
        mode=mode,
        episodes=episodes,
        melody_length=melody_length,
        mood_vector=music_scores,
        batch_size=64,
        reward_weight_overrides=reward_overrides,
    )
    (
        actions,
        melody,
        durations,
        velocities,
        pitch_actions,
        duration_actions,
        velocity_actions,
        events,
        reward_breakdowns,
    ) = generate_melody(env, agent)

    metrics = calculate_melody_metrics(
        actions=actions,
        melody=melody,
        durations=durations,
        velocities=velocities,
        pitch_actions=pitch_actions,
        mode=mode,
        base_notes=env.base_notes,
    )
    reward_summary = summarize_reward_breakdowns(reward_breakdowns)
    selection_score = calculate_selection_score(
        reward_summary["total_reward"],
        metrics,
        mode=mode,
    )
    note_names = convert_to_note_names(melody)

    generation_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    generation_dir = GENERATION_ROOT / generation_id
    generation_dir.mkdir(parents=True, exist_ok=True)
    midi_filename = generation_dir / f"generated_{mode}.mid"
    export_melody_to_midi(
        melody,
        filename=midi_filename,
        durations=durations,
        velocities=velocities,
        tempo=MODE_PROFILES[mode]["tempo"],
        instrument_name=MODE_PROFILES[mode]["instrument"],
        chord_progression=MODE_PROFILES[mode]["chord_progression"],
    )
    save_reward_csv(rewards, generation_dir / "episode_rewards.csv")
    if training_metrics and "episode_losses" in training_metrics:
        save_loss_csv(training_metrics["episode_losses"], generation_dir / "episode_losses.csv")
    save_reward_breakdown_csv(reward_breakdowns, generation_dir / "generated_reward_breakdown.csv")

    payload = {
        "generation_id": generation_id,
        "image_path": str(image_path),
        "raw_label": raw_label,
        "music_mode": mode,
        "music_scores": music_scores,
        "classifier": classifier_result,
        "episodes": episodes,
        "melody_length": melody_length,
        "seed": seed,
        "reward_weight_overrides": reward_overrides,
        "reward_weights": env.reward_weights,
        "training_metrics": {
            key: value for key, value in (training_metrics or {}).items()
            if key != "episode_losses"
        },
        "actions": actions,
        "pitch_actions": pitch_actions,
        "duration_actions": duration_actions,
        "velocity_actions": velocity_actions,
        "events": events,
        "melody": melody,
        "durations": durations,
        "velocities": velocities,
        "note_names": note_names,
        "preview_events": _build_preview_events(melody, durations, velocities),
        "reward_summary": reward_summary,
        "metrics": metrics,
        "selection_score": selection_score,
        "midi_file": str(midi_filename),
        "midi_url": f"/generated/{generation_id}/{midi_filename.name}",
        "sample_dir": str(generation_dir),
        "feedback_log_jsonl": str(FEEDBACK_LOG_JSONL),
        "feedback_log_csv": str(FEEDBACK_LOG_CSV),
    }

    with open(generation_dir / "generation.json", "w", encoding="utf-8") as json_file:
        json.dump(payload, json_file, ensure_ascii=False, indent=2)

    return payload


def load_generation(generation_id):
    generation_path = GENERATION_ROOT / generation_id / "generation.json"
    if not generation_path.exists():
        raise FileNotFoundError(f"Unknown generation: {generation_id}")
    with open(generation_path, "r", encoding="utf-8") as json_file:
        return json.load(json_file)


class FeedbackGUIHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        return

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            _text_response(self, INDEX_HTML)
            return
        if self.path.startswith("/generated/"):
            self._serve_generated_file()
            return
        _text_response(self, "Not found", "text/plain; charset=utf-8", status=404)

    def do_POST(self):
        if self.path == "/api/generate":
            self._handle_generate()
            return
        if self.path == "/api/feedback":
            self._handle_feedback()
            return
        _json_response(self, {"error": "Not found"}, status=404)

    def _handle_generate(self):
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length)
            form = _parse_multipart_form(self.headers, body)
            image_field = form.get("image")
            if image_field is None or not image_field.get("filename"):
                _json_response(self, {"error": "image is required"}, status=400)
                return

            image_path = _save_upload(image_field)
            episodes = _safe_int(form.get("episodes", {}).get("value"), 800, 1, 20000)
            melody_length = _safe_int(form.get("melody_length", {}).get("value"), 32, 8, 96)
            seed = _safe_int(form.get("seed", {}).get("value"), 42, 0, 999999999)
            payload = generate_from_image(image_path, episodes, melody_length, seed)
            _json_response(self, payload)
        except Exception as error:
            _json_response(self, {"error": str(error)}, status=500)

    def _handle_feedback(self):
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length).decode("utf-8")
            data = json.loads(body)
            generation = load_generation(data["generation_id"])
            feedback_record = append_feedback({
                "generation_id": generation["generation_id"],
                "music_mode": generation["music_mode"],
                "raw_label": generation["raw_label"],
                "image_path": generation["image_path"],
                "midi_file": generation["midi_file"],
                "selection_score": generation["selection_score"],
                "metrics": generation["metrics"],
                "reward_summary": generation["reward_summary"],
                "classifier": generation["classifier"],
                "feedback": data["feedback"],
                "comment": data.get("comment", ""),
            })
            _json_response(self, feedback_record)
        except Exception as error:
            _json_response(self, {"error": str(error)}, status=500)

    def _serve_generated_file(self):
        parts = [unquote(part) for part in self.path.split("/") if part]
        if len(parts) != 3:
            _text_response(self, "Not found", "text/plain; charset=utf-8", status=404)
            return
        _, generation_id, filename = parts
        file_path = GENERATION_ROOT / generation_id / filename
        if not file_path.exists() or not file_path.is_file():
            _text_response(self, "Not found", "text/plain; charset=utf-8", status=404)
            return
        content_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
        body = file_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def parse_args():
    parser = argparse.ArgumentParser(description="Run the EmotionRL feedback GUI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    return parser.parse_args()


def main():
    args = parse_args()
    GUI_ROOT.mkdir(parents=True, exist_ok=True)
    UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    GENERATION_ROOT.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((args.host, args.port), FeedbackGUIHandler)
    print(f"EmotionRL feedback GUI: http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
