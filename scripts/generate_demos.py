#!/usr/bin/env python3
"""Build reproducible demo assets from recorded OSWorld trajectories.

First import: python scripts/generate_demos.py --source ../iclr-dart-page_1129/trajectories
Regenerate the videos from checked-in assets: python scripts/generate_demos.py
Requires Pillow and imageio-ffmpeg. No model inference or simulated actions.
"""
import argparse
from functools import lru_cache
import json
import math
from pathlib import Path
import re
import subprocess

from PIL import Image, ImageDraw, ImageFont
import imageio_ffmpeg

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "public" / "demos"
SPECS = [
    ("vibrance", "554785e9", "GIMP", "Make the colors pop", "Enhance a photo with richer, more vibrant colors.", "01"),
    ("slides", "5c1a6c3d", "LibreOffice Impress", "Style a presentation", "Refine a slide title with bold type, size, and an underline.", "02"),
    ("resize", "d16c99dc", "GIMP", "Resize with precision", "Resize a husky photo to 512 px high, keeping its proportions.", "03"),
]


def import_sources(source):
    index_path = source / "trajectories.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else None
    demos = []
    for slug, prefix, app, title, description, number in SPECS:
        if index is not None:
            record = next(x for x in index if x["id"].startswith(prefix))
            source_folder = source / record["id"]
            messages = json.loads((source / (record["id"] + ".json")).read_text())
        else:
            source_folder = next(source.glob(prefix + "*"))
            messages = json.loads((source_folder / "final_messages.json").read_text())
            instruction = next(
                part["text"].split("## User Instruction", 1)[1].strip()
                for message in messages if message["role"] == "user"
                for part in message["content"]
                if "## User Instruction" in part.get("text", "")
            )
            record = {"id": source_folder.name, "task_id": source_folder.name.split("_")[0],
                      "instruction": instruction, "reward": (source_folder / "reward.txt").read_text().strip(),
                      "os": "Ubuntu 22.04"}
        if float(record["reward"]) != 1.0:
            raise ValueError(f"Expected a successful archive example: {record['id']}")
        folder = OUT / slug
        folder.mkdir(parents=True, exist_ok=True)
        steps, screenshot = [], None
        for message in messages:
            content = message.get("content", [])
            if not isinstance(content, list):
                continue
            for part in content:
                if part.get("type") == "image_url":
                    screenshot = part["image_url"]
            if message["role"] != "assistant" or not screenshot:
                continue
            text = "\n".join(p.get("text", "") for p in content)
            thought, _, action = text.partition("Action:")
            action = action.strip()
            original = Image.open(source_folder / screenshot).convert("RGB")
            width, height = original.size
            original.thumbnail((1440, 810), Image.Resampling.LANCZOS)
            dest = f"{len(steps):02d}.webp"
            original.save(folder / dest, "WEBP", quality=87)
            points = [[round(int(x) / width * 100, 3), round(int(y) / height * 100, 3)]
                      for x, y in re.findall(r"<\|box_start\|>\((\d+),\s*(\d+)\)", action)]
            steps.append({"image": f"demos/{slug}/{dest}", "thought": thought.removeprefix("Thought:").strip(),
                          "action": action, "point": points[0] if points else None,
                          "endPoint": points[1] if len(points) > 1 else None})
        demos.append({"id": slug, "number": number, "app": app, "title": title,
                      "description": description, "instruction": record["instruction"],
                      "taskId": record["task_id"], "sourceId": record["id"],
                      "reward": float(record["reward"]), "os": record["os"],
                      "secondsPerStep": 2, "video": f"demos/{slug}.mp4",
                      "poster": f"demos/{slug}-poster.webp", "preview": steps[-1]["image"],
                      "posterStep": len(steps) - 1, "steps": steps})
    (OUT / "manifest.json").write_text(json.dumps(demos, indent=2) + "\n")
    return demos


@lru_cache(maxsize=16)
def font(size, bold=False):
    candidates = [
        Path(f"/System/Library/Fonts/Supplemental/Arial{' Bold' if bold else ''}.ttf"),
        Path(f"/usr/share/fonts/truetype/dejavu/DejaVuSans{'-Bold' if bold else ''}.ttf"),
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default(size=size)


def short_action(action):
    name = action.split("(")[0]
    labels = {"click": "Click", "left_double": "Double-click", "right_single": "Right-click", "drag": "Drag", "hotkey": "Keyboard shortcut", "type": "Type", "scroll": "Scroll", "wait": "Wait for the application", "finished": "Task complete"}
    label = labels.get(name, name)
    if name in ("hotkey", "type"):
        match = re.search(r"(?:key|content)='(.*?)'", action)
        if match:
            label += "  /  " + match[1].replace("\\n", " Enter")
    return label[:82]


@lru_cache(maxsize=3)
def screenshot_frame(path):
    with Image.open(ROOT / "public" / path) as image:
        return image.convert("RGB").resize((1200, 675), Image.Resampling.LANCZOS)


def action_badge(draw, text, x, y):
    badge_font = font(18, True)
    width = draw.textbbox((0, 0), text, font=badge_font)[2] + 28
    x = max(48, min(x, 1232 - width))
    y = max(88, min(y, 707))
    draw.rounded_rectangle((x, y, x + width, y + 38), radius=9,
                           fill="#0b1712", outline="#c5f277", width=2)
    draw.text((x + 14, y + 9), text, font=badge_font, fill="#eaffd6")


def draw_action_marker(canvas, step, index, phase):
    """Annotate recorded coordinates; only the annotation moves, not the screenshot."""
    overlay = Image.new("RGBA", canvas.size)
    draw = ImageDraw.Draw(overlay)
    point = step["point"]
    if point:
        x, y = 40 + point[0] * 12, 80 + point[1] * 6.75
        end = step.get("endPoint")
        if end:
            ex, ey = 40 + end[0] * 12, 80 + end[1] * 6.75
            draw.line((x, y, ex, ey), fill="#08110d", width=11)
            draw.line((x, y, ex, ey), fill="#c5f277", width=5)
            angle = math.atan2(ey - y, ex - x)
            arrow = [(ex, ey)] + [(ex - 18 * math.cos(angle + offset), ey - 18 * math.sin(angle + offset)) for offset in (-0.5, 0.5)]
            draw.line(arrow[1:] + [arrow[0], arrow[1]], fill="#08110d", width=8)
            draw.polygon(arrow, fill="#c5f277")
            draw.ellipse((ex-12, ey-12, ex+12, ey+12), outline="#08110d", width=6)
            draw.ellipse((ex-12, ey-12, ex+12, ey+12), outline="#ffffff", width=3)
            travel = (1 - math.cos(phase * math.pi)) / 2
            mx, my = x + (ex-x) * travel, y + (ey-y) * travel
            draw.ellipse((mx-6, my-6, mx+6, my+6), fill="#ffffff", outline="#08110d", width=2)
        pulse = (phase * 2) % 1
        r = 29 + 15 * pulse
        draw.ellipse((x-32, y-32, x+32, y+32), fill=(196, 243, 120, 45))
        draw.ellipse((x-r, y-r, x+r, y+r), outline=(196, 243, 120, int(220 * (1-pulse))), width=4)
        draw.ellipse((x-23, y-23, x+23, y+23), outline="#061009", width=10)
        draw.ellipse((x-23, y-23, x+23, y+23), outline="#c5f277", width=5)
        draw.ellipse((x-6, y-6, x+6, y+6), fill="#ffffff", outline="#08110d", width=2)
        label = f"{index+1:02d}  {short_action(step['action']).upper()}"
        label_width = draw.textbbox((0, 0), label, font=font(18, True))[2] + 28
        label_x = x + 42 if x + 42 + label_width < 1232 else x - 42 - label_width
        action_badge(draw, label, label_x if not end else x - label_width / 2,
                     y - 19 if not end else y - 70)
    else:
        label = short_action(step["action"]).upper()
        if step["action"].startswith("hotkey"):
            key = re.search(r"key='([^']+)'", step["action"])
            if key:
                label = "KEYBOARD   " + " + ".join(key[1].upper().split())
        width = draw.textbbox((0, 0), label, font=font(18, True))[2] + 28
        action_badge(draw, label, 1216 - width, 699)
    canvas.paste(Image.alpha_composite(canvas.convert("RGBA"), overlay).convert("RGB"))


def render_frame(demo, index, phase=0):
    step = demo["steps"][index]
    canvas = Image.new("RGB", (1280, 860), "#0b1115")
    draw = ImageDraw.Draw(canvas)
    draw.text((38, 24), "DART", font=font(26, True), fill="#c5f277")
    draw.text((118, 28), "/ GUI", font=font(19), fill="#c2ccd0")
    draw.text((320, 29), demo["app"] + "  /  " + demo["title"], font=font(18), fill="#e7ece9")
    draw.text((1045, 31), "TRAJECTORY REPLAY", font=font(12, True), fill="#94a3a7")
    screen = screenshot_frame(step["image"])
    canvas.paste(screen, (40, 80))
    draw.rectangle((39, 79, 1240, 755), outline="#344448", width=1)
    draw_action_marker(canvas, step, index, phase)
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle((40, 769, 140, 805), radius=6, fill="#c5f277")
    draw.text((51, 776), f"{index+1:02d} / {len(demo['steps']):02d}", font=font(20, True), fill="#10180d")
    draw.text((160, 776), short_action(step["action"]), font=font(22, True), fill="#edf2eb")
    draw.text((40, 812), "Recorded OSWorld screenshots  /  2 seconds per step  /  Original timing condensed", font=font(13), fill="#98aaa9")
    draw.text((1020, 812), "DART-GUI-7B", font=font(13, True), fill="#c5f277")
    draw.rectangle((0, 854, 1280, 859), fill="#273333")
    draw.rectangle((0, 854, int(1280 * (index + phase) / len(demo["steps"])), 859), fill="#c5f277")
    return canvas


def encode(demo):
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-vcodec", "rawvideo", "-pix_fmt", "rgb24", "-s", "1280x860", "-r", "12", "-i", "-", "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(OUT / f"{demo['id']}.mp4")]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for index in range(len(demo["steps"])):
        # Keep screenshot content unchanged; animate only the recorded action marker.
        for tick in range(24):
            frame = render_frame(demo, index, tick/23)
            proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError("FFmpeg failed")
    render_frame(demo, demo.get("posterStep", len(demo["steps"])-1), 1).save(OUT / f"{demo['id']}-poster.webp", quality=90)
    print(f"Generated {demo['id']}.mp4 ({len(demo['steps'])*2}s)", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    demos = import_sources(args.source.resolve()) if args.source else json.loads((OUT / "manifest.json").read_text())
    for demo in demos:
        encode(demo)
    # Keep the same full trajectory and 2-second pacing in the portable GIF.
    frames = [render_frame(demos[0], i, tick/4).resize((896, 602), Image.Resampling.LANCZOS)
              for i in range(len(demos[0]["steps"])) for tick in range(4)]
    frames[0].save(OUT / "dart-gui-demo.gif", save_all=True, append_images=frames[1:], duration=500, loop=0, optimize=True)
    print("Generated dart-gui-demo.gif", flush=True)


if __name__ == "__main__":
    main()
