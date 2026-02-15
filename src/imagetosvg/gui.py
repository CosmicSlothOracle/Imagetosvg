from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from .config import DetailPreset, PipelineConfig
from .pipeline import VectorizationPipeline

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD  # type: ignore
except ImportError:  # optional
    DND_FILES = None
    TkinterDnD = None


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Imagetosvg - Precision Vector Reconstruction")
        self.root.geometry("1100x700")

        self.input_path: Path | None = None
        self.preview_original: ImageTk.PhotoImage | None = None
        self.preview_result: ImageTk.PhotoImage | None = None

        self.detail = tk.StringVar(value=DetailPreset.HIGH.value)
        self._build()

    def _build(self) -> None:
        controls = ttk.Frame(self.root, padding=10)
        controls.pack(fill=tk.X)

        ttk.Button(controls, text="Select Image", command=self._pick_file).pack(side=tk.LEFT)
        ttk.Button(controls, text="Run", command=self._run).pack(side=tk.LEFT, padx=8)

        ttk.Label(controls, text="Detail").pack(side=tk.LEFT, padx=(20, 4))
        ttk.Combobox(
            controls,
            textvariable=self.detail,
            values=[d.value for d in DetailPreset],
            state="readonly",
            width=8,
        ).pack(side=tk.LEFT)

        self.status = ttk.Label(controls, text="Drop an image or click Select Image")
        self.status.pack(side=tk.LEFT, padx=16)

        pane = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        pane.pack(fill=tk.BOTH, expand=True)

        self.orig_label = ttk.Label(pane)
        self.svg_label = ttk.Label(pane)
        pane.add(self.orig_label, weight=1)
        pane.add(self.svg_label, weight=1)

        if TkinterDnD and DND_FILES:
            self.root.drop_target_register(DND_FILES)
            self.root.dnd_bind("<<Drop>>", self._drop)

    def _pick_file(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg *.webp")])
        if path:
            self._load_input(Path(path))

    def _drop(self, event: tk.Event) -> None:
        data = str(event.data).strip("{}")
        if data:
            self._load_input(Path(data))

    def _load_input(self, path: Path) -> None:
        self.input_path = path
        self.status.configure(text=f"Selected: {path.name}")
        self.preview_original = self._to_photo(path)
        self.orig_label.configure(image=self.preview_original)

    def _to_photo(self, path: Path) -> ImageTk.PhotoImage:
        image = Image.open(path)
        image.thumbnail((520, 640))
        return ImageTk.PhotoImage(image)

    def _run(self) -> None:
        if not self.input_path:
            messagebox.showwarning("Missing input", "Please select an image first.")
            return

        self.status.configure(text="Processing...")
        self.root.update_idletasks()

        try:
            config = PipelineConfig(detail=DetailPreset(self.detail.get()), output_dir=Path("output"))
            pipeline = VectorizationPipeline(config)
            result = pipeline.run(self.input_path)[0]
        except Exception as exc:
            messagebox.showerror("Vectorization failed", str(exc))
            self.status.configure(text="Failed")
            return

        png_preview = self._render_svg_preview(result.output)
        self.preview_result = ImageTk.PhotoImage(png_preview)
        self.svg_label.configure(image=self.preview_result)

        if result.report:
            self.status.configure(text=f"Done: SSIM={result.report.ssim:.4f} | MSE={result.report.mse:.2f}")
        else:
            self.status.configure(text="Done: validation skipped (CairoSVG unavailable)")

    def _render_svg_preview(self, path: Path) -> Image.Image:
        try:
            import cairosvg
        except ImportError:
            messagebox.showwarning("Preview unavailable", "Install cairosvg for SVG preview rendering.")
            if self.input_path:
                return self._to_pil(self.input_path)
            return Image.new("RGB", (520, 640), "#222")

        png_bytes = cairosvg.svg2png(url=str(path))
        arr = cv2.imdecode(np.frombuffer(png_bytes, np.uint8), cv2.IMREAD_COLOR)
        if arr is None:
            return Image.new("RGB", (520, 640), "#222")
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(arr).resize((520, 640))

    def _to_pil(self, path: Path) -> Image.Image:
        image = Image.open(path)
        image.thumbnail((520, 640))
        return image


def launch_gui() -> None:
    if TkinterDnD:
        root = TkinterDnD.Tk()
    else:
        root = tk.Tk()
    App(root)
    root.mainloop()
