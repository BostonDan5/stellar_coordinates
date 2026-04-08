import argparse
import json
import math
import os
import tkinter as tk
from tkinter import font as tkfont
from tkinter import messagebox, ttk

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

STAR_FILE = "data/hyg_v42.csv"  # From https://www.astronexus.com/projects/hyg
CONSTELLATIONS_FILE = "data/constellations.json"  # From https://github.com/pirtleshell/constellations

ZODIAC_ABBREVIATIONS = ["Ari", "Tau", "Gem", "Cnc", "Leo", "Vir", "Lib", "Sco", "Sgr", "Cap", "Aqr", "Psc"]

OUTPUT_DIR = "output"

CANVAS_SIZE = 800
MARGIN = 15
TITLE_HEIGHT = 35  # vertical space reserved for the constellation name
CROSSHAIR_ARM = 5
MAX_STARS = 100  # maximum stars shown per constellation


def load_data():
    hyg_df = pd.read_csv(STAR_FILE)
    hyg_df = hyg_df[1:]  # Remove the first row, which is an entry for Sol

    with open(CONSTELLATIONS_FILE) as f:
        con_df = pd.DataFrame(json.load(f))

    return hyg_df, con_df


def build_constellation_list(con_df):
    """Return (zodiac, others) as lists of (abbr, name), others sorted alphabetically."""
    abbr_to_name = dict(zip(con_df["abbr"], con_df["name"]))
    zodiac = [(a, abbr_to_name[a]) for a in ZODIAC_ABBREVIATIONS if a in abbr_to_name]
    zodiac_set = set(ZODIAC_ABBREVIATIONS)
    others = sorted(
        [(a, n) for a, n in abbr_to_name.items() if a not in zodiac_set],
        key=lambda x: x[1],
    )
    return zodiac, others


def project(ra, dec):
    """Equirectangular projection of RA (hours) / dec (degrees) to (x, y).

    RA increases east; on a sky map east is to the left, so x is negated.
    Returns arrays in degrees-equivalent units, y increasing upward.
    """
    ra = ra.copy()
    if ra.max() - ra.min() > 12.0:
        ra[ra < 12.0] += 24.0

    ra_deg = ra * 15.0  # hours → degrees
    dec_rad = np.radians(dec.mean())
    x = -ra_deg * math.cos(dec_rad)  # negated so east is left
    y = dec
    return x, y


def normalize(x, y):
    """Scale (x, y) arrays to fit the canvas, preserving aspect ratio."""
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    x_span = x_max - x_min or 1.0
    y_span = y_max - y_min or 1.0

    avail_w = CANVAS_SIZE - 2 * MARGIN
    avail_h = CANVAS_SIZE - 2 * MARGIN - TITLE_HEIGHT

    scale = min(avail_w / x_span, avail_h / y_span)

    cx = CANVAS_SIZE / 2
    cy = TITLE_HEIGHT + MARGIN + avail_h / 2

    x_mid = (x_min + x_max) / 2
    y_mid = (y_min + y_max) / 2

    sx = cx + (x - x_mid) * scale
    sy = cy - (y - y_mid) * scale  # canvas y increases downward
    return sx, sy


def draw_crosshair(canvas, x, y, arm=CROSSHAIR_ARM):
    """Draw an 8-arm crosshair (4 lines at 45° intervals)."""
    for angle_deg in (0, 45, 90, 135):
        rad = math.radians(angle_deg)
        dx = math.cos(rad) * arm
        dy = math.sin(rad) * arm
        canvas.create_line(x - dx, y - dy, x + dx, y + dy, fill="black")


def star_label(row):
    """Build the list label for a star row: bf field + (proper name) if present."""
    bf = str(row["bf"]).strip() if pd.notna(row["bf"]) else ""
    proper = str(row["proper"]).strip() if pd.notna(row["proper"]) else ""
    if bf and proper:
        return f"{bf} ({proper})"
    if bf:
        return bf
    if proper:
        return f"({proper})"
    return "(unnamed)"


class App(tk.Tk):
    def __init__(self, hyg_df, con_df, combined=False):
        super().__init__()
        self.title("Stellar Coordinates")
        self.resizable(False, False)

        self.hyg_df = hyg_df
        self.con_df = con_df
        self.combined = combined
        self._mode = "constellations"    # "constellations" | "stars"
        self._current_stars = None      # DataFrame of stars for current constellation
        self._current_con_name = None   # name of the currently displayed constellation

        self._build_ui()

    def _build_ui(self):
        # ── Left panel ────────────────────────────────────────────────
        self.left = tk.Frame(self, width=220)
        self.left.pack(side=tk.LEFT, fill=tk.Y, padx=(8, 0), pady=8)
        self.left.pack_propagate(False)

        # Button area — packed at top, only visible in star mode
        self.btn_frame = tk.Frame(self.left)
        # not packed yet

        self.back_btn = tk.Button(
            self.btn_frame, text="Constellations",
            font=("TkDefaultFont", 9),
            command=self._show_constellations,
            anchor="w",
        )
        self.back_btn.pack(fill=tk.X, pady=(0, 2))

        self.export_btn = tk.Button(
            self.btn_frame, text="Export PNG",
            font=("TkDefaultFont", 9),
            command=self._export_png,
            anchor="w",
        )
        self.export_btn.pack(fill=tk.X)

        # Title label — always visible
        self.header_label = tk.Label(
            self.left, text="Constellations",
            font=("TkDefaultFont", 11, "bold"),
            anchor="w",
        )
        self.header_label.pack(fill=tk.X, pady=(4, 4))

        list_frame = tk.Frame(self.left)
        list_frame.pack(fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL)
        self.listbox = tk.Listbox(
            list_frame,
            yscrollcommand=scrollbar.set,
            selectmode=tk.MULTIPLE,
            activestyle="none",
            font=("TkDefaultFont", 10),
        )
        scrollbar.config(command=self.listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._populate_constellation_list()
        self.listbox.bind("<Button-1>", self._on_listbox_click)

        # ── Right canvas ──────────────────────────────────────────────
        right = tk.Frame(self)
        right.pack(side=tk.LEFT, padx=8, pady=8)

        self.canvas = tk.Canvas(right, width=CANVAS_SIZE, height=CANVAS_SIZE, bg="white")
        self.canvas.pack()
        self._label_font = tkfont.Font(family="TkDefaultFont", size=8)

    # ── Constellation mode ────────────────────────────────────────────

    def _populate_constellation_list(self):
        zodiac, others = build_constellation_list(self.con_df)
        self._items = []
        self.listbox.delete(0, tk.END)

        for abbr, name in zodiac:
            self.listbox.insert(tk.END, f"  {name}")
            self._items.append(("con", abbr, name))

        self.listbox.insert(tk.END, "─" * 22)
        self._items.append(None)

        for abbr, name in others:
            self.listbox.insert(tk.END, f"  {name}")
            self._items.append(("con", abbr, name))

    def _show_constellations(self):
        self._mode = "constellations"
        self._current_stars = None
        self.header_label.config(text="Constellations")
        self.btn_frame.pack_forget()
        self._populate_constellation_list()
        self.canvas.delete("all")

    # ── Star mode ─────────────────────────────────────────────────────

    def _show_stars(self, abbr, name):
        self._mode = "stars"
        self._current_con_name = name
        stars = (
            self.hyg_df[self.hyg_df["con"] == abbr]
            .copy()
            .assign(mag=lambda df: pd.to_numeric(df["mag"], errors="coerce"))
            .sort_values("mag")
            .head(MAX_STARS)
        )
        self._current_stars = stars

        self.btn_frame.pack(fill=tk.X, pady=(0, 2), before=self.header_label)
        self.header_label.config(text=f"{name} — top {MAX_STARS}")

        self._items = []
        self.listbox.delete(0, tk.END)
        for _, row in stars.iterrows():
            self.listbox.insert(tk.END, f"  {star_label(row)}")
            self._items.append(("star", row))

        # Auto-select through the last star that has a proper name
        proper_indices = [
            i for i, (_, row) in enumerate(self._items)
            if pd.notna(row["proper"]) and str(row["proper"]).strip()
        ]
        initial_idx = proper_indices[-1] if proper_indices else 0
        self._apply_star_selection(initial_idx)

    def _draw_stars(self, stars, name):
        self.canvas.delete("all")

        self.canvas.create_text(
            CANVAS_SIZE // 2, TITLE_HEIGHT // 2,
            text=name, fill="black",
            font=("TkDefaultFont", 14, "bold"),
        )

        if stars.empty:
            return

        ra = stars["ra"].astype(float).values
        dec = stars["dec"].astype(float).values

        x, y = project(ra, dec)
        sx, sy = normalize(x, y)

        label_gap = CROSSHAIR_ARM + 3
        label_h = self._label_font.metrics("linespace")

        for (cx, cy), (_, row) in zip(zip(sx, sy), stars.iterrows()):
            draw_crosshair(self.canvas, cx, cy)
            text = star_label(row)
            text_w = self._label_font.measure(text)
            lx = cx + label_gap
            ly = cy
            anchor = "w"
            if lx + text_w > CANVAS_SIZE - MARGIN:
                lx = min(cx, CANVAS_SIZE - MARGIN - text_w)
                ly = cy + label_gap + label_h / 2
                anchor = "w"
            self.canvas.create_text(
                lx, ly,
                text=text,
                anchor=anchor,
                fill="blue",
                font=self._label_font,
            )

    def _render_image(self, stars, name, draw_crosshairs=True, draw_labels=True):
        """Render stars and/or labels into a new PIL Image and return it."""
        img = Image.new("RGB", (CANVAS_SIZE, CANVAS_SIZE), "white")
        draw = ImageDraw.Draw(img)

        try:
            title_font = ImageFont.truetype("arial.ttf", 14)
            label_font = ImageFont.truetype("arial.ttf", 8)
        except OSError:
            title_font = ImageFont.load_default()
            label_font = ImageFont.load_default()

        # Title
        bbox = draw.textbbox((0, 0), name, font=title_font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text((CANVAS_SIZE // 2 - tw // 2, TITLE_HEIGHT // 2 - th // 2),
                  name, fill="black", font=title_font)

        if not stars.empty:
            ra = stars["ra"].astype(float).values
            dec = stars["dec"].astype(float).values
            x, y = project(ra, dec)
            sx, sy = normalize(x, y)

            label_gap = CROSSHAIR_ARM + 3

            for (cx, cy), (_, row) in zip(zip(sx, sy), stars.iterrows()):
                if draw_crosshairs:
                    for angle_deg in (0, 45, 90, 135):
                        rad = math.radians(angle_deg)
                        dx = math.cos(rad) * CROSSHAIR_ARM
                        dy = math.sin(rad) * CROSSHAIR_ARM
                        draw.line([(cx - dx, cy - dy), (cx + dx, cy + dy)], fill="black")

                if draw_labels:
                    text = star_label(row)
                    tbbox = draw.textbbox((0, 0), text, font=label_font)
                    text_w = tbbox[2] - tbbox[0]
                    text_h = tbbox[3] - tbbox[1]
                    lx = cx + label_gap
                    ly = cy - text_h / 2
                    if lx + text_w > CANVAS_SIZE - MARGIN:
                        lx = min(cx, CANVAS_SIZE - MARGIN - text_w)
                        ly = cy + label_gap
                    draw.text((lx, ly), text, fill="blue", font=label_font)

        return img

    def _export_png(self):
        sel = self.listbox.curselection()
        if not sel:
            return
        stars = self._current_stars.iloc[: sel[-1] + 1]
        name = self._current_con_name
        safe_name = name.replace(" ", "_")

        os.makedirs(OUTPUT_DIR, exist_ok=True)

        if self.combined:
            img = self._render_image(stars, name, draw_crosshairs=True, draw_labels=True)
            path = os.path.join(OUTPUT_DIR, f"{safe_name}.png")
            img.save(path)
            messagebox.showinfo("Exported", f"Saved {path}")
        else:
            stars_img = self._render_image(stars, name, draw_crosshairs=True, draw_labels=False)
            names_img = self._render_image(stars, name, draw_crosshairs=False, draw_labels=True)
            stars_path = os.path.join(OUTPUT_DIR, f"{safe_name}_stars.png")
            names_path = os.path.join(OUTPUT_DIR, f"{safe_name}_names.png")
            stars_img.save(stars_path)
            names_img.save(names_path)
            messagebox.showinfo("Exported", f"Saved:\n{stars_path}\n{names_path}")

    def _apply_star_selection(self, idx):
        """Select rows 0..idx in the listbox and redraw that subset on the canvas."""
        self.listbox.selection_clear(0, tk.END)
        self.listbox.selection_set(0, idx)
        self.listbox.see(idx)
        selected_stars = self._current_stars.iloc[: idx + 1]
        self._draw_stars(selected_stars, self._current_con_name)

    # ── Click handler ─────────────────────────────────────────────────

    def _on_listbox_click(self, event):
        idx = self.listbox.nearest(event.y)
        if idx < 0 or idx >= len(self._items):
            return "break"
        item = self._items[idx]

        if self._mode == "constellations":
            if item is None:
                return "break"
            _, abbr, name = item
            self.listbox.selection_clear(0, tk.END)
            self.listbox.selection_set(idx)
            self._show_stars(abbr, name)
        elif self._mode == "stars":
            self._apply_star_selection(idx)

        return "break"


def main():
    parser = argparse.ArgumentParser(description="Stellar Coordinates viewer")
    parser.add_argument("--combined", action="store_true",
                        help="Export stars and labels into a single PNG (default: separate files)")
    args = parser.parse_args()

    hyg_df, con_df = load_data()
    app = App(hyg_df, con_df, combined=args.combined)
    app.mainloop()


if __name__ == "__main__":
    main()
