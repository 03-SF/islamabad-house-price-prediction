"""
Property Price Prediction System — GUI + CLI
-------------------------------------------
This file provides a polished Tkinter GUI (left form / right panel) and a
CLI fallback. It loads `best_model.pkl` and `encoders.pkl` produced by the
training pipeline and exposes an easy-to-use interface for predicting prices.

Run:
    python prediction_system.py

Files required in the same folder:
 - best_model.pkl
 - encoders.pkl
 - feature_names.txt (optional)
 - preprocessed_data.csv (optional, used for market snapshot)

"""

from __future__ import annotations
import os
import pickle
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np

ROOT = os.path.dirname(__file__)


def area_to_marla(area_str: str | None) -> float | None:
    if area_str is None:
        return None
    s = str(area_str).strip().lower()
    if not s or s in ("unknown", "nan"):
        return None
    try:
        if "kanal" in s:
            return float(s.replace("kanal", "").strip()) * 20.0
        if "marla" in s:
            return float(s.replace("marla", "").strip())
        return float(s)
    except Exception:
        return None


def format_pkr(x: float) -> str:
    try:
        return f"PKR {int(round(x)):,}"
    except Exception:
        return str(x)


def load_artifacts():
    model_p = os.path.join(ROOT, "best_model.pkl")
    enc_p = os.path.join(ROOT, "encoders.pkl")
    feat_p = os.path.join(ROOT, "feature_names.txt")

    if not os.path.exists(model_p):
        raise FileNotFoundError("best_model.pkl not found — run model_training.py first")
    if not os.path.exists(enc_p):
        raise FileNotFoundError("encoders.pkl not found — run data_preprocessing.py first")

    with open(model_p, "rb") as f:
        model = pickle.load(f)
    with open(enc_p, "rb") as f:
        enc = pickle.load(f)

    feature_names = None
    if os.path.exists(feat_p):
        with open(feat_p, "r") as f:
            feature_names = [ln.strip() for ln in f if ln.strip()]

    return model, enc, feature_names


def make_feature_vector(inputs: dict, encoders: dict, feature_names: list | None):
    # Build vector in same order used in preprocessing
    x = []

    area_m = area_to_marla(inputs.get("Area"))
    if area_m is None:
        area_m = encoders.get("median_area_marla", 5.0)
    x.append(float(area_m))

    x.append(float(inputs.get("Bedrooms", 0)))
    x.append(float(inputs.get("Bathrooms", 0)))

    # location
    le_loc = encoders.get("Location")
    locs = encoders.get("Location_grouped_classes") or []
    loc = inputs.get("Location")
    if loc not in locs:
        loc = "Other"
    try:
        loc_e = int(le_loc.transform([loc])[0])
    except Exception:
        loc_e = 0
    x.append(float(loc_e))

    # property type
    le_type = encoders.get("Property Type")
    t = inputs.get("Property Type") or "House"
    try:
        te = int(le_type.transform([t])[0])
    except Exception:
        te = 0
    x.append(float(te))

    # property age
    try:
        by = int(inputs.get("Built Year"))
        age = max(0, 2025 - by)
    except Exception:
        age = 2025 - int(encoders.get("median_year", 2000))
    x.append(float(age))

    for k in ["Parking", "Servant Qtrs", "Store Rooms", "Kitchens"]:
        try:
            x.append(float(int(inputs.get(k, 0))))
        except Exception:
            x.append(0.0)

    for k in ["Drawing Room", "Dining Room", "Study Room", "Prayer Room", "Lounge/Sitting"]:
        x.append(1.0 if inputs.get(k) else 0.0)

    if feature_names is not None:
        if len(x) < len(feature_names):
            x += [0.0] * (len(feature_names) - len(x))
        elif len(x) > len(feature_names):
            x = x[: len(feature_names)]

    return np.array(x, dtype=float).reshape(1, -1)


class PredictorGUI:
    def __init__(self, model, encoders, feature_names):
        self.model = model
        self.enc = encoders
        self.feature_names = feature_names

        self.root = tk.Tk()
        self.root.title("Property Price Predictor")
        self.root.geometry("980x560")

        # Theme colors
        self._BG = '#eefaf0'       # page background (very light green)
        self._ACCENT = '#2f8f4a'   # primary green accent
        self._CARD = '#ffffff'     # white cards/inputs

        # Configure ttk style for a light-green theme
        self.style = ttk.Style()
        try:
            self.style.theme_use('clam')
        except Exception:
            pass

        self.style.configure('TFrame', background=self._BG)
        self.style.configure('TLabel', background=self._BG)
        self.style.configure('TLabelframe', background=self._BG)
        self.style.configure('TLabelframe.Label', font=('Segoe UI', 10, 'bold'), foreground=self._ACCENT, background=self._BG)
        self.style.configure('TEntry', fieldbackground=self._CARD)
        self.style.configure('TCombobox', fieldbackground=self._CARD)
        self.style.configure('TSpinbox', fieldbackground=self._CARD)

        self.style.configure('Accent.TButton', background=self._ACCENT, foreground='white', font=('Segoe UI', 11, 'bold'))
        self.style.map('Accent.TButton', background=[('active', '#267b3e')])

        self.style.configure('Price.TLabel', background='#f7fff7', foreground=self._ACCENT, font=('Segoe UI', 18, 'bold'))

        self.root.configure(bg=self._BG)

        self._build_ui()

    def _build_ui(self):
        main = ttk.Frame(self.root, padding=12)
        main.pack(fill="both", expand=True)

        left = ttk.Frame(main)
        left.pack(side="left", fill="both", expand=True, padx=(0, 12))

        right = ttk.Frame(main, width=320)
        right.pack(side="right", fill="y")

        # Property Basics
        basics = ttk.LabelFrame(left, text="Property Basics", padding=10)
        basics.pack(fill="x", pady=6)

        ttk.Label(basics, text="Area size (e.g. 5 Marla, 1 Kanal)").grid(column=0, row=0, sticky='w')
        self.area_var = tk.StringVar()
        ttk.Entry(basics, textvariable=self.area_var, width=30).grid(column=1, row=0, padx=6)

        ttk.Label(basics, text="Built year").grid(column=0, row=1, sticky='w', pady=6)
        years = [str(y) for y in range(1950, 2026)][::-1]
        self.year_var = tk.StringVar(value='2024')
        ttk.Combobox(basics, values=years, textvariable=self.year_var, width=12).grid(column=1, row=1, sticky='w')

        ttk.Label(basics, text="Property Type").grid(column=0, row=2, sticky='w', pady=6)
        types = list(self.enc.get('Property Type').classes_) if self.enc.get('Property Type', None) is not None else ["House"]
        self.type_var = tk.StringVar(value=types[0])
        ttk.Combobox(basics, values=types, textvariable=self.type_var, width=18).grid(column=1, row=2, sticky='w')

        ttk.Label(basics, text="Location").grid(column=0, row=3, sticky='w', pady=6)
        locs = self.enc.get('Location_grouped_classes') or ['Other']
        self.loc_var = tk.StringVar(value=locs[0])
        ttk.Combobox(basics, values=locs, textvariable=self.loc_var, width=36).grid(column=1, row=3, sticky='w')

        # Core Layout
        core = ttk.LabelFrame(left, text='Core Layout', padding=10)
        core.pack(fill='x', pady=6)

        ttk.Label(core, text='Bedrooms').grid(column=0, row=0, sticky='w')
        self.bed_var = tk.IntVar(value=3)
        bed_frame = ttk.Frame(core)
        bed_frame.grid(column=1, row=0, sticky='w')
        ttk.Button(bed_frame, text='-', width=2, command=lambda: self._spin(self.bed_var, -1)).pack(side='left')
        ttk.Label(bed_frame, textvariable=self.bed_var, width=3, anchor='center').pack(side='left')
        ttk.Button(bed_frame, text='+', width=2, command=lambda: self._spin(self.bed_var, 1)).pack(side='left')

        ttk.Label(core, text='Bathrooms').grid(column=2, row=0, sticky='w', padx=(12,0))
        self.bath_var = tk.IntVar(value=2)
        bath_frame = ttk.Frame(core)
        bath_frame.grid(column=3, row=0, sticky='w')
        ttk.Button(bath_frame, text='-', width=2, command=lambda: self._spin(self.bath_var, -1)).pack(side='left')
        ttk.Label(bath_frame, textvariable=self.bath_var, width=3, anchor='center').pack(side='left')
        ttk.Button(bath_frame, text='+', width=2, command=lambda: self._spin(self.bath_var, 1)).pack(side='left')

        # Detailed Amenities
        details = ttk.LabelFrame(left, text='Detailed Amenities', padding=10)
        details.pack(fill='x', pady=6)
        self.park_var = tk.IntVar(value=0)
        ttk.Label(details, text='Parking').grid(column=0, row=0)
        ttk.Spinbox(details, from_=0, to=10, textvariable=self.park_var, width=4).grid(column=1, row=0)

        self.serv_var = tk.IntVar(value=0)
        ttk.Label(details, text='Servant Quarters').grid(column=2, row=0)
        ttk.Spinbox(details, from_=0, to=10, textvariable=self.serv_var, width=4).grid(column=3, row=0)

        self.store_var = tk.IntVar(value=0)
        ttk.Label(details, text='Store Rooms').grid(column=0, row=1, pady=6)
        ttk.Spinbox(details, from_=0, to=10, textvariable=self.store_var, width=4).grid(column=1, row=1)

        self.kit_var = tk.IntVar(value=1)
        ttk.Label(details, text='Kitchens').grid(column=2, row=1)
        ttk.Spinbox(details, from_=0, to=10, textvariable=self.kit_var, width=4).grid(column=3, row=1)

        # Extra Rooms
        extra = ttk.LabelFrame(left, text='Extra Rooms', padding=10)
        extra.pack(fill='x', pady=6)
        self.draw_var = tk.IntVar(value=1)
        ttk.Checkbutton(extra, text='Drawing Room', variable=self.draw_var).grid(column=0, row=0, padx=6)
        self.dine_var = tk.IntVar(value=1)
        ttk.Checkbutton(extra, text='Dining Room', variable=self.dine_var).grid(column=1, row=0, padx=6)
        self.study_var = tk.IntVar(value=0)
        ttk.Checkbutton(extra, text='Study Room', variable=self.study_var).grid(column=2, row=0, padx=6)
        self.pray_var = tk.IntVar(value=0)
        ttk.Checkbutton(extra, text='Prayer Room', variable=self.pray_var).grid(column=3, row=0, padx=6)
        self.lounge_var = tk.IntVar(value=1)
        ttk.Checkbutton(extra, text='Lounge', variable=self.lounge_var).grid(column=4, row=0, padx=6)

        # Right panel
        right.pack_propagate(False)
        top_right = ttk.Frame(right, padding=12)
        top_right.pack(fill='both', expand=True)

        self.predict_btn = ttk.Button(top_right, text='Predict Price Now', command=self.on_predict, style='Accent.TButton')
        self.predict_btn.pack(fill='x', pady=(0,12))
        self.price_display = ttk.Label(top_right, text='—', style='Price.TLabel', anchor='center', padding=12)
        self.price_display.pack(fill='x', pady=(0,12))

        snap = ttk.LabelFrame(top_right, text='Market Snapshot', padding=10)
        snap.pack(fill='both', expand=True)

        self.regional_var = tk.StringVar(value='Regional Avg: —')
        ttk.Label(snap, textvariable=self.regional_var, font=('Segoe UI', 11)).pack(anchor='w')

        self.demand_var = tk.StringVar(value='Market demand: —')
        ttk.Label(snap, textvariable=self.demand_var, font=('Segoe UI', 10)).pack(anchor='w', pady=(6,0))

        self._load_market_snapshot()

    def _spin(self, var, delta):
        var.set(max(0, var.get() + delta))

    def _load_market_snapshot(self):
        try:
            import pandas as pd
            p = os.path.join(ROOT, 'preprocessed_data.csv')
            if os.path.exists(p):
                df = pd.read_csv(p)
                avg = int(df['Price (PKR)'].mean())
                self.regional_var.set(f'Regional Avg: PKR {avg:,}')
                d = min(10, round(len(df) / 50, 1))
                self.demand_var.set(f'Market demand (local): {d}/10')
        except Exception:
            pass

    def on_predict(self):
        raw = {
            'Area': self.area_var.get(),
            'Bedrooms': self.bed_var.get(),
            'Bathrooms': self.bath_var.get(),
            'Location': self.loc_var.get(),
            'Property Type': self.type_var.get(),
            'Built Year': self.year_var.get(),
            'Parking': self.park_var.get(),
            'Servant Qtrs': self.serv_var.get(),
            'Store Rooms': self.store_var.get(),
            'Kitchens': self.kit_var.get(),
            'Drawing Room': bool(self.draw_var.get()),
            'Dining Room': bool(self.dine_var.get()),
            'Study Room': bool(self.study_var.get()),
            'Prayer Room': bool(self.pray_var.get()),
            'Lounge/Sitting': bool(self.lounge_var.get()),
        }
        try:
            x = make_feature_vector(raw, self.enc, self.feature_names)
            pred = self.model.predict(x)[0]
            self.price_display.config(text=format_pkr(pred))
        except Exception as e:
            messagebox.showerror('Prediction error', str(e))


def run_gui(model, encoders, feature_names):
    app = PredictorGUI(model, encoders, feature_names)
    app.root.mainloop()


def run_cli(model, encoders, feature_names):
    print('Interactive CLI predictor — press Enter to accept default values')
    area = input('Area (e.g. "5 Marla" or "1 Kanal"): ').strip() or ''
    beds = input('Bedrooms [3]: ').strip() or '3'
    baths = input('Bathrooms [2]: ').strip() or '2'
    year = input('Built Year [2015]: ').strip() or '2015'
    locs = encoders.get('Location_grouped_classes') or ['Other']
    print('Choose Location (or type):')
    for i, l in enumerate(locs[:40], 1):
        print(f'  {i}. {l}')
    loc = input(f'Location [Other]: ').strip() or 'Other'
    types = list(encoders.get('Property Type').classes_) if encoders.get('Property Type', None) is not None else ['House']
    prop_type = input(f'Property Type [{types[0]}]: ').strip() or types[0]
    park = input('Parking [0]: ').strip() or '0'
    serv = input('Servant Qtrs [0]: ').strip() or '0'
    store = input('Store Rooms [0]: ').strip() or '0'
    kits = input('Kitchens [1]: ').strip() or '1'
    draw = input('Drawing Room? y/N: ').strip().lower().startswith('y')
    dine = input('Dining Room? y/N: ').strip().lower().startswith('y')
    study = input('Study Room? y/N: ').strip().lower().startswith('y')
    pray = input('Prayer Room? y/N: ').strip().lower().startswith('y')
    lounge = input('Lounge/Sitting? y/N: ').strip().lower().startswith('y')

    raw = {
        'Area': area,
        'Bedrooms': beds,
        'Bathrooms': baths,
        'Location': loc,
        'Property Type': prop_type,
        'Built Year': year,
        'Parking': park,
        'Servant Qtrs': serv,
        'Store Rooms': store,
        'Kitchens': kits,
        'Drawing Room': draw,
        'Dining Room': dine,
        'Study Room': study,
        'Prayer Room': pray,
        'Lounge/Sitting': lounge,
    }
    x = make_feature_vector(raw, encoders, feature_names)
    pred = model.predict(x)[0]
    print('\nPredicted price:', format_pkr(pred))


if __name__ == '__main__':
    model, encoders, feature_names = load_artifacts()
    try:
        run_gui(model, encoders, feature_names)
    except Exception as e:
        print('GUI failed — falling back to CLI. Error:', e)
        run_cli(model, encoders, feature_names)
