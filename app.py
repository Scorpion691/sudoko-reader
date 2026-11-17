"""Sudoku Reader - desktop launcher.  One button runs the whole project and opens the dashboard.

    python app.py            (or double-click SudokuReader.bat / the .exe built by build_exe.bat)

Steps behind the button: make sure the MNIST data and the trained network exist (downloads / trains them
the first time), run the integrated demo (NN -> AC-3 -> search -> CSP), optionally refresh the three
experiment tables, then build outputs/dashboard.html and open it in the browser.
"""
import contextlib
import io
import os
import queue
import sys
import threading
import traceback
import webbrowser

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
_site = os.path.join(sys.prefix, "Lib", "site-packages")  # some Python installs do not put this on the path
if os.path.isdir(_site) and _site not in sys.path:
    sys.path.append(_site)

import customtkinter as ctk  # noqa: E402

from eval.dashboard import STUDENT  # noqa: E402

DASH = os.path.join(ROOT, "outputs", "dashboard.html")
BG, CARD, BLUE, BLUE2, GREEN, RED, MUTED = "#060b28", "#0d1540", "#0075ff", "#3d8bff", "#01b574", "#f0605d", "#a0aec0"


def run_pipeline(seed, mis, exps, status=lambda s: None):
    """Everything the Run button does (also used by --selftest)."""
    status("Checking data and the trained network...")
    if not os.path.exists(os.path.join(ROOT, "data", "mnist19.npz")):
        status("Downloading and preparing MNIST (first run only)...")
        from data import make_data
        make_data.main(0, True)
    if not os.path.exists(os.path.join(ROOT, "nn", "model_mlp.npz")):
        status("Training the digit classifier (first run only, about 20 s)...")
        from nn import train
        train.main(10, 0)
    if exps:
        status("Running the 3 experiments (about 2 minutes)...")
        from eval import experiments as E
        env = E.setup()
        E.exp1(env, 5, 20)
        E.exp2(5, 10)
        E.exp3(env, 5, 20)
    status("Running the pipeline: NN -> AC-3 -> search -> CSP -> dashboard...")
    import run_demo
    run_demo.main(seed, mis, 30, open_browser=False)


class QWriter(io.TextIOBase):
    """Turns print() output of the worker thread into messages for the log box."""

    def __init__(self, q):
        self.q = q

    def write(self, s):
        if s:
            self.q.put(("log", s))
        return len(s or "")


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        ctk.set_appearance_mode("dark")
        self.title("Sudoku Reader")
        self.geometry("840x680")
        self.minsize(720, 600)
        self.configure(fg_color=BG)
        self.q = queue.Queue()
        self.running = False
        self._build()
        self.after(80, self._poll)

    # ------------------------------------------------------------------ UI
    def _build(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(3, weight=1)

        head = ctk.CTkFrame(self, fg_color=CARD, corner_radius=20)
        head.grid(row=0, column=0, sticky="ew", padx=22, pady=(22, 12))
        ctk.CTkLabel(head, text="SUDOKU READER", font=ctk.CTkFont(size=26, weight="bold")).pack(anchor="w", padx=22, pady=(18, 0))
        ctk.CTkLabel(head, text="%s  -  %s" % (STUDENT["code"], STUDENT["project"]), text_color="#cfe1ff",
                     font=ctk.CTkFont(size=14)).pack(anchor="w", padx=22)
        ctk.CTkLabel(head, text="%s   |   ID %s" % (STUDENT["name"], STUDENT["id"]), text_color=MUTED,
                     font=ctk.CTkFont(size=13)).pack(anchor="w", padx=22, pady=(2, 16))

        opt = ctk.CTkFrame(self, fg_color=CARD, corner_radius=20)
        opt.grid(row=1, column=0, sticky="ew", padx=22, pady=6)
        opt.grid_columnconfigure(4, weight=1)
        ctk.CTkLabel(opt, text="Seed", text_color=MUTED).grid(row=0, column=0, padx=(22, 8), pady=16)
        self.seed = ctk.CTkEntry(opt, width=70, justify="center")
        self.seed.insert(0, "3")
        self.seed.grid(row=0, column=1, pady=16)
        ctk.CTkLabel(opt, text="Injected misreads", text_color=MUTED).grid(row=0, column=2, padx=(24, 8))
        self.mis = ctk.CTkSegmentedButton(opt, values=["1", "2", "3", "4"], selected_color=BLUE, selected_hover_color=BLUE2,
                                          unselected_color="#0b1240")
        self.mis.set("2")
        self.mis.grid(row=0, column=3)
        self.open_after = ctk.CTkCheckBox(opt, text="Open the dashboard when done", fg_color=BLUE, hover_color=BLUE2)
        self.open_after.select()
        self.open_after.grid(row=1, column=0, columnspan=4, sticky="w", padx=22, pady=(0, 6))
        self.exps = ctk.CTkCheckBox(opt, text="Also re-run the 3 experiments (about 2 minutes)", fg_color=BLUE, hover_color=BLUE2)
        self.exps.grid(row=2, column=0, columnspan=4, sticky="w", padx=22, pady=(0, 16))

        act = ctk.CTkFrame(self, fg_color="transparent")
        act.grid(row=2, column=0, sticky="ew", padx=22, pady=8)
        act.grid_columnconfigure(0, weight=1)
        self.btn = ctk.CTkButton(act, text="▶   Run", height=58, corner_radius=16, fg_color=BLUE, hover_color=BLUE2,
                                 font=ctk.CTkFont(size=20, weight="bold"), command=self.run)
        self.btn.grid(row=0, column=0, sticky="ew")
        self.open_btn = ctk.CTkButton(act, text="Open dashboard", height=58, width=170, corner_radius=16, fg_color="#0b1240",
                                      hover_color="#16206a", state="disabled", command=self.open_dash)
        self.open_btn.grid(row=0, column=1, padx=(12, 0))
        self.bar = ctk.CTkProgressBar(act, mode="indeterminate", progress_color=BLUE2, fg_color="#0b1240", height=6)
        self.bar.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(14, 0))
        self.bar.set(0)
        self.status = ctk.CTkLabel(act, text="Ready.", text_color=MUTED, anchor="w")
        self.status.grid(row=2, column=0, columnspan=2, sticky="w", pady=(6, 0))

        self.log = ctk.CTkTextbox(self, fg_color="#04081f", text_color="#cfe1ff", corner_radius=16,
                                  font=ctk.CTkFont(family="Consolas", size=12))
        self.log.grid(row=3, column=0, sticky="nsew", padx=22, pady=(6, 22))
        self.log.configure(state="disabled")
        if os.path.exists(DASH):
            self.open_btn.configure(state="normal")

    # ------------------------------------------------------------------ actions
    def _say(self, text, color=MUTED):
        self.status.configure(text=text, text_color=color)

    def _append(self, s):
        self.log.configure(state="normal")
        self.log.insert("end", s)
        self.log.see("end")
        self.log.configure(state="disabled")

    def open_dash(self):
        if os.path.exists(DASH):
            webbrowser.open("file:///" + DASH.replace("\\", "/"))

    def run(self):
        if self.running:
            return
        try:
            seed, mis = int(self.seed.get()), int(self.mis.get())
        except ValueError:
            self._say("Seed must be a whole number.", RED)
            return
        self.running = True
        self.btn.configure(state="disabled", text="Running...")
        self.open_btn.configure(state="disabled")
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")
        self.bar.start()
        self._say("Working...", BLUE2)
        args = (seed, mis, bool(self.exps.get()))
        threading.Thread(target=self._work, args=args, daemon=True).start()

    def _work(self, seed, mis, exps):
        try:
            with contextlib.redirect_stdout(QWriter(self.q)), contextlib.redirect_stderr(QWriter(self.q)):
                run_pipeline(seed, mis, exps, lambda s: self.q.put(("status", s)))
            self.q.put(("done", None))
        except Exception:
            self.q.put(("log", "\n" + traceback.format_exc()))
            self.q.put(("error", None))

    def _poll(self):
        try:
            while True:
                kind, val = self.q.get_nowait()
                if kind == "log":
                    self._append(val)
                elif kind == "status":
                    self._say(val, BLUE2)
                elif kind in ("done", "error"):
                    self.running = False
                    self.bar.stop()
                    self.bar.set(1 if kind == "done" else 0)
                    self.btn.configure(state="normal", text="▶   Run again")
                    if kind == "done":
                        self._say("Done. The dashboard is ready.", GREEN)
                        self.open_btn.configure(state="normal")
                        if self.open_after.get():
                            self.open_dash()
                    else:
                        self._say("Something failed - see the log below.", RED)
        except queue.Empty:
            pass
        self.after(80, self._poll)


if __name__ == "__main__":
    if sys.stdout is None:  # windowed (no console) mode
        sys.stdout = sys.stderr = io.StringIO()
    if "--selftest" in sys.argv:  # headless check of what the button does; result goes to outputs/selftest.txt
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                run_pipeline(3, 2, False)
            result = "OK\n" + ("dashboard exists: %s" % os.path.exists(DASH))
        except Exception:
            result = "FAILED\n" + traceback.format_exc()
        with open(os.path.join(ROOT, "outputs", "selftest.txt"), "w", encoding="utf-8") as f:
            f.write(result + "\n--- log tail ---\n" + buf.getvalue()[-600:])
        sys.exit(0)
    App().mainloop()
