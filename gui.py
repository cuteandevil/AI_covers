#!/usr/bin/env python
"""
Graphical user interface for AI Cover Generator.
Provides a simple Tkinter frontend for the command-line tool.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import subprocess
import threading
import queue
import os
import sys


class CoverGeneratorGUI:
    def __init__(self, master):
        self.master = master
        master.title("AI Cover Generator GUI")
        master.geometry("600x500")
        master.resizable(True, True)

        # Variables
        self.input_file = tk.StringVar()
        self.target_speaker_file = tk.StringVar()
        self.output_file = tk.StringVar()
        self.device = tk.StringVar(value="auto")  # auto, cpu, cuda
        self.save_accompaniment = tk.BooleanVar(value=False)
        # Source separation is controlled via config.yaml; we expose a checkbox to enable/disable
        # by modifying config.yaml temporarily (optional feature)
        self.source_separation = tk.BooleanVar(value=False)

        self.queue = queue.Queue()
        self.process = None

        self.create_widgets()
        self.poll_queue()

    def create_widgets(self):
        # Input file
        ttk.Label(self.master, text="Input Audio:").grid(row=0, column=0, padx=10, pady=5, sticky="w")
        ttk.Entry(self.master, textvariable=self.input_file, width=50).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(self.master, text="Browse", command=self.browse_input).grid(row=0, column=2, padx=5, pady=5)

        # Target speaker
        ttk.Label(self.master, text="Target Speaker Audio:").grid(row=1, column=0, padx=10, pady=5, sticky="w")
        ttk.Entry(self.master, textvariable=self.target_speaker_file, width=50).grid(row=1, column=1, padx=5, pady=5)
        ttk.Button(self.master, text="Browse", command=self.browse_target).grid(row=1, column=2, padx=5, pady=5)

        # Output file
        ttk.Label(self.master, text="Output Cover:").grid(row=2, column=0, padx=10, pady=5, sticky="w")
        ttk.Entry(self.master, textvariable=self.output_file, width=50).grid(row=2, column=1, padx=5, pady=5)
        ttk.Button(self.master, text="Browse", command=self.browse_output).grid(row=2, column=2, padx=5, pady=5)

        # Device selection
        ttk.Label(self.master, text="Device:").grid(row=3, column=0, padx=10, pady=5, sticky="w")
        device_combo = ttk.Combobox(self.master, textvariable=self.device, values=["auto", "cpu", "cuda"], width=10, state="readonly")
        device_combo.grid(row=3, column=1, padx=5, pady=5, sticky="w")

        # Checkboxes
        ttk.Checkbutton(self.master, text="Save accompaniment (if source separation enabled)", variable=self.save_accompaniment).grid(row=4, column=0, columnspan=3, padx=10, pady=5, sticky="w")
        ttk.Checkbutton(self.master, text="Enable source separation (Demucs)", variable=self.source_separation).grid(row=5, column=0, columnspan=3, padx=10, pady=5, sticky="w")

        # Run button
        self.run_button = ttk.Button(self.master, text="Generate Cover", command=self.start_generation)
        self.run_button.grid(row=6, column=0, columnspan=3, pady=20)

        # Progress/log area
        ttk.Label(self.master, text="Log:").grid(row=7, column=0, padx=10, pady=5, sticky="nw")
        self.log_area = scrolledtext.ScrolledText(self.master, width=70, height=15, wrap=tk.WORD)
        self.log_area.grid(row=8, column=0, columnspan=3, padx=10, pady=5, sticky="nsew")
        self.log_area.configure(state='disabled')

        # Configure grid weights
        self.master.grid_columnconfigure(1, weight=1)
        self.master.grid_rowconfigure(8, weight=1)

    def browse_input(self):
        filename = filedialog.askopenfilename(
            title="Select input audio",
            filetypes=[("Audio files", "*.wav *.mp3 *.flac *.ogg"), ("All files", "*.*")]
        )
        if filename:
            self.input_file.set(filename)

    def browse_target(self):
        filename = filedialog.askopenfilename(
            title="Select target speaker audio",
            filetypes=[("Audio files", "*.wav *.mp3 *.flac *.ogg"), ("All files", "*.*")]
        )
        if filename:
            self.target_speaker_file.set(filename)

    def browse_output(self):
        filename = filedialog.asksaveasfilename(
            title="Save output cover",
            defaultextension=".wav",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")]
        )
        if filename:
            self.output_file.set(filename)

    def log_message(self, message):
        self.log_area.configure(state='normal')
        self.log_area.insert(tk.END, message + "\n")
        self.log_area.see(tk.END)
        self.log_area.configure(state='disabled')

    def start_generation(self):
        # Validate inputs
        if not self.input_file.get():
            messagebox.showerror("Error", "Please select input audio file")
            return
        if not self.target_speaker_file.get():
            messagebox.showerror("Error", "Please select target speaker audio file")
            return
        if not self.output_file.get():
            messagebox.showerror("Error", "Please specify output file")
            return

        # Disable run button during processing
        self.run_button.configure(state='disabled')
        self.log_message("Starting generation...")

        # Build command
        cmd = [sys.executable, "main.py"]
        cmd.extend(["--input", self.input_file.get()])
        cmd.extend(["--target_speaker", self.target_speaker_file.get()])
        cmd.extend(["--output", self.output_file.get()])
        cmd.extend(["--config", "config.yaml"])
        # Device
        if self.device.get() != "auto":
            cmd.extend(["--device", self.device.get()])
        # Save accompaniment
        if self.save_accompaniment.get():
            cmd.append("--save_accompaniment")

        # TODO: Handle source separation via config override if needed
        # For now, we rely on config.yaml; user can enable/disable there.

        # Run in separate thread
        thread = threading.Thread(target=self.run_command, args=(cmd,))
        thread.daemon = True
        thread.start()

    def run_command(self, cmd):
        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                cwd=r"D:\AI_covers"
            )
            # Read output line by line
            for line in self.process.stdout:
                self.queue.put(line.strip())
            self.process.wait()
            self.queue.put(f"Process exited with code {self.process.returncode}")
        except Exception as e:
            self.queue.put(f"Error: {e}")
        finally:
            self.queue.put("__DONE__")  # Signal completion

    def poll_queue(self):
        try:
            while True:
                line = self.queue.get_nowait()
                if line == "__DONE__":
                    self.run_button.configure(state='normal')
                    self.log_message("Generation finished.")
                    break
                else:
                    self.log_message(line)
        except queue.Empty:
            pass
        self.master.after(100, self.poll_queue)


def main():
    root = tk.Tk()
    app = CoverGeneratorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()