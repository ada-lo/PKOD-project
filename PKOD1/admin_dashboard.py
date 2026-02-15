import customtkinter as ctk
import json
import os
import time
from datetime import datetime
import config

# --- FILE PATHS (use backend config) ---
STATE_FILE = config.OCCUPANCY_STATE_FILE
COMMAND_FILE = config.COMMAND_FILE

# seconds after which data is considered stale
STALE_AFTER_SEC = 5

# --- THEME SETUP ---
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


# ---------- ATOMIC COMMAND WRITER ----------
def atomic_write_command(data):
    tmp = COMMAND_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, COMMAND_FILE)


# --- SAFE SNAPSHOT LOADER ---
def load_snapshot():
    if not os.path.exists(STATE_FILE):
        return None, "No snapshot file yet"
    try:
        with open(STATE_FILE, "r") as f:
            data = json.load(f)
        return data, None
    except Exception as e:
        return None, f"Failed to read snapshot: {e}"


# --- DERIVED VIEW (do not trust stored counts) ---
def derive_view(data: dict) -> dict:
    vehicle_states = data.get("vehicle_states", [])

    active = [
        v for v in vehicle_states if v.get("has_entered") and not v.get("has_exited")
    ]

    exited = [v for v in vehicle_states if v.get("has_exited")]

    return {
        "occupancy": len(active),
        "active": active,
        "exited": exited,
        "entry_count": data.get("entry_count", 0),
        "exit_count": data.get("exit_count", 0),
        "last_update": data.get("last_update", 0),
        "audit": data.get("audit", []),
    }


def is_stale(last_update: float) -> bool:
    if not last_update:
        return True
    return (time.time() - last_update) > STALE_AFTER_SEC


class ParkingAdminApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Window Setup
        self.title("Parking Admin Dashboard")
        self.geometry("1000x650")

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # -------- LEFT SIDEBAR --------
        self.sidebar = ctk.CTkFrame(self, width=220, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")

        self.logo_label = ctk.CTkLabel(
            self.sidebar,
            text="ADMIN\nDASHBOARD",
            font=ctk.CTkFont(size=20, weight="bold"),
        )
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        # Controls
        ctk.CTkLabel(self.sidebar, text="Manual Controls").grid(
            row=1, column=0, padx=20, pady=(20, 5), sticky="w"
        )

        self.btn_reset = ctk.CTkButton(
            self.sidebar,
            text="RESET SYSTEM",
            fg_color="#c0392b",
            hover_color="#e74c3c",
            command=self.reset_system,
        )
        self.btn_reset.grid(row=2, column=0, padx=20, pady=10)

        self.btn_full = ctk.CTkButton(
            self.sidebar,
            text="FORCE FULL",
            fg_color="#d35400",
            hover_color="#e67e22",
            command=self.force_full,
        )
        self.btn_full.grid(row=3, column=0, padx=20, pady=10)

        self.input_set_occ = ctk.CTkEntry(
            self.sidebar, placeholder_text="Set Occupancy (e.g. 15)"
        )
        self.input_set_occ.grid(row=4, column=0, padx=20, pady=(20, 5))

        self.btn_set_occ = ctk.CTkButton(
            self.sidebar, text="SET VALUE", command=self.set_occupancy
        )
        self.btn_set_occ.grid(row=5, column=0, padx=20, pady=5)

        self.status_label = ctk.CTkLabel(
            self.sidebar, text="Status: Waiting", text_color="gray"
        )
        self.status_label.grid(row=6, column=0, padx=20, pady=(40, 10))

        # -------- MAIN AREA --------
        self.main = ctk.CTkFrame(self, fg_color="transparent")
        self.main.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
        self.main.grid_columnconfigure((0, 1, 2), weight=1)

        self.card_occ = self.create_stat_card(0, "OCCUPANCY", "0")
        self.card_in = self.create_stat_card(1, "TOTAL ENTRY", "0")
        self.card_out = self.create_stat_card(2, "TOTAL EXIT", "0")

        ctk.CTkLabel(self.main, text="Capacity Usage").grid(
            row=1, column=0, columnspan=3, sticky="w"
        )
        self.progress = ctk.CTkProgressBar(self.main)
        self.progress.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(0, 20))
        self.progress.set(0)

        ctk.CTkLabel(
            self.main,
            text="Audit Log",
            font=ctk.CTkFont(size=16, weight="bold"),
        ).grid(row=3, column=0, columnspan=3, sticky="w")

        self.log_frame = ctk.CTkScrollableFrame(self.main, height=300)
        self.log_frame.grid(row=4, column=0, columnspan=3, sticky="nsew")

        self.after(1000, self.update_dashboard)

    # ---------- UI HELPERS ----------
    def create_stat_card(self, col, title, value):
        frame = ctk.CTkFrame(self.main, corner_radius=10)
        frame.grid(row=0, column=col, padx=10, pady=10, sticky="nsew")

        ctk.CTkLabel(frame, text=title).pack(pady=(10, 0))
        val_label = ctk.CTkLabel(
            frame, text=value, font=ctk.CTkFont(size=36, weight="bold")
        )
        val_label.pack(pady=10)
        return val_label

    # ---------- COMMANDS ----------
    def send_command(self, command, value=0):
        payload = {
            "id": f"cmd_{int(time.time())}",
            "command": command,
            "value": value,
            "ts": time.time(),
        }
        atomic_write_command(payload)
        self.status_label.configure(text=f"Sent: {command}", text_color="yellow")

    def reset_system(self):
        self.send_command("RESET_SYSTEM")

    def force_full(self):
        self.send_command("FORCE_FULL")

    def set_occupancy(self):
        try:
            val = int(self.input_set_occ.get())
            self.send_command("SET_OCCUPANCY", val)
        except ValueError:
            self.status_label.configure(text="Invalid value", text_color="red")

    # ---------- DATA UPDATE ----------
    def read_state(self):
        data, err = load_snapshot()
        return data

    def update_dashboard(self):
        data, err = load_snapshot()
        if err:
            # No snapshot yet or read failed
            self.card_occ.configure(text=f"0/{getattr(config, 'MAX_CAPACITY', 80)}")
            self.card_in.configure(text="0")
            self.card_out.configure(text="0")
            self.progress.set(0)
            self.update_audit([])
            self.status_label.configure(text=err, text_color="gray")
        else:
            view = derive_view(data)

            occ = view.get("occupancy", 0)
            ent = view.get("entry_count", 0)
            ex = view.get("exit_count", 0)

            max_cap = data.get("max_capacity", getattr(config, "MAX_CAPACITY", 80))

            self.card_occ.configure(text=f"{occ}/{max_cap}")
            self.card_in.configure(text=str(ent))
            self.card_out.configure(text=str(ex))

            ratio = min(occ / max_cap, 1.0) if max_cap > 0 else 0
            self.progress.set(ratio)

            self.update_audit(view.get("audit", []))

            last = view.get("last_update", 0)
            if is_stale(last):
                self.status_label.configure(text="Stale data", text_color="yellow")
            else:
                self.status_label.configure(text="Connected", text_color="green")

        self.after(1000, self.update_dashboard)


    def update_audit(self, audit):
        for w in self.log_frame.winfo_children():
            w.destroy()

        for ev in reversed(audit[-15:]):
            ts = datetime.fromtimestamp(ev["ts"]).strftime("%H:%M:%S")
            reason = ev.get("reason", "unknown").upper()
            occ = ev.get("occupancy", 0)

            row = ctk.CTkFrame(self.log_frame)
            row.pack(fill="x", pady=2)

            color = "green" if "ENTRY" in reason else "red" if "EXIT" in reason else "white"

            ctk.CTkLabel(row, text=f"[{ts}]", width=80).pack(side="left", padx=5)
            ctk.CTkLabel(row, text=reason, text_color=color, width=120).pack(
                side="left"
            )
            ctk.CTkLabel(row, text=f"Occupancy: {occ}").pack(side="right", padx=10)


if __name__ == "__main__":
    app = ParkingAdminApp()
    app.mainloop()
