#!/usr/bin/env python3
"""
FaceMatch - experimental face similarity search across Instagram follower/following lists.

Pipeline:
  1. Detect faces in a target image and pick one (by index or by clicking on it).
  2. Log in to Instagram and collect the followers/following of one or more profiles.
  3. For every account, download the profile picture (and optionally recent posts),
     extract face embeddings and compute the cosine similarity to the target face.
  4. Print a ranked list, a list of accounts that could not be compared, and
     optionally export both to CSV.

Similarity scores are NOT proof of identity.
"""

import argparse
import csv
import logging
import sys
import time
import tkinter as tk
from collections import Counter
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Optional

import cv2
import numpy as np
import requests
from deepface import DeepFace
from instagrapi import Client
from PIL import Image, ImageTk

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s %(message)s")
log = logging.getLogger("facematch")

DETECTOR_ORDER = ["retinaface", "mtcnn", "opencv"]  # tried in this order until one finds a face
MODEL_NAME = "Facenet512"
SLEEP_BETWEEN_REQUESTS = 1.5
DOWNLOAD_RETRIES = 2        # 2 retries = 3 attempts per image
MAX_PREVIEW_SIZE = 1000     # longest edge of the face selection window (pixels)

# Instagram media types
MEDIA_PHOTO = 1
MEDIA_CAROUSEL = 8          # 2 = video/reel, skipped

# Reasons shown in the "NOT COMPARABLE" report
REASON_NO_PROFILE_PIC = "No profile picture"
REASON_NO_FACE = "No face detected"
REASON_DOWNLOAD_FAILED = "Download failed permanently"
REASON_POSTS_UNAVAILABLE = "Posts unavailable and no usable image"
REASON_PRIVATE_NO_FACE = "Private account without detectable face"
REASON_OTHER = "Other/unexpected error"

# Status values returned by process_user()
STATUS_OK = "ok"
STATUS_DOWNLOAD_FAILED = "download_failed"
STATUS_NOT_COMPARABLE = "not_comparable"


# --------------------------------------------------------------------------- helpers

def short(text: str, limit: int = 120) -> str:
    """Trim long error messages for console output (CSV keeps the full text)."""
    text = (text or "").replace("\n", " ")
    return text if len(text) <= limit else text[:limit - 3] + "..."


def bring_to_front(window: tk.Tk) -> None:
    """Force the window to the foreground and keep it on top until it has focus."""
    window.update_idletasks()
    window.deiconify()
    window.attributes("-topmost", True)
    window.lift()
    window.focus_force()

    def _release_on_focus(_event=None):
        # Normal behaviour again once the window actually received keyboard focus.
        window.attributes("-topmost", False)

    window.bind("<FocusIn>", _release_on_focus)

    # Some window managers ignore the first lift()/focus_force() call,
    # so retry a few times shortly after the window becomes visible.
    def _retry(attempts=3):
        if attempts == 0:
            return
        window.lift()
        window.focus_force()
        window.after(200, _retry, attempts - 1)

    window.after(100, _retry)

    # Safety net: never leave the window permanently "always on top".
    window.after(5000, lambda: window.attributes("-topmost", False))



# --------------------------------------------------------------------------- GUI: settings

class StartupDialog:
    """Tkinter form that collects all run parameters when they were not passed on the CLI."""

    REQUIRED = ("username", "password", "target_image", "account")

    def __init__(self, defaults: argparse.Namespace):
        self.defaults = defaults
        self.result: Optional[dict] = None
        self.vars: dict = {}
        self.root = tk.Tk()
        self.root.title("FaceMatch - Settings")
        self.root.resizable(False, False)
        self._build()
        bring_to_front(self.root)

    def _add_row(self, row: int, label: str, key: str, value="", show=None, browse=None):
        tk.Label(self.root, text=label, anchor="w").grid(row=row, column=0, sticky="w", padx=8, pady=3)
        var = tk.StringVar(value="" if value is None else str(value))
        tk.Entry(self.root, textvariable=var, width=48, show=show).grid(
            row=row, column=1, sticky="we", padx=4, pady=3)
        if browse:
            tk.Button(self.root, text="...", width=3,
                      command=lambda: self._browse(var, browse)).grid(row=row, column=2, padx=4)
        self.vars[key] = var

    def _browse(self, var: tk.StringVar, kind: str):
        if kind == "open":
            path = filedialog.askopenfilename(
                title="Select target image",
                filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp"), ("All files", "*.*")])
        else:
            path = filedialog.asksaveasfilename(
                title="Save results CSV", defaultextension=".csv",
                filetypes=[("CSV", "*.csv")])
        if path:
            var.set(path)

    def _build(self):
        d = self.defaults
        self._add_row(0, "Instagram username *", "username", d.username)
        self._add_row(1, "Instagram password *", "password", d.password, show="*")
        self._add_row(2, "Target image *", "target_image", d.target_image, browse="open")

        self.vars["select"] = tk.BooleanVar(value=bool(d.select))
        tk.Checkbutton(self.root, text="Pick the face interactively (click on it)",
                       variable=self.vars["select"]).grid(row=3, column=1, sticky="w", padx=4)

        self._add_row(4, "Person index (if not interactive)", "person_index", d.person_index)
        self._add_row(5, "Account(s) to scan, comma-separated *", "account", d.account)

        tk.Label(self.root, text="Mode").grid(row=6, column=0, sticky="w", padx=8, pady=3)
        self.vars["mode"] = tk.StringVar(value=d.mode)
        ttk.Combobox(self.root, textvariable=self.vars["mode"], state="readonly", width=45,
                     values=["followers", "following"]).grid(row=6, column=1, sticky="w", padx=4, pady=3)

        self._add_row(7, "Max accounts per profile", "limit", d.limit)
        self._add_row(8, "Posts per account (0 = profile picture only)", "posts_per_user", d.posts_per_user)
        self._add_row(9, "Match threshold (cosine)", "threshold", d.threshold)
        self._add_row(10, "Results CSV (optional)", "csv", d.csv, browse="save")
        self._add_row(11, "Session file", "session_file", d.session_file)

        buttons = tk.Frame(self.root)
        buttons.grid(row=12, column=0, columnspan=3, pady=10)
        tk.Button(buttons, text="Start", width=12, command=self._on_start).pack(side="left", padx=6)
        tk.Button(buttons, text="Cancel", width=12, command=self.root.destroy).pack(side="left", padx=6)
        self.root.bind("<Return>", lambda _e: self._on_start())

    def _on_start(self):
        v = {k: (var.get().strip() if isinstance(var, tk.StringVar) else var.get())
             for k, var in self.vars.items()}

        missing = [k for k in self.REQUIRED if not v[k]]
        if missing:
            messagebox.showerror("Missing input", "Please fill in: " + ", ".join(missing),
                                 parent=self.root)
            return
        if not Path(v["target_image"]).exists():
            messagebox.showerror("File not found", f"Target image not found:\n{v['target_image']}",
                                 parent=self.root)
            return
        try:
            v["person_index"] = int(v["person_index"] or 0)
            v["limit"] = int(v["limit"] or 100)
            v["posts_per_user"] = int(v["posts_per_user"] or 0)
            v["threshold"] = float(v["threshold"] or 0.55)
        except ValueError:
            messagebox.showerror("Invalid number",
                                 "Person index, limit and posts per account must be integers; "
                                 "threshold must be a number.", parent=self.root)
            return

        v["csv"] = v["csv"] or None
        v["session_file"] = v["session_file"] or "ig_session.json"
        self.result = v
        self.root.destroy()

    def run(self) -> Optional[dict]:
        self.root.mainloop()
        return self.result


# --------------------------------------------------------------------------- GUI: face picker

class FaceSelector:
    """
    Shows the target image with a numbered box around every face found by get_embeddings()
    and returns the index of the face the user clicked. Because the boxes come from the same
    detection run as the embeddings, the numbers match --person-index.
    """

    def __init__(self, image_path: str, faces: list[dict]):
        self.image_path = image_path
        self.faces = faces
        self.img_cv = None
        self.selected_index: Optional[int] = None
        self.scale = 1.0

    def draw_rectangles(self):
        for i, face in enumerate(self.faces):
            x, y, w, h = face["area"]
            cv2.rectangle(self.img_cv, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(self.img_cv, str(i), (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    def on_click(self, event):
        # Map the click from the (possibly downscaled) preview back to original pixels
        px, py = event.x / self.scale, event.y / self.scale
        for i, face in enumerate(self.faces):
            x, y, w, h = face["area"]
            if x <= px <= x + w and y <= py <= y + h:
                self.selected_index = i
                print(f"Face #{i} selected.")
                self.root.destroy()
                return
        print("No face hit. Please click inside a green rectangle.")

    def select_face(self) -> Optional[int]:
        self.img_cv = cv2.imread(self.image_path)
        if self.img_cv is None:
            log.error("Could not load image: %s", self.image_path)
            return None
        self.draw_rectangles()

        pil_img = Image.fromarray(cv2.cvtColor(self.img_cv, cv2.COLOR_BGR2RGB))

        # Downscale large images so the window fits on screen
        longest = max(pil_img.width, pil_img.height)
        if longest > MAX_PREVIEW_SIZE:
            self.scale = MAX_PREVIEW_SIZE / longest
            pil_img = pil_img.resize((int(pil_img.width * self.scale),
                                      int(pil_img.height * self.scale)))

        self.root = tk.Tk()
        self.root.title("Click the face you want to search for")
        self.img_tk = ImageTk.PhotoImage(pil_img)  # keep a reference or Tk drops the image

        canvas = tk.Canvas(self.root, width=pil_img.width, height=pil_img.height)
        canvas.pack()
        canvas.create_image(0, 0, anchor=tk.NW, image=self.img_tk)
        canvas.bind("<Button-1>", self.on_click)

        bring_to_front(self.root)
        print("\nSelection window opened. Click the green rectangle of the desired face.")
        self.root.mainloop()
        return self.selected_index


# --------------------------------------------------------------------------- Instagram

def login(username: str, password: str, session_file: str) -> Client:
    cl = Client()
    session_path = Path(session_file)
    if session_path.exists():
        try:
            cl.load_settings(session_path)
            log.info("Saved session loaded.")
        except Exception as e:
            log.warning("Could not load session: %s", e)
            cl = Client()

    # With a loaded session, login() only verifies it (one request) and logs in again
    # automatically if Instagram has expired it. It also sets the username that
    # instagrapi shows in its request log.
    cl.login(username, password)
    cl.dump_settings(session_path)
    log.info("Login successful, session saved.")
    return cl


def get_users(client: Client, account_username: str, limit: int, mode: str) -> list[dict]:
    user_id = client.user_id_from_username(account_username)

    if mode == "following":
        users_dict = client.user_following(user_id, amount=limit)
    else:
        users_dict = client.user_followers(user_id, amount=limit)

    users = []
    for uid, user in users_dict.items():
        pic = getattr(user, "profile_pic_url", "")
        pic_hd = getattr(user, "profile_pic_url_hd", "")
        users.append({
            "id": uid,
            "username": user.username,
            "full_name": getattr(user, "full_name", ""),
            "profile_pic_url": str(pic) if pic else "",          # 150 x 150 px thumbnail
            "profile_pic_url_hd": str(pic_hd) if pic_hd else "",  # usually empty in follower lists
            "is_private": getattr(user, "is_private", False),
        })
    return users


def resolve_full_size_profile_pic(client: Client, user: dict) -> None:
    """
    Replace the 150 x 150 px profile picture from the follower list with the full-size
    version and refresh the private flag. Costs one extra API request per account.
    On failure the thumbnail is kept and the retry pass tries again.
    """
    if user.get("profile_pic_resolved"):
        return
    if user["profile_pic_url_hd"]:
        user["profile_pic_url"] = user["profile_pic_url_hd"]
        user["profile_pic_resolved"] = True
        return
    try:
        details = client.user_info(user["id"])
    except Exception as e:
        log.warning("    Full-size profile picture unavailable, using 150 px thumbnail: %s",
                    short(str(e)))
        return
    if details.profile_pic_url_hd:
        user["profile_pic_url"] = str(details.profile_pic_url_hd)
    user["is_private"] = details.is_private
    user["profile_pic_resolved"] = True


def get_user_posts(client: Client, user_id: str, count: int) -> tuple[list[str], Optional[str]]:
    """
    Return (image URLs, error text). Error text is None if the request succeeded.
    Photos are used as they are. Of a carousel only the first photo is used, so a carousel
    costs as much as a single post. Videos and reels are skipped.
    """
    urls = []
    try:
        medias = client.user_medias(user_id, amount=count)
        for media in medias:
            media_type = getattr(media, "media_type", MEDIA_PHOTO)
            url = None
            if media_type == MEDIA_PHOTO:
                url = getattr(media, "thumbnail_url", None)
            elif media_type == MEDIA_CAROUSEL:
                for item in getattr(media, "resources", None) or []:
                    if getattr(item, "media_type", None) == MEDIA_PHOTO and item.thumbnail_url:
                        url = item.thumbnail_url
                        break
            if url:
                urls.append(str(url))
    except Exception as e:
        log.debug("Posts of %s unavailable: %s", user_id, e)
        return urls, str(e)
    return urls, None


def download_image(session: requests.Session, url: str, save_path: Path,
                   retries: int = DOWNLOAD_RETRIES) -> Optional[Path]:
    for attempt in range(retries + 1):
        try:
            resp = session.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()
            save_path.write_bytes(resp.content)
            log.info("Download OK: %s -> %s", url, save_path)
            return save_path
        except Exception as e:
            log.warning("Download attempt %d failed: %s", attempt + 1, e)
            time.sleep(1 + attempt)

    log.error("Download failed permanently: %s", url)
    return None


# --------------------------------------------------------------------------- face recognition

def get_embeddings(image_path: Path, detectors: list[str]) -> tuple[Optional[list[dict]], str]:
    """Return ([{embedding, area}], detector_name). Each entry is one detected face."""
    for detector in detectors:
        try:
            result = DeepFace.represent(
                img_path=str(image_path),
                model_name=MODEL_NAME,
                detector_backend=detector,
                enforce_detection=True,
                align=True,
            )
            faces = []
            for r in result:
                a = r.get("facial_area", {})
                faces.append({
                    "embedding": np.array(r["embedding"]),
                    "area": (a.get("x", 0), a.get("y", 0), a.get("w", 0), a.get("h", 0)),
                })
            if faces:
                log.info("Detector %s: %d face(s) found in %s", detector, len(faces), image_path)
                return faces, detector
            log.info("Detector %s: no face found in %s", detector, image_path)
        except Exception as e:
            log.debug("Detector %s failed: %s", detector, e)

    log.warning("No detector could find a face in %s", image_path)
    return None, ""


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    return float(np.dot(a, b))



def process_user(client: Client, session: requests.Session, user: dict,
                 target_emb: np.ndarray, work_dir: Path, posts_per_user: int) -> dict:
    """
    Compare one account against the target face.

    Returned status:
      STATUS_OK               -> score/source filled, note may describe partial data
      STATUS_DOWNLOAD_FAILED  -> at least one download failed and nothing could be
                                 compared -> candidate for the retry pass
      STATUS_NOT_COMPARABLE   -> reason/detail filled
    """
    username = user["username"]
    info = {
        "username": username,
        "full_name": user["full_name"],
        "is_private": user["is_private"],
        "status": STATUS_NOT_COMPARABLE,
        "reason": "",
        "detail": "",
        "note": "",
        "score": -1.0,
        "source": "",
    }

    try:
        resolve_full_size_profile_pic(client, user)
        info["is_private"] = user["is_private"]
        log.info("    Private: %s", user["is_private"])

        has_profile_pic = bool(user["profile_pic_url"])
        urls = [user["profile_pic_url"]] if has_profile_pic else []
        posts_error = None
        if posts_per_user > 0:
            post_urls, posts_error = get_user_posts(client, user["id"], posts_per_user)
            urls += post_urls

        download_failed_any = False
        profile_pic_failed = False
        face_found_any = False
        compared = False
        best_score, best_source = -1.0, ""

        for j, url in enumerate(urls):
            img_path = work_dir / f"{username}_{j}.jpg"
            if not download_image(session, url, img_path):
                download_failed_any = True
                if j == 0 and has_profile_pic:
                    profile_pic_failed = True
                continue

            try:
                faces, _ = get_embeddings(img_path, DETECTOR_ORDER)
            finally:
                # Delete the image right after analysis, whether or not a face was found.
                img_path.unlink(missing_ok=True)
            if not faces:
                continue
            face_found_any = True

            for k, face in enumerate(faces):
                score = cosine_similarity(target_emb, face["embedding"])
                compared = True
                log.info("    Score %.4f for %s (face %d)", score, img_path, k + 1)
                if score > best_score:
                    best_score, best_source = score, str(img_path)

        # Classification - order defines priority
        if compared:
            note = ""
            if posts_error:
                note = f"Posts unavailable ({posts_error}) - profile picture only"
            elif profile_pic_failed:
                note = "Profile picture download failed - posts only"
            info.update(status=STATUS_OK, score=best_score, source=best_source, note=note)
        elif download_failed_any:
            info.update(status=STATUS_DOWNLOAD_FAILED, reason=REASON_DOWNLOAD_FAILED)
        elif not urls:
            if posts_error:
                info.update(reason=REASON_POSTS_UNAVAILABLE, detail=posts_error)
            else:
                info.update(reason=REASON_NO_PROFILE_PIC)
        elif not face_found_any:
            if user["is_private"]:
                info.update(reason=REASON_PRIVATE_NO_FACE, detail=posts_error or "")
            elif posts_error:
                info.update(reason=REASON_POSTS_UNAVAILABLE, detail=posts_error)
            else:
                info.update(reason=REASON_NO_FACE)
        else:
            info.update(reason=REASON_OTHER, detail="Unexpected state without comparison")

    except Exception as e:
        log.warning("    Unexpected error for @%s: %s", username, e)
        info.update(status=STATUS_NOT_COMPARABLE, reason=REASON_OTHER, detail=str(e))

    return info


# --------------------------------------------------------------------------- main

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Face similarity search across Instagram follower/following lists. "
                    "If required arguments are missing, a settings window opens.")
    parser.add_argument("--username", help="Instagram login")
    parser.add_argument("--password", help="Instagram password")
    parser.add_argument("--target-image", help="Path to the target image")
    parser.add_argument("--person-index", type=int, default=0,
                        help="0 = first detected face, 1 = second ... (ignored with --select)")
    parser.add_argument("--select", action="store_true",
                        help="Pick the face interactively by clicking on it")
    parser.add_argument("--account", help="Instagram account or comma-separated list")
    parser.add_argument("--mode", choices=["followers", "following"], default="followers")
    parser.add_argument("--limit", type=int, default=100, help="Max accounts per profile")
    parser.add_argument("--posts-per-user", type=int, default=0,
                        help="Also check the last N posts per account (0 = profile picture only)")
    parser.add_argument("--threshold", type=float, default=0.55,
                        help="Cosine similarity threshold for 'possible match'")
    parser.add_argument("--csv", help="Results CSV (not-comparable accounts go to <name>_not_comparable.csv)")
    parser.add_argument("--session-file", default="ig_session.json",
                        help="File used to persist the Instagram session")
    parser.add_argument("--no-gui", action="store_true",
                        help="Never open the settings window; fail if required arguments are missing")
    args = parser.parse_args()

    if any(getattr(args, key) is None for key in StartupDialog.REQUIRED):
        if args.no_gui:
            parser.error("--username, --password, --target-image and --account are required with --no-gui")
        values = StartupDialog(args).run()
        if values is None:
            log.info("Cancelled by user.")
            sys.exit(0)
        for key, value in values.items():
            setattr(args, key, value)
    return args


def main():
    args = parse_args()

    target_path = Path(args.target_image)
    if not target_path.exists():
        log.error("Target image not found: %s", target_path)
        sys.exit(1)

    # ---------- target face ----------
    log.info("Analysing target image %s ...", target_path)
    target_faces, detector = get_embeddings(target_path, DETECTOR_ORDER)
    if not target_faces:
        log.error("No face found in the target image.")
        sys.exit(1)

    chosen = None
    if args.select:
        index = FaceSelector(str(target_path), target_faces).select_face()
        if index is not None:
            chosen = target_faces[index]
        else:
            log.warning("No face selected - using face index %d.", args.person_index)

    if chosen is None:
        if args.person_index >= len(target_faces):
            log.error("Person index %d out of range (0-%d).", args.person_index, len(target_faces) - 1)
            sys.exit(1)
        chosen = target_faces[args.person_index]

    target_emb = chosen["embedding"]
    log.info("Using face at %s (%d detected, detector: %s).", chosen["area"], len(target_faces), detector)

    # ---------- Instagram ----------
    log.info("Logging in as %s ...", args.username)
    try:
        client = login(args.username, args.password, args.session_file)
    except Exception as e:
        log.error("Login failed: %s", e)
        sys.exit(1)

    all_users, seen = [], set()
    for account in args.account.split(","):
        account = account.strip()
        if not account:
            continue
        log.info("Loading up to %d '%s' of @%s ...", args.limit, args.mode, account)
        try:
            for u in get_users(client, account, args.limit, args.mode):
                if u["username"] not in seen:
                    seen.add(u["username"])
                    all_users.append(u)
        except Exception as e:
            log.warning("Account @%s could not be loaded: %s", account, e)
    log.info("%d unique accounts loaded.", len(all_users))

    work_dir = Path("downloaded_pics")
    work_dir.mkdir(exist_ok=True)
    session = requests.Session()
    results: list[dict] = []
    not_comparable: list[dict] = []
    failed_downloads: list[dict] = []

    # ---------- pass 1 ----------
    for i, user in enumerate(all_users, 1):
        log.info("[%d/%d] Checking @%s ...", i, len(all_users), user["username"])

        info = process_user(client, session, user, target_emb, work_dir, args.posts_per_user)

        if info["status"] == STATUS_OK:
            results.append(info)
            if info["note"]:
                log.warning("    Partial data: %s", short(info["note"]))
        elif info["status"] == STATUS_DOWNLOAD_FAILED:
            failed_downloads.append(user)
            log.warning("    Download failed - @%s will be retried at the end.", user["username"])
        else:
            not_comparable.append(info)
            log.info("    Not comparable: %s", info["reason"])

        time.sleep(SLEEP_BETWEEN_REQUESTS)

    # ---------- pass 2: accounts whose downloads failed ----------
    if failed_downloads:
        log.info("Retry pass for %d account(s) with failed downloads ...", len(failed_downloads))
        for i, user in enumerate(failed_downloads, 1):
            log.info("[Retry %d/%d] Checking @%s ...", i, len(failed_downloads), user["username"])

            info = process_user(client, session, user, target_emb, work_dir, args.posts_per_user)

            if info["status"] == STATUS_OK:
                results.append(info)
                log.info("    Retry successful: score %.4f", info["score"])
            else:
                if info["status"] == STATUS_DOWNLOAD_FAILED:
                    info["status"] = STATUS_NOT_COMPARABLE
                    info["detail"] = "failed again in retry pass"
                not_comparable.append(info)
                log.info("    Still not comparable: %s", info["reason"])

            time.sleep(SLEEP_BETWEEN_REQUESTS)

    results.sort(key=lambda r: r["score"], reverse=True)

    # ---------- output: ranking ----------
    print("\n" + "=" * 78)
    print("RESULTS (sorted by similarity - NOT proof of identity!)")
    print("=" * 78)

    if not results:
        print("No comparable accounts found.")
    else:
        for rank, r in enumerate(results, 1):
            marker = "  <-- possible match" if r["score"] >= args.threshold else ""
            print(f"{rank:>3}. @{r['username']:<30} Score: {r['score']:.4f}  "
                  f"Private: {r['is_private']}  Source: {r['source']}{marker}")
            if r["note"]:
                print(f"      Note: {short(r['note'])}")

    # ---------- output: not comparable ----------
    print("\n" + "=" * 78)
    print("NOT COMPARABLE")
    print("=" * 78)

    if not not_comparable:
        print("All loaded accounts could be compared.")
    else:
        for nc in not_comparable:
            detail = f" ({short(nc['detail'])})" if nc["detail"] else ""
            print(f"  @{nc['username']:<30} Private: {str(nc['is_private']):<5}  "
                  f"Reason: {nc['reason']}{detail}")

        print("\nSummary by reason:")
        for reason, n in Counter(nc["reason"] for nc in not_comparable).most_common():
            print(f"  {n:>4} x {reason}")

    print(f"\nTotal: {len(all_users)} accounts loaded, {len(results)} compared, "
          f"{len(not_comparable)} not comparable.")

    # ---------- CSV ----------
    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "rank", "username", "full_name", "is_private", "score", "possible_match",
                "source", "note"])
            writer.writeheader()
            for rank, r in enumerate(results, 1):
                writer.writerow({
                    "rank": rank,
                    "username": r["username"],
                    "full_name": r["full_name"],
                    "is_private": r["is_private"],
                    "score": f"{r['score']:.4f}",
                    "possible_match": "yes" if r["score"] >= args.threshold else "",
                    "source": r["source"],
                    "note": r["note"],
                })
        log.info("Results saved to %s.", args.csv)

        csv_path = Path(args.csv)
        nc_path = csv_path.with_name(f"{csv_path.stem}_not_comparable.csv")
        with open(nc_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "username", "full_name", "is_private", "reason", "detail"])
            writer.writeheader()
            for nc in not_comparable:
                writer.writerow({
                    "username": nc["username"],
                    "full_name": nc["full_name"],
                    "is_private": nc["is_private"],
                    "reason": nc["reason"],
                    "detail": nc["detail"],
                })
        log.info("Not-comparable accounts saved to %s.", nc_path)

    print("\n[!] Similarity is not proof of identity. Verify every match manually.")


if __name__ == "__main__":
    main()
