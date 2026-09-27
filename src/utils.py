import re
import time
import unicodedata
import pandas as pd


def normalize_text(x):
    """Normalize text by converting to lowercase, decomposing unicodes, and removing extra spaces."""
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    x = str(x).lower()
    x = unicodedata.normalize("NFKD", x)
    x = re.sub(r"[^\w\s]", " ", x, flags=re.UNICODE)
    x = re.sub(r"\s+", " ", x).strip()
    return x


def normalize_name(x):
    """Normalize business name with standardized corporate suffix replacements."""
    x = normalize_text(x)
    replacements = {
        r"\bprivate limited\b": "pvt ltd",
        r"\bprivate\b": "pvt",
        r"\blimited\b": "ltd",
        r"\bcorporation\b": "corp",
        r"\bcompany\b": "co",
        r"\bincorporated\b": "inc",
    }
    for pattern, sub in replacements.items():
        x = re.sub(pattern, sub, x)
    return x.strip()


def normalize_address(x):
    """Normalize address with standardized street/location abbreviations."""
    x = normalize_text(x)
    replacements = {
        r"\broad\b": "rd",
        r"\bstreet\b": "st",
        r"\bavenue\b": "ave",
        r"\bboulevard\b": "blvd",
        r"\blane\b": "ln",
        r"\bbuilding\b": "bldg",
        r"\bapartment\b": "apt",
        r"\bfloor\b": "flr",
        r"\bnumber\b": "no",
    }
    for pattern, sub in replacements.items():
        x = re.sub(pattern, sub, x)
    return x.strip()


class ProgressTracker:
    """Helper class to print clean periodic progress updates with percentage and ETA."""

    def __init__(self, total, stage_name, update_interval_sec=10):
        self.total = max(1, total)
        self.stage_name = stage_name
        self.interval = update_interval_sec
        self.start_time = time.time()
        self.last_print_time = 0
        self.current = 0

    def update(self, count=1):
        self.current += count
        now = time.time()
        if now - self.last_print_time >= self.interval or self.current >= self.total:
            self.last_print_time = now
            elapsed = now - self.start_time
            pct = (self.current / self.total) * 100
            speed = self.current / elapsed if elapsed > 0 else 0
            eta = (self.total - self.current) / speed if speed > 0 else 0
            print(
                f"[{self.stage_name}] Progress: {self.current:,}/{self.total:,} "
                f"({pct:.1f}%) | Speed: {speed:,.1f}/s | Elapsed: {elapsed:.0f}s | ETA: {eta:.0f}s",
                flush=True
            )