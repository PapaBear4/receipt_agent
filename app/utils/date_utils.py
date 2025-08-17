from datetime import datetime
from dateutil import parser


def normalize_date_to_mmddyyyy(raw: str) -> str:
    raw = (raw or "").strip()
    if not raw:
        return ""
    dt = None
    # Try common receipt formats first to reduce ambiguity
    for fmt in ("%m/%d/%Y", "%m/%d/%y", "%Y-%m-%d", "%m-%d-%Y", "%m.%d.%Y"):
        try:
            dt = datetime.strptime(raw, fmt)
            break
        except Exception:
            continue
    if dt is None:
        try:
            dt = parser.parse(raw, dayfirst=False, yearfirst=False, fuzzy=True)
        except Exception:
            return ""
    return dt.strftime("%m/%d/%Y")


def parse_date_to_unix(raw: str) -> int:
    raw = (raw or "").strip()
    if not raw:
        return 0
    dt = None
    for fmt in ("%m/%d/%Y", "%m/%d/%y", "%Y-%m-%d", "%m-%d-%Y", "%m.%d.%Y"):
        try:
            dt = datetime.strptime(raw, fmt)
            break
        except Exception:
            continue
    if dt is None:
        try:
            dt = parser.parse(raw, dayfirst=False, yearfirst=False, fuzzy=True)
        except Exception:
            return 0
    try:
        return int(dt.timestamp())
    except Exception:
        return 0
