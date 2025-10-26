from __future__ import annotations

from datetime import date, time, datetime, timedelta
from typing import Optional, Union

from ..values import Value


# ---- Helpers -----------------------------------------------------------------

def _ensure_date(v: Union[date, str]) -> date:
    if isinstance(v, date) and not isinstance(v, datetime):
        return v
    if isinstance(v, str):
        # try ISO first
        try:
            return date.fromisoformat(v)
        except Exception:
            pass
        # common fallback formats
        for fmt in ("%d.%m.%Y", "%m/%d/%Y", "%Y/%m/%d"):
            try:
                return datetime.strptime(v, fmt).date()
            except Exception:
                continue
    raise TypeError(f"Unsupported date value: {v!r}")


def _ensure_time(v: Union[time, str]) -> time:
    if isinstance(v, time):
        return v
    if isinstance(v, str):
        # try ISO-like variants
        for fmt in ("%H:%M:%S", "%H:%M"):
            try:
                return datetime.strptime(v, fmt).time()
            except Exception:
                continue
    raise TypeError(f"Unsupported time value: {v!r}")


def _ensure_datetime(v: Union[datetime, str]) -> datetime:
    if isinstance(v, datetime):
        return v
    if isinstance(v, str):
        # ISO 8601 (without timezone)
        try:
            return datetime.fromisoformat(v)
        except Exception:
            pass
        # common fallbacks
        for fmt in ("%Y-%m-%d %H:%M:%S", "%d.%m.%Y %H:%M:%S", "%m/%d/%Y %H:%M:%S"):
            try:
                return datetime.strptime(v, fmt)
            except Exception:
                continue
    raise TypeError(f"Unsupported datetime value: {v!r}")


# ---- Generators --------------------------------------------------------------

class RandomDate(Value[Union[date, str]]):
    """Generates random dates within an inclusive range.

    Args:
        start: start boundary (inclusive), as `date` or string (ISO or common formats)
        stop: end boundary (inclusive), as `date` or string
        step_days: step size in days (must be >= 1)
        strftime: if provided, result is formatted as string with this pattern; otherwise `date`.
    """

    def __init__(
            self,
            start: Union[date, str],
            stop: Union[date, str],
            step_days: int = 1,
            strftime: Optional[str] = None,
    ) -> None:
        super().__init__(None)
        d1, d2 = _ensure_date(start), _ensure_date(stop)
        if d2 < d1:
            raise ValueError("stop date must be >= start date")
        if step_days < 1:
            raise ValueError("step_days must be >= 1")
        self._start = d1
        self._stop = d2
        self._step = step_days
        self._fmt = strftime

    def __call__(self) -> Union[date, str]:
        delta_days = (self._stop - self._start).days
        steps = delta_days // self._step
        offset = 0 if steps == 0 else __import__("random").randint(0, steps) * self._step
        result = self._start + timedelta(days=offset)
        return result.strftime(self._fmt) if self._fmt else result


class RandomTime(Value[Union[time, str]]):
    """Generates random times within an inclusive range (same day).

    Args:
        start: start boundary (inclusive), as `time` or string (e.g., "09:00" or "09:00:00")
        stop: end boundary (inclusive), as `time` or string
        step_seconds: step size in seconds (>= 1)
        strftime: if provided, return string; otherwise `time`.
    """

    def __init__(
            self,
            start: Union[time, str],
            stop: Union[time, str],
            step_seconds: int = 1,
            strftime: Optional[str] = None,
    ) -> None:
        super().__init__(None)
        t1, t2 = _ensure_time(start), _ensure_time(stop)
        s1 = t1.hour * 3600 + t1.minute * 60 + t1.second
        s2 = t2.hour * 3600 + t2.minute * 60 + t2.second
        if s2 < s1:
            raise ValueError("stop time must be >= start time")
        if step_seconds < 1:
            raise ValueError("step_seconds must be >= 1")
        self._start = s1
        self._stop = s2
        self._step = step_seconds
        self._fmt = strftime

    def __call__(self) -> Union[time, str]:
        steps = (self._stop - self._start) // self._step
        offset = 0 if steps == 0 else __import__("random").randint(0, steps) * self._step
        seconds = self._start + offset
        h, rem = divmod(seconds, 3600)
        m, s = divmod(rem, 60)
        result = time(hour=h, minute=m, second=s)
        return result.strftime(self._fmt) if self._fmt else result


class RandomDateTime(Value[Union[datetime, str]]):
    """Generates random datetimes within an inclusive range.

    Args:
        start: start boundary (inclusive), as `datetime` or string (ISO / common formats)
        stop: end boundary (inclusive)
        step_seconds: step size in seconds (>= 1)
        strftime: if provided, return string; otherwise `datetime`.
    """

    def __init__(
            self,
            start: Union[datetime, str],
            stop: Union[datetime, str],
            step_seconds: int = 1,
            strftime: Optional[str] = None,
    ) -> None:
        super().__init__(None)
        d1, d2 = _ensure_datetime(start), _ensure_datetime(stop)
        if d2 < d1:
            raise ValueError("stop datetime must be >= start datetime")
        if step_seconds < 1:
            raise ValueError("step_seconds must be >= 1")
        self._start = d1
        self._stop = d2
        self._step = step_seconds
        self._fmt = strftime

    def __call__(self) -> Union[datetime, str]:
        total_seconds = int((self._stop - self._start).total_seconds())
        steps = total_seconds // self._step
        offset = 0 if steps == 0 else __import__("random").randint(0, steps) * self._step
        result = self._start + timedelta(seconds=offset)
        return result.strftime(self._fmt) if self._fmt else result
