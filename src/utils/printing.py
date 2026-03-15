"""Training progress printing utilities."""

import sys
from typing import Dict, Any, Optional


class Columns:
    SMALL = 10
    MEDIUM = 14
    LARGE = 20


def make_header() -> str:
    return (
        "Round".ljust(Columns.SMALL)
        + "│ Mode".ljust(Columns.SMALL)
        + "│ Clients".ljust(Columns.SMALL)
        + "│ Loss".ljust(Columns.SMALL)
        + "│ Acc".ljust(Columns.SMALL)
        + "│ DSNR".ljust(Columns.SMALL)
        + "│ Time".ljust(Columns.MEDIUM)
    )


def make_line(
    round_num: int,
    total_rounds: int,
    is_train: bool,
    num_clients: int,
    loss: float,
    accuracy: float,
    dsnr: Optional[float] = None,
    round_time: Optional[float] = None,
    total_time: Optional[float] = None,
) -> str:
    mode_str = "Train" if is_train else "Eval"
    
    dsnr_str = f"{dsnr:.2f}" if dsnr is not None else "N/A"
    
    if round_time is not None:
        time_str = format_time(round_time)
    else:
        time_str = "--:--"
    
    return (
        f"[{round_num:>3}/{total_rounds}]".ljust(Columns.SMALL)
        + f"│ {mode_str}".ljust(Columns.SMALL)
        + f"│ {num_clients:>3}".ljust(Columns.SMALL)
        + f"│ {loss:>7.4f}".ljust(Columns.SMALL)
        + f"│ {accuracy:>6.4f}".ljust(Columns.SMALL)
        + f"│ {dsnr_str:>6}".ljust(Columns.SMALL)
        + f"│ {time_str}".ljust(Columns.MEDIUM)
    )


def format_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins:02d}:{secs:02d}"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours:02d}:{mins:02d}:00"


def print_header() -> None:
    line = make_header()
    sys.stdout.write("\n" + line + "\n")
    sys.stdout.write("─" * len(line) + "\n")
    sys.stdout.flush()


def print_line(
    round_num: int,
    total_rounds: int,
    is_train: bool,
    num_clients: int,
    loss: float,
    accuracy: float,
    dsnr: Optional[float] = None,
    round_time: Optional[float] = None,
    total_time: Optional[float] = None,
    persistent: bool = False,
) -> None:
    line = make_line(
        round_num, total_rounds, is_train, num_clients,
        loss, accuracy, dsnr, round_time, total_time
    )
    
    if persistent:
        sys.stdout.write(line + "\n")
    else:
        sys.stdout.write("\r" + line)
    
    sys.stdout.flush()


def print_summary(
    total_time: float,
    best_accuracy: float,
    final_round: int,
    convergence_round: Optional[int] = None,
) -> None:
    sys.stdout.write("\n\n")
    sys.stdout.write("=" * 50 + "\n")
    sys.stdout.write("Training Complete\n")
    sys.stdout.write("=" * 50 + "\n")
    sys.stdout.write(f"Total rounds: {final_round}\n")
    sys.stdout.write(f"Best accuracy: {best_accuracy:.4f}\n")
    sys.stdout.write(f"Total time: {format_time(total_time)}\n")
    if convergence_round is not None:
        sys.stdout.write(f"Convergence round: {convergence_round}\n")
    sys.stdout.write("=" * 50 + "\n")
    sys.stdout.flush()
