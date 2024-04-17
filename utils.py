import dataclasses
import pathlib
from typing import Optional

from tkinter import filedialog
import tkinter as tk

def enforce_between_0_and_1(value: float) -> None:
    if not (0 <= value <= 1):
        raise ValueError(f"Value {value} must be between 0 and 1")


def convert_percentage_if_necessary(value: float) -> float:
    """
    If a number is above 1, divide it by 100.
    """
    if value <= 1:
        return value
    else:
        return value / 100


def is_floatable(string: str) -> bool:
    try:
        float(string)
        return True
    except ValueError:
        return False
    


@dataclasses.dataclass()
class LastVisitedFolder:
    last_location: pathlib.Path

    def update(self, new_location: pathlib.Path) -> None:
        self.last_location = new_location


results_folder_tracker = LastVisitedFolder(pathlib.Path.home())

def check_valid_hdx_file(path: Optional[str]) -> None:
    if path is None:
        raise ValueError("No path supplied.")
    filepath = pathlib.Path(path)
    if not filepath.exists():
        raise IOError(f"File {path} does not exist")
    if not filepath.suffix == ".csv":
        raise IOError(f"File {path} must be a csv file.")


def file_browser() -> Optional[str]:
    """
    Opens a small window to allow a user to select a results file.
    """
    try:
        root = tk.Tk()
        results_file = filedialog.askopenfilenames(
            parent=root,
            initialdir=results_folder_tracker.last_location,
            initialfile="tmp",
            filetypes=[("CSV", "*.csv"), ("All files", "*")],
        )[0]

        check_valid_hdx_file(results_file)
        results_folder_tracker.update(pathlib.Path(results_file).parent)
        return results_file

    finally:
        root.destroy()

