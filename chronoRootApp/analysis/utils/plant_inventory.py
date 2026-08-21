"""Scan Analysis/ plant slots for the Overview / Overlay plant list."""

import os
import pathlib

from .fileUtilities import (
    convertFromPathSafe,
    get_latest_result_dir,
    load_result_metadata,
    normalize_factor_value,
)
from analysis.utils.report_utils import natural_key as natural_keys


def read_log_summary(log_path):
    """Return (date, error_rate) without reading the whole log file."""
    date = ""
    error_rate = ""
    try:
        with open(log_path, "r") as f:
            first = f.readline()
            date = first.replace("Analysis completed: ", "").strip()
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - 4096), os.SEEK_SET)
            tail = f.read()
        for line in reversed(tail.splitlines()):
            if "Error rate:" in line:
                try:
                    error_rate = round(float(line.split(":")[-1].strip()), 4)
                except ValueError:
                    error_rate = ""
                break
    except OSError:
        pass
    return date, error_rate


def scan_analysis_plants(analysis_root):
    """
    Walk Analysis/*/*/*/* and return row tuples:

    (experiment, rpi, camera, plant, plate_condition, extra_variable,
     error_rate, status, date, active_path, plant_slot)
    """
    rows = []
    if not analysis_root or not os.path.isdir(analysis_root):
        return rows

    pathlib_dir = pathlib.Path(analysis_root)
    plant_slots = sorted(
        pathlib_dir.glob("*/*/*/*"), key=lambda p: natural_keys(str(p))
    )
    plant_slots = [str(p) for p in plant_slots if p.is_dir()]

    for plant_slot in plant_slots:
        rel_path = os.path.relpath(plant_slot, analysis_root)
        split = rel_path.split(os.path.sep)
        if len(split) < 4:
            continue

        experiment = convertFromPathSafe(split[0])
        rpi = split[1]
        camera = split[2]
        plant = split[3]

        result_dir = get_latest_result_dir(plant_slot)
        if result_dir is None:
            status = "Not finished"
            date = ""
            error_rate = ""
            plate_condition = ""
            extra_variable = ""
            active_path = plant_slot
        else:
            meta = load_result_metadata(result_dir)
            plate_condition = normalize_factor_value(meta.get("PlateCondition", ""))
            extra_variable = normalize_factor_value(meta.get("ExtraVariable", ""))
            active_path = result_dir

            log_path = os.path.join(result_dir, "log.txt")
            if os.path.exists(log_path):
                date, error_rate = read_log_summary(log_path)
                status = "Finished"
            else:
                date = ""
                error_rate = ""
                status = "Not finished"

        rows.append(
            (
                experiment,
                rpi,
                camera,
                plant,
                plate_condition,
                extra_variable,
                error_rate,
                status,
                date,
                active_path,
                plant_slot,
            )
        )
    return rows
