#!/usr/bin/env python3
"""Deterministically generate data/courses.xlsx for the Excel Narrator Agent demo.

No randomness: the rows below are fixed so the classroom demo always gets the
same answers. Notably, "Advanced Robotics" has the highest enrollment (58
students) of any row in the Enrollments sheet -- the demo query "Which course
has the highest enrollment?" relies on that being a crisp, unique fact.

Run:
    .venv/bin/python scripts/make_sample_xlsx.py
"""

from __future__ import annotations

from pathlib import Path

from openpyxl import Workbook

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = REPO_ROOT / "data" / "courses.xlsx"

ENROLLMENTS_HEADERS = [
    "Course",
    "Instructor",
    "Quarter",
    "Students Enrolled",
    "Satisfaction",
]
ENROLLMENTS_ROWS = [
    ["Robotics 101", "Alex Chen", "2025 Q1", 42, 4.5],
    ["Advanced Robotics", "Priya Nair", "2025 Q1", 39, 4.6],
    ["Python for Kids", "Jordan Lee", "2025 Q1", 48, 4.7],
    ["Intro to 3D Printing", "Sam Okafor", "2025 Q2", 33, 4.3],
    ["Circuit Design Basics", "Jordan Lee", "2025 Q2", 29, 4.2],
    ["Robotics 101", "Alex Chen", "2025 Q2", 45, 4.6],
    ["Advanced Robotics", "Priya Nair", "2025 Q3", 58, 4.9],  # highest enrollment
    ["Python for Kids", "Jordan Lee", "2025 Q3", 51, 4.8],
    ["Intro to Arduino", "Sam Okafor", "2025 Q3", 27, 4.1],
    ["Wearable Tech Design", "Priya Nair", "2025 Q4", 34, 4.4],
    ["Intro to 3D Printing", "Sam Okafor", "2026 Q1", 36, 4.5],
    ["Advanced Robotics", "Priya Nair", "2026 Q2", 41, 4.7],
]

EQUIPMENT_HEADERS = ["Item", "Category", "Quantity", "Location"]
EQUIPMENT_ROWS = [
    ["3D Printer (Creality Ender 3)", "Fabrication", 6, "Innovation Lab"],
    ["micro:bit Kit", "Electronics", 24, "STEM Classroom A"],
    ["Soldering Station", "Electronics", 10, "Innovation Lab"],
    ["Robotics Kit (VEX IQ)", "Robotics", 15, "STEM Classroom B"],
    ["Laser Cutter", "Fabrication", 2, "Innovation Lab"],
    ["Raspberry Pi 5", "Computing", 20, "STEM Classroom A"],
    ["Safety Goggles", "Safety", 50, "Storage Closet"],
    ["Multimeter", "Electronics", 12, "Innovation Lab"],
]


def build_workbook() -> Workbook:
    wb = Workbook()

    enrollments = wb.active
    enrollments.title = "Enrollments"
    enrollments.append(ENROLLMENTS_HEADERS)
    for row in ENROLLMENTS_ROWS:
        enrollments.append(row)

    equipment = wb.create_sheet("Equipment")
    equipment.append(EQUIPMENT_HEADERS)
    for row in EQUIPMENT_ROWS:
        equipment.append(row)

    return wb


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    workbook = build_workbook()
    workbook.save(OUTPUT_PATH)
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
