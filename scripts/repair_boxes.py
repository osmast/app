#!/usr/bin/env python3
"""Report (and optionally fix) bounding boxes that leave their image.

    python scripts/repair_boxes.py                          # every dataset, dry run
    python scripts/repair_boxes.py --dataset bb              # one dataset, dry run
    python scripts/repair_boxes.py --dataset bb --apply      # write the fix

A box is stored in source-image pixels. A corner outside ``0..width`` /
``0..height`` can only come from a drag that left the frame, which the canvas used
to allow - the labeller now pulls the corner back onto the border, so no *new*
box can be stored that way, but a corpus labelled before the fix still carries
them (in ``datasets/bb``: 11 of 526).

Dry run is the default because this rewrites ground truth: nothing is deleted,
the corners are clamped onto the image, and any box that would collapse to
nothing is listed for a manual re-draw instead. Re-export the dataset afterwards
(``python run.py dataset`` -> *Export dataset...*) so the trainer sees the
repaired coordinates.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from daphnia import paths
from daphnia.db import connection, queries
from daphnia.maintenance import clamp_corpus_boxes


def _datasets(conn, name=None) -> list:
    """The datasets to inspect: one by name, or every one there is."""
    rows = [dict(row) for row in queries.list_datasets(conn)]
    if name:
        wanted = str(name).strip().lower()
        rows = [row for row in rows
                if str(row.get("name", "")).strip().lower() == wanted
                or str(row.get("dataset_id")) == name]
        if not rows:
            known = ", ".join(str(row.get("name")) for row in
                              queries.list_datasets(conn))
            raise SystemExit(f"No dataset named '{name}'. Known: {known or 'none'}")
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Report and clamp bounding boxes that leave their image.")
    parser.add_argument("--dataset", help="dataset name or id (default: all)")
    parser.add_argument("--apply", action="store_true",
                        help="write the clamped corners (default: dry run)")
    parser.add_argument("--db", default=str(paths.DB_PATH),
                        help=f"database to open (default: {paths.DB_PATH})")
    args = parser.parse_args(argv)

    with connection(args.db) as conn:
        datasets = _datasets(conn, args.dataset)
        if not datasets:
            print("No datasets in this database.")
            return 0

        total_found = 0
        for dataset in datasets:
            result = clamp_corpus_boxes(
                conn, dataset["dataset_id"], apply=args.apply)
            found = result["found"]
            total_found += len(found)
            print(f"\n{dataset['name']} ({dataset['dataset_id']})")
            if not found:
                print("  all bounding boxes are inside their image.")
                continue
            for item in found:
                before = ", ".join(f"{value:.1f}" for value in item["before"])
                after = ", ".join(f"{value:.1f}" for value in item["after"])
                print(f"  {Path(item['file_path']).name}"
                      f"  specimen {item['specimen_index']}"
                      f"   [{before}]  ->  [{after}]")
            print(f"  {len(found)} box(es) outside their image, "
                  f"{len(result['degenerate'])} too small to clamp.")
            if result["degenerate"]:
                print("  Re-draw these by hand in the Dataset surface:")
                for item in result["degenerate"]:
                    print(f"    {Path(item['file_path']).name}"
                          f" specimen {item['specimen_index']}")

        verb = "clamped" if args.apply else "would be clamped"
        print(f"\n{total_found} box(es) found; {verb} on --apply."
              + ("" if args.apply else
                 "  Nothing was written - pass --apply to fix them."))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
