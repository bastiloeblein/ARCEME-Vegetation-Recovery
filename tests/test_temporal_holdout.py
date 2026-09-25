"""Regression tests for the fixed chronological holdout protocol."""

from __future__ import annotations

import csv
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from model.cv_splits import create_chronological_holdout, create_spacetime_folds


def _write_metadata(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = ["DisNo.", "year", "koppen_geiger", "pheno_season_name"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _path(root: Path, cube_id: str) -> str:
    return str(root / f"{cube_id}_postprocessed.zarr")


class ChronologicalHoldoutTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.metadata_path = self.root / "metadata.csv"
        self.rows = [
            {
                "DisNo.": "2099-A",
                "year": 2019,
                "koppen_geiger": "Cfa",
                "pheno_season_name": "Summer",
            },
            {
                "DisNo.": "1900-B",
                "year": 2020,
                "koppen_geiger": "BWh",
                "pheno_season_name": "Hotter",
            },
            {
                "DisNo.": "1900-C",
                "year": 2021,
                "koppen_geiger": "Cfb",
                "pheno_season_name": "Autumn",
            },
            {
                "DisNo.": "1900-D",
                "year": 2022,
                "koppen_geiger": "Aw",
                "pheno_season_name": "Wet",
            },
            {
                "DisNo.": "1900-E",
                "year": 2023,
                "koppen_geiger": "Dfb",
                "pheno_season_name": "Winter",
            },
        ]
        _write_metadata(self.metadata_path, self.rows)
        self.paths = [_path(self.root, row["DisNo."]) for row in self.rows]

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_uses_metadata_years_and_returns_disjoint_chronological_ranges(self) -> None:
        result = create_chronological_holdout(
            list(reversed(self.paths)),
            self.metadata_path,
            train_end_year=2021,
            validation_years=[2023, 2022],
        )

        fold = result["folds"][0]
        train_files = fold["train_files"]
        validation_files = fold["val_files"]

        self.assertEqual(train_files, sorted(self.paths[:3]))
        self.assertEqual(validation_files, sorted(self.paths[3:]))
        self.assertTrue(set(train_files).isdisjoint(validation_files))
        self.assertIn(_path(self.root, "2099-A"), train_files)
        self.assertEqual(fold["num_train"], 3)
        self.assertEqual(fold["num_val"], 2)

        metadata = result["metadata"]
        self.assertEqual(metadata["split_type"], "temporal_holdout")
        self.assertEqual(metadata["train_years"], [2019, 2020, 2021])
        self.assertEqual(metadata["validation_years"], [2022, 2023])
        self.assertEqual(metadata["year_counts"], {
            "2019": 1,
            "2020": 1,
            "2021": 1,
            "2022": 1,
            "2023": 1,
        })

    def test_is_deterministic_and_independent_of_input_order(self) -> None:
        first = create_chronological_holdout(
            self.paths,
            self.metadata_path,
            train_end_year=2021,
            validation_years=[2022, 2023],
        )
        second = create_chronological_holdout(
            [self.paths[2], self.paths[4], self.paths[0], self.paths[3], self.paths[1]],
            self.metadata_path,
            train_end_year="2021",
            validation_years=["2023", "2022"],
        )

        self.assertEqual(first, second)

    def test_counts_only_quality_filtered_paths(self) -> None:
        rejected_path = self.paths[1]
        eligible_paths = [path for path in self.paths if path != rejected_path]

        result = create_chronological_holdout(
            eligible_paths,
            self.metadata_path,
            train_end_year=2021,
            validation_years=[2022, 2023],
        )

        fold = result["folds"][0]
        self.assertEqual(result["metadata"]["total_cubes"], 4)
        self.assertEqual(fold["num_train"], 2)
        self.assertEqual(fold["num_val"], 2)
        self.assertNotIn(rejected_path, fold["train_files"] + fold["val_files"])

    def test_rejects_overlapping_or_nonchronological_years(self) -> None:
        with self.assertRaisesRegex(ValueError, "not chronological"):
            create_chronological_holdout(
                self.paths,
                self.metadata_path,
                train_end_year=2022,
                validation_years=[2022, 2023],
            )

        with self.assertRaisesRegex(ValueError, "at least one year"):
            create_chronological_holdout(
                self.paths,
                self.metadata_path,
                train_end_year=2021,
                validation_years=[],
            )

    def test_rejects_unassigned_years_and_empty_subsets(self) -> None:
        with self.assertRaisesRegex(ValueError, "outside the configured temporal split"):
            create_chronological_holdout(
                self.paths,
                self.metadata_path,
                train_end_year=2020,
                validation_years=[2022, 2023],
            )

        validation_only = self.paths[3:]
        with self.assertRaisesRegex(ValueError, "training subset is empty"):
            create_chronological_holdout(
                validation_only,
                self.metadata_path,
                train_end_year=2021,
                validation_years=[2022, 2023],
            )

        training_only = self.paths[:3]
        with self.assertRaisesRegex(ValueError, "validation subset is empty"):
            create_chronological_holdout(
                training_only,
                self.metadata_path,
                train_end_year=2021,
                validation_years=[2022, 2023],
            )

    def test_rejects_missing_metadata_and_duplicate_cube_ids(self) -> None:
        unknown = _path(self.root, "UNKNOWN")
        with self.assertRaisesRegex(ValueError, "No metadata row"):
            create_chronological_holdout(
                self.paths + [unknown],
                self.metadata_path,
                train_end_year=2021,
                validation_years=[2022, 2023],
            )

        duplicate_id_paths = [
            str(self.root / "2099-A.zarr"),
            str(self.root / "2099-A_postprocessed.zarr"),
            *self.paths[1:],
        ]
        with self.assertRaisesRegex(ValueError, "same ARCEME event ID"):
            create_chronological_holdout(
                duplicate_id_paths,
                self.metadata_path,
                train_end_year=2021,
                validation_years=[2022, 2023],
            )

    def test_existing_climate_grouped_split_remains_group_disjoint(self) -> None:
        rows = []
        paths = []
        for index, climate in enumerate(["Af", "Af", "BWh", "BWh", "Cfb", "Cfb"]):
            cube_id = f"cube-{index}"
            rows.append(
                {
                    "DisNo.": cube_id,
                    "year": 2020,
                    "koppen_geiger": climate,
                    "pheno_season_name": "Summer",
                }
            )
            paths.append(_path(self.root, cube_id))
        metadata_path = self.root / "grouped_metadata.csv"
        _write_metadata(metadata_path, rows)

        with redirect_stdout(io.StringIO()):
            result = create_spacetime_folds(
                paths,
                metadata_path,
                spacevar="koppen_geiger",
                timevar=None,
                k=3,
                seed=777,
                show=False,
            )
        path_to_climate = {
            _path(self.root, row["DisNo."]): row["koppen_geiger"] for row in rows
        }

        validation_union = set()
        for fold in result["folds"]:
            train_climates = {path_to_climate[path] for path in fold["train_files"]}
            validation_climates = {
                path_to_climate[path] for path in fold["val_files"]
            }
            self.assertTrue(train_climates.isdisjoint(validation_climates))
            validation_union.update(fold["val_files"])
        self.assertEqual(validation_union, set(paths))


if __name__ == "__main__":
    unittest.main()
