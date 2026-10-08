#!/usr/bin/env python3
"""DCCT-NIDS Block 09A-A-R2: versioned reconstruction of revised Compact cohort.

Preserve historical Block 08A files and counts unchanged. This versioned dataset
uses the current 10 Parquet inputs, original 78-column exact chronological
deduplication SQL and original frozen training-fitted preprocessing SQL.
Historical source 6,319,955 flows; current cohort 6,320,003 flows.
Validate expected +45/+2/+1 capture-file drifts and split-specific counts.
The reconstructed cohort is scientifically NON-EQUIVALENT to the historical
tree-baseline cohort unless row-level equality can independently be proven.
No training, no credential handling, no Git actions.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
DATA = Path("/kaggle/input/datasets/dhoogla/csecicids2018")
CACHE = ROOT / "data" / "model_ready" / "compact_r2_6320003"
WORK = ROOT / "data" / "interim"
TABLE = ROOT / "outputs" / "tables"
METRICS = ROOT / "outputs" / "metrics"
MANIFESTS = ROOT / "outputs" / "manifests"
REPORTS = ROOT / "reports" / "experiment_notes"
EXPECTED_RAW = 6_659_532
EXPECTED_DEDUP = 6_320_003
ORIGINAL_DEDUP = 6_319_955
EXPECTED_DELTAS = {2: 45, 6: 2, 7: 1}
EXPECTED_SPLITS = {"train": 4_202_458, "validation": 1_145_712, "test": 971_833}
BLOCK = "09A-A-R2"


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def hash_file(path):
    result = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def quote(identifier):
    return '"' + str(identifier).replace('"', '""') + '"'


def sql_literal(value):
    return "'" + str(value).replace("'", "''") + "'"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    start = time.monotonic()
    for directory in [CACHE, WORK, TABLE, METRICS, MANIFESTS, REPORTS]:
        directory.mkdir(parents=True, exist_ok=True)

    source_paths = {
        "dedup_sql": ROOT / "scripts/block_04_exact_deduplication.sql",
        "canonical_schema": ROOT / "configs/datasets/cse_cic_ids2018_canonical_schema.json",
        "original_dedup_by_file": ROOT / "outputs/tables/table_04_01_exact_deduplication_by_file.csv",
        "provenance_sql": ROOT / "scripts/block_06_reconstruct_provenance.sql",
        "transform_sql": ROOT / "scripts/block_08a_compact_preprocessing.sql",
        "tax": ROOT / "configs/datasets/cse_cic_ids2018_attack_taxonomy.json",
        "preprocessor": ROOT / "configs/preprocessing/block_08a_compact_strict_preprocessor.json",
        "features": ROOT / "configs/datasets/cse_cic_ids2018_features_compact.json",
        "chronology": ROOT / "outputs/tables/table_06_03_strict_chronological_file_split.csv",
        "canonical_distribution": ROOT / "outputs/tables/table_08a_03_compact_split_class_distribution.csv",
        "prior_checksums": ROOT / "outputs/manifests/block_08a_compact_split_checksums.json",
        "prior_families": ROOT / "outputs/tables/table_06_01_chronological_family_distribution.csv",
    }
    for name, path in source_paths.items():
        require(path.is_file(), f"Missing authoritative source {name}: {path}")

    schema_sql = source_paths["dedup_sql"].read_text(encoding="utf-8")
    prefix = schema_sql.split("\nFROM (", 1)[0]
    canonical = re.findall(r'"([^"]+)"', prefix)
    require(len(canonical) == 78 and len(set(canonical)) == 78, f"Expected 78 canonical columns; got {len(canonical)}")
    require(canonical[-1] == "Label", "Dedup source must end with Label")
    canonical_schema = load_json(source_paths["canonical_schema"])
    canonical_types = {item["column_name"]: item["canonical_type"]
                       for item in canonical_schema["columns"]}
    require(set(canonical_types) == set(canonical), "Canonical schema mismatch")
    require(sum(typ == "int64" for typ in canonical_types.values()) == 40,
            "Expected 40 canonical integer columns")
    require(sum(typ in ("double", "float64") for typ in canonical_types.values()) == 37,
            "Expected 37 canonical double columns")
    require(canonical_types["Label"] == "string", "Expected string label schema")

    expected_retained_by_file = {}
    with open(source_paths["original_dedup_by_file"], newline="", encoding="utf-8") as stream:
        for line in csv.DictReader(stream):
            expected_retained_by_file[int(line["file_order"])] = int(line["retained_rows"])
    require(len(expected_retained_by_file) == 10, "Incomplete original per-file retention audit")

    taxonomy = load_json(source_paths["tax"])["taxonomy"]
    require("Benign" in taxonomy and len(taxonomy) == 15, "Unexpected taxonomy")
    features = load_json(source_paths["features"])["features"]
    fitted = load_json(source_paths["preprocessor"])
    require(len(features) == 32 and features == fitted["features"], "Compact features/preprocessing order mismatch")
    expected_fitting_rows = int(fitted["fitting_scope"]["training_rows"])
    require(expected_fitting_rows == 4_202_411, "Unexpected frozen fitting-row scope")

    source_files = []
    with open(source_paths["chronology"], newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            path = DATA / row["file_name"]
            require(path.is_file(), f"Missing Kaggle input file: {path}")
            source_files.append({
                "file_order": int(row["file_order"]),
                "source_file": row["file_name"],
                "capture_date": row["capture_date"],
                "split": row["strict_chronological_split"],
                "expected_raw": int(row["raw_rows"]),
                "path": path,
            })
    source_files.sort(key=lambda x: x["file_order"])
    require([x["file_order"] for x in source_files] == list(range(1, 11)),
            "Expected 10 chronologically ordered raw files")

    expected_group_counts = {}
    with open(source_paths["canonical_distribution"], newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            key = (row["split"], row["attack_family"], row["fine_grained_label"])
            expected_group_counts[key] = int(row["row_count"])
    reference_sha = {
        x["split"]: x["sha256"] for x in load_json(source_paths["prior_checksums"])["files"]
    }
    original_per_file = {}
    with open(source_paths["prior_families"], newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            key = (int(row["file_order"]), row["attack_family"])
            original_per_file[key] = int(row["row_count"])

    expected_columns = [
        "row_id", "file_order", "source_file", "capture_date",
        "fine_grained_label", "attack_family", "binary_label", *features
    ]
    require(len(expected_columns) == 39, "Unexpected output column count")

    # In a new Kaggle session all three files are missing. If all three
    # already exist we audit them first, rather than replace valuable data.
    existing = {s: CACHE / f"{s}_compact.parquet" for s in ("train", "validation", "test")}
    present = {s: path.is_file() for s, path in existing.items()}
    if any(present.values()) and not all(present.values()):
        raise RuntimeError(
            "Partial model-ready cache detected. Preserve it and remove/relocate "
            "the incomplete outputs before a clean, all-split reconstruction."
        )
    reconstruct = not all(present.values())
    raw_sources = []

    database = WORK / "block_09a_a_r2_versioned_compact.duckdb"
    if reconstruct:
        require(not database.exists(), f"Leftover recovery DuckDB file exists: {database}. Investigate before retry.")
        free_gb = shutil.disk_usage(ROOT).free / (1024 ** 3)
        print(f"Available working storage: {free_gb:.1f} GiB", flush=True)
        require(free_gb >= 7, "At least 7 GiB free space is recommended for full-row deduplication.")
        print("Versioned cohort destination:", CACHE, flush=True)
        print("Original Block 08A Compact directory will not be modified.", flush=True)

        db = duckdb.connect(str(database))
        try:
            db.execute("SET preserve_insertion_order = true")
            db.execute("SET threads = 4")
            db.execute("SET memory_limit = '6GB'")
            print("Phase 1/4: Stream ten raw Parquet files into aligned DuckDB columns.", flush=True)
            for i, meta in enumerate(source_files, 1):
                arrow_schema = pq.ParquetFile(meta["path"]).schema_arrow
                input_names = arrow_schema.names
                require(len(input_names) == 78 and set(input_names) == set(canonical),
                        f"Raw schema names do not match frozen 78-column definition: {meta['source_file']}")
                actual_raw = int(pq.ParquetFile(meta["path"]).metadata.num_rows)
                require(actual_raw == meta["expected_raw"],
                        f"Raw-row mismatch in {meta['source_file']}: {actual_raw} != {meta['expected_raw']}")
                # Match the frozen Block 02A/02B canonical schema precisely:
                # int64 for integer-only columns, float64 for mixed and float
                # columns, and dictionary-decoded VARCHAR for labels. The
                # previous all-DOUBLE cast was NOT the original schema.
                cast_for = {"int64": "BIGINT", "double": "DOUBLE",
                            "float64": "DOUBLE", "string": "VARCHAR"}
                expressions = [
                    f"CAST({quote(column)} AS {cast_for[canonical_types[column]]}) "
                    f"AS {quote(column)}"
                    for column in canonical
                ]
                raw_sources.append({
                    "file_order": meta["file_order"],
                    "file_name": meta["source_file"],
                    "sha256": hash_file(meta["path"]),
                    "size_bytes": meta["path"].stat().st_size,
                    "raw_rows": actual_raw,
                })
                path_sql = sql_literal(meta["path"].as_posix())
                source_sql = (
                    "SELECT " + ", ".join(expressions) +
                    f", {meta['file_order']}::INTEGER AS file_order" +
                    f", {sql_literal(meta['source_file'])}::VARCHAR AS source_file" +
                    f", {sql_literal(meta['capture_date'])}::DATE AS capture_date" +
                    ", ROW_NUMBER() OVER ()::BIGINT AS ingestion_order" +
                    f" FROM read_parquet({path_sql})"
                )
                if i == 1:
                    db.execute("CREATE TABLE raw_with_provenance AS " + source_sql)
                else:
                    db.execute("INSERT INTO raw_with_provenance " + source_sql)
                print(f"  [{i:02d}/10] {meta['source_file']}: {actual_raw:,} rows", flush=True)

            raw_count = int(db.execute("SELECT COUNT(*) FROM raw_with_provenance").fetchone()[0])
            require(raw_count == EXPECTED_RAW, f"Unexpected raw count: {raw_count:,}")
            print("Phase 2/4: Apply the committed full 78-column chronological deduplication.", flush=True)
            provenance_sql = source_paths["provenance_sql"].read_text(encoding="utf-8")
            # The original SQL is the authoritative duplicate definition;
            # its partition and retention order are executed without changes.
            db.execute(provenance_sql)
            retained = int(db.execute("SELECT COUNT(*) FROM retained_with_provenance").fetchone()[0])
            retained_by_file = {
                int(order): int(count)
                for order, count in db.execute(
                    "SELECT file_order, COUNT(*) FROM retained_with_provenance "
                    "GROUP BY file_order ORDER BY file_order"
                ).fetchall()
            }
            diagnostics = [
                {"file_order": order,
                 "original_retained": expected_retained_by_file[order],
                 "current_retained": retained_by_file.get(order, 0),
                 "difference": retained_by_file.get(order, 0) - expected_retained_by_file[order],
                 "expected_difference": EXPECTED_DELTAS.get(order, 0)}
                for order in range(1, 11)
            ]
            dedup_audit_path = TABLE / "table_09a_a_r2_01_capture_retention_deltas.csv"
            with dedup_audit_path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(diagnostics[0]))
                writer.writeheader()
                writer.writerows(diagnostics)
            print("Current capture-row differences from historical Block 04:", flush=True)
            for row in diagnostics:
                if row["difference"]:
                    print(f"  capture {row['file_order']:02d}: "
                          f"original {row['original_retained']:,}; "
                          f"current {row['current_retained']:,}; "
                          f"delta {row['difference']:+,}", flush=True)
            require(retained == EXPECTED_DEDUP,
                    f"Current cohort unexpectedly contains {retained:,}, "
                    f"expected {EXPECTED_DEDUP:,}. No outputs materialized.")
            require(all(row["difference"] == row["expected_difference"] for row in diagnostics),
                    "One or more capture-file count differences changed since "
                    "Diagnostic 02/03; refusing to materialize revised cohort.")
            require(sum(row["difference"] for row in diagnostics)
                    == EXPECTED_DEDUP - ORIGINAL_DEDUP,
                    "Current vs original cohort drift failed accounting")
            print("  ✓ All per-capture deltas match independent diagnostics.", flush=True)
            db.execute("DROP TABLE raw_with_provenance")
            db.execute("CHECKPOINT")
            print(f"  Retained {retained:,}; removed {raw_count - retained:,} duplicates", flush=True)

            print("Phase 3/4: Reconstruct labels, temporal splits, and global flow-row IDs.", flush=True)
            family_case = "CASE CAST(\"Label\" AS VARCHAR) " + " ".join(
                f"WHEN {sql_literal(raw)} THEN {sql_literal(t['attack_family'])}"
                for raw, t in taxonomy.items()
            ) + " ELSE NULL END"
            bin_case = "CASE CAST(\"Label\" AS VARCHAR) " + " ".join(
                f"WHEN {sql_literal(raw)} THEN {sql_literal(t['binary_label'])}"
                for raw, t in taxonomy.items()
            ) + " ELSE NULL END"
            split_case = "CASE " + " ".join(
                f"WHEN file_order = {m['file_order']} THEN {sql_literal(m['split'])}"
                for m in source_files
            ) + " ELSE NULL END"
            db.execute(
                "CREATE VIEW compact_aligned AS SELECT "
                "ROW_NUMBER() OVER (ORDER BY file_order, ingestion_order)::BIGINT AS row_id, "
                "file_order, source_file, capture_date, "
                "CAST(\"Label\" AS VARCHAR) AS fine_grained_label, "
                f"{family_case} AS attack_family, "
                f"{bin_case} AS binary_label, "
                f"{split_case} AS split, "
                + ", ".join(quote(c) for c in canonical if c != "Label")
                + " FROM retained_with_provenance"
            )
            per_file = db.execute(
                "SELECT file_order, attack_family, COUNT(*) AS rows "
                "FROM compact_aligned GROUP BY file_order, attack_family "
                "ORDER BY file_order, attack_family"
            ).fetchall()
            observed_per_file = {(int(f), str(a)): int(n) for f, a, n in per_file}
            require(all(a != "None" for _, a in observed_per_file),
                    "One or more rows have unmapped attack-family labels.")
            family_delta_records = [
                {
                    "file_order": f,
                    "attack_family": a,
                    "original_rows": original_per_file.get((f, a), 0),
                    "current_rows": observed_per_file.get((f, a), 0),
                    "difference": observed_per_file.get((f, a), 0)
                                  - original_per_file.get((f, a), 0),
                }
                for (f, a) in sorted(set(observed_per_file) | set(original_per_file))
            ]
            require(sum(x["difference"] for x in family_delta_records) == 48,
                    "Unexpected cross-file family count drift")
            family_delta_path = TABLE / "table_09a_a_r2_02_capture_family_deltas.csv"
            with family_delta_path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(family_delta_records[0]))
                writer.writeheader()
                writer.writerows(family_delta_records)
            print("  ✓ All family labels valid; explicit historical differences recorded.", flush=True)

            print("Phase 4/4: Apply frozen Block 08A transforms to the NEW cohort.", flush=True)
            transform_sql = source_paths["transform_sql"].read_text(encoding="utf-8")
            # No hard-coded transform values: use the original 32-feature SQL verbatim.
            select_sql = transform_sql[transform_sql.index("\nSELECT\n") + 1:].strip().rstrip(";")
            require("FROM compact_aligned" in select_sql and "'<train|validation|test>'" in select_sql,
                    "Unexpected authoritative Block 08A SQL template.")

            for split, dest in existing.items():
                query = select_sql.replace("'<train|validation|test>'", sql_literal(split))
                temporary = dest.with_suffix(".parquet.part")
                require(not temporary.exists(), f"Leftover temporary export exists: {temporary}")
                db.execute(
                    f"COPY ({query}) TO {sql_literal(temporary.as_posix())} "
                    "(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)"
                )
                require(temporary.stat().st_size > 0, f"Empty export: {temporary}")
                temporary.replace(dest)
                print(f"  Saved {split}: {dest.stat().st_size / 2**20:.2f} MiB", flush=True)
        finally:
            db.close()
            if database.exists():
                database.unlink()
            wal = database.with_suffix(database.suffix + ".wal")
            if wal.exists():
                wal.unlink()
    else:
        print("All three versioned Compact Parquets already present; auditing without overwriting.", flush=True)
        for meta in source_files:
            raw_sources.append({
                "file_order": meta["file_order"],
                "file_name": meta["source_file"],
                "sha256": hash_file(meta["path"]),
                "size_bytes": meta["path"].stat().st_size,
                "raw_rows": meta["expected_raw"],
            })
        dedup_audit_path = TABLE / "table_09a_a_r2_01_capture_retention_deltas.csv"
        family_delta_path = TABLE / "table_09a_a_r2_02_capture_family_deltas.csv"
        require(dedup_audit_path.exists() and family_delta_path.exists(),
                "Existing R2 data have no matching provenance audit. "
                "Do not label them verified.")

    # Independent verification from exported Parquet files.
    print("Validating restored Parquet row counts, exact class distributions, schemas, and finite values...", flush=True)
    audit_db = duckdb.connect(":memory:")
    audit_db.execute("SET threads = 4")
    actual_group_counts = {}
    audits = []
    for split, dest in existing.items():
        pf = pq.ParquetFile(dest)
        names = pf.schema_arrow.names
        require(names == expected_columns,
                f"{split} column order mismatch. Actual: {names}; expected: {expected_columns}")
        n = pf.metadata.num_rows
        require(n == EXPECTED_SPLITS[split],
                f"{split}: current-cohort rows {n:,} != required {EXPECTED_SPLITS[split]:,}")
        finite_conditions = [
            f"SUM(CASE WHEN {quote(f)} IS NULL OR NOT isfinite(CAST({quote(f)} AS DOUBLE)) THEN 1 ELSE 0 END)"
            for f in features
        ]
        raw_path = sql_literal(dest.as_posix())
        counts = audit_db.execute(
            "SELECT attack_family, fine_grained_label, COUNT(*) "
            f"FROM read_parquet({raw_path}) GROUP BY 1, 2"
        ).fetchall()
        for family, fine, count in counts:
            actual_group_counts[(split, str(family), str(fine))] = int(count)
        invalid_counts = audit_db.execute(
            "SELECT " + ", ".join(finite_conditions) +
            f" FROM read_parquet({raw_path})"
        ).fetchone()
        invalid_cells = sum(int(x or 0) for x in invalid_counts)
        row_ids = audit_db.execute(
            f"SELECT MIN(row_id), MAX(row_id), COUNT(DISTINCT row_id), "
            f"COUNT(*) - COUNT(DISTINCT row_id) FROM read_parquet({raw_path})"
        ).fetchone()
        mapping_invalid = int(audit_db.execute(
            f"SELECT COUNT(*) FROM read_parquet({raw_path}) "
            "WHERE attack_family IS NULL OR binary_label IS NULL "
            "OR fine_grained_label IS NULL "
            "OR (fine_grained_label = 'Benign' AND binary_label != 'Benign') "
            "OR (fine_grained_label != 'Benign' AND binary_label != 'Attack')"
        ).fetchone()[0])
        require(mapping_invalid == 0,
                f"{split}: {mapping_invalid} missing/invalid label mappings")
        require(row_ids[2] == n and row_ids[3] == 0,
                f"{split}: duplicate or missing row IDs")
        require(row_ids[1] - row_ids[0] + 1 == n,
                f"{split}: row IDs are not contiguous")
        require(invalid_cells == 0, f"{split}: {invalid_cells} non-finite cells")
        digest = hash_file(dest)
        audits.append({
            "split": split, "rows": n, "columns": len(names),
            "row_id_min": row_ids[0], "row_id_max": row_ids[1],
            "invalid_feature_cells": invalid_cells, "file_size_bytes": dest.stat().st_size,
            "sha256": digest, "prior_sha256": reference_sha[split],
            "bytewise_matches_block_08a": digest == reference_sha[split],
        })
        print(f"  {split}: {n:,} rows; invalid cells={invalid_cells}; "
              f"original byte SHA {'MATCH' if digest == reference_sha[split] else 'DIFF'}", flush=True)
    audit_db.close()

    require(set(actual_group_counts) == set(expected_group_counts),
            "Class/split membership changed; review current input revision before training.")
    group_deltas = [
        {
            "split": split,
            "attack_family": family,
            "fine_grained_label": fine,
            "historical_rows": expected_group_counts[(split, family, fine)],
            "current_rows": actual_group_counts[(split, family, fine)],
            "difference": actual_group_counts[(split, family, fine)]
                          - expected_group_counts[(split, family, fine)],
        }
        for split, family, fine in sorted(actual_group_counts)
    ]
    require(
        {s: sum(x["difference"] for x in group_deltas if x["split"] == s)
         for s in EXPECTED_SPLITS}
        == {"train": 47, "validation": 1, "test": 0},
        "Unexpected class-wise split-count differences from historical cohort"
    )
    require(all(x["difference"] == 0 for x in group_deltas if x["split"] == "test"),
            "TEST CLASS COUNTS changed despite equal total. Refusing to proceed.")
    group_delta_path = TABLE / "table_09a_a_r2_04_fine_grained_class_deltas.csv"
    with group_delta_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(group_deltas[0]))
        writer.writeheader()
        writer.writerows(group_deltas)
    ranges = sorted((r["row_id_min"], r["row_id_max"], r["split"]) for r in audits)
    for prev, nxt in zip(ranges, ranges[1:]):
        require(prev[1] < nxt[0], f"Overlapping row-id ranges: {prev[2]} and {nxt[2]}")
    require(sum(r["rows"] for r in audits) == EXPECTED_DEDUP, "Total rows across splits mismatch")
    require(ranges == [(1, EXPECTED_SPLITS["train"], "train"),
                       (EXPECTED_SPLITS["train"]+1,
                        EXPECTED_SPLITS["train"]+EXPECTED_SPLITS["validation"], "validation"),
                       (EXPECTED_SPLITS["train"]+EXPECTED_SPLITS["validation"]+1,
                        EXPECTED_DEDUP, "test")],
            f"Unexpected global row-ID boundaries: {ranges}")
    require(all(r["rows"] > 0 for r in audits), "Empty split detected")

    audit_csv = TABLE / "table_09a_a_r2_03_versioned_split_integrity.csv"
    with audit_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(audits[0]))
        writer.writeheader()
        writer.writerows(audits)

    class_csv = TABLE / "table_09a_a_r2_05_versioned_class_distribution.csv"
    with class_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["split", "attack_family", "fine_grained_label", "rows"])
        for (split, family, fine), n in sorted(actual_group_counts.items()):
            writer.writerow([split, family, fine, n])

    matching = sum(r["bytewise_matches_block_08a"] for r in audits)
    status = "verified_distinct_cohort_not_historical_matched"
    manifest = {
        "block": BLOCK, "status": status, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(DATA), "original_raw_rows": EXPECTED_RAW,
        "historical_retained_rows": ORIGINAL_DEDUP,
        "current_retained_rows": EXPECTED_DEDUP,
        "delta_from_historical": EXPECTED_DEDUP - ORIGINAL_DEDUP,
        "split_row_counts": EXPECTED_SPLITS,
        "dataset_revision": "current_kaggle_6320003_frozen_08a_transform",
        "raw_file_checksums_sha256": raw_sources,
        "strict_temporal_protocol_unchanged": True,
        "reconstruction_ran": reconstruct,
        "canonical_feature_count": len(features),
        "dedup_columns": len(canonical),
        "original_sql_and_config_sha256": {
            key: hash_file(path) for key, path in source_paths.items()
        },
        "outputs": audits,
        "historical_fine_grained_class_delta_audited": True,
        "all_per_file_deltas_match_diagnostics": True,
        "row_id_reconstruction": "1-based global ROW_NUMBER over (file_order, ingestion_order)",
        "row_identity_verified_against_original_block_08a": False,
        "identical_cohort_to_historical_tree_baselines": False,
        "frozen_preprocessor_fitting_scope": "historical Block 08A training cohort only",
        "publication_caution": "NEW COHORT: full-row chronological deduplication "
        "yields 48 extra records. Counts/splits/labels/features validated only "
        "for current Kaggle inputs. NO matched-cohort performance inference "
        "against 08B/08C/08D permitted without retraining those baselines. "
        "Original Block 08A training-fitted preprocessing parameters reused "
        "unaltered; they are not refitted on the new cohort.",
        "next_block": "09A-B-R2",
        "elapsed_seconds": time.monotonic() - start,
    }
    manifest_path = MANIFESTS / "block_09a_a_r2_versioned_compact_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    summary_path = METRICS / "metric_09a_a_r2_versioned_compact_summary.json"
    summary_path.write_text(json.dumps({
        "block": BLOCK, "status": status, "total_rows": EXPECTED_DEDUP, "historical_rows": ORIGINAL_DEDUP,
        "identical_to_historical_cohort": False, "splits": audits
    }, indent=2) + "\n", encoding="utf-8")

    report_path = REPORTS / "block_09a_a_r2_versioned_compact.md"
    report_path.write_text(
        "# Block 09A-A-R2 — Versioned Compact reconstruction\n\n"
        f"- Status: **{status}**\n"
        "- Current Kaggle raw input: 6,659,532 flows (10 capture files)\n"
        "- Exact 78-column chronological deduplication: "
        f"**{EXPECTED_DEDUP:,} retained flows**\n"
        f"- Historical Block 08A: **{ORIGINAL_DEDUP:,} retained flows**\n"
        "- Delta: **+48** (file 02 +45, file 06 +2, file 07 +1)\n"
        "- Changed split sizes: train **4,202,458**, validation **1,145,712**, "
        "test **971,833**\n"
        "- Canonical 39 columns and finite Compact features: **validated**\n"
        "- Test fine-grained class counts versus original: **unchanged**\n"
        "- 32 features use exact frozen Block 08A transformation SQL: **yes**\n"
        "- New dataset path: `data/model_ready/compact_r2_6320003/`\n"
        "- Raw Parquet SHA256 signatures recorded in versioned JSON manifest.\n"
        "- Historical feature preprocessing stats **not refitted**: "
        "original train-only fitted statistics reused as documented.\n\n"
        "## Exported splits\n\n"
        "| Split | Rows | Historical same-file SHA match |\n"
        "|---|---:|---|\n"
        + "".join(f"| {r['split']} | {r['rows']:,} | "
                  f"{r['bytewise_matches_block_08a']} |\n" for r in audits)
        + "\n## Scientific interpretation\n\n"
        "**This is not the historical tree-baseline cohort.** "
        "Data input revisions or old reconstruction details are unresolved. "
        "Matching exact test class counts does not prove equal row identities. "
        "Do NOT interpret new MLP results as a controlled paired comparison "
        "against historical 08B–08E tree results. For matched comparisons, "
        "retrain all baselines on this revision with a newly fitted training-only "
        "preprocessor, or recover and verify the original exact data cohort.\n",
        encoding="utf-8",
    )
    required_outputs = [
        audit_csv, class_csv, group_delta_path, family_delta_path,
        dedup_audit_path, manifest_path, summary_path, report_path
    ]
    for path in required_outputs:
        require(path.is_file() and path.stat().st_size > 0,
                f"Missing report or table: {path}")
    print("\n" + "=" * 86)
    print("DCCT-NIDS BLOCK 09A-A-R2 — VERSIONED COHORT COMPLETE")
    print("=" * 86)
    print(f"New strict-temporal cohort: {EXPECTED_DEDUP:,} rows")
    print("Split counts: 4,202,458 / 1,145,712 / 971,833")
    print("Original baseline data retained: 6,319,955 (not overwritten)")
    print(f"Original full-Parquet byte SHA matches: {matching}/3")
    print(f"Validation status: {status}")
    print(f"Runtime: {manifest['elapsed_seconds'] / 60:.2f} minutes")
    print("Manifest:", manifest_path.relative_to(ROOT))
    print("Large Parquet files are Git-ignored; NO Git push performed.")
    print("WARNING: historical tree baselines are NOT cohort-matched.")
    print("=" * 86)


if __name__ == "__main__":
    main()
