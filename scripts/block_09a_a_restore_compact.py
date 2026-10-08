#!/usr/bin/env python3
"""DCCT-NIDS Block 09A-A: reconstruct ignored Compact model-ready Parquet splits.

Uses original Block 04/06 duplicate definition, Block 05A taxonomy,
Block 06 chronological protocol, and original frozen Block 08A SQL.
Does NOT refit or recompute preprocessing parameters. No model training.
See validation report: passing counts/labels does not certify byte-identical
outputs or identical internal row-id hashes.
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
CACHE = ROOT / "data" / "model_ready" / "compact"
WORK = ROOT / "data" / "interim"
TABLE = ROOT / "outputs" / "tables"
METRICS = ROOT / "outputs" / "metrics"
MANIFESTS = ROOT / "outputs" / "manifests"
REPORTS = ROOT / "reports" / "experiment_notes"
EXPECTED_RAW = 6_659_532
EXPECTED_DEDUP = 6_319_955
BLOCK = "09A-A"


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

    database = WORK / "block_09a_a_compact_restoration.duckdb"
    if reconstruct:
        require(not database.exists(), f"Leftover recovery DuckDB file exists: {database}. Investigate before retry.")
        free_gb = shutil.disk_usage(ROOT).free / (1024 ** 3)
        print(f"Available working storage: {free_gb:.1f} GiB", flush=True)
        require(free_gb >= 7, "At least 7 GiB free space is recommended for full-row deduplication.")

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
                 "reconstructed_retained": retained_by_file.get(order, 0),
                 "difference": retained_by_file.get(order, 0) - expected_retained_by_file[order]}
                for order in range(1, 11)
            ]
            dedup_audit_path = TABLE / "table_09a_a_00_dedup_retention_diagnostics.csv"
            with dedup_audit_path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(diagnostics[0]))
                writer.writeheader()
                writer.writerows(diagnostics)
            if retained != EXPECTED_DEDUP or any(row["difference"] != 0 for row in diagnostics):
                print("RECOVERY DEDUPLICATION DIAGNOSTICS:", flush=True)
                for row in diagnostics:
                    if row["difference"] != 0:
                        print(f"  capture {row['file_order']:02d}: expected "
                              f"{row['original_retained']:,}, observed "
                              f"{row['reconstructed_retained']:,}, delta "
                              f"{row['difference']:+,}", flush=True)
                raise RuntimeError(
                    f"Deduplication does not match Block 04: {retained:,} "
                    f"versus {EXPECTED_DEDUP:,}. Diagnostic saved at "
                    f"{dedup_audit_path}; do not proceed to ML training."
                )
            print("  ✓ All 10 per-file retained counts match authoritative Block 04", flush=True)
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
            require(observed_per_file == original_per_file,
                    "Full-row deduplication or family provenance does not match Block 06 "
                    "(per-file/per-family counts differ).")

            print("Phase 4/4: Execute frozen Block 08A SQL and materialize three splits.", flush=True)
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
        print("All three Compact Parquets already present; auditing existing cache without overwriting it.", flush=True)

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
        require(row_ids[2] == n and row_ids[3] == 0,
                f"{split}: duplicate or missing row IDs")
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

    require(actual_group_counts == expected_group_counts,
            "Critical: regenerated fine-grained class distribution differs from the original Block 08A."
            " Do not train until discrepancy is resolved.")
    ranges = sorted((r["row_id_min"], r["row_id_max"], r["split"]) for r in audits)
    for prev, nxt in zip(ranges, ranges[1:]):
        require(prev[1] < nxt[0], f"Overlapping row-id ranges: {prev[2]} and {nxt[2]}")
    require(sum(r["rows"] for r in audits) == EXPECTED_DEDUP, "Total rows across splits mismatch")
    require(all(r["rows"] > 0 for r in audits), "Empty split detected")

    audit_csv = TABLE / "table_09a_a_01_compact_restoration_integrity.csv"
    with audit_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(audits[0]))
        writer.writeheader()
        writer.writerows(audits)

    class_csv = TABLE / "table_09a_a_02_restored_family_class_distribution.csv"
    with class_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["split", "attack_family", "fine_grained_label", "rows"])
        for (split, family, fine), n in sorted(actual_group_counts.items()):
            writer.writerow([split, family, fine, n])

    matching = sum(r["bytewise_matches_block_08a"] for r in audits)
    status = "bytewise_verified" if matching == 3 else "logical_counts_and_schema_verified_bytewise_unmatched"
    manifest = {
        "block": BLOCK, "status": status, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(DATA), "original_raw_rows": EXPECTED_RAW,
        "original_retained_rows": EXPECTED_DEDUP,
        "reconstruction_ran": reconstruct,
        "canonical_feature_count": len(features),
        "dedup_columns": len(canonical),
        "original_sql_and_config_sha256": {
            key: hash_file(path) for key, path in source_paths.items()
        },
        "outputs": audits,
        "all_family_class_counts_match_block_08a": True,
        "all_per_file_family_counts_match_block_06": bool(reconstruct),
        "row_id_reconstruction": "1-based global ROW_NUMBER over (file_order, ingestion_order)",
        "row_identity_verified_against_original_block_08a": matching == 3,
        "publication_caution": "If restored Parquet SHA-256 differs from Block 08A, "
        "column schema, class counts and finite values are verified but identical row_id "
        "assignment, internal holdout membership, float bits and file serialization "
        "are NOT proven. Preserve this distinction in cross-model comparisons.",
        "next_block": "09A-B",
        "elapsed_seconds": time.monotonic() - start,
    }
    manifest_path = MANIFESTS / "block_09a_a_compact_restoration_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    summary_path = METRICS / "metric_09a_a_compact_restoration_summary.json"
    summary_path.write_text(json.dumps({
        "block": BLOCK, "status": status, "total_rows": EXPECTED_DEDUP,
        "original_checksum_matches": matching, "splits": audits
    }, indent=2) + "\n", encoding="utf-8")

    report_path = REPORTS / "block_09a_a_compact_restoration.md"
    report_path.write_text(
        "# Block 09A-A — Compact dataset restoration\n\n"
        f"- Reconstructed from Kaggle raw data: **{reconstruct}**\n"
        f"- Observed split class distributions match the original Block 08A exactly: **yes**\n"
        f"- Canonical 39-column schema and finite features: **pass**\n"
        f"- Reproduced 6,319,955 deduplicated flows: **yes**\n"
        f"- Original file bytes match: **{matching}/3**\n"
        "- Frozen Block 08A preprocessing SQL used as-is; no validation/test fitting.\n"
        "- The global flow ID is recreated from chronological file order and "
        "per-file ingestion order. Identical internal holdout membership is "
        "not established unless original split file hashes match.\n\n"
        "## Recovered splits\n\n"
        + "| Split | Rows | SHA-256 original match |\n|---|---:|---|\n"
        + "".join(f"| {r['split']} | {r['rows']:,} | {r['bytewise_matches_block_08a']} |\n" for r in audits)
        + "\n## Scientific caution\n\n"
        "The original raw dataset contains capture-specific artifacts, and the prior "
        "training-internal holdout was derived after fitting the full-training "
        "preprocessing/feature policy. Do not treat internal nearly perfect scores "
        "as independent zero-day generalization evidence. Official test remains untouched.\n",
        encoding="utf-8",
    )
    for path in [audit_csv, class_csv, manifest_path, summary_path, report_path, TABLE / "table_09a_a_00_dedup_retention_diagnostics.csv"]:
        require(path.is_file() and path.stat().st_size > 0, f"Missing output: {path}")

    print("\n" + "=" * 86)
    print("DCCT-NIDS BLOCK 09A-A — DATASET RESTORATION COMPLETE")
    print("=" * 86)
    print(f"Logical integrity: verified; fine-grained labels: exact")
    print(f"Original full-Parquet byte SHA matches: {matching}/3")
    print(f"Restore status: {status}")
    print(f"Runtime: {manifest['elapsed_seconds'] / 60:.2f} minutes")
    print("Manifest:", manifest_path.relative_to(ROOT))
    print("Large Parquet files remain Git-ignored; lightweight audit artifacts are ready to push.")
    print("=" * 86)


if __name__ == "__main__":
    main()
