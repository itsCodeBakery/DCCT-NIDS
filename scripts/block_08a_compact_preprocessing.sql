-- DCCT-NIDS Block 8A Compact preprocessing
-- Generated: 2026-07-10T07:25:51.288861+00:00
--
-- All medians, quartiles, log1p decisions, centers, and scales
-- were fitted exclusively on the strict chronological training
-- partition.

SELECT
    row_id,
    file_order,
    source_file,
    capture_date,
    fine_grained_label,
    attack_family,
    binary_label,
    CAST(((ln(1.0 + (CASE WHEN "Fwd Header Length" IS NULL OR NOT isfinite(CAST("Fwd Header Length" AS DOUBLE)) THEN 72 ELSE CAST("Fwd Header Length" AS DOUBLE) END))) - 4.290459441148391) / 1.4163266482187655 AS FLOAT) AS "Fwd Header Length",
            CAST(((ln(1.0 + (CASE WHEN "Subflow Fwd Bytes" IS NULL OR NOT isfinite(CAST("Subflow Fwd Bytes" AS DOUBLE)) THEN 161 ELSE CAST("Subflow Fwd Bytes" AS DOUBLE) END))) - 5.0875963352323836) / 3.7970930387541695 AS FLOAT) AS "Subflow Fwd Bytes",
            CAST(((ln(1.0 + (CASE WHEN "Fwd Packet Length Mean" IS NULL OR NOT isfinite(CAST("Fwd Packet Length Mean" AS DOUBLE)) THEN 49.055557250976562 ELSE CAST("Fwd Packet Length Mean" AS DOUBLE) END))) - 3.9131335335829585) / 2.9189451510722715 AS FLOAT) AS "Fwd Packet Length Mean",
            CAST(((ln(1.0 + (CASE WHEN "Fwd Packet Length Max" IS NULL OR NOT isfinite(CAST("Fwd Packet Length Max" AS DOUBLE)) THEN 103 ELSE CAST("Fwd Packet Length Max" AS DOUBLE) END))) - 4.6443908991413725) / 3.4746248502169723 AS FLOAT) AS "Fwd Packet Length Max",
            CAST(((ln(1.0 + (CASE WHEN "Bwd Packet Length Mean" IS NULL OR NOT isfinite(CAST("Bwd Packet Length Mean" AS DOUBLE)) THEN 121.13636016845703 ELSE CAST("Bwd Packet Length Mean" AS DOUBLE) END))) - 4.8051381268634472) / 5.3981627015177525 AS FLOAT) AS "Bwd Packet Length Mean",
            CAST(((CASE WHEN "Bwd Packet Length Max" IS NULL OR NOT isfinite(CAST("Bwd Packet Length Max" AS DOUBLE)) THEN 302 ELSE CAST("Bwd Packet Length Max" AS DOUBLE) END) - 302) / 964 AS FLOAT) AS "Bwd Packet Length Max",
            CAST(((ln(1.0 + (CASE WHEN "Subflow Bwd Bytes" IS NULL OR NOT isfinite(CAST("Subflow Bwd Bytes" AS DOUBLE)) THEN 312 ELSE CAST("Subflow Bwd Bytes" AS DOUBLE) END))) - 5.7462031905401529) / 6.8721281013389861 AS FLOAT) AS "Subflow Bwd Bytes",
            CAST(((CASE WHEN "Init Fwd Win Bytes" IS NULL OR NOT isfinite(CAST("Init Fwd Win Bytes" AS DOUBLE)) THEN 8192 ELSE CAST("Init Fwd Win Bytes" AS DOUBLE) END) - 8192) / 7967 AS FLOAT) AS "Init Fwd Win Bytes",
            CAST(((CASE WHEN "Fwd Packet Length Std" IS NULL OR NOT isfinite(CAST("Fwd Packet Length Std" AS DOUBLE)) THEN 49.217205047607422 ELSE CAST("Fwd Packet Length Std" AS DOUBLE) END) - 49.217205047607422) / 222.62332153320312 AS FLOAT) AS "Fwd Packet Length Std",
            CAST(((ln(1.0 + (CASE WHEN "Bwd Header Length" IS NULL OR NOT isfinite(CAST("Bwd Header Length" AS DOUBLE)) THEN 92 ELSE CAST("Bwd Header Length" AS DOUBLE) END))) - 4.5325994931532563) / 2.5301632413761213 AS FLOAT) AS "Bwd Header Length",
            CAST(((CASE WHEN "Bwd Packet Length Std" IS NULL OR NOT isfinite(CAST("Bwd Packet Length Std" AS DOUBLE)) THEN 178.97857666015625 ELSE CAST("Bwd Packet Length Std" AS DOUBLE) END) - 178.97857666015625) / 406.1217041015625 AS FLOAT) AS "Bwd Packet Length Std",
            CAST(((CASE WHEN "Packet Length Std" IS NULL OR NOT isfinite(CAST("Packet Length Std" AS DOUBLE)) THEN 203.73719787597656 ELSE CAST("Packet Length Std" AS DOUBLE) END) - 203.73719787597656) / 315.16635227203369 AS FLOAT) AS "Packet Length Std",
            CAST(((ln(1.0 + (CASE WHEN "Packet Length Mean" IS NULL OR NOT isfinite(CAST("Packet Length Mean" AS DOUBLE)) THEN 102.42222595214844 ELSE CAST("Packet Length Mean" AS DOUBLE) END))) - 4.6388198901569533) / 1.5531981941112245 AS FLOAT) AS "Packet Length Mean",
            CAST(((ln(1.0 + (CASE WHEN "Avg Packet Size" IS NULL OR NOT isfinite(CAST("Avg Packet Size" AS DOUBLE)) THEN 122.75 ELSE CAST("Avg Packet Size" AS DOUBLE) END))) - 4.8182634014487995) / 1.3998081186896054 AS FLOAT) AS "Avg Packet Size",
            CAST(((CASE WHEN "Fwd IAT Mean" IS NULL OR NOT isfinite(CAST("Fwd IAT Mean" AS DOUBLE)) THEN 43396 ELSE CAST("Fwd IAT Mean" AS DOUBLE) END) - 43396) / 1105845.75 AS FLOAT) AS "Fwd IAT Mean",
            CAST(((CASE WHEN "Fwd IAT Total" IS NULL OR NOT isfinite(CAST("Fwd IAT Total" AS DOUBLE)) THEN 161896 ELSE CAST("Fwd IAT Total" AS DOUBLE) END) - 161896) / 4478520.5 AS FLOAT) AS "Fwd IAT Total",
            CAST(((CASE WHEN "Fwd IAT Max" IS NULL OR NOT isfinite(CAST("Fwd IAT Max" AS DOUBLE)) THEN 91911 ELSE CAST("Fwd IAT Max" AS DOUBLE) END) - 91911) / 4244315.5 AS FLOAT) AS "Fwd IAT Max",
            CAST(((ln(1.0 + (CASE WHEN "Total Backward Packets" IS NULL OR NOT isfinite(CAST("Total Backward Packets" AS DOUBLE)) THEN 3 ELSE CAST("Total Backward Packets" AS DOUBLE) END))) - 1.3862943611198906) / 0.916290731874155 AS FLOAT) AS "Total Backward Packets",
            CAST(((CASE WHEN "Packet Length Max" IS NULL OR NOT isfinite(CAST("Packet Length Max" AS DOUBLE)) THEN 788 ELSE CAST("Packet Length Max" AS DOUBLE) END) - 788) / 917 AS FLOAT) AS "Packet Length Max",
            CAST(((ln(1.0 + (CASE WHEN "Fwd Packets/s" IS NULL OR NOT isfinite(CAST("Fwd Packets/s" AS DOUBLE)) THEN 4.5388364791870117 ELSE CAST("Fwd Packets/s" AS DOUBLE) END))) - 1.7117844568941256) / 4.6921460178280991 AS FLOAT) AS "Fwd Packets/s",
            CAST(((CASE WHEN "Flow IAT Mean" IS NULL OR NOT isfinite(CAST("Flow IAT Mean" AS DOUBLE)) THEN 138156.671875 ELSE CAST("Flow IAT Mean" AS DOUBLE) END) - 138156.671875) / 692857.27075195312 AS FLOAT) AS "Flow IAT Mean",
            CAST(((CASE WHEN "Init Bwd Win Bytes" IS NULL OR NOT isfinite(CAST("Init Bwd Win Bytes" AS DOUBLE)) THEN 211 ELSE CAST("Init Bwd Win Bytes" AS DOUBLE) END) - 211) / 256 AS FLOAT) AS "Init Bwd Win Bytes",
            CAST(((ln(1.0 + (CASE WHEN "Bwd Packets/s" IS NULL OR NOT isfinite(CAST("Bwd Packets/s" AS DOUBLE)) THEN 2.8281521797180176 ELSE CAST("Bwd Packets/s" AS DOUBLE) END))) - 1.3423822271678221) / 4.2496372704284688 AS FLOAT) AS "Bwd Packets/s",
            CAST(((ln(1.0 + (CASE WHEN "Bwd IAT Min" IS NULL OR NOT isfinite(CAST("Bwd IAT Min" AS DOUBLE)) THEN 21 ELSE CAST("Bwd IAT Min" AS DOUBLE) END))) - 3.0910424533583161) / 6.8448154792082629 AS FLOAT) AS "Bwd IAT Min",
            CAST(((CASE WHEN "Flow Packets/s" IS NULL OR NOT isfinite(CAST("Flow Packets/s" AS DOUBLE)) THEN 8.1262824288999997 ELSE CAST("Flow Packets/s" AS DOUBLE) END) - 8.1262824288999997) / 390.08222367022063 AS FLOAT) AS "Flow Packets/s",
            CAST(((CASE WHEN "Fwd IAT Min" IS NULL OR NOT isfinite(CAST("Fwd IAT Min" AS DOUBLE)) THEN 80 ELSE CAST("Fwd IAT Min" AS DOUBLE) END) - 80) / 10089 AS FLOAT) AS "Fwd IAT Min",
            CAST(((ln(1.0 + (CASE WHEN "Flow IAT Std" IS NULL OR NOT isfinite(CAST("Flow IAT Std" AS DOUBLE)) THEN 17500.8046875 ELSE CAST("Flow IAT Std" AS DOUBLE) END))) - 9.7700592795945873) / 13.386686635252222 AS FLOAT) AS "Flow IAT Std",
            CAST(((ln(1.0 + (CASE WHEN "Bwd IAT Total" IS NULL OR NOT isfinite(CAST("Bwd IAT Total" AS DOUBLE)) THEN 13248 ELSE CAST("Bwd IAT Total" AS DOUBLE) END))) - 9.4916773568681236) / 13.989207109439725 AS FLOAT) AS "Bwd IAT Total",
            CAST(((ln(1.0 + (CASE WHEN "Bwd IAT Max" IS NULL OR NOT isfinite(CAST("Bwd IAT Max" AS DOUBLE)) THEN 11875 ELSE CAST("Bwd IAT Max" AS DOUBLE) END))) - 9.382274835883651) / 13.767485600949156 AS FLOAT) AS "Bwd IAT Max",
            CAST(((ln(1.0 + (CASE WHEN "Flow Bytes/s" IS NULL OR NOT isfinite(CAST("Flow Bytes/s" AS DOUBLE)) THEN 751.29465439379999 ELSE CAST("Flow Bytes/s" AS DOUBLE) END))) - 6.623128074856762) / 6.3829229501907818 AS FLOAT) AS "Flow Bytes/s",
            CAST(((ln(1.0 + (CASE WHEN "Bwd IAT Mean" IS NULL OR NOT isfinite(CAST("Bwd IAT Mean" AS DOUBLE)) THEN 5561.33349609375 ELSE CAST("Bwd IAT Mean" AS DOUBLE) END))) - 8.6237729927399691) / 12.584000979157588 AS FLOAT) AS "Bwd IAT Mean",
            CAST(((ln(1.0 + (CASE WHEN "Bwd IAT Std" IS NULL OR NOT isfinite(CAST("Bwd IAT Std" AS DOUBLE)) THEN 176.00852966308594 ELSE CAST("Bwd IAT Std" AS DOUBLE) END))) - 5.1761979215996456) / 12.617469656313903 AS FLOAT) AS "Bwd IAT Std"
FROM compact_aligned
WHERE split = '<train|validation|test>'
ORDER BY row_id;
