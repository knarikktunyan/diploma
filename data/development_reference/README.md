Original 80-signal development reference (20 per class).

raw/, processed/, dataset.csv, feature_analysis/ and models/ preserve the original data and 90% holdout results. src/ preserves the previous generator and configuration. original_sha256.json records the original file checksums. This snapshot is synthetic, not circuit simulation output.

The original raw/normal folder also contained normal_001_analysis.csv, a user GUI export. It is preserved in its original relative location and included in the checksum manifest, but is not one of the 80 time/amplitude signal CSVs. When restoring the original raw dataset for batch processing, copy only the numbered signal CSVs.
