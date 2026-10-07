"""Ordered, versioned model features; legacy CSV datasets retain schema 1."""
from config import FEATURE_NAMES
from features import extract_features
from local_features import LOCAL_FEATURE_NAMES, extract_local_features

FEATURE_SCHEMAS = {1: FEATURE_NAMES, 2: FEATURE_NAMES + LOCAL_FEATURE_NAMES}
CURRENT_FEATURE_SCHEMA = 2


def feature_names(version):
    if version not in FEATURE_SCHEMAS:
        raise ValueError(f'Unsupported feature schema version: {version}')
    return list(FEATURE_SCHEMAS[version])


def analysis_features(signal, sampling_rate, version=1):
    feature_names(version)
    values = extract_features(signal, sampling_rate)
    if version == 2:
        values.update(extract_local_features(signal))
    return values
