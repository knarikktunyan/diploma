# Real simulation validation — awaiting data

No identified SPICE/Synopsys simulation traces or circuit source files were found
in the workspace during this stage. All reported ML results remain synthetic.
Real-circuit performance has not been measured. The application already imports
generic CSV signals; no simulator connection or converter was added.

## Input contract

Export a simulation trace to CSV with numeric `time` in seconds and `amplitude`
in a documented, consistent physical unit. Values must be finite, time must be
strictly increasing and uniformly sampled, and schema-2 analysis needs at least
32 samples. Adaptive simulator timesteps will be rejected by FFT analysis;
arrange a documented uniform export/resampling procedure before import. Do not
silently remove noise, harmonic content or transient anomalies to satisfy a label.

The synthetic training population uses 500 samples at 100 kHz, frequencies around
800–1200 Hz and amplitude around 0.8–1.2. The reader accepts other uniform rates,
but acceptance is not evidence of generalization. Spectral energy depends on
record length, windowed features depend on record duration, and crossing periods
are unreliable when too few cycles are present. Keep units, duration and frequency
range comparable for an initial experiment; document every conversion.

## Provenance and ground truth

Preserve the original simulation outputs and record, alongside each trace:

- Simulator name/version, circuit identifier and source/netlist reference.
- Operating conditions, parameter/fault changes, sampling procedure and units.
- An independently justified class label; do not label from the model prediction.
- Circuit/run grouping, so correlated traces from the same circuit/run do not
  cross training and test boundaries if future real-data training is introduced.

Some circuit behavior may not fit these four synthetic classes. Record ambiguous
or mixed faults explicitly rather than forcing a convenient label. Ground truth
must come from the simulation setup and engineering assessment.

## Initial validation procedure

Use the GUI to load, analyze, inspect waveform/spectrum, classify with both models
and export. The same model can be checked independently:

```bash
venv/bin/python src/prediction.py /path/to/trace.csv --export /path/to/result.csv
venv/bin/python src/prediction.py /path/to/trace.csv \
  --model models/time_local/svm.pkl --export /path/to/svm_result.csv
```

Compare predictions with independently established labels. Report class support,
per-class precision/recall/F1, macro scores and confusion matrices on a frozen
cohort. Inspect failures and compare distributions against the synthetic training
population. Preserve the present synthetic baseline and models before adapting
anything. Do not tune on the real heldout cohort and then reuse its scores as an
unbiased final evaluation.

This procedure is ready to use when actual traces become available. It is not a
completed real-data validation result.
