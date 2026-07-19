# Simulation data layout

The full simulation dataset is not bundled. The loader expects the following Frabjous-style structure:

```text
data/simulated/simfrb/
  simdata/
    type_A/SNR_15/simulatefrbs_type_A.npz
    type_A/SNR_25/simulatefrbs_type_A.npz
    ...
    type_C2/SNR_100/simulatefrbs_type_C2.npz
  remove_type_C1_SNR_<snr>.txt   # optional exclusion lists
  type_C2_SNR_<snr>.txt          # optional exclusion lists
```

Required classes are `A`, `B`, `C`, `C1`, and `C2`; required S/N levels are `15`, `25`, `35`, `50`, and `100`. Each NPZ file must contain a single array or use `arr_0`, with shape `(N, 256, 256)`.
