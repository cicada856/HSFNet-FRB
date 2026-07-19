# Processed CHIME data

- `chime_interp_frbs_data.npz`: 535 independently normalized `256 x 256` waterfall arrays (`uint8`).
- `chime_labels.txt`: internal labels A/B/C/C1/C2 in array order.
- `chime_file_order.txt`: source event filenames only; local directory prefixes were removed.
- `chime_interp_frbs_msnr.txt`: accompanying S/N metadata retained from the project data package.

Class counts are A=163, B=270, C=62, C1=20, and C2=20. The arrays are derived from CHIME/FRB data products and are not covered by this repository's MIT code license.
