# public_datasets/

Supplementary training volume — real, published, peer-reviewed datasets,
clearly labelled as distinct from the team's own self-collected traces
(which live in `../raw_traces/`, never here).

| Dataset | Provides |
|---|---|
| OxIOD | 158 sequences, 42.5 km / 14.7 hrs, 5 users, 4 phone placements, Vicon ground truth |
| RoNIN | 42+ hrs, 100 subjects, 3 Android device types, placement-invariant |
| TLIO | 400 sequences, ~60 hrs, headset-mounted IMU, high-rate ground truth |
| RIDI | 10 subjects, 4 placements, Visual-Inertial SLAM ground truth |
| Google Smartphone Decimeter Challenge | Real smartphone GNSS raw measurements + IMU |

Download links and per-dataset notes (license, format, subset used) belong
here once each dataset is pulled in.

## OxIOD — how to get it

Hosted at http://deepio.cs.ox.ac.uk/ (Oxford).

1. Open the page and find the **"Dataset (1.01G)"** link — it opens a
   Google Form (`forms.gle/wjE7u5AonoyyrgXJ7`) asking for basic details
   (name / email / intended use). This is Oxford's standard access-request
   step, not a paid license — no one on the team has submitted it yet.
2. Submit the form; the download link is emailed back (check spam).
3. Unzip into `public_datasets/oxiod/` — organized by
   participant/phone-placement folders, each with synced IMU + Vicon
   ground-truth CSVs (its own schema, not the `ml-residual/` data contract
   below — write a small adapter if/when this is wired into training).
4. Cite in the dossier appendix / final report:
   - Chen et al., *"OxIOD: The Dataset for Deep Inertial Odometry,"* arXiv:1809.07491.
   - Chen et al., *"Deep Learning based Pedestrian Inertial Navigation: Methods, Dataset and On-Device Inference,"* IEEE Internet of Things Journal.

## RoNIN / TLIO / RIDI / Google Smartphone Decimeter Challenge

Not yet requested/downloaded. Add a subsection here (same shape as OxIOD's above)
once you pull one in — link, access process, target subfolder, citation.

## Status

- [ ] OxIOD requested
- [ ] OxIOD downloaded → `public_datasets/oxiod/`
- [ ] RoNIN
- [ ] TLIO
- [ ] RIDI
- [ ] Google Smartphone Decimeter Challenge
