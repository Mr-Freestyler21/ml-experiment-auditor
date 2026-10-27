# F1 demo project

The audit target for the main demo. It predicts whether a Formula 1 driver pits on the
next lap, from 52,471 laps of the 2023 and 2024 seasons.

**Provenance:** this project predates the hackathon. `train.py` is adapted from the
notebook in [f1-pit-stop-prediction](https://github.com/Mr-Freestyler21/f1-pit-stop-prediction),
keeping the original modelling code and random split unchanged so the auditor has the
original evaluation to examine. `data/laps.csv` is the F1 Strategy Dataset v2 from Kaggle,
copied unchanged from that repository.

Run it on its own:

```bash
python demo/f1/train.py
```
