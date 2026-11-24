# CASG: Coverage-Aware Structured Grading

Code and data accompanying the paper *Coverage-Aware Structured Grading: Embedding
Institutional Distribution Constraints in LLM Scoring*.

CASG makes institutional score-distribution requirements part of the scoring
objective. Each question is decomposed into scoring points and sub-points, an LLM
assesses how much of each sub-point a response covers, a power function with
exponent alpha maps that coverage to credit, and alpha is fitted by continuous
optimization against a loss defined on the institutional targets for difficulty P,
discrimination D, and standard deviation sigma.

## Repository layout

```
code/
  scripts/           scoring pipeline, alpha optimization, cross-class and cross-major runs
    b1/              cross-class transfer and recalibration (Class B1)
    food/            cross-major validation (Food Engineering, Class A1)
  prompts/           few-shot baseline prompt (FEW-LLM)
  loss_configs/      loss definitions as JSON, selected with --config
  figures/           scripts that generate every figure in the paper
data/
  raw/               questions, rubrics, keypoint weights, anonymized score tables
  results/           intermediate and final result files
  food_class/        cross-major alpha sweep, aggregate only
```

## Key entry points

| Purpose | Command |
|---|---|
| Main experiment, alpha optimization | `python code/scripts/alpha_per_question_analysis.py --config pdsigma01` |
| Two-scale comparison over five losses | `python code/scripts/casg_renormalized_compare.py` |
| Cross-class transfer and recalibration | `python code/scripts/compute_b1_table5_renorm16.py` |
| Few-shot baseline on Class B1 | `python code/scripts/run_b1_llm_v2_baseline.py` |
| Cross-major alpha sweep | `python code/scripts/food/build_alpha_sweep_table.py` |
| Figures | `python code/scripts/figures/plot_alpha_tuning.py` and siblings |

The reported exponent for the development course is alpha* = 0.8246 with
L(alpha*) = 0.4375, computed on the renormalized 16-question weights. The
cross-class run gives alpha*_B1 = 0.5846 after recalibration, and the cross-major
run selects alpha* = 0 on the short-answer scale.

## Reproduced values

Running the entry points above on the data shipped here yields:

| Setting | Result |
|---|---|
| Development course, renormalized weights | alpha* = 0.8246, L = 0.4375, P = 0.2000, D = 0.1754, sigma = 7.769, r = 0.6511, QWK10 = 0.6012, MAE = 5.485 |
| Class B1, transfer at alpha*_B2 | P = 0.236, D = 0.191, sigma = 8.33, r = 0.659, MAE = 5.90 |
| Class B1, recalibrated at alpha*_B1 | alpha* = 0.5846, P = 0.200, D = 0.175, sigma = 7.80, r = 0.674, QWK10 = 0.554, MAE = 4.63 |
| Food Engineering, short-answer scale at alpha = 0 | P = 0.426, D = 0.603, sigma = 23.53, r = 0.931, QWK10 = 0.900, MAE = 7.24 |

Note that `code/scripts/alpha_sweep_from_xlsx.py` doubles as a library
(`parse_xlsx`, `load_deps`, `compute_casg`, `metrics_for`). Its own `main()` uses a
legacy absolute-deviation loss that yields alpha = 1.8559, which is not the result
reported in the paper. Reproduce paper numbers with the relative-deviation
configurations in `code/loss_configs/`, in particular `pdsigma01.json`.

## Data availability and privacy

`data/raw/` contains questions, rubrics, scoring-point definitions, and score
tables in which every student appears only as a sequential pseudonym of the form
`S001`. The pseudonym key is held by the authors and is not published.

Student free-text answers are deliberately excluded from this repository. The
answer texts, which would be required to re-run the coverage assessment from
scratch, cannot be shared publicly because they contain identifiable coursework.
They are available upon reasonable request from the corresponding author, subject
to institutional approval for data sharing, as stated in the paper. All aggregate
metrics, per-question results, and alpha sweeps needed to check every number in
the paper are included.

## License

Code is released under the MIT License, see `LICENSE`. Data files are released
under CC BY 4.0, except where the privacy statement above applies.