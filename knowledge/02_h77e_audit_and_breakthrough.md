# Correction and withdrawal: earlier H77E analysis draft

> **This document's previous contents are withdrawn as an unreliable historical draft. Do not cite its causal or leaderboard claims.** It asserted that the H33 TIFF itself was organizer-confirmed at 0.2778 and stated multiple detailed family-wide mechanisms and quantitative results as established facts without adequate source-to-claim verification. It also contained a mathematical claim that binary sparse rasters universally dominate calibrated probability maps; that assertion is not established by the supplied metric definition and is withdrawn.

## Correct evidence status

- The public DrivenData leaderboard snapshot available to this repository showed **0.2778 ORGANIZER-CONFIRMED for participant `extradr19`'s leaderboard row** and **0.4413 ORGANIZER-CONFIRMED for the first-place row**. A row score does not identify an uploaded filename or SHA-256.
- The attribution of 0.2778 to `GEMSDOE32`'s named `h33-h33-2-b2-20261004T220000Z-e5eb6e7e-zeros.tif` remains **OWNER-REPORT**. The GEMSDOE32 site states it has no organizer receipt for that file. No file-to-score receipt is present here.
- The H33 TIFF's file properties and dot construction are forensic measurements; they do not establish why any leaderboard entry scored as it did or the location of hidden faults.
- The H77E and later lane holdout results are local hide-and-recover measurements. They are not competition scores and must always be presented with evaluator, withheld-positive count, and 95% CI.

## Replacement materials

- Corrected H33 metric/mechanism analysis with provenance caveats: [`research/why-h33-scored-02778.md`](../research/why-h33-scored-02778.md)
- H82 new-hypothesis ranking, official-source links, and obtainability limits: [`research/h82_hypotheses.md`](../research/h82_hypotheses.md)
- H82 negative lane report: [`docs/h82.html`](../docs/h82.html)
- H82 full evidence: [`evidence/h82_run_card.json`](../evidence/h82_run_card.json)

## Official links reviewed

- Competition task and metric: <https://www.drivendata.org/competitions/306/competition-doe-gems/page/967/>
- Public leaderboard (scores identify leaderboard rows, not source TIFFs): <https://www.drivendata.org/competitions/306/competition-doe-gems/leaderboard/>
- GEMSDOE32 page cited by the owner: <https://buffedlizard55-lab.github.io/GEMSDOE32/docs/index.html>

**Disposition:** H33's marginal-coverage/thinning explanation is plausible under a distance-tolerant metric, but neither file-to-score attribution nor a causal explanation is proven. H82 is negative and not approved to submit. No claim that GEMSDOE77 will beat an organizer score is warranted.
