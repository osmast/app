# Accuracy — the error budget of one length

What a length from this app is worth, term by term: **every error source, its formula, its
magnitude on this rig, and what the app does about it.** The point of the document is a
single question asked in five ways —

> *I measured 2314 µm. How accurate is that? Which part of the error is the animal, which
> part is the optics, and which part is the model? And does the app know?*

The short answer, measured: **the reported `±` is the scale's relative error and nothing else
— and in neither path is it the largest term.** On the *Study* path (a human places both
landmarks) the **human's click** dominates: 73.4 µm against a 5.9 µm `±`, i.e. the true budget
is **13× the number on screen** (§3.1). On the *live Predict* path the **pose model** dominates:
224 µm against a 10 µm `±`, i.e. **23×** (§3.2) — its error is measured offline by
`scripts/train_rtmpose.py --evaluate` and is deliberately **not** added to the number on screen;
the surface says so in words (`pages/predict.py::_pose_summary`, pinned by
`tests/test_predict.py`).

**Changed 2026-10-22:** the app now reports the total *beside* the `±` on every measuring
surface (`units.total_uncertainty_um` → `total +/- …`), stores it with the result
(`corpus_labels.total_uncertainty_um`) and exports it. So the first half of that short answer
still holds — the `±` is the scale term alone — but the budget no longer has to be derived by
hand to be seen. What is still **not** in the number is the pose model's own error (S8, named
as a separate number), the repeat spread (S7, its own label on Predict) and everything §6
lists.

Rules this document obeys (`memory-bank/mlContext.md`): **law 5** — a number is reported with
its provenance or not at all; **law 8** — a failure is named, never silent; **§8.4** — say
what was *not* measured out loud. This file is the error *ledger*; `docs/calibration.md` is
how the scale is obtained, `docs/rtmdet.md` §5 and `docs/rtmpose.md` §3 are the model
metrics, and `memory-bank/mlContext.md` is the discipline behind all three.

---

## 0. The budget at a glance

Symbols are defined in §1. Every percentage here is **relative** — it scales with the animal —
unless the column says *absolute*.

| # | Source | Kind | Formula | Worth on this rig | What the app does |
| --- | --- | --- | --- | --- | --- |
| **S1** | Scale from a **clicked span** | systematic, relative | $\delta = \dfrac{2c}{s}\cdot 100$, $c = 1.0\ \text{px}$ | $14.7\ \text{px} \to \pm 13.61\,\%$; $147\ \text{px} \to \pm 1.36\,\%$ | derived, stored, `WEAK` above $2\,\%$, shown **with the route** it came from |
| **S2** | Scale from the **printed lattice** | systematic, relative | $\delta_{\text{fit}} = \dfrac{\operatorname{median}(\varepsilon_b)}{\bar{p}}\cdot 100$, $\varepsilon_b = \dfrac{1.4826\,\mathrm{MAD}}{\sqrt{k_b}}$ | $\delta_{\text{fit}} = 2.49\,\%$ on the eyepiece photo | derived, stored, displayed |
| **S3** | The **print itself** | systematic, relative | $\delta_{\text{paper}} = 0.5\,\%$ | the floor under every grid calibration | combined **in quadrature** with S2 |
| **S4** | **Drift / distortion** | systematic, **position-dependent** | $\dfrac{\max_i \lvert p_i - \bar{p}\rvert}{\bar{p}}\cdot 100$ | $5.1$–$6.3\,\%$, i.e. $102$–$126\ \text{µm}$ on $2\ \text{mm}$ | its **own line**, never in the `±`; a *model* stays *unmeasured* |
| **S5** | **Quantisation** (the pixel grid) | random, **absolute** | $\sigma_{\text{quant}} = \dfrac{0.408}{S}$ | $36.7\ \text{µm}/\text{px} \to \pm 15.0\ \text{µm}$ | **not** reported; derivable from $S$ alone |
| **S6** | **Clicking** the measured line | random, relative | $\dfrac{u_{\text{click}}}{L} = \dfrac{2c}{s}$ | $u_{\text{click}} = \dfrac{2c}{S} = 73.4\ \text{µm}$ | **not** reported on a measurement |
| **S7** | **Repeatability** (rotation, pose jitter) | random, relative | $sd = \sqrt{\dfrac{\sum (L_i - \bar{L})^2}{n-1}}$, $SE = \dfrac{sd}{\sqrt{n}}$ | live; on Predict only | measured and displayed: `mean ±SE (n=…)` |
| **S8** | **Pose model** | mixed, **heavy-tailed** | $d_k = \lVert \hat{p}_k - g_k \rVert$, then PCK@$t$ and OKS | median $3.2\ \text{px}$, p90 $37.5\ \text{px}$, PCK@5px $0.67$, OKS $0.56$; length error median $3.0\ \text{px} \approx 227\ \text{µm}$ | measured **offline**; the surface names it a *separate number* |
| **S9** | **Detector** | mixed | greedy $\mathrm{IoU} \ge 0.5$ match, then $P$, $R$, $F_1$, $\mathrm{AP}_{50}$ | $P\,0.958 / R\,0.852 / F_1\,0.902 / \mathrm{AP}_{50}\,0.848$ at $\ge 0.35$ | shown live (boxes, score, threshold); **not** propagated |
| **S10** | **Age** of the scale (a knock) | systematic, **step in time** | not a σ — a flag on $t > 12\ \text{h}$ | the live profile is $29\ \text{h}$ old | `RE-VERIFY` in every window's footer |
| **S11** | **Frame quality** (focus, motion) | selection bias | $q = \operatorname{Var}(\nabla^2 I)$, keep $\arg\max_i q_i$ | picks the best of a burst | sharpest frame kept; a *recording* replaces the lottery |
| **S12** | **Identity** (which animal) | association error | associate iff $\mathrm{IoU} \ge 0.30$; innermost box owns the midpoint | a swap is *visible* | exposed as a jump in $sd$; unattached rows listed as *nobody* |
| **S13** | **Reference mis-declared** (wrong pitch) | systematic, **factor** error | $S \propto 1/p$ | $0.5\ \text{mm}$ filed as $1\ \text{mm} \to 2\times$ every length | one pitch field in four places, stored as `target_type` + `actual_distance_um`; $>2\times$ jump needs confirmation |
| **S14** | **Plausibility** (a number that cannot be an animal) | — | **no gate exists** | $18$ rows read $6.6$–$56\ \text{mm}$; $440$ read over $4\ \text{mm}$ (§5 d) | **nothing flags them** |


---

## 1. How the terms combine — two rules, and the one that is missing

### Symbols used from here on

| Symbol | Meaning | Where it lives |
| --- | --- | --- |
| $L$ | the length, in **µm** — the number being reported | `corpus_labels.length_um` |
| $s$ | the same length, in **pixels** | `corpus_labels.length_px` |
| $S$ | the scale actually applied, in **px/µm** | `corpus_labels.px_per_um` |
| $\delta$ | the relative scale uncertainty, in **percent** | `corpus_labels.uncertainty_pct` |
| $p$ | the printed pitch of the reference target, in µm | `grid_pitch_um()` → `actual_distance_um` |
| $c$ | how well one landmark can be placed, in pixels | `CLICK_ERROR_PX = 1.0` |
| $n$ | how many repeats a mean rests on | `Track.count` |
| $\sigma$ | a standard deviation (1σ) | — |
| $u$ | an uncertainty carried *into* a result | — |

**Rule 1 — the scale error is relative, so it is not a constant number of µm.** The scale is
what turns pixels into µm, so an error on it multiplies the animal:

$$L \;=\; \frac{s}{S} \qquad\Longrightarrow\qquad \Delta L \;=\; L \cdot \frac{\delta}{100}$$

For $\delta = \pm 0.50\,\%$ that is $\pm 5\,\text{µm}$ on a 1 mm animal and $\pm 10\,\text{µm}$
on a 2 mm one — so a `±` quoted **without the length it belongs to** is meaningless, and every
readout prints the pair together: `2314.0 +/- 11.6 um (2.314 mm)`
(`daphnia/units.uncertainty_um`, `OpticalProfile.uncertainty_um`, `.range_um`).

**Rule 2 — independent terms combine in quadrature:**

$$u \;=\; \sqrt{\,u_1^2 + u_2^2 + \dots\,}$$

which is exactly what the code already does for the scale's two components:
`np.hypot(statistical_pct, paper_tolerance_pct)` (`vision/lattice.py`, `vision/grid.py`,
`vision/points.py` — all three). Quadrature is valid only for **independent** terms: the
print's stretch and the fit's noise are independent; the drift is a function of *where you
looked*, so it is kept out of the sum.

**What was missing — the total, now reported (2026-10-22).** Until this round the app reported
**one** term of a five-term budget. The complete combination is

$$u_{\text{total}}(L) \;=\; \sqrt{\,u_{\text{scale}}^2 + u_{\text{click}}^2 + \sigma_{\text{quant}}^2 + u_{\text{model}}^2 + u_{\text{repeat}}^2\,}$$

and it now exists in code: `units.total_uncertainty_um(um, px_per_um, uncertainty_pct, *,
extra_um=0.0)` (`daphnia/units.py`, with `OpticalProfile.total_uncertainty_um` as the
object-shaped door). Three terms it derives itself, from what the row already carries; the
other two are passed in by a caller that measured them (`extra_um`), because they are not
properties of the scale.

| Term | Formula | Source | Reported today? |
| --- | --- | --- | --- |
| $u_{\text{scale}}$ | $L\cdot\dfrac{\delta}{100}$ | S1–S3 | **yes** — this *is* the `±` on screen |
| $u_{\text{click}}$ | $L\cdot\dfrac{2c}{s} = \dfrac{2c}{S}$ | S6 | **yes** — inside `total +/-`, on every measuring screen |
| $\sigma_{\text{quant}}$ | $\dfrac{0.41}{S}$ | S5 | **yes** — inside `total +/-` |
| $u_{\text{model}}$ | the pose model's length error | S8 | no (offline only; pass it as `extra_um`, and **off** the click term) |
| $u_{\text{repeat}}$ | $\dfrac{sd}{\sqrt{n}}$ | S7 | only on Predict, and as a *separate* label (`extra_um` when combining) |

Where it is shown, stored and exported: the readout is
`2314.0 +/- 6.8 um (2.314 mm)   total +/- 75.2 um (3.2 %)` (Bench, Dataset, Annotate/Study and
the Calibrate panel's example line, all through `units.format_total_uncertainty`); the number
is written by the one write path as `corpus_labels.total_uncertainty_um` and travels in
`labels.csv`/`labels.json`/the Bench CSV and in the corpus export; and a label written before
the column existed has it derived from the row's own scale, its `profile_id`'s immutable
calibration, or not at all (`connection.backfill_total_uncertainty`, which names what it
changed).

Three honest caveats about that sum. **(a)** $u_{\text{model}}$ is **not** a σ: the per-keypoint
error is heavy-tailed (median 3.2 px, p90 37.5 px), so a quadrature sum built on the median
understates the tail — both numbers are statements about one distribution and both have to
travel. **(b)** Until `pipeline/derive.py` consumes the pose model, a **study** number has no
$u_{\text{model}}$ term at all (a human is the model), while a **Predict** number has one the
app never adds. The two surfaces are therefore *not* comparable number-for-number, and §5
shows by how much. This is stated in the code, not inferred: *"a study number, and any
combined uncertainty, still needs that step"* (`memory-bank/activeContext.md`). **(c)** The
sum is still **not** the whole error of a measurement: the drift (S4, a position), the age of
the scale (S10, a step) and the annotator's own variance (S6's second half, unmeasured) are
outside it on purpose, and §6 lists them.

---

## 2. The sources, one at a time

### S1 — Scale from a clicked span (the fallback route)

`daphnia/vision/calibration.py` — one landmark is worth $c$ pixels, so a span between two of
them is worth twice that:

$$c = \texttt{CLICK_ERROR\_PX} = 1.0\ \text{px}
\qquad\qquad
\delta_{\text{span}}(s) \;=\; \frac{2c}{s}\cdot 100 \ \ [\%]$$

Derivation: each landmark is placed with about $c$ pixels of imprecision, so the *span*
between two of them is uncertain by **twice** that — a line is two points, not one. Relative
to the span, that is the calibration's relative error, which is why clicking a longer span is
arithmetic rather than advice:

| span $s$ | what it is at 2× | $\delta_{\text{span}}$ | $\Delta L$ on a 2 mm animal |
| --- | --- | --- | --- |
| $14.7\ \text{px}$ | one 0.1 mm pitch | $\pm 13.61\,\%$ | $\pm 272\,\text{µm}$ |
| $147\ \text{px}$ | ten of them | $\pm 1.36\,\%$ | $\pm 27.2\,\text{µm}$ |
| $1470\ \text{px}$ | a 1 mm square, ten cells | $\pm 0.14\,\%$ | $\pm 2.7\,\text{µm}$ |

$2c/s$ is a **conservative envelope**, not a 1σ: for two independent clicks of σ = $c$ the
span's σ is $\sqrt{2}\,c/s$ (≈ 0.71 of the quoted figure). The code quotes the larger number
deliberately — this project may not claim less uncertainty than it has (`SR-OPTICS-005`,
`SR-OPTICS-015`). Reproduced live (§7): $14.7 \to 13.61\,\%$, $147 \to 1.36\,\%$.

**Handling:** derived, displayed **with its route** (`pages/calibration.py::_uncertainty_text`),
stored as `calibration_profiles.uncertainty_pct`, flagged `WEAK` above
`CALIBRATION_WEAK_UNCERTAINTY_PCT = 2.0 %`. One caveat it states itself: this route measures
the *photograph* and does **not** include the print's own tolerance (S3). A scale the operator
*types in* has no derivation at all — it is stored as a **claim** (`source = hand-entered`,
`SR-OPTICS-029`), with no `calibration_records` row and no proof photo, and the panel says so
rather than printing `±0.00 %` (`SR-OPTICS-005`).

### S2 — Scale from the printed lattice (the shipped route)

`vision/lattice.py`, three steps, each with its own maths.

**Per band.** The period is the robust median of a 24-px band's dip spacings, and the band's
own error is the **standard error of that median**:

$$p_b = \operatorname{median}(\text{inliers}), \qquad
\text{inliers} = \{\, \text{spacing} : |\text{spacing} - p_b| \le 0.25\,p_b \,\}$$

$$\text{scatter}_b \;=\; \frac{\operatorname{median}\bigl(|\text{inliers} - p_b|\bigr)}{p_b}
\;\le\; 0.15 \quad\text{(else the band is not a lattice)}$$

$$\varepsilon_b \;=\; \frac{1.4826 \cdot \operatorname{median}\bigl(|\text{inliers} - p_b|\bigr)}{\sqrt{k_b}}
\qquad\text{[px]},\qquad k_b = \#\,\text{inliers}$$

$1.4826 \cdot \mathrm{MAD}$ is the robust σ of a normal distribution and $1/\sqrt{k}$ is the
standard error of a mean, so $\varepsilon_b$ answers *"how well did this band pin its own
period"*, in pixels.

**Per field.** A band is used only if it agrees with the field's consensus
(`CONSENSUS_TOLERANCE = 0.25` — which tolerates the real 4–6 % drift and rejects a band that
locked onto half or twice the pitch), both line families must contribute at least
`MIN_BANDS = 3` bands each and agree within `MAX_ANISOTROPY_PCT = 3 %`, and every band must
pass its own tests: `MIN_PERIODICITY = 0.5` (its autocorrelation at that lag — measured on
this rig: lattice 0.67–0.83, pure noise 0.1–0.2), `MIN_DEPTH_SIGMA = 1.5` and
`DIP_DEPTH_FRACTION = 0.25` (a dip must be a *line*, not the profile's own scatter),
`MIN_LINES = 5`, and `MIN_PERIOD_PX = 6 … MAX_PERIOD_PX = 500`. Then

$$\bar{p} \;=\; \operatorname{median}\bigl(\{p_b\}\bigr)
\qquad\qquad
\delta_{\text{fit}} \;=\; \frac{\operatorname{median}\bigl(\{\varepsilon_b\}\bigr)}{\bar{p}}\cdot 100
\ \ [\%]$$

$$\boxed{\; \delta \;=\; \sqrt{\,\delta_{\text{fit}}^2 + \delta_{\text{paper}}^2\,}\; } \qquad \delta_{\text{paper}} = 0.5\,\% \ \ \text{(S3)}$$

**Measured on this rig's own eyepiece photo** (2026-10-19, on the live
`PROF_LEICADM500_2X_IPHONE_1.0X_INFINITY` chain):

$$\bar{p} \;=\; 27.239\ \tfrac{\text{px}}{\text{cell}} \quad (10\ \text{bands}),
\qquad
\delta_{\text{fit}} \;=\; 2.49\,\%,
\qquad
\text{tilt} \;=\; -9.00^\circ$$

$$\delta \;=\; \sqrt{2.49^2 + 0.5^2}\ \% \;=\; 2.54\,\%
\qquad\Longrightarrow\qquad
S \;=\; \frac{\bar{p}}{p} \;=\; \frac{27.239}{1000} \;=\; 0.027239\ \tfrac{\text{px}}{\text{µm}}$$

and the store holds exactly that row: `0.027264 px/um`, `±0.2934 %`, verified 2026-10-21 — the
same paper, the same $27.264\ \text{px}/\text{cell}$.

Band-to-band agreement is **not** fit accuracy: $\bar{p}$ is one printed cell's pixel size, so
it inherits the print's tolerance (S3) and the sensor's sampling (S5) as floors that no
averaging removes. And "agreement" is being measured *of the same estimator*: a band that
reads half the pitch off a pale print's split shoulders (measured here: $13.4\ \text{px}$
where the paper answers $26.8\ \text{px}$) is rejected by consensus rather than averaged in —
$\delta_{\text{fit}}$ therefore measures **precision**, and only S3 and S5 bound its
**accuracy**.

### S3 — The print is the floor ($0.5\,\%$)

`PAPER_PITCH_TOLERANCE_PCT = 0.5` (`vision/grid.py`): *how closely a printed 1 mm spacing
matches 1 mm*. Printed paper is not a length standard — sheet stretch, printer scaling and a
photocopy at "fit to page" are all far worse — so the term is **stated, combined and never
averaged away** (`SR-OPTICS-018`):

$$\delta_{\text{paper}} \;=\; 0.5\,\%
\qquad\Longrightarrow\qquad
\delta \;=\; \sqrt{\delta_{\text{fit}}^2 + \delta_{\text{paper}}^2}
\;\;\xrightarrow[\ \delta_{\text{fit}} \to 0 \ ]{}\;\; 0.5\,\%$$

That is the floor under **every** grid calibration, and it is why five decimals of
$\bar{p}$ in the panel do not mean five decimals of accuracy. Lowering it is possible but
never automatic: the tolerance is a parameter,
`uncertainty_pct(paper_tolerance_pct=…)`, and the only honest way to lower it is to **verify
the sheet once against a stage micrometer** — an external length standard — after which the
smaller number is a measurement rather than a hope. **Status: a stated constant, not a
measurement of this sheet.**

Note the route asymmetry the panel also states: a **clicked span** (S1) measures the
*photograph* and does **not** include this term; only the grid route does.

### S4 — Drift: the scale is not constant across the field

Two independent measurements of the same fact, and neither is an uncertainty of the *centre*
scale. `vision/lattice.py` reads it as an envelope over bands;
`vision/points.py` (`_drift_pct`, and `_drift_from_values` for a checkerboard) reads it as
the centre against the edge:

$$\Delta_{\text{drift}}^{\text{lattice}} \;=\; \frac{\max_i \bigl|p_i - \bar{p}\bigr|}{\bar{p}}\cdot 100$$

$$\Delta_{\text{drift}}^{\text{dots}} \;=\;
\frac{\Bigl|\operatorname{median}\bigl(s \mid r \ge 0.75\,r_{\max}\bigr)
      - \operatorname{median}\bigl(s \mid r \le 0.5\,r_{\max}\bigr)\Bigr|}
     {\operatorname{median}\bigl(s \mid r \le 0.5\,r_{\max}\bigr)}\cdot 100$$

with $r$ the distance from the field's centre, $s$ the local spacing, and **at least 6 dots in
each ring** (`MIN_POINTS_FOR_DRIFT`) or nothing is compared.

The first form is a **maximum absolute deviation**, so it is an envelope and **not a σ** — read
it as *"no band disagreed by more than this"*, which is deliberately pessimistic and does not
shrink with more bands. Measured on this rig: $5.10\,\%$ (5.1–6.3 % across the photo set),
which is a real length:

$$\Delta L_{\text{drift}} \;=\; L\cdot\frac{5.1}{100} \;=\; 2000 \cdot 0.051 \;=\; 102\ \text{µm}
\qquad\text{on a } 2\,\text{mm animal}$$

— roughly $10\times$ the print's own tolerance, which is what the `WEAK` verdict is about. The
Calibrate screen gives it **its own line**, never folded into $\pm$:

> `Drift: ± 5.1 % across the field (centre → edge)` — *"Nothing is wrong with the number — it
> is a place in the frame, not an uncertainty."*

**Why it is not a quadrature term.** $\Delta_{\text{drift}}$ is a function of *where* the
specimen sat, so it is a systematic offset at a known place, not a random error to be averaged
away — and no functional form is fitted to it, because fitting one *is* the distortion model
that does not exist yet. Hence: report the envelope, and advise keeping specimens central
(`SR-OPTICS-036`). The **model** that would remove the term is measured to 0.511 px rms by
discorpy's documented flow on this very photo but **fails the mirror test** (an exact pixel
permutation: coefficients differ by 0.82, the centre lands 184 px off) and disagrees with the
direct spacing measurement (8.1–9.8 % against ~5.7 %), so it is **not stored and not applied**
(`SR-OPTICS-032/033/034`, `docs/calibration.md` §1–§3).

### S5 — Quantisation: the pixel grid itself

Both endpoints can only sit on the pixel lattice, so each landmark carries a uniform half-pixel
error. A length is a *difference* of two of them:

$$\operatorname{Var}(\text{one endpoint}) = \frac{(1\ \text{px})^2}{12} = \frac{1}{12}\ \text{px}^2
\qquad\text{(uniform over } \pm 0.5\ \text{px)}$$

$$\sigma_{\text{quant}} \;=\; \sqrt{2\cdot\frac{1}{12}}\ \text{px} \;=\; \frac{1}{\sqrt{6}}\ \text{px} \;\approx\; 0.408\ \text{px}
\qquad\Longrightarrow\qquad
\sigma_{\text{quant}}[\text{µm}] \;=\; \frac{0.408}{S}$$

This term is **not** reported anywhere, yet it needs no new measurement: it is a constant over
$S$, and both are already in the row. It is also the practical form of `SR-OPTICS-009`
(neonate-grade $\approx 21.2\ \text{µm}/\text{px}$, Agatz 2015):

| chain | $S$ (px/µm) | $1\ \text{px}$ is | $\sigma_{\text{quant}}$ | verdict against $\delta = 0.5\,\%$ |
| --- | --- | --- | --- | --- |
| eyepiece grid, this rig | $0.027264$ | $36.7\ \text{µm}$ | $\pm 15.0\ \text{µm}$ | **quantisation dominates** the $\pm 10\ \text{µm}$ on 2 mm |
| live profile `…_135641` | $0.0134$ | $74.6\ \text{µm}$ | $\pm 30.4\ \text{µm}$ | 3× the calibration term |
| fine preset, live DB | $0.121055$ | $8.3\ \text{µm}$ | $\pm 3.4\ \text{µm}$ | the print dominates — better than neonate grade |
| the accuracy target | $0.0472$ | $21.2\ \text{µm}$ | $\pm 8.6\ \text{µm}$ | *not enforced* (`SR-OPTICS-009`, proposed) |

**Status: derived and shown — inside `total +/-`.** It needs no new measurement (`S` is in
the row), so `units.quantisation_um(S)` computes it and `units.total_uncertainty_um` combines
it; the panel prints the sum, and the number is stored with the result. What the app *also*
still does is state the consequence in words — `productContext.md`'s honesty rule: *at
0.147 px/µm, 1 px ≈ 6.8 µm, so the readout must not imply more than it knows*.

### S6 — Clicking the endpoints of a measurement (human routes)

The same $2c$ that calibrates a reference span (S1) applies to the line being *measured* — but
here it is a term of the result rather than of the calibration:

$$\frac{u_{\text{click}}}{L} \;=\; \frac{2c}{s} \quad\text{(relative, dimensionless)}
\qquad\Longrightarrow\qquad
u_{\text{click}} \;=\; L\cdot\frac{2c}{s} \;=\; \frac{2c}{S} \ \ [\text{µm}]$$

so in **absolute** terms it collapses to a constant: the percentage and the length cancel — a
longer line is a longer animal *and* a smaller relative error. As a percentage of the animal,
$u_{\text{click}}/L = 2c/s$. Worked on the eyepiece chain ($S = 0.027264$):

$$\Delta L_{\text{click}} \;=\; \frac{2}{0.027264} \;=\; 73.4\ \text{µm}
\qquad\text{however long the animal is}$$

That is a large absolute term (it is half a millimetre of "where did I click"), and it is
**not reported on a measurement** — only on a *calibration* span. It is also exactly the term
the grid route removes for the scale, by measuring a period instead of clicking: the estimator
has no click. **Status: derived for calibrations *and* for measurements** — `units.click_uncertainty_um(S)`
is the dominant term of `total +/-` on every human-placed length, and it is a *constant* in µm
(73.4 µm on the eyepiece chain) because the percentage and the length cancel.



### S7 — Repeatability: the animal is not constant, and neither is the pose

One frame is one **sample**. The pose model jitters and the animal rotates, so a live
measurement of the same animal differs frame to frame, and the mean of $n$ repeats has a
standard error. `vision/tracking.py` accumulates $\sum L_i$ and $\sum L_i^2$ in one pass, so the
statistics are exact and O(1) per frame:

$$\bar{L} \;=\; \frac{1}{n}\sum_{i=1}^{n} L_i
\qquad\qquad
sd \;=\; \sqrt{\frac{\sum_{i=1}^{n}\bigl(L_i - \bar{L}\bigr)^2}{n-1}}
\qquad\qquad
SE \;=\; \frac{sd}{\sqrt{n}}$$

$$n = 1 \;\Longrightarrow\; sd = 0,\ \ SE = 0
\qquad\text{— one sample has no measured spread, and inventing one is refused}$$

The sample ($n-1$) denominator is deliberate: these repeats *are* a sample of the poses this
animal will take, and the mean of that sample is the number reported, so $sd/\sqrt{n}$ is the
uncertainty of that number. Verified by hand in `tests/test_tracking.py`:
$L = 10,\ 20\ \text{px} \Rightarrow \bar{L} = 15$, $sd = \sqrt{50} = 7.0711$, $SE = 5.0$; four
identical repeats $\Rightarrow sd = 0$; more repeats $\Rightarrow$ a smaller $SE$.

Because tracks average **pixel** lengths and the conversion happens at *display* time, a scale
switch re-labels without re-running the models, and one track can never mix two scales:

$$u_{\text{repeat}}[\text{µm}] \;=\; \frac{SE}{S}
\qquad\text{displayed as } \texttt{"2314 +/- 12 um (n=9)"}$$

The honest limit: a greedy IoU association can hand one animal's history to another, so the
spread is reported **with** the average — a swap then shows up as a jump in $sd$
(`tests/test_tracking.py` pins a crossing pair at $sd > 30\ \text{px}$) instead of hiding inside
a plausible mean. **Status: measured and displayed — on Predict only.** Bench, Dataset and Study
take one line per animal: no $n$, no spread.

### S8 — The pose model: the error that *becomes* the length

`daphnia/vision/rtmpose/`, scored in the **source-image pixel frame** (law 1 — one object owns
the mapping and its `invert`, `PoseInput`):

$$d_k \;=\; \bigl\lVert \hat{p}_k - g_k \bigr\rVert_2 \ \ [\text{px}]
\qquad\qquad
\mathrm{PCK@}t \;=\; \frac{1}{NK}\,\#\bigl\{\, d_k \le t \,\bigr\}$$

$$\mathrm{OKS} \;=\; \frac{1}{K}\sum_{k=1}^{K}
\exp\!\left(-\,\frac{d_k^{2}}{2\,A\,\sigma_k^{2}}\right),
\qquad \sigma_k = 0.05,\quad A = \text{box area in px}^{2}$$

$$\text{length error} \;=\; \bigl|\hat{L} - L\bigr| \;=\; \frac{\bigl|\hat{d} - d\bigr|}{S}
\ \ [\text{µm}] \qquad \hat{d}, d = \text{predicted and true head→tail distance}$$

**Measured** (`models/rtmpose_daphnia.pt`, 250 epochs, best epoch 135, `--split-by group`, on
**one held-out capture** — 101 samples, `b03` — device MPS, 2026-10-17):

| quantity | value |
| --- | --- |
| per-keypoint error, median | $3.2\ \text{px}$ (head $2.2$, tail $3.9$) |
| per-keypoint error, p90 | $37.5\ \text{px}$ |
| $\mathrm{PCK@}5\text{px}$ | $0.67$ |
| $\mathrm{OKS}$ (our sigmas) | $0.56$ |
| head→tail length error, median | $3.0\ \text{px}$ — recorded as $227\ \text{µm}$ (which implies $75.7\ \text{µm}/\text{px}$; at the documented $74.8$ it is $224\ \text{µm}$) |

Provenance, as law 5 requires: the sigmas are **ours**, so the OKS is comparable with another
run of `scripts/train_rtmpose.py --evaluate` on this corpus **and with nothing else**; the split
is one capture, so epoch-to-epoch differences cannot be resolved (`mlContext.md` §6 trap 8); and
the p90 shows a heavy tail — $37.5\ \text{px} \;=\; 2806\ \text{µm}$ at $74.8\ \text{µm}/\text{px}$,
twelve times the median. Determinism was checked separately: six identical passes on one frame
returned $5360.0\ \text{µm}$ every time, so frame-to-frame wobble is the animal and the
photograph, not the arithmetic.

**How it is handled:** measured offline and *named* rather than added —
`(scale +/- 0.50% only - the model's own error is a separate number)`. **Status: measured
offline, never inside a displayed `±`.**

### S9 — The detector: which animal, and how confident

`daphnia/vision/rtmdet/`. The metric, stated before the model (law 1): matching happens in
**original image pixels**, with **both** predictions and labels mapped back through the
letterbox first.

$$\mathrm{IoU}(B_p, B_g) \;=\; \frac{|B_p \cap B_g|}{|B_p \cup B_g|}
\qquad\qquad
\text{match} \iff \mathrm{IoU} \ge 0.5 \ \ (\text{greedy, score-descending})$$

$$P \;=\; \frac{TP}{TP + FP}
\qquad\quad
R \;=\; \frac{TP}{TP + FN}
\qquad\quad
F_1 \;=\; \frac{2PR}{P + R}$$

$$\mathrm{AP}_{50} \;=\; \sum_i \max\bigl(0,\ r_i - r_{i-1}\bigr)\cdot
\tilde{P}(r_i),
\qquad \tilde{P} = \text{monotone envelope of the precision/recall curve}$$

**Measured — two different runs, and they must not be mixed** (`progress.md` known issue 17:
the docs' table and the checkpoint on disk are *not* the same run):

*Run A — the documented 30-epoch run* (`docs/rtmdet.md` §5, `datasets/bb`: 158 frames / 332
boxes from 10 captures, validated on 15 frames / 54 boxes):

| score $\ge$ | P | R | F1 | detections/frame | animals/frame |
| --- | --- | --- | --- | --- | --- |
| 0.05 (AP's threshold) | 0.011 | 0.926 | 0.022 | 300 (capped) | 3.6 |
| 0.35 (**the surface's default**) | 0.608 | 0.833 | 0.703 | 4.9 | 3.6 |
| 0.50 | 0.727 | 0.148 | 0.246 | 0.7 | 3.6 |

*Run B — the checkpoint actually on disk*, re-measured 2026-10-17 on MPS with
`scripts/train_rtmdet.py --evaluate --checkpoint models/rtmdet_daphnia.pt --score-threshold …`,
same one-capture validation set:

$$P = 0.958,\quad R = 0.852,\quad F_1 = 0.902,\quad \mathrm{AP}_{50} = 0.848
\qquad (\text{score} \ge 0.35)$$
$$P = 1.000,\quad R = 0.296
\qquad (\text{score} \ge 0.50)$$

**This is not COCO mAP**: no pycocotools, no 101-point interpolation, no per-class average — so
the number is comparable with another run of *this script* on *this corpus* and nothing else,
and the report prints that warning itself. The score is **not calibrated** (recall collapses
0.83 → 0.15 between thresholds 0.35 and 0.50), so a `0.4` means *"probably"*, not *"no"*.

A detector error reaches the length only **indirectly** — through which box the pose model is
asked about — and that propagation is **not measured**: the pipeline has no end-to-end
length-error budget for "the box was wrong by $x$ px". **Status: shown live (boxes, score,
threshold), named in the docs, not propagated.**

### S10 — Age: a knock is not a σ, it is a step

A bumped adapter changes the scale **without changing any recorded value**, so age cannot be
expressed as a standard deviation — the error is zero until the knock and then a constant
offset of unknown size. The app therefore treats it as a **flag on a threshold**:

$$\text{age} \;=\; \frac{t_{\text{now}} - t_{\text{verified}}}{3600}\ \ [\text{h}]
\qquad\qquad
\texttt{RE-VERIFY} \iff \text{age} > \texttt{CALIBRATION\_MAX\_AGE\_HOURS} = 12$$

$$w(t) \;=\; \begin{cases} 0, & t \le 12\ \text{h} \\ \text{unknown offset}, & t > 12\ \text{h} \end{cases}$$

Measured live in `zoo.db` (2026-10-22): the active profile was verified 29 h before reading, so
every window's footer says `RE-VERIFY`, and a second profile in the same store differs from the
first by **11×** ($0.147$ against $0.0134\ \text{px}/\text{µm}$). **How it is handled:** stated
on every surface (`StatusFooter.summary`, `SR-OPTICS-013/020`); the drift *per hour* is not
measured, so no rate is claimed.

### S11 — Frame quality: which frame gets measured at all

`daphnia/vision/quality.py` scores a frame by the variance of its Laplacian — a focus proxy
cheap enough to run inline (measured 2.5 ms/frame on a 1280×720 frame):

$$q(I) \;=\; \operatorname{Var}\Bigl(\nabla^{2}\,\mathrm{grey}(I)\Bigr)
\qquad\qquad
i^{\star} \;=\; \arg\max_{i \in \text{burst}} q_i$$

The selected frame's `sharpness` is stored on the media asset, so *why* a frame was chosen is
recoverable — but this term is a **selection bias**, not a random error: choosing the sharpest
frame is not neutral with respect to size (a blurred animal's apparent length differs from a
sharp one's), and that correlation is **not measured anywhere**. What the app does about the
*acquisition* end of it: a burst of `BURST_SIZE = 5` frames, and — when five frames are a
lottery on a well that drifts in and out of focus — a **recording** whose length the operator
decides (`SR-STUDY2-013`, up to `RECORD_MAX_FRAMES = 300` ≈ 10 s ≈ 360 MB; measured 17.2 ms to
write and 2.5 ms to score one frame, ≈ 51 frames/s of headroom against a 30 fps stream).
**Status: applied, never reported as an uncertainty.**

### S12 — Identity: which animal a number is *about*

Two rules, both stated once and owned by one object (`mlContext.md` law 1):

$$\text{associate}(\text{measurement}, \text{track}) \iff
\mathrm{IoU}\bigl(m.\text{box},\ \text{track}.\text{last}.\text{box}\bigr) \ge
\texttt{DEFAULT\_IOU} = 0.30 \quad\text{(greedy, best-first, deterministic)}$$

$$\text{individual}(m) \;=\; \arg\min_{\text{boxes } b \ni m.\text{midpoint}} \operatorname{area}(b),
\qquad\text{or } \texttt{None} \text{ (listed as nobody, never attached)}$$

$P(\text{swap})$ — the probability that a crossing pair is mis-assigned — is **not measured**;
what is measured is its *symptom*: a swap inflates the affected track's $sd$ (S7), which is why
the spread is always reported beside the mean. On the Bench/Dataset side the mis-assignment is
a *listing* problem, not a numeric one, and the `None` case is named rather than silent
(`vision/individuals.py`, law 8).

### S13 — The reference mis-declared: a **factor** error

This is the one error that multiplies the answer, and it is an operator/paper error, not an
optical one. The scale is the pixel period divided by the *stated* pitch:

$$S = \frac{\bar{p}_{\text{px}}}{p_{\text{stated}}}
\qquad\Longrightarrow\qquad
\frac{S'}{S} \;=\; \frac{p_{\text{stated}}}{p_{\text{true}}}
\qquad\Longrightarrow\qquad
\frac{\Delta L}{L} \;=\; \left|\frac{p_{\text{true}}}{p_{\text{stated}}} - 1\right|\cdot 100$$

$$\text{a } 0.5\ \text{mm sheet filed as } 1\ \text{mm}: \qquad
\frac{\Delta L}{L} = \left|\frac{500}{1000} - 1\right| = 100\,\%$$

i.e. **every** later measurement is $2\times$ the truth — and no fit error (S2), no drift (S4)
and no repetition (S7) will reveal it, because the whole field is consistently wrong. Guards:

* one **kind** of target only, so a wrong *slide* cannot be selected (`SR-OPTICS-019`);
* the pitch is **one field** used in four places — detection, the clicked-span arithmetic
  $s_{\text{squares}}\times p$, the stored `target_type` and `actual_distance_um` — so changing
  it re-scales the detection already on screen instead of asking for a second photo
  (`SR-OPTICS-030`);
* the **hand method counts squares**, not a pitch, which deletes the "wrong pitch for the wrong
  slide" class entirely (`SR-OPTICS-027`);
* activation is guarded: a switch of more than $2\times$ must be confirmed with both values
  named (`SCALE_JUMP_FACTOR = 2.0`, `SR-OPTICS-008`) — the guard written after one `profile_id`
  silently carried $0.783$ and $0.147\ \text{px}/\text{µm}$.

$$\frac{S_{\text{new}}}{S_{\text{old}}} = \frac{0.147}{0.0134} \;=\; 11 \;\gg\; 2
\qquad\Longrightarrow\qquad \text{confirmation required}$$

**Status: guarded at entry, and the pitch is stored per record** — but nothing checks a stored
row against the pitch afterwards (see S14).

### S14 — Plausibility: a number that cannot be an animal

There is **no gate** on a measured length anywhere in the domain layer (grep: no
`plausible`, no species range, no `MAX_LENGTH`). Every other source in this table is either
derived, displayed, refused or named — this one is *none of those*, and §5(d) shows what that
costs on the live store: **18 rows reading 6.6–56 mm**, physically impossible as *Daphnia magna*
body lengths ($\sim 0.5$–$3\ \text{mm}$), with nothing in the app to flag them. The two candidate
explanations — a line drawn across a plate rather than an animal, or a scale applied from the
wrong chain — are *not distinguishable from the stored row alone*, which is itself the finding.
**Status: open.** A plausibility band per study (organism → length range) plus a
"line spans $x\,\%$ of the frame" note would have caught all 18 without touching the maths.

---

## 3. The budget, combined — how accurate *is* the number on screen?

Two worked cases, both on chains that exist in the live store, both for $L = 2000\ \text{µm}$.
The point of the section is the last row of each table: **the reported `±` is not the total**,
and by how much it differs depends on the chain.

### 3.1 Study path — a human places both landmarks

Chain: the eyepiece grid, $S = 0.027264\ \text{px}/\text{µm}$, $\delta = 0.2934\,\%$; at
$36.7\ \text{µm}/\text{px}$ a 2 mm animal is a $54.5\ \text{px}$ line, so $c = 1\ \text{px}$ is
$2\,\%$ of the animal per endpoint.

$$u_{\text{scale}} = L\cdot\frac{\delta}{100} = 2000 \cdot 0.002934 = 5.87\ \text{µm}$$
$$u_{\text{click}} = \frac{2c}{S} = \frac{2}{0.027264} = 73.4\ \text{µm}$$
$$\sigma_{\text{quant}} = \frac{0.408}{S} = 15.0\ \text{µm}$$
$$u_{\text{model}} = 0 \quad\text{(a human is the model; the annotator's own variance is unmeasured)}$$

$$u_{\text{total}} = \sqrt{5.87^{2} + 73.4^{2} + 15.0^{2}} = \sqrt{5640} = 75.1\ \text{µm}
\qquad (3.8\,\% \text{ of the animal})$$

| term | value | $u_{\text{term}}/u_{\text{total}}$ (magnitudes — the column does **not** sum to 100) |
| --- | --- | --- |
| scale (S1–S3) | $5.87\ \text{µm}$ | $7.8\,\%$ — **this is the `±` on screen** |
| clicking (S6) | $73.4\ \text{µm}$ | $97.7\,\%$ — dominates |
| quantisation (S5) | $15.0\ \text{µm}$ | $20.0\,\%$ |
| model | $0$ | — |
| **total** | $\mathbf{75.1\ \text{µm}}$ | the `±` on screen is $5.87$ — **13× smaller**; since 2026-10-22 the screen prints this total *beside* it |

### 3.2 Predict path — the model places both landmarks

Chain: the wide-lens profile the pose model was scored on, $S = 0.01336\ \text{px}/\text{µm}$
($74.8\ \text{µm}/\text{px}$), $\delta = 0.50\,\%$. No human click; instead the pose model's own
error, from `--evaluate` (median $3.0\ \text{px}$, p90 $37.5\ \text{px}$):

$$u_{\text{scale}} = 2000 \cdot 0.005 = 10.0\ \text{µm}
\qquad\qquad
\sigma_{\text{quant}} = \frac{0.408}{0.01336} = 30.5\ \text{µm}
\qquad\qquad
u_{\text{model}}^{\text{median}} = 3.0\ \text{px} \cdot 74.8 = 224\ \text{µm}$$

$$u_{\text{total}}^{\text{median}} = \sqrt{10.0^{2} + 30.5^{2} + 224^{2}} = \sqrt{51200} = 226\ \text{µm}
\qquad (11.3\,\% \text{ of the animal})$$

$$\text{at the p90: } \quad u_{\text{model}} = 37.5\ \text{px}\cdot 74.8 = 2805\ \text{µm}
\;\Longrightarrow\;
u_{\text{total}}^{\text{p90}} \approx 2805\ \text{µm} \qquad (140\,\% \text{ of the animal})$$

| term | value | $u_{\text{term}}/u_{\text{total}}$ (magnitudes — does **not** sum to 100) |
| --- | --- | --- |
| scale (S1–S3) | $10.0\ \text{µm}$ | $4.4\,\%$ — **this is the `±` on screen** |
| quantisation (S5) | $30.5\ \text{µm}$ | $13.5\,\%$ |
| click | $0$ | — (no human) |
| model (S8), median | $224\ \text{µm}$ | $99.1\,\%$ — dominates |
| **total** | $\mathbf{226\ \text{µm}}$ | the `±` on screen is $10$ — **23× smaller**; the model's term is still *not* added, because the Predict surface names it as a separate number |

The heavy tail is the operative fact: a *median* says "the typical animal is pinned to 11 %",
and the *p90* says "one animal in ten is off by more than its own length". That is why the two
numbers travel together and why `mlContext.md` §6 trap 8 forbids quoting a best-of instead of a
spread.

### 3.3 What the screen says, side by side

| quantity | Study path (3.1) | Predict path (3.2) |
| --- | --- | --- |
| the `±` shown / stored | $\pm 5.9\ \text{µm}$ | $\pm 10.0\ \text{µm}$ |
| `total +/-` printed beside it (since 2026-10-22) | $75.1\ \text{µm}$ | $226\ \text{µm}$ — **only if a caller passes the model term**; the Predict surface itself prints none |
| factor between them | $\times 13$ | $\times 23$ |
| what dominates | the human click | the pose model |
| can it be compared with the other surface? | **no** — different terms, different meaning | **no** |

That last row is the practical consequence, and it is a **policy** statement, not a bug: `SR-STUDY`
says the Study surface is *"the only surface that may produce publication numbers"*, and Predict
is the surface where you *watch* the detector. Two `±` from two surfaces are not two estimates
of the same quantity, and comparing them is exactly the "two numbers silently compared" failure
`memory-bank/mlContext.md` §5 takes from **torchmetrics**' metric-protocol discipline.

---

## 4. How each source is handled — the policy, in one table

Four mechanisms are in use, and every source falls into exactly one of them:

1. **Derive and display it** — the number is computed, stored with the result and shown where it
   is read (`units.format_with_uncertainty` for the scale term, `units.format_total_uncertainty`
   for the whole budget, the Calibrate panel, the footer).
2. **Name it as a separate number** — it is measured *offline*, is real, and is deliberately
   **not** folded into the displayed `±`, because folding it would be a claim about the other
   surface's number (S8, S9). It may be *passed into* the total by a caller that measured it
   (`units.total_uncertainty_um(extra_um=…)`), which is the opposite of folding it in silently:
   the caller states it.
3. **Refuse or flag instead of guessing** — a wrong scale is worse than no scale, so the
   estimator refuses; a scale that may have moved is flagged; a number with no scale is labelled
   unscaled (`SR-OPTICS-017`, `SR-OPTICS-010`, `X-N05`).
4. **Nothing yet** — derived or derivable, and simply not built (S14, the per-position term).
   These are the honest gaps, listed again in §6.

| Source | Derived? | Shown where | Stored with the result? | Gate / flag | Requirement |
| --- | --- | --- | --- | --- | --- |
| **S1** span scale | yes $\dfrac{2c}{s}$ | Calibrate panel, **with its route** | `calibration_profiles.uncertainty_pct` → per row | `WEAK` if $> 2\,\%$; help says "click a longer span" | `SR-OPTICS-005`, `-015`, `-012` |
| **S2** lattice scale | yes (robust MAD/median) | panel: `Period`, readings count, tilt, tooltip | same | refuses a non-lattice; consensus rejects a bad band | `SR-OPTICS-016`, `-035` |
| **S3** print floor | stated $0.5\,\%$ | panel names the route that includes it | inside `uncertainty_pct` | — | `SR-OPTICS-018` |
| **S4** drift | yes (envelope / inner-vs-outer) | **own line** + `WEAK` verdict + tooltip | **no** (not on the row) | `WEAK` above the print's $0.5\,\%$; distortion says *unmeasured* | `SR-OPTICS-036`, `-032` |
| **S5** quantisation | yes $\dfrac{0.408}{S}$ | inside `total +/-` on every measuring screen | `corpus_labels.total_uncertainty_um` (via the sum) | the resolution target is **proposed**, not enforced | `SR-OPTICS-009` |
| **S6** click on a line | yes $\dfrac{2c}{S}$ | inside `total +/-` on every measuring screen; the calibration panel states the same term for its span | `corpus_labels.total_uncertainty_um` | — | `SR-OPTICS-005` (calibration spans) |
| **S7** repeatability | yes ($sd$, $SE$) | **Predict** labels: `mean ±SE (n=…)` | no (live session) | $n = 1 \Rightarrow$ no spread is invented | `mlContext.md` §3, §8 |
| **S8** pose model | offline (`--evaluate`) | named: *"a separate number"* | `models.metrics_json` on the model row | a prediction is never stored as ground truth | `SR-PREDICT-N01`, law 7 |
| **S9** detector | offline (`--evaluate`) | live boxes, score, threshold | `models.metrics_json`, `classes` | score threshold is the operator's choice, no fallback | `docs/rtmdet.md` §5, law 8 |
| **S10** age / knock | yes (hours) | **every** window's footer | `last_verified_at` | `RE-VERIFY` past 12 h | `SR-OPTICS-013`, `-020` |
| **S11** frame quality | yes (Laplacian variance) | the strip / the label | media asset `sharpness` | one kept frame per target; a recording when 5 frames are a lottery | `SR-STUDY-006`, `SR-STUDY2-013` |
| **S12** identity | yes (IoU, midpoint) | the list, one line per animal | `individual_index` | `None` = *nobody*, printed as such | `vision/individuals.py`, law 8 |
| **S13** wrong pitch | guarded at entry | panel states the pitch; it is one field in four places | `target_type`, `actual_distance_um` | $> 2\times$ jump needs confirmation naming both values | `SR-OPTICS-008`, `-019`, `-027`, `-030` |
| **S14** plausibility | **no** | nowhere | no | **none** | **open** |

**The one asymmetry worth knowing:** measures of the *machine* are handled everywhere and
identically (S1–S4, S10 — the shared `StatusFooter` carries the scale, its `±`, its age and the
`RE-VERIFY` flag into **every** window, `X-N09`). Measures of the *measurement* moved, in the
2026-10-22 round, from "not carried at all" to **carried inside the total**: S5 and S6 are now
derived, printed and stored per row, so a stored number says what it is worth and not only how
it was scaled. Still not carried: the per-position part of S4 (the drift is a place, not a
number on the row), the annotator's own variance, and S14 — which is carried nowhere because it
does not exist. So today's stored row is a complete record of **the scale that was applied**, a
good record of **how accurate the number is**, and no record of **where in the frame** it was
taken.

---

## 5. Worked examples

### (a) Repeats buy accuracy at $1/\sqrt{n}$ — and only there

Suppose one animal is measured four times on the live feed with $L = 2000\ \text{µm}$,
$\delta = 0.50\,\%$, and the animal's own rotation gives $sd = 40\ \text{µm}$:

$$n=1:\quad SE = \frac{0}{1} = 0 \ \ \text{(no spread is invented — the label shows the frame's own value)}$$
$$n=4:\quad SE = \frac{40}{\sqrt{4}} = 20\ \text{µm}
\qquad\qquad
n=16:\quad SE = \frac{40}{4} = 10\ \text{µm}$$

$$u_{\text{total}}(n=4) = \sqrt{10^2 + 0^2 + 30.5^2 + 224^2 + 20^2} = 227\ \text{µm}$$

The lesson is arithmetical: repeats shrink the **random** term (S7) and cannot touch the
systematic ones (S1–S4) or the model's median error (S8) — $224\ \text{µm}$ does not move, however
many frames you average. That is why "measure it a few times" is not a substitute for a
calibrated scale or a better model, and why the app reports the pair instead of a tighter-looking
single number.

### (b) A weak calibration, end to end

The fallback route with a short span: $s = 14.7\ \text{px}$ (one 0.1 mm pitch at 2×), so
$\delta = 13.61\,\%$ (S1), stored as a `WEAK` calibration. On a **neonate**,
$L = 700\ \text{µm}$, on the eyepiece chain $S = 0.027264$:

$$u_{\text{scale}} = 700 \cdot 0.1361 = 95.3\ \text{µm}
\qquad
u_{\text{click}} = \frac{2}{0.027264} = 73.4\ \text{µm}
\qquad
\sigma_{\text{quant}} = 15.0\ \text{µm}$$

$$u_{\text{total}} = \sqrt{95.3^2 + 73.4^2 + 15.0^2} = \sqrt{14694} = 121\ \text{µm}
\qquad (17\,\% \text{ of a 700 µm animal})$$

This is the case `SR-OPTICS-005` was written for: a single short pitch makes the scale
$\sim 27\times$ less certain than a magic $0.5\,\%$ would claim, and the app says so with a `WEAK`
flag, the route, and the advice ("click a longer span" — a $147\ \text{px}$ span takes
$\delta$ from $13.61$ to $1.36\,\%$).

### (c) Off-centre: the drift is bigger than the quoted `±`

Same 2 mm adult, $S = 0.027264$, measured **at the edge** of the field, where the field's own
readings move by $\Delta_{\text{drift}} = 5.1\,\%$:

$$\bigl|\Delta L_{\text{drift}}\bigr| \;\le\; L\cdot\frac{5.1}{100} \;=\; 102\ \text{µm}
\qquad\text{against}\qquad
u_{\text{scale}} = 5.87\ \text{µm}$$

So the *systematic* term the app deliberately keeps out of the `±` is **17× larger** than the
`±` it does print — and its sign is decided by *where in the frame* the specimen sat, which the
store keeps only as pixels (`p1_x, p1_y`) with no term in the row that says "this one was
off-centre". Both facts are by design: the drift is reported as its own line, and the advice is
to keep specimens central. The consequence to state out loud is that **a `±` from a measurement
taken at the edge is not comparable with one taken at the centre**, and nothing in the row says
which you have.

### (d) Audit of the live store — the one source with no handling at all (S14)

Every row is written by **one** path (`repository.save_corpus_label`) and records the scale that
was applied, so the arithmetic can be *audited* rather than assumed. Read straight from
`zoo.db` (2026-10-22, 550 labelled rows, all `source = manual`):

```sql
SELECT round(px_per_um,5) AS S, round(1.0/px_per_um,2) AS um_per_px,
       count(*) AS n, round(min(length_um),1) AS min_um, round(max(length_um),1) AS max_um
FROM corpus_labels WHERE length_um IS NOT NULL GROUP BY round(px_per_um,5) ORDER BY n DESC;
```

| $S$ (px/µm) | µm/px | rows | min $L$ | max $L$ |
| --- | --- | --- | --- | --- |
| $0.01336$ | $74.84$ | 520 | $0.0$ | $18642.6\ \text{µm}$ |
| $0.01005$ | $99.50$ | 18 | $6629.9$ | $56363.4\ \text{µm}$ |
| $0.02726$ | $36.68$ | 12 | $1070.4$ | $6497.6\ \text{µm}$ |

**440 of the 550 rows read more than $4\ \text{mm}$**, and the arithmetic says the line, not the
scale, is the explanation:

$$\frac{s}{S}: \qquad 566.5\ \text{px} / 0.01005 = 56363\ \text{µm} = 56.4\ \text{mm}
\qquad\text{but also}\qquad
566.5\ \text{px} / 0.027264 = 20.8\ \text{mm}
\quad\text{and}\quad
566.5\ \text{px} / 0.01336 = 42.4\ \text{mm}$$

A *Daphnia magna* body length is roughly $0.5$–$3\ \text{mm}$, so $56.4\ \text{mm}$ is $19$–$113\times$
too long — and it stays absurd under **every** stored scale, i.e. it is not a mis-scaled animal,
it is a line that is not an animal ($566.5\ \text{px}$ is $29\,\%$ of the frame's width). Two
candidate stories (a line drawn across a plate; a label made on a non-animal image) are **not
distinguishable from the row**, and that is the finding: the store records the **scale** applied
and the **endpoints** drawn, but no field says *what was on the frame* and no field says whether
the result is plausible for the organism. One row is degenerate: $s = 0.0\ \text{px}$, stored as
$L = 0.0\ \text{µm}$ without complaint.

**What would have caught it** (a proposal, *not* existing behaviour): a plausibility band per
study — the organism's own length range, since `studies.organism` is already stored —
$$\underline{L} \;\le\; L \;\le\; \overline{L}
\qquad\text{plus}\qquad
s \;\le\; f\cdot\min(W, H),\ \ f \approx 0.5
\qquad\text{plus}\qquad
s > 0$$

applied as a **named** warning (law 8: a failure is named, never silent), never as a silent
clamp — the row would still be stored, with the flag travelling beside it.

---

## 6. What is *not* measured — stated out loud (`mlContext.md` §8.4)

1. **The total uncertainty of a length, on the Predict path.** The human path is now combined,
   stored and printed (§1: `units.total_uncertainty_um`, `corpus_labels.total_uncertainty_um`),
   but **no surface passes the pose model's own error into it** — Predict still names that term
   as a separate number, because folding it in would make a Predict number look like a Study
   number (§3.3). Until `pipeline/derive.py` consumes the pose model, a combined budget for a
   model-placed length exists only when a caller asks for it (`extra_um=…`, `click_error_px=0.0`).
2. **The annotator's own variance** on the endpoint definition (*anterior margin of the compound
   eye → base of the apical spine*, spine excluded, lateral view). No inter-annotator study
   exists, on any surface, for any trait — and it is a term the pose model is *scored against*,
   so it is inside every `d` and every `PCK` as well.
3. **The per-position term on a measurement.** The drift is measured per calibration (§2 S4) and
   displayed; a row stores one `px_per_um` and no position term, so a number taken at the edge is
   not flagged as such (§5 c).
4. **The model's error inside a displayed number.** Measured offline (S8), named as separate,
   never added — and for the detector (S9) the propagation into a length error is not measured at
   all.
5. **The distortion model.** Measured to 0.511 px rms by discorpy's documented flow, rejected by
   the mirror test and by disagreement with the direct spacing measurement, **not stored**
   (`SR-OPTICS-032/033/034`).
6. **Any accuracy claim on more than one held-out capture.** Both models were scored on a single
   capture (15 frames / 54 boxes for the detector, 101 samples for the pose), so epoch-to-epoch
   differences cannot be resolved (`SR-DATASET-010`).
7. **Longitudinal deviation.** Mean $\pm$ $sd$ per individual per timepoint is not implemented —
   `pipeline/aggregate.growth_trajectory` raises `NotImplementedError` — so growth variance
   against measurement noise cannot be computed in-app; it can be computed from the corpus export
   (`px_per_um`, `uncertainty_pct`, `length_um`, `p1/p2`), but the **Study** CSV still drops the
   landmarks (`EXPORT_COLUMNS`, known issue 4).
8. **Selection bias.** Sharpest-frame selection (S11) is a choice correlated with focus, and its
   effect on a length is not measured.
9. **Plausibility.** No gate (S14), so an impossible row is stored and reported like any other.
10. **The lab's Mac, a nearly full disk, a page of wells drawing its frames.** Every timing in
    this document was taken on the M4, offscreen.

---

## 7. Reproducing every number in this document

```bash
conda activate z-app

# S1 — the clicked-span arithmetic (stdlib only)
python -c "from daphnia.vision.calibration import relative_uncertainty_pct as r; \
print('%.2f%%  %.2f%%' % (r(14.7), r(147.0)))"        # -> 13.61%  1.36%

# S2/S3/S4 — the scale's components, kept apart and combined in quadrature
python -c "from daphnia.vision.lattice import LatticeEstimate as E; \
e = E(period_px=27.239, angle_deg=-9.0, bands=10, spread_pct=5.10, statistical_pct=2.49); \
print('fit %.2f%%  drift %.2f%%  delta %.2f%%' % \
      (e.statistical_uncertainty_pct(), e.spread_pct, e.uncertainty_pct()))"
# -> fit 2.49%  drift 5.10%  delta 2.54%

# S5/S6 — the terms that need no new measurement
python -c "S=0.027264; print('quant +/-%.1f um   click +/-%.1f um' % (0.408/S, 2/S))"
# -> quant +/-15.0 um   click +/-73.4 um

# S1/S5/S6 combined — the whole budget of one length, as the surfaces print it
python -c "from daphnia import units; \
print(units.format_total_uncertainty(2000.0, 0.027264, 0.2934)); \
print(units.format_total_uncertainty(2000.0, 0.01336, 0.50, extra_um=224.0, \
      click_error_px=0.0))"
# -> total +/- 75.1 um (3.8 %)      (the Study path, §3.1)
# -> total +/- 226.3 um (11.3 %)    (the Predict path, §3.2 - model term passed in, click off)
python -m unittest tests.test_units -v          # the two §3 cases, hand-computed

# S1/S5/S6 as stored — what a row says it is worth (the column is added by `migrate()`,
# so launch the app once with this build before running it)
sqlite3 -header -column zoo.db \
  "select round(px_per_um,5) S, count(*) n, round(min(total_uncertainty_um),1) lo,
          round(max(total_uncertainty_um),1) hi
   from corpus_labels group by round(px_per_um,5) order by n desc;"

# S1 stored / S14 audit — what the rows actually say
sqlite3 -header -column zoo.db \
  "select round(px_per_um,5) S, round(1.0/px_per_um,2) um_per_px, count(*) n,
          round(min(length_um),1) min_um, round(max(length_um),1) max_um
   from corpus_labels where length_um is not null
   group by round(px_per_um,5) order by n desc;"

# S7 — repeatability, hand-computed in the suite
python -m unittest tests.test_tracking -v

# S8 / S9 — the model's own error, with provenance (corpus, split, threshold, epoch, device)
python scripts/train_rtmpose.py --evaluate --checkpoint models/rtmpose_daphnia.pt
python scripts/train_rtmdet.py --evaluate --checkpoint models/rtmdet_daphnia.pt \
       --score-threshold 0.35
make train-check                      # the plane: torch + MPS + a fwd/bwd pass
python scripts/calibration_probe.py recordings/calibration/<photo>.png --coef 3   # §S4's model
```

Values in this document were re-derived on 2026-10-22: $\delta_{\text{span}}(14.7) = 13.61\,\%$,
$\delta_{\text{span}}(147) = 1.36\,\%$, $\delta = \sqrt{2.49^2+0.5^2} = 2.54\,\%$,
$\sigma_{\text{quant}} = 15.0\ \text{µm}$, $u_{\text{click}} = 73.4\ \text{µm}$ on the eyepiece
chain, and the live store's three scale groups (520 / 18 / 12 rows).

---

## 8. Traceability, and what this document is not

| This document | Requirement | Code | Test that pins it |
| --- | --- | --- | --- |
| S1, S6 | `SR-OPTICS-005`, `-015` | `vision/calibration.relative_uncertainty_pct` | `tests/test_calibrations.py` |
| S2, S4 | `SR-OPTICS-016`, `-035`, `-036` | `vision/lattice.py`, `vision/points.py` | `tests/test_lattice.py`, `tests/test_points.py` |
| S3 | `SR-OPTICS-018` | `vision/grid.PAPER_PITCH_TOLERANCE_PCT` | `tests/test_grid.py` |
| S4 (model absent) | `SR-OPTICS-032`, `-033`, `-034` | `docs/calibration.md`, `scripts/calibration_probe.py` | probe + mirror test |
| S5 | `SR-OPTICS-009` (proposed) | `units.quantisation_um`, `units.total_uncertainty_um` | `tests/test_units.py::TotalUncertaintyTests` |
| S5+S6 combined (the total) | law 5, `docs/accuracy.md` §1 | `units.total_uncertainty_um` / `format_total_uncertainty`, `vision/calibration.OpticalProfile.total_uncertainty_um`, `db/repository.save_corpus_label`, `db/connection.backfill_total_uncertainty` | `tests/test_units.py`, `tests/test_calibrations.py` (stored + migrated), `tests/test_dataset.py` (exported) |
| S7 | `mlContext.md` §3, §8 | `vision/tracking.py` | `tests/test_tracking.py` |
| S8 | `SR-PREDICT-013`, law 5 | `vision/rtmpose/{losses,train}.py` | `tests/test_rtmpose.py` (oracle, blind stub) |
| S9 | law 1, law 5 | `vision/rtmdet/train.py` | `tests/test_rtmdet.py::EvaluationTests` |
| S10 | `SR-OPTICS-013`, `-020` | `ui/widgets/status_footer.py` | `tests/test_ui_smoke.py` |
| S11 | `SR-STUDY-006`, `SR-STUDY2-013` | `vision/quality.py` | `tests/test_quality.py`, `tests/test_study2.py` |
| S12 | law 8 | `vision/individuals.py` | `tests/test_individuals.py`, `tests/test_tracking.py` |
| S13 | `SR-OPTICS-008`, `-019`, `-027`, `-030` | `ui/pages/calibration.py` | `tests/test_calibration.py` |
| S14 | **open** | — | — |

**What this document is not.**

* **It is not an accuracy claim for the app.** It is the *ledger*: what each term is, how it is
  derived, and whether it is reported. The only accuracy figures in it that come from a model are
  quoted **with** their corpus, split, threshold, epoch and device — and both come from a single
  held-out capture.
* **It is not a replacement for `docs/calibration.md`** (how a scale is *obtained*), for
  `docs/rtmdet.md` §5 and `docs/rtmpose.md` §3 (the model metrics), or for
  `memory-bank/mlContext.md` (the discipline). It is the one place where the terms are put side
  by side and combined.
* **It does not introduce a new number.** Every value here is either read from a stored row, from
  a constant in the code, or from a documented measurement; §7 is how each one is reproduced.
* **Nothing here is corrected by hand.** A metric that changes (a new checkpoint, a re-measured
  corpus, a new constant) makes this document *stale*, and the honest response is to re-measure
  and update it — not to adjust a number so the tables keep agreeing. `progress.md` known issue 17
  is the precedent: the docs' table and the checkpoint on disk were different runs, and the fix
  was to measure, not to reconcile.

