# SplatWorld3D — first external-world result

## Question

Can the address-conditioned operator plate from SplatWorld3 be used as a compact 3-D novel-view representation when the teacher is no longer another operator plate, but an **external rendered 3-D world**?

This is the first test in the repo where camera pose is given physical meaning. The renderer is the analytic sphere/cube/checkered-floor world adapted from the older `worldrank.py` line. The operator does not see depth, 3-D hit points, or held-out images.

The important result is mixed:

> The address-conditioned operator feature map is a good smooth interpolator of the orbit, but **learning the material `g` does not improve the held-out views**, and a still simpler raw camera-coordinate model performs better.

So this experiment does **not** show that the plate learned 3-D. It does establish the first external-world baseline and identifies what the next test must remove.

---

## Design

Eight training camera azimuths:

```text
0, 45, 90, 135, 180, 225, 270, 315 degrees
```

Eight untouched halfway test views:

```text
22.5, 67.5, 112.5, 157.5, 202.5, 247.5, 292.5, 337.5 degrees
```

All camera addresses live on the same operator-address radius:

```text
a(phi) = (2 cos(phi), 2 sin(phi), 0)
```

Therefore the radial / fovea-shaped center-to-periphery envelope discovered in the earlier atlas cannot distinguish the train and test angles.

A six-dimensional PCA image basis is built from **training views only**. Test images never enter the basis, plate fitting, ridge selection, or model selection.

---

## Attempt 1 — in-sample material fitting fails

The first version fitted a small affine readout from the plate response to the six image-basis coefficients and changed `g` by conservative edge-to-edge transfers to minimize training-view coefficient error.

Held-out relative image RMSE:

| method | held-out error |
|---|---:|
| learned operator plate | **0.574539** |
| plain angular pixel interpolation | **0.496992** |
| nearest training image | **0.583746** |
| train-basis oracle projection | **0.426942** |

The learned plate only barely beat nearest-frame and lost clearly to ordinary image interpolation.

A diagnostic then showed why the fitting rule was untrustworthy:

```text
untrained operator       0.562413
learned operator         0.574539
relative effect of g     -2.156%   (worse)
```

while the affine readout design matrix had condition number

```text
untrained g   2864
learned g     1274
```

The material moved substantially (`||Delta g|| = 1.3224`, 71 accepted conservative edits) while held-out prediction got worse. The original objective was therefore mostly rewarding an almost-singular fit to eight training views.

---

## Attempt 2 — train `g` for leave-one-view-out interpolation

The correction was fixed before reading the new held-out result:

- candidate `g` values are scored by leave-one-training-view-out prediction across the eight training cameras;
- ridge is selected only from `{0.01, 0.1, 1.0}` by that training-only LOO score;
- five operator features plus an intercept are used;
- the eight halfway views remain completely untouched until the end.

Result:

| method | held-out error |
|---|---:|
| **untrained operator features** | **0.440043** |
| LOO-trained `g` | **0.441968** |
| plain angular pixel interpolation | **0.496992** |
| periodic Fourier baseline | **0.516521** |
| nearest training image | **0.583746** |
| train-basis oracle projection | **0.426942** |

This is a much better interpolator than the naive version, but the key comparison fails:

```text
untrained g : 0.440043
trained g   : 0.441968
```

The LOO objective itself improved (`9.8496 -> 9.4345`) and the material changed strongly (`||Delta g|| = 1.6040`, 75 accepted edits), but true held-out error became slightly worse.

Preregistered verdict:

```text
EXTERNAL_WORLD_OPERATOR_INTERPOLATION_NOT_YET_SHOWN
```

That verdict refers specifically to **learning the material substrate**. It does not erase the fact that the fixed operator coordinates interpolate the held-out views well.

---

## Attempt 3 — is the fixed operator feature map special?

The untrained operator features were surprisingly effective, so they were attacked with matched smooth-feature controls. Every model sees the same eight training images, uses the same training-only image basis, and chooses ridge using training LOO only.

### Simple controls

| feature map | held-out error |
|---|---:|
| raw camera address `[cos(phi), sin(phi)]` | **0.431122** |
| operator features, 32-seed median | **0.439921** |
| random smooth 5-D features, 32-seed median | **0.443817** |
| plain pixel interpolation | **0.496992** |
| 5-D harmonic features | **0.516620** |
| train-basis oracle projection | **0.426942** |

The raw two-dimensional camera coordinate is best among the learned/readout models and lands very close to the training-basis oracle.

That is decisive for this first scene:

> The external orbit is simple enough that a regularized linear map from `(cos phi, sin phi)` already explains nearly all of the recoverable training-basis view variation.

The operator is therefore **not needed** to explain the success on this orbit.

### One interesting secondary observation

The untrained operator feature map is unusually stable across initialization seeds:

```text
operator5, 32 seeds
median  0.439921
mean    0.439928
min     0.439311   (diagnostic only; not selected)
max     0.440401

random5, 32 seeds
median  0.443817
mean    0.462766
min     0.433978   (diagnostic only; not selected)
max     0.532228
```

So the resolvent family supplies a consistent smooth coordinate embedding, whereas arbitrary smooth random features are much more variable. But that is a weaker observation than a learned 3-D representation, and the raw camera coordinate still wins.

---

## What survived

The same-family sparse-world result showed that sparse observations can change one shared plate and improve untaught addresses when the target world itself belongs to the reachable operator family.

The first external-world experiment now shows something different:

```text
external 3-D orbit
    -> smooth novel-view interpolation is possible
    -> fixed operator coordinates work well
    -> but raw camera coordinates work even better
    -> changing g does not improve true held-out views
```

Therefore we have **not** crossed from operator-family system identification to learned external 3-D geometry yet.

This is a useful failure because it rules out an easy demonstration. A camera orbit around a simple scene is too globally smooth to tell whether a shared operator substrate contains anything beyond a convenient nonlinear coordinate map.

---

## What the next test must require

The next world should make simple global pose interpolation insufficient. The old `worldplus.py` room is useful here because it already contains the needed ingredients:

- camera **translation**, not only azimuth on one orbit;
- occlusion and disocclusion;
- nearby viewpoints with qualitatively different visibility;
- routes through a 2-D floor plan;
- deliberately aliased views;
- counterfactual camera positions that were not traversed during training.

The clean next question is therefore:

> Can one shared `g`, queried by `(x,z,yaw)`, use sparse observed routes to predict **counterfactual views at unvisited positions** better than direct pose features, local interpolation, and matched random-feature controls?

If the learned material still cannot beat the fixed/untrained coordinates there, then `g` has not earned world-memory status. If it does, the result will no longer be explainable by a one-dimensional smooth camera orbit.

---

## Honest status

Established by this experiment:

```text
external 3-D images                    YES
held-out intermediate camera views     YES
fixed-radius / fovea control            YES
operator coordinates interpolate well   YES
operator beats plain pixel interpolation YES (regularized fixed features)
operator is necessary                   NO
learning g helps                        NO
learned 3-D world in g                  NOT SHOWN
```

That is where SplatWorld3D starts.
