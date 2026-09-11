# Sparse World Interpolation — result

## Question

Can a shared operator plate infer the **untaught parts of an address-conditioned world** from only a few taught addresses, rather than merely reproducing the radial / fovea-shaped center-to-periphery envelope already built into the address loading?

This experiment was preregistered in `sparse_world.py` before the result was read.

## Why the fovea likeness is a confound

The earlier committed `operator_atlas/atlas_after_data.npz` really does have a strong radial envelope:

- `corr(dM, radius) = 0.8871`
- `corr(dM, radius^2) = 0.8715`
- `corr(cond(A), radius) = 0.8312`
- `corr(|c|, radius) = 0.8836`

A general quadratic fit to `dM(x,y)` gives approximately

```text
0.0507 x^2 + 0.0647 y^2 - 0.00212 xy + ...
R^2 = 0.7986
```

So the atlas is genuinely fovea-shaped in the descriptive sense: the center is a low-deformation / well-conditioned chart and the periphery moves toward high gain and the old SplatWorld fire regime.

But that envelope is **not learned**. It follows largely from the construction

```text
D_i(a) proportional to node_coord_i dot a
```

with roughly sphere-distributed node coordinates. Moving farther from the origin grows several diagonal detunings at once. The radial trend is therefore a structural prior of the address mechanism, not evidence that a biological fovea emerged.

The next experiment was designed so that this radial trend cannot explain success.

---

## Design

One hidden teacher plate starts from the same initial material as the student and receives 80 deterministic conservative material transfers of size `0.04`.

The teacher and student both have:

```text
operator dimension                 6
material edges                    12
independent material DOF          11   (because sum(g) is conserved)
```

The four taught addresses are all at **exactly radius 2**:

```text
(+2,  0, 0)
( 0, +2, 0)
(-2,  0, 0)
( 0, -2, 0)
```

Held-out evaluation includes 60 intermediate angles around the **same radius-2 ring**, plus a dense interior grid. Therefore ordinary center/periphery falloff supplies no information about which held-out angle should have which response.

Three students are compared.

### FULL positive control

All six operator-response components are supplied at all four anchors:

```text
4 addresses x 6 components = 24 scalar observations
```

This asks whether the constrained material family can recover the hidden teacher at all.

### SPARSE experiment

The student receives only two fixed random linear projections of the six-component response at each anchor:

```text
4 addresses x 2 measurements = 8 scalar observations
```

That is deliberately fewer observations than the **11 independent material degrees of freedom**. The sparse measurements do not uniquely specify the teacher material.

### SHUFFLED kill control

Exactly the same eight sparse target measurements are cyclically assigned to the wrong physical addresses.

All fitting uses the same derivative-free rule: try conservative local edge-to-edge material transfers, retain the best improving transfer, and reduce the dither size on a fixed schedule. No gradient, decoder pixel, teacher material, held-out address, or held-out image participates in fitting.

---

## Result

The exact Python 3.11 and 3.12 CI runs agree.

### Held-out same-radius ring

Relative operator-response field error:

| system | held-out radius-2 ring error |
|---|---:|
| untouched student | **0.248868** |
| **SPARSE: 8 scalar observations** | **0.077162** |
| FULL positive control | **0.005681** |
| SHUFFLED-address control | **0.448567** |
| pure radius-only predictor | **0.965725** |

Thus the underdetermined sparse fit removes about **69.0% of the original held-out ring error**:

```text
0.248868 -> 0.077162
```

while shuffled address labels make the reconstruction substantially worse than doing nothing.

A predictor that knows only radius is almost useless on this ring (`0.965725`) because the teacher has strong angular structure even though radius is fixed.

### Dense held-out grid

Relative field error over the dense central grid:

| system | grid error |
|---|---:|
| untouched student | **0.244299** |
| **SPARSE** | **0.076172** |
| FULL | **0.006656** |
| SHUFFLED | **0.478586** |

So the same sparse fit generalizes away from the four taught points as well.

### Directional agreement on the untaught ring

Median cosine similarity between the predicted and teacher six-component response vectors:

| system | median cosine |
|---|---:|
| untouched student | 0.976070 |
| **SPARSE** | **0.997479** |
| FULL | 0.999988 |
| SHUFFLED | 0.911341 |

### It did not simply recover the hidden material

The sparse student remains far from the teacher in material space:

```text
||g_sparse - g_teacher|| = 0.418296
```

while the full positive control gets much closer:

```text
||g_full - g_teacher|| = 0.076076
```

Yet the sparse student's held-out field error is already only `0.077`.

That matters because the sparse result is not merely "four observations happened to identify the exact hidden plate." Eight scalar constraints are fewer than the eleven independent conserved-material variables, and the learned plate is materially different from the teacher. What generalizes is the **operator field induced by the shared physical parameterization**.

### Conservation

All systems retain the same total material to numerical precision:

```text
sum(g) = 3.7745606621765675   (base)
       = 3.774560662176568    (teacher)
       = 3.774560662176568    (sparse)
       = 3.7745606621765675   (full)
       = 3.774560662176567    (shuffled)
```

---

## Preregistered verdict

All fixed checks passed:

```text
PASS  full positive control
PASS  sparse beats untouched base by >2x
PASS  sparse beats shuffled-address control by >2x
PASS  sparse beats radius-only prediction by >3x
PASS  material conservation
PASS  sparse observations < independent material DOF
```

The code records:

```text
SPARSE_SAME_RADIUS_INTERPOLATION_SURVIVES_FOVEA_CONTROL
```

---

## What this means

This is stronger than the first operator atlas result.

The first atlas showed:

```text
one g -> many smoothly related M_g(a)
```

and one material edit coherently deformed the whole family.

This experiment now shows, inside a physically reachable teacher family:

```text
four sparse observations
        ->
change one shared g
        ->
untaught addresses become substantially more correct
```

Even when the untaught test points have **exactly the same radius as the taught points**.

So the interpolation cannot be explained by the fovea-shaped radial envelope alone. Physical address matters beyond radius, and changing the shared substrate transfers information between addresses.

The most compact statement is:

> **Sparse local evidence can reshape one shared operator substrate so that unobserved parts of its addressed transformation field move toward the same hidden world.**

That is not the same as storing four outputs and interpolating their pixels. The student updates only one conserved material vector `g`; every held-out operator is regenerated from that changed substrate.

---

## What it does *not* mean

The hidden teacher is generated by the same model family as the student. This is therefore a controlled **system-identification / operator-family interpolation** result, not evidence that arbitrary real images can be organized this way.

The address coordinates still do not mean physical camera pose.

The teacher's targets are operator responses, not four independently supplied photographs.

The fovea likeness remains a useful structural analogy, but the radial envelope itself is largely imposed by `D(a)` and should not be described as an emergent fovea.

The real next wall is semantic teaching: give a few addresses observations that come from an external coherent object or transformation, rather than from another plate in the same family, and ask whether the untaught addresses acquire the missing geometry.

---

Reproduce the numeric experiment:

```bash
python sparse_world.py --no_render
```

Render the real decoder comparison:

```bash
python sparse_world.py --cpu
```

The render writes teacher, untouched, sparse, full and shuffled atlases plus per-address error maps and exact JSON/NPZ receipts.
