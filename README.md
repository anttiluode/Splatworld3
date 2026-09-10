# SplatWorld3 — one plate, many views

![Whole operator atlas](operator_atlas/atlas_before_faces.png)

`SplatWorld3` inserts the **ObjektiYksi operator-family idea** into the existing SplatWorld face manifold.

The persistent object is not one matrix per view. It is one small material vector `g`:

```text
                         query a=(x,y,z)
                                |
                                v
persistent material g -> dense M_g(a) -> local rig B(z0) -> decoder -> face
```

with

```text
M_g(a) = [K(g) + D(a) + i gamma I]^-1 .
```

Different queries expose different dense matrices from the **same stored `g`**.

**[Read the working paper: One Plate, Many Views](PAPER.md)**

---

## First whole-world atlas result

The live app originally showed one query plus four neighboring queries. `operator_atlas.py` now asks the more revealing question:

> **What entire visual landscape is exposed by one persistent `g`?**

The committed experiment uses a `10 x 10` grid over

```text
x,y in [-5,+5]
z = 0
motion_gain = 9
```

so all 100 tiles below come from one plate queried at different addresses.

![Atlas before one plate mutation](operator_atlas/atlas_before_faces.png)

The white number in each tile is the relative operator distance

```text
||M_g(a)-M_g(0)|| / ||M_g(0)|| .
```

The matching operator diagnostics are:

![Operator atlas metrics before mutation](operator_atlas/atlas_before_metrics.png)

Three things are visible at once:

- a broad central region of smooth, recognizable face transitions;
- increasingly different / higher-gain operator slices toward the outside;
- the old SplatWorld **fire** where a locally measured face chart is driven far outside its calibrated region.

This is not merely a visual impression. In the six-dimensional operator-response space, adjacent atlas cells have only **33.8% of the mean separation of arbitrary pairs**. PCA puts **78.21%** of atlas variation in the first two directions, **89.41%** in three, and **96.69%** in four.

At the same time, large operator distance and poor conditioning are strongly linked:

```text
corr(relative matrix distance, cond(A)) = 0.9645
corr(relative matrix distance, |c|)     = 0.8820
```

So the outer spectacular regions are not automatically "hidden worlds". Much of the edge behavior is strong operator response plus extrapolation of a local decoder chart.

---

## Scratch the plate once

The atlas was then regenerated after **one** conservative material transfer:

```text
g_8 -= 0.12
g_5 += 0.12
sum(g) before = 3.9431981293
sum(g) after  = 3.9431981293
```

No pixel is inspected when choosing or applying the material edit.

The new world is:

![Atlas after one material mutation](operator_atlas/atlas_after_faces.png)

The absolute visual change is:

![Absolute pixel change after one material mutation](operator_atlas/atlas_difference.png)

and the operator-response displacement over address space is:

![Per-address operator coefficient displacement](operator_atlas/atlas_mutation_metrics.png)

One local edit produces

```text
mean |delta c| across 100 addresses = 0.8437
max  |delta c|                      = 4.7430
mean absolute pixel change          = 11.64 / 255
largest tile mean pixel change      = 56.97 / 255
```

But the atlas is **not scrambled**. Before/after spatial maps remain highly correlated:

```text
relative operator-distance map : 0.9934
condition-number map            : 0.9890
coefficient-norm map            : 0.9797
```

For horizontally/vertically adjacent addresses, the median cosine similarity of the six-dimensional mutation vectors is **0.915**.

That is the central result currently worth keeping:

> **One small persistent material object generates a continuous address-conditioned landscape of transformations; one local conserved edit deforms that whole landscape in a structured way rather than independently replacing its outputs.**

That is the limited computational sense in which the **holographic plate** analogy has earned something here.

It is still an analogy, not a claim of optical holography or a learned 3-D world.

---

## Reproduce the atlas

```bash
pip install -r requirements.txt
python operator_plate.py
python splatworld3.py --selftest
python operator_atlas.py --selftest
python operator_atlas.py --address_span 5 --motion_gain 9 --grid 10 --mutate --show
```

The committed result directory is [`operator_atlas/`](operator_atlas/):

```text
atlas_before_faces.png
atlas_before_metrics.png
atlas_before_data.npz
atlas_after_faces.png
atlas_after_metrics.png
atlas_after_data.npz
atlas_difference.png
atlas_mutation_metrics.png
atlas_mutation_summary.json
```

The PNGs are the visual record. The NPZs contain the exact address grid, coefficient vectors, relative matrix distances, condition numbers, coefficient norms, material vector and operator scale. The JSON freezes the mutation and aggregate changes.

---

## Live explorer

```bash
python splatworld3.py
```

For the deliberately extreme regime that first made widely separated addressed states obvious:

```bash
python splatworld3.py --address_span 5 --motion_gain 9 --raw
```

The large face is the current query. Four thumbnails show nearby queries of the same plate.

The current UI corrects an important failure in the first operator build: the operator was changing mathematically but its response was too small to see, while the raw matrix heatmap exaggerated small changes by rescaling every frame. The current version matches operator/direct coefficient budgets, shows actual `|c|`, displays `Re[M_g(a)-M_g(0)]`, and makes **M** select one material-conserving edit with a visible operator-family effect.

### Controls

- **drag** — move abstract query `x,y`
- **right drag** — finer query movement
- **mouse wheel** or **[ / ]** — move the third query coordinate
- **O** — operator / direct-coordinate A/B
- **M** — one visible high-impact conservative material transfer
- **U** — undo latest material transfer
- **G** — restore startup material
- **ENTER** — commit current output as a new identity and re-measure its local rig
- **N / P** — new / previous committed identity; plate persists
- **L** — identity-lock / raw decoder A/B
- **A** — automatic query motion
- **R** — return query to origin
- **K** — save plate to `operator_plate.npz`
- **S** — screenshot
- **Q / Esc** — quit

The old direct SplatWorld2 lineage remains available:

```bash
python splatworld2.py
```

---

## What the two nested objects are

SplatWorld2 measures a local transport-like basis around a committed face:

```text
B(z0) = locally measured visual directions.
```

SplatWorld3 inserts the operator family before that basis:

```text
c(a) = scale * response(M_g(a))
z(a) = z0 + c(a) B(z0).
```

So there are two state dependencies:

```text
g  determines which operator slice an address exposes
z0 determines what that operator response means inside the face decoder
```

The address therefore has **structure but no learned semantics yet**. Moving through `(x,y,z)` can change pose, hair, apparent identity, apparent sex, background, scale, or mixtures of them. The present plate was never taught that `x` should mean yaw or that neighboring coordinates must preserve one person.

---

## What is actually claimed

Supported here:

- one persistent `g` exposes many different dense operators;
- the addressed response family has coherent neighborhoods rather than behaving like independent random slots;
- the family is low-dimensional enough on this plane that four PCs explain about **96.7%** of coefficient variance;
- one local material-conserving edit changes many addressed operators simultaneously;
- the resulting deformation field is structured and the large-scale operator landscape largely survives.

Not established:

- `x,y,z` are not certified camera or physical-space coordinates;
- the atlas is not a reconstructed 3-D head;
- different tiles do not necessarily preserve one identity;
- the outer fire regions are not evidence of hidden physics;
- `M` is a visibility perturbation, not a biological or learned credit-assignment rule;
- one atlas from one seed is not a universal result.

The limited claim is therefore:

> **SplatWorld3 makes one-shared-substrate / many-address-conditioned-operators visible as a globally deformable landscape inside an existing learned image manifold.**

---

## What comes next

The next experiment should not merely increase gain or add another numbered gate.

Use several safe addresses in the central non-fire region, give them target views of **one identity**, and allow only `g` to change. Then render the complete atlas again.

The important test is the **untrained space between the target addresses**.

If only the trained points look right, the plate is multiplexed operator memory.

If intermediate addresses organize themselves into coherent unseen transitions — for example

```text
left profile -> three-quarter -> front -> three-quarter -> right profile
```

— then the stronger idea becomes testable:

```text
few observations -> one shared plate -> an interpolated addressed world.
```

That is where `SplatWorld3` goes next if the operator earns it.

---

The decoder was trained from CelebA; check the dataset's own terms before commercial use of model outputs.
