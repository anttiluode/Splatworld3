# SplatWorld3 — one plate, many views

![SplatWorld2 control](splatworld2.png)

`SplatWorld3` inserts the **ObjektiYksi operator-family idea** into the existing SplatWorld face manifold.

The persistent object is not a matrix per view. It is one small material vector `g`:

```text
                         query a=(x,y,z)
                                |
                                v
persistent material g -> dense M_g(a) -> local rig B(z0) -> decoder -> face
```

with

```text
M_g(a) = [K(g) + D(a) + i gamma I]^-1
```

Different queries expose different dense matrices from the same `g`. The matrix response is then projected into the local transport-like basis measured around the currently committed face.

This is a **visual operator-family experiment**, not a claim that the CelebA model contains a true 3-D head or world.

## Important v1.1 correction: the first UI hid the effect

The first operator build was mathematically changing `M_g(a)`, but its coupling into the SplatWorld local rig was much too weak to see. On the startup plate, the four operator-response norms were only about `0.10–0.16`, while a random conservative `M` mutation changed them by only about `1e-3` in norm. At the same time the raw-matrix heatmap rescaled itself every frame, making tiny matrix changes look visually dramatic.

That was a bad instrument, not an interesting null result.

The current version therefore:

- calibrates operator coefficients to the **same local-motion budget** as the direct-coordinate control;
- makes the four preview faces sample a **neighborhood around the current address**, so dragging moves the whole visible family;
- shows `|c|` for the current query and each preview;
- displays `Re[M_g(a)-M_g(0)]` instead of a misleading independently normalized raw matrix;
- makes **M** choose one conservative edge-to-edge transfer with high operator-family impact in the currently displayed neighborhood;
- prints the actual per-view coefficient displacement `|dc|` caused by that material edit.

The scale is fixed for a plate state while you mutate it, so a material edit cannot hide itself by causing an automatic rescaling.

## What to look at

Run:

```bash
pip install -r requirements.txt
python splatworld3.py --selftest
python splatworld3.py
```

The old control remains:

```bash
python splatworld2.py
```

In **OPERATOR** mode, drag around. The big face is the current query. The four smaller faces are nearby queries of the **same plate**, not fixed thumbnails. They should now visibly move as the query neighborhood moves.

Press **O** to switch to **DIRECT** mode. In DIRECT mode the large face deliberately bypasses `g`; changing `g` should not affect that large face. The previews still show the operator family.

Press **M** in OPERATOR mode. `M` performs exactly one material-conserving transfer:

```text
g_src -= delta
g_dst += delta
sum(g) unchanged
```

but chooses the source/destination edge pair that most changes the operator responses at the four currently displayed nearby queries. It does **not** inspect pixels and it is **not learning**. It is simply a visibility probe: one local change to `g`, chosen so we can actually see what changing the shared substrate does to the family.

The console and HUD report the mean and per-view `|dc|` caused by that one edit.

Press **U** to undo it exactly. Press **G** to restore the startup plate.

## Controls

- **drag** — move abstract query `x,y`
- **right drag** — finer query movement
- **mouse wheel** or **[ / ]** — move the third query coordinate
- **O** — operator / direct-coordinate A/B
- **M** — one visible high-impact conservative material transfer
- **U** — undo latest material transfer
- **G** — restore startup material
- **ENTER** — commit current output as new identity and re-measure its local rig
- **N / P** — new / previous committed identity; plate persists
- **L** — identity-lock / raw decoder A/B
- **A** — automatic query motion
- **R** — return query to origin
- **K** — save plate to `operator_plate.npz`
- **S** — screenshot
- **Q / Esc** — quit

A saved plate can be reloaded:

```bash
python splatworld3.py --plate operator_plate.npz
```

## What is actually being tested

SplatWorld2 already showed that the strongest local transport-like directions depend on the committed identity. One face may expose yaw-like motion while another exposes hair, expression, scale, or mixtures.

SplatWorld3 keeps that measured local rig:

```text
B(z0) = locally measured transport-like basis
```

but replaces direct mouse coefficients with an address-conditioned operator response:

```text
c(a) = scale * response(M_g(a))
z(a) = z0 + c(a) B(z0)
```

This gives two nested state dependencies:

```text
g determines which operator slice the query exposes
z0 determines what that operator response means visually
```

The question is therefore not “does this make a pretty face?” It is:

> **Does one shared substrate generate a coherent family of visual transformations when it is queried continuously?**

Useful observations would include coherent local motion, repeatable loops in query space, several nearby views deforming together after one `g` edit, or stable structure surviving when the same plate is carried to a new committed identity.

If operator mode remains arbitrary or visually no better than direct control, the operator layer has not bought us anything.

## Honest boundary

The coordinates `x,y,z` are abstract query coordinates. They are **not certified camera position, depth, head pose, or disentangled semantic axes**.

The four previews are not four true photographs of one 3-D person. They are outputs of the same learned face decoder after four nearby queries have been passed through one shared operator substrate and the current local face rig.

`M` is not a learning rule. It intentionally chooses a high-impact conservative material perturbation so the shared-substrate consequence can be seen rather than buried below the decoder's visual resolution.

The current limited claim is:

> **One persistent `g` generates many address-conditioned dense operators, and SplatWorld3 makes their shared deformation directly inspectable inside an existing visual manifold.**

The next interesting step, only if the live behavior earns it, is to train `g` on several visual relations and see whether the spaces between trained queries become coherent rather than memorized islands.

---

`operator_plate.py` is standalone NumPy and contains its own physical/operator sanity checks. `splatworld2.py` remains the untouched direct-control lineage. The decoder was trained from CelebA; check the dataset's terms before commercial use of model outputs.
