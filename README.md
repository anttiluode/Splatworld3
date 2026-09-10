# SplatWorld3 — one plate, many views

![SplatWorld2 control](splatworld2.png)

`Splatworld3` is the branch where the **ObjektiYksi operator-family idea is inserted into the visual SplatWorld face manifold**.

The old `splatworld2.py` is intentionally still here as the control. It does:

```text
committed identity z0
        +
local measured transport rig B(z0)
        +
direct coefficients a
        ->
face
```

The new `splatworld3.py` puts one persistent shared object in the middle:

```text
                         address a=(x,y,z)
                                |
                                v
persistent material g -> dense M_g(a) -> local rig B(z0) -> decoder -> face
```

The key point is that **there is no stored matrix per address**.

There is one small material vector `g`. From it the program rebuilds a dense address-conditioned operator

```text
M_g(a) = [ K(g) + D(a) + i gamma I ]^-1
```

and samples that operator to obtain coefficients in the currently measured local face rig.

So the experiment is deliberately trying to make this idea visible:

> **one persistent substrate can expose a family of different transformations depending on how it is queried.**

That is the limited sense in which the "holographic plate" analogy is useful here. It is not optical holography and it is not yet a 3-D world model.

## What you should see

The main window is still the familiar SplatWorld face.

The right side now shows:

- four faces produced at four fixed addresses from the **same** `g`,
- the real part of the current dense matrix `M_g(a)`,
- a bar-code-like view of the persistent material couplings,
- how far the current matrix has moved from the matrix exposed at the origin.

Drag the mouse. You are no longer directly moving on two chosen latent axes. You are moving an **address**. The address changes the matrix, the matrix changes the coefficients injected into the local transport basis, and the decoder makes that visible as a face change.

The origin is centered so

```text
a = (0,0,0) -> committed identity
```

and nearby addresses expose nearby operator slices.

The third coordinate is controlled with the mouse wheel or `[` / `]`.

## The visually important button: `M`

Press **M** once.

It performs one tiny conservative material edit:

```text
one edge loses delta g
another edge gains delta g
sum(g) stays constant
```

Only the shared substrate changed.

Then look at the main face **and the four corner-address previews**.

If they all move, that is the point of this version:

```text
change one piece of g
        ->
change the family { M_g(a) }
        ->
several addressed manifestations move together
```

Press **U** to undo the edit and **G** to restore the original plate.

This is not learning yet. It is the simplest visual demonstration that one stored object is upstream of several effective transformations.

## Why this came after ObjektiYksi

`ObjektiYksi` established the useful computational distinction:

```text
ordinary layer:
    store W
    y = W x

operator substrate:
    store g
    query address a
    expose M_g(a)
    y = M_g(a) x
```

Its later experiments also showed that two addressed dense operators can share one conserved substrate, and that apparent interference between them need not imply incompatibility.

`SplatWorld3` does **not** copy those numerical claims onto faces. It borrows the object and asks a new question:

> what does an address-conditioned operator family look like when every change can be watched directly in an already-trained visual manifold?

## Relationship to SplatWorld2

SplatWorld2 found that different committed faces expose different local transport-like directions. One face may have a local mode that looks like yaw; another may expose hair, expression, scale, or a mixture.

SplatWorld3 keeps that local measurement:

```text
B = measured local transport-like basis around z0
```

but instead of letting the mouse choose coefficients directly, it uses

```text
c(a) = sample( M_g(a) )
z(a) = z0 + gain * c(a) B
```

This gives two nested sources of state dependence:

```text
shared plate g determines the address-conditioned operator
current identity z0 determines what those coefficients mean visually
```

That is why pressing **N** is interesting. A new identity gets a newly measured local rig, but **the operator plate survives**. The same plate is now being interpreted through a different local visual Jacobian.

## Run

The existing ONNX decoder is already in the repository.

```bash
pip install -r requirements.txt
python splatworld3.py --selftest
python splatworld3.py
```

Automatic address motion:

```bash
python splatworld3.py --auto
```

The old direct explorer remains available:

```bash
python splatworld2.py
```

## Controls

- **drag** — move address `x,y`
- **right drag** — finer address motion
- **mouse wheel** or **[ / ]** — move the third address coordinate
- **O** — operator / direct-coordinate A/B
- **M** — one conservative mutation of shared material `g`
- **U** — undo the latest material mutation
- **G** — reset material to the startup plate
- **ENTER** — commit the currently produced face as the new identity; re-probe its local rig while keeping the plate
- **N** — new random identity; plate persists
- **P** — previous committed identity; plate persists
- **L** — identity-lock / raw decoder A/B
- **A** — automatic address motion
- **R** — return address to the origin
- **K** — save the current plate as `operator_plate.npz`
- **S** — save the current UI frame
- **Q / Esc** — quit

A saved plate can be reloaded:

```bash
python splatworld3.py --plate operator_plate.npz
```

## What the operator actually is

`operator_plate.py` is intentionally small and independent of the face model.

`g` contains positive edge couplings on a small reciprocal graph. Those couplings form a Laplacian-like stiffness matrix `K(g)`. The address changes diagonal loading and slightly shifts the carrier coordinate. The program then solves the resulting dense inverse response.

The visual face control uses a fixed complex illumination vector to sample the matrix. The response is centered at the origin and bounded before it is projected into the local SplatWorld basis.

That means the current implementation is a **toy operator substrate**, not a learned physical model of faces.

Its standalone self-test checks that:

1. changing address changes the exposed matrix,
2. the same address deterministically gives the same matrix,
3. one local material transfer conserves total material,
4. that one transfer changes an exposed response,
5. derivative-free conservative dithering can fit several matrix slices produced by a reachable hidden teacher plate.

Run it directly with:

```bash
python operator_plate.py
```

## What would count as interesting now

Do not ask whether the UI looks futuristic. Ask what the family does.

The first things worth watching are:

- Do nearby addresses produce coherent motion or unrelated repainting?
- Do loops in address space approximately return to the same visual state?
- When `g` is mutated once, do several addresses deform in related ways rather than independently?
- Does the same saved plate retain recognizable structure when it is carried to a different committed identity/local rig?
- Does operator mode produce behavior qualitatively different from the direct-coordinate control under **O**?

If none of that is interesting, the operator layer is unnecessary and SplatWorld2 is the better program.

If coherent structure does appear, then the next step is **not another numbered gate**. It is to let the plate learn from several visually defined address/view relations and see whether the spaces between the trained addresses become a coherent manifold.

## Honest boundary

The three coordinates called `x,y,z` here are **abstract query coordinates**. They are not certified camera position, head pose, physical space, or disentangled semantic axes.

The four preview faces are not four true views of one 3-D person. They are four outputs generated by interrogating the same operator substrate at four addresses and then passing those responses through the local face rig.

So the current claim is deliberately small:

> **SplatWorld3 makes one-shared-substrate / many-address-conditioned-operators visible inside an existing learned image manifold.**

The interesting question is what kind of visual geometry, if any, emerges from that arrangement.

---

`Splatworld2.py` and `splatworld2.png` remain as the untouched control lineage. The decoder was trained from CelebA; check the dataset's own terms before commercial use of model outputs.
