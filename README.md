# First Snow

An interactive snow-crystal growth demo for Genesis House, built on a
WebGL2 port of the Gravner–Griffeath cellular-automaton snow crystal
model. Every crystal shown is a real simulation output — nothing here is
hand-authored geometry.

- **[index.html](index.html)** — drag the puck on the dark circular plane
  to steer temperature and supersaturation.
- **[sliders.html](sliders.html)** — the same engine, controlled with
  three named sliders (Branchiness, Density, Asymmetry) instead of a 2D
  drag surface.

Live at: `https://sitara-systems.github.io/first-snow/`

## Attribution

The underlying growth model implementation is adapted from
[vishnubob/snowflake](https://github.com/vishnubob/snowflake) (MIT
licensed), itself an implementation of Gravner & Griffeath, "Modeling
Snow Crystal Growth II: A mesoscopic lattice map with plausible dynamics"
(2008). This demo is a from-scratch WebGL2 fragment-shader port of that
model's update rule, not a direct copy of the reference's code.

This is a prototype built for an internal creative-director review, not a
production deliverable.
