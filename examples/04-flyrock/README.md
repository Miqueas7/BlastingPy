# Flyrock

Hole table -> per-hole ranges of the published models -> governing hole -> clearance distance -> Roth for one hole.

```
python flyrock.py
```

The hole table is synthetic (six invented holes of 165 mm, one with a charge column shorter than 1 m, one vertical); it
says nothing about any real blast. `k = 27` is the Richards & Moore value for hard competent rock; you supply your own.
Nothing is written to disk. The output shows the table of ranges with `max_model_m` (the largest range across the
hole-specific models of each hole) and `lundborg_max_m` (Lundborg's maximum, which depends only on the diameter: the
same for every hole), the warnings (among them the holes where Lundborg's maximum is more than twice the hole-specific
models), the worst hole, which governs the clearance of the blast, and the clearance for people and equipment with and
without the Peruvian regulatory minimum, computed from the largest hole-specific range. One line shows the clearance
with Lundborg's diameter-only maximum, for contrast: the models disagree widely, and choosing one is the engineer's
decision.
Then the minimum stemming for a target range, Roth's launch velocity for one hole and the published critiques.
