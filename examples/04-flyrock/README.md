# Flyrock

Hole table -> per-hole ranges of the published models -> governing hole -> clearance distance -> Roth for one hole.

```
python flyrock.py
```

The hole table is synthetic (six invented holes of 165 mm, one with a charge column shorter than 1 m, one vertical); it
says nothing about any real blast. `k = 27` is the Richards & Moore value for hard competent rock; you supply your own.
Nothing is written to disk. The output shows the table of ranges with `max_model_m` (the largest range across the
models of each hole), the warnings (among them the holes where the models disagree by more than a factor of 2), the
worst hole, which governs the clearance of the blast, and the clearance for people and equipment with and without the
Peruvian regulatory minimum, computed from the largest range across the models. One line shows the clearance with the
Richards & Moore range alone, for contrast: the models disagree widely, and choosing one is the engineer's decision.
Then the minimum stemming for a target range, Roth's launch velocity for one hole and the published critiques.
