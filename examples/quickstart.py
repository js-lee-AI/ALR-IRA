"""DFlash, KD, erase, ALR and ALR + IRA on toy Markov targets. CPU only, no downloads, under a second."""

import alr_ira

# Five toy worlds, each with an order-2 target, a corpus written by another chain and
# 200 held-out prompts. Every objective trains the same 3,200 blocks per world.
results = alr_ira.run_toy(worlds=5, seed=0)
print(alr_ira.format_results(results))
