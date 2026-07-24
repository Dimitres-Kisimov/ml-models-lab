# sku-text-classifier

fastText-style classifier: hashed character n-grams (3-5) into an
`nn.EmbeddingBag(mean)` followed by a linear softmax. Trains in seconds and
degrades gracefully to unseen part numbers.

- **Build:** deterministic FNV-1a feature hashing of boundary-marked char
  n-grams (`hashing.py`, pure Python and unit-tested) -> EmbeddingBag ->
  linear layer.
- **Improvement:** a regex sub-token split of alphanumerics runs *before*
  hashing (`M8x40` -> `m8`, `x40`, `8`, `40`) so a never-before-seen SKU is
  represented by its shape; inverse-frequency class weights stop rare
  categories being drowned by common ones.
- **Data:** 12 categories x ~300 templated descriptions with deliberate
  cross-category vocabulary overlap (shared materials, brands, sizes), so the
  task is not trivially separable (`mllab/synth.make_sku_text`).
- **Metric:** macro-F1 on a stratified split (not accuracy) + a row-normalised
  confusion matrix figure.

Measured (seed 5): macro-F1 0.963 vs a majority-class baseline of 0.013.

## Sources

- fastText / bag of tricks - Joulin et al., 2016 - https://arxiv.org/abs/1607.01759
- TextCNN - Kim, EMNLP 2014 - https://arxiv.org/abs/1408.5882
- Character-level convolutional networks - Zhang, Zhao & LeCun, NIPS 2015 - https://arxiv.org/abs/1509.01626
- The feature-hashing trick - Weinberger et al., ICML 2009 - https://arxiv.org/abs/0902.2206
