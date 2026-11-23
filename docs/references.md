# References

Papers used by the code. The PDFs in `references/` are kept only for the four methods that are
actually implemented; the PPO paper was removed because there is no reinforcement learning here.

| Paper | Local file | Used by |
|---|---|---|
| Chen & Guestrin, *XGBoost: A Scalable Tree Boosting System*, KDD 2016 | `references/xgboost_chen_guestrin_2016.pdf` | `EnsemblePredictor.train_xgb` |
| Koch, Zemel & Salakhutdinov, *Siamese Neural Networks for One-shot Image Recognition*, ICML 2015 | `references/siamese_koch_2015.pdf` | twin-tower player comparison |
| Zhang & Ma, *Ensemble Machine Learning* | `references/ensemble_stacking_zhang_2021.pdf` | NN + tree blending rationale |
| Thorp, *The Kelly Criterion in Blackjack, Sports Betting and the Stock Market* | `references/kelly_thorp_2008.pdf` | fractional Kelly reference strategy |

Data sources are documented in [`data-sources.md`](data-sources.md); the dataset data card is
[`dataset.md`](dataset.md).
