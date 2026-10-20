# Visual Panda Lift archive

- Read README.md, RESULTS.md, docs/DEVELOPMENT.md and docs/REPRODUCING.md first.
- Original sources in frozen/ must remain byte-identical; published runner has separately documented release fixes.
- Preserve datasets, models, trajectories and failures. Do not rerun consumed final tests or label task50000–50099 as unseen for new experiments.
- Original experiment complete: six models,60000updates,100physical tasks,1200final replays,1800record audit. BC97.67%,DP98.78%,paired95%[-0.44,+3.00]points; no confirmed advantage.
- Data/checkpoints/environments/private logs stay local. No robot/environment changes or automatic new training.
- Any new experiment must fix its scope,budget and independent holdout first. Personal learning/mastery is distinct from tool-assisted experimental completion.
