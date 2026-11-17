| Method | Mean Reward | Loss Rate % | Jain's Index | Type |
| --- | --- | --- | --- | --- |
| Baseline: equal | -11.655 +/- 0.951 | 0.115% | 0.9873 | Static baseline |
| Baseline: proportional | -3.467 +/- 0.660 | 0.000% | 1.0000 | Static baseline |
| Learner: sac (ep 1000) | -27.421 +/- 3.438 | Logged per step | Dynamic policy | Continual DRL |
| Learner: sac_ewc (ep 1000) | -29.984 +/- 7.252 | Logged per step | Dynamic policy | Continual DRL |
