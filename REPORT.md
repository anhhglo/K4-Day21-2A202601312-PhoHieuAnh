# MLflow experiment results

| Configuration | n_estimators | learning_rate | max_depth | F1 score | Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | 100 | 0.1 | 3 | 0.7109004739336493 | 0.878 |
| Weak comparison | 50 | 0.05 | 2 | 0.6051282051282051 | 0.846 |
| Strong candidate | 200 | 0.1 | 5 | 0.7149321266968326 | 0.874 |

## Production parameters

The selected run is the strong candidate: `n_estimators: 200`,
`learning_rate: 0.1`, and `max_depth: 5`. Its F1 score of
`0.7149321266968326` is the highest recorded score and exceeds the required
`0.65` threshold.

The approximately 75/25 imbalance permits a majority-only accuracy near 0.752
while positive-class F1 is 0, so F1 protects the minority outcome that the
service must detect.
