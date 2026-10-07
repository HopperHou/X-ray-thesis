# B2 / AXIS Plan B full-training report

Status: VALID_COMPLETE

Six runs completed E50; 4,150 optimizer steps per arm. E50 is the sole primary checkpoint.

## Official development metrics

|Model|AP50 (%)|mAP50–95 (%)|H (%)|
|---|---:|---:|---:|
|A0-R|53.286486|27.355653|9.837038|
|B2_seed42|53.354758|27.561786|10.068795|
|AXIS_seed42|53.357673|27.551901|10.049955|
|B2_seed43|53.306074|27.585844|10.115609|
|AXIS_seed43|53.307777|27.589171|10.121212|
|B2_seed44|53.374605|27.556051|10.071997|
|AXIS_seed44|53.373047|27.562526|10.088708|

## All fixed metrics (%)

|Model|mAP50_95|AP50|AP75|AP80|AP85|AP90|AP95|H|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|A0-R|27.355653|53.286486|24.251998|15.450566|7.459391|1.898164|0.125069|9.837038|
|B2_seed42|27.561786|53.354758|24.436671|16.147823|7.540665|2.093403|0.125415|10.068795|
|AXIS_seed42|27.551901|53.357673|24.398238|16.096322|7.542797|2.093587|0.118828|10.049955|
|B2_seed43|27.585844|53.306074|24.644030|16.093505|7.543264|2.157574|0.139671|10.115609|
|AXIS_seed43|27.589171|53.307777|24.661527|16.098596|7.597451|2.108816|0.139671|10.121212|
|B2_seed44|27.556051|53.374605|24.723422|15.903620|7.661872|1.991118|0.079954|10.071997|
|AXIS_seed44|27.562526|53.373047|24.714316|16.003565|7.661847|1.991058|0.072753|10.088708|

## Three-seed mean and sample SD (%)

|Arm / statistic|mAP50_95|AP50|AP75|AP80|AP85|AP90|AP95|H|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|B2 / mean|27.567894|53.345146|24.601374|16.048316|7.581934|2.080698|0.115013|10.085467|
|B2 / sample_SD|0.015808|0.035262|0.148058|0.128220|0.069241|0.083952|0.031188|0.026153|
|AXIS / mean|27.567866|53.346166|24.591360|16.066161|7.600698|2.064487|0.110417|10.086625|
|AXIS / sample_SD|0.019201|0.034123|0.169318|0.054222|0.059591|0.064046|0.034242|0.035674|

## Paired differences (percentage points)

|Comparison|3-seed mean mAP difference|Patient-bootstrap conditional 95% CI|
|---|---:|---:|
|AXIS_minus_A0-R|+0.212213|[+0.057255, +0.377980]|
|AXIS_minus_B2|-0.000028|[-0.016569, +0.012908]|
|B2_minus_A0-R|+0.212241|[+0.060337, +0.379422]|

## Interpretation boundary

AXIS − A0-R evaluates the complete system. AXIS − B2 evaluates the additional axis geometry term. Report all signs, all seeds, and sample SD; no result direction changes VALID_COMPLETE.

Development has been repeatedly exposed. These conditional confidence intervals do not remove model-selection bias or establish independent external generalization. Test data were not read. Seed42 E50 is the predeclared representative; no best epoch or seed selection.

## Files

See comparison_summary.json, bootstrap_summary.json, per_class.csv, convergence.csv, pairing_check.json, evaluation_check.json and gpu_holder_restore_status.json.
