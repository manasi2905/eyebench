"""Create a report-ready comparison of all PoTeC stacking ablations."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


RESULTS_ROOT = Path('results')
REGIME_LABELS = {
    'seen_subject_unseen_item': 'Unseen text',
    'unseen_subject_seen_item': 'Unseen reader',
    'unseen_subject_unseen_item': 'Both unseen',
}
EXPERIMENT_ORDER = [
    'Original 17-feature stack',
    'Four proposal features',
    'Four features + reading speed',
    'Heterogeneous tabular bases',
    'Heterogeneous + RoBERTEye-F',
    'Heterogeneous + PLM-AS-RM',
]


def read_original_results() -> pd.DataFrame:
    results = pd.read_csv(
        RESULTS_ROOT / 'stacking_potec_de' / 'metrics_summary.csv'
    )
    results = results[results['eval_type'] == 'test'].copy()
    results['experiment'] = 'Original 17-feature stack'
    results['meta_learner'] = 'Logistic Regression'
    return results


def read_tuned_results(feature_set: str, experiment: str) -> pd.DataFrame:
    results = pd.read_csv(
        RESULTS_ROOT
        / 'stacking_potec_de_tuned'
        / feature_set
        / 'metrics_summary.csv'
    )
    results = results[
        (results['eval_type'] == 'test')
        & (results['model'] == 'stacking_ensemble')
    ].copy()
    results['experiment'] = experiment
    results['meta_learner'] = 'Logistic Regression'
    return results


def read_neural_fusion_results(
    directory_name: str,
    experiment: str,
) -> pd.DataFrame:
    results = pd.read_csv(RESULTS_ROOT / directory_name / 'metrics_summary.csv')
    results = results.copy()
    results['experiment'] = experiment
    results['meta_learner'] = results['meta_learner'].map(
        {'logistic_regression': 'Logistic Regression', 'mlp': 'Tiny MLP'}
    )
    return results


def collect_results() -> pd.DataFrame:
    frames = [
        read_original_results(),
        read_tuned_results('core', 'Four proposal features'),
        read_tuned_results(
            'core_plus_reading_speed',
            'Four features + reading speed',
        ),
        read_tuned_results('heterogeneous', 'Heterogeneous tabular bases'),
        read_neural_fusion_results(
            'stacking_potec_de_neural_fusion',
            'Heterogeneous + RoBERTEye-F',
        ),
        read_neural_fusion_results(
            'stacking_potec_de_neural_fusion_PLMASfArgs',
            'Heterogeneous + PLM-AS-RM',
        ),
    ]
    results = pd.concat(frames, ignore_index=True)
    results['experiment'] = pd.Categorical(
        results['experiment'],
        categories=EXPERIMENT_ORDER,
        ordered=True,
    )
    columns = [
        'experiment',
        'meta_learner',
        'eval_regime',
        'folds',
        'mean_auroc',
        'std_auroc',
        'mean_accuracy',
        'mean_balanced_accuracy',
        'mean_threshold_tuned_accuracy',
        'mean_threshold_tuned_balanced_accuracy',
    ]
    for column in columns:
        if column not in results:
            results[column] = pd.NA
    return results[columns].sort_values(
        ['experiment', 'meta_learner', 'eval_regime']
    )


def create_markdown_report(results: pd.DataFrame) -> str:
    table = results.pivot_table(
        index=['experiment', 'meta_learner'],
        columns='eval_regime',
        values=['mean_auroc', 'std_auroc'],
        observed=True,
    )
    lines = [
        '# Extended PoTeC stacking comparison',
        '',
        'All values are four-fold test AUROC (mean ± standard deviation).',
        '',
        '| Experiment | Meta-learner | Unseen text | Unseen reader | Both unseen | Average |',
        '|---|---|---:|---:|---:|---:|',
    ]
    for experiment, meta_learner in table.index:
        values = []
        means = []
        for regime in REGIME_LABELS:
            mean = table.loc[(experiment, meta_learner), ('mean_auroc', regime)]
            std = table.loc[(experiment, meta_learner), ('std_auroc', regime)]
            values.append(f'{mean:.3f} ± {std:.3f}')
            means.append(mean)
        lines.append(
            f'| {experiment} | {meta_learner} | '
            f'{values[0]} | {values[1]} | {values[2]} | '
            f'{sum(means) / len(means):.3f} |'
        )
    lines.extend(
        [
            '',
            '## Interpretation',
            '',
            '- Model-specific feature views improve the tabular stack for unseen readers and the fully unseen regime.',
            '- PLM-AS-RM plus Logistic Regression has the strongest average neural-fusion result.',
            '- The tiny MLP does not improve consistently over Logistic Regression and has higher variance.',
            '- Validation-selected thresholds remain unstable; AUROC is the primary metric.',
            '- Neural fusion is a holdout-stacking ablation: the meta-learner is trained on validation predictions and evaluated on test predictions.',
            '- Because the neural-input experiments were motivated after inspecting benchmark results, they should be described as exploratory rather than confirmatory.',
            '',
        ]
    )
    return '\n'.join(lines)


def main() -> None:
    output_dir = RESULTS_ROOT / 'stacking_potec_de_report'
    output_dir.mkdir(parents=True, exist_ok=True)
    results = collect_results()
    results.to_csv(output_dir / 'model_comparison.csv', index=False)
    (output_dir / 'REPORT.md').write_text(
        create_markdown_report(results),
        encoding='utf-8',
    )
    print(create_markdown_report(results))


if __name__ == '__main__':
    main()
