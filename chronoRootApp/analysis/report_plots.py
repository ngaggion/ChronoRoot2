"""Comparison-mode plots paired with statistical report outputs."""

import os

import matplotlib.pyplot as plt
import seaborn as sns

from .stats_utils import (
    _mode_spec,
    _usable_groups,
    ensure_factor_columns,
    get_enabled_comparison_modes,
)
from .utils.fileUtilities import normalize_factor_value, UNSPECIFIED_FACTOR
from .utils.report_paths import append_report_index, comparison_plot_path, comparison_stats_path
from .utils.report_style import legend_title_for_hue, palette_for_hue

plt.switch_backend('agg')


def _plot_title(metric_label, spec):
    return f'{metric_label} — {spec["header"]}'


def _should_skip_mode(data, mode, conf):
    spec = _mode_spec(mode, conf=conf)
    if spec['group_col'] == 'PlateCondition' and len(_usable_groups(data['PlateCondition'].unique())) < 2:
        return True
    if spec['group_col'] == 'ExtraVariable' and len(_usable_groups(data['ExtraVariable'].unique())) < 2:
        return True
    if spec['stratify_col'] == 'PlateCondition' and len(_usable_groups(data['PlateCondition'].unique())) < 1:
        return True
    if spec['stratify_col'] == 'ExtraVariable' and len(_usable_groups(data['ExtraVariable'].unique())) < 1:
        return True
    if spec['stratify_col'] == 'Experiment' and len(_usable_groups(data['Experiment'].unique())) < 1:
        return True
    return False


def _palette_kwargs(data, hue_col, conf):
    palette = palette_for_hue(data, hue_col)
    kwargs = {}
    if palette:
        kwargs['palette'] = palette
    return kwargs


def _set_hue_legend(ax, hue_col, conf):
    leg = ax.get_legend()
    if leg is not None:
        leg.set_title(legend_title_for_hue(hue_col, conf))


def _lineplot_kwargs(x_col, y_col, hue_col, facet_col=None, data=None, conf=None):
    kwargs = {
        'x': x_col,
        'y': y_col,
        'hue': hue_col,
        'errorbar': 'se',
        'height': 4,
        'aspect': 1.2,
        'kind': 'line',
    }
    if facet_col:
        kwargs['col'] = facet_col
    if data is not None and conf is not None:
        kwargs.update(_palette_kwargs(data, hue_col, conf))
    return kwargs


def _catplot_kwargs(data, x, y, hue, col, conf, kind='box'):
    kwargs = {
        'data': data, 'x': x, 'y': y, 'hue': hue, 'col': col,
        'kind': kind, 'height': 4, 'aspect': 1.2,
    }
    kwargs.update(_palette_kwargs(data, hue, conf))
    return kwargs


def plot_comparison_mode(conf, data, metric, mode, output_path, *,
                         x_col='ElapsedTime (h)', metric_label=None, title=None,
                         module=None, metric_slug_name=None, analysis_type='temporal',
                         register_index=True):
    """Save a line plot matching the grouping used for a comparison mode."""
    data = ensure_factor_columns(data)
    if _should_skip_mode(data, mode, conf):
        return False

    spec = _mode_spec(mode, conf=conf)
    metric_label = metric_label or metric
    title = title or _plot_title(metric_label, spec)

    plt.ioff()
    try:
        if mode == 'by_genotype':
            g = sns.relplot(data=data, **_lineplot_kwargs(x_col, metric, 'Experiment', data=data, conf=conf))
        elif mode == 'genotype_by_plate':
            g = sns.relplot(
                data=data, **_lineplot_kwargs(x_col, metric, 'Experiment', 'PlateCondition', data=data, conf=conf),
                col_order=sorted(
                    [v for v in data['PlateCondition'].unique() if normalize_factor_value(v) != UNSPECIFIED_FACTOR],
                    key=str,
                ),
            )
        elif mode == 'genotype_by_extra':
            g = sns.relplot(
                data=data, **_lineplot_kwargs(x_col, metric, 'Experiment', 'ExtraVariable', data=data, conf=conf),
                col_order=sorted(
                    [v for v in data['ExtraVariable'].unique() if normalize_factor_value(v) != UNSPECIFIED_FACTOR],
                    key=str,
                ),
            )
        elif mode == 'by_plate_condition':
            g = sns.relplot(data=data, **_lineplot_kwargs(x_col, metric, 'PlateCondition', data=data, conf=conf))
        elif mode == 'by_extra_variable':
            g = sns.relplot(data=data, **_lineplot_kwargs(x_col, metric, 'ExtraVariable', data=data, conf=conf))
        elif mode == 'plate_within_genotype':
            g = sns.relplot(
                data=data, **_lineplot_kwargs(x_col, metric, 'PlateCondition', 'Experiment', data=data, conf=conf),
                col_order=sorted(data['Experiment'].unique(), key=str),
            )
        elif mode == 'extra_within_genotype':
            g = sns.relplot(
                data=data, **_lineplot_kwargs(x_col, metric, 'ExtraVariable', 'Experiment', data=data, conf=conf),
                col_order=sorted(data['Experiment'].unique(), key=str),
            )
        else:
            return False

        g.fig.suptitle(title, y=1.02)
        g.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close('all')

        if register_index and module and metric_slug_name:
            stats_path = comparison_stats_path(
                os.path.dirname(output_path), mode, metric_slug=metric_slug_name,
            )
            append_report_index(
                conf, module, metric_slug_name, analysis_type, mode,
                plot_file_path=output_path,
                stats_file_path=stats_path if os.path.exists(stats_path) else None,
                description=title,
            )
        return True
    except Exception:
        plt.close('all')
        return False


def plot_scalar_comparison_mode(conf, data, metric, mode, base_dir, *,
                                module=None, metric_slug_name=None, analysis_type='scalar',
                                metric_label=None, x_col='Experiment', plot_kind='box'):
    """Box/swarm plot for scalar metrics (FPCA PCs, etc.)."""
    data = ensure_factor_columns(data)
    if _should_skip_mode(data, mode, conf):
        return False

    spec = _mode_spec(mode, conf=conf)
    metric_label = metric_label or metric
    title = _plot_title(metric_label, spec)
    output_path = comparison_plot_path(base_dir, mode, metric_slug=metric_slug_name or '')
    plt.ioff()

    try:
        if mode == 'by_genotype':
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.boxplot(data=data, x=x_col, y=metric, hue='Experiment', ax=ax,
                        **_palette_kwargs(data, 'Experiment', conf))
            ax.set_title(title)
            _set_hue_legend(ax, 'Experiment', conf)
        elif mode == 'genotype_by_plate':
            g = sns.catplot(**_catplot_kwargs(data, 'Experiment', metric, 'Experiment', 'PlateCondition', conf, plot_kind))
            g.fig.suptitle(title, y=1.02)
            g.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close('all')
            return True
        elif mode == 'genotype_by_extra':
            g = sns.catplot(**_catplot_kwargs(data, 'Experiment', metric, 'Experiment', 'ExtraVariable', conf, plot_kind))
            g.fig.suptitle(title, y=1.02)
            g.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close('all')
            return True
        elif mode == 'by_plate_condition':
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.boxplot(data=data, x='PlateCondition', y=metric, hue='PlateCondition', ax=ax,
                        **_palette_kwargs(data, 'PlateCondition', conf))
            ax.set_title(title)
            _set_hue_legend(ax, 'PlateCondition', conf)
        elif mode == 'by_extra_variable':
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.boxplot(data=data, x='ExtraVariable', y=metric, hue='ExtraVariable', ax=ax,
                        **_palette_kwargs(data, 'ExtraVariable', conf))
            ax.set_title(title)
            _set_hue_legend(ax, 'ExtraVariable', conf)
        elif mode == 'plate_within_genotype':
            g = sns.catplot(**_catplot_kwargs(data, 'PlateCondition', metric, 'PlateCondition', 'Experiment', conf, plot_kind))
            g.fig.suptitle(title, y=1.02)
            g.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close('all')
            return True
        elif mode == 'extra_within_genotype':
            g = sns.catplot(**_catplot_kwargs(data, 'ExtraVariable', metric, 'ExtraVariable', 'Experiment', conf, plot_kind))
            g.fig.suptitle(title, y=1.02)
            g.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close('all')
            return True
        else:
            return False

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close('all')
        return True
    except Exception:
        plt.close('all')
        return False


def plot_interval_comparison_mode(conf, data, metric, mode, base_dir, *,
                                   x_col='Day', module=None, metric_slug_name=None,
                                   analysis_type='interval', metric_label=None):
    """Violin/swarm plot for interval-based metrics (convex hull, angles)."""
    data = ensure_factor_columns(data)
    if _should_skip_mode(data, mode, conf):
        return False

    spec = _mode_spec(mode, conf=conf)
    metric_label = metric_label or metric
    title = _plot_title(metric_label, spec)
    output_path = comparison_plot_path(base_dir, mode, metric_slug=metric_slug_name or '')
    plt.ioff()

    try:
        if mode == 'by_genotype':
            fig, ax = plt.subplots(figsize=(8, 6))
            exp_palette = _palette_kwargs(data, 'Experiment', conf)
            sns.violinplot(data=data, x=x_col, y=metric, hue='Experiment', ax=ax, inner=None, **exp_palette)
            sns.swarmplot(data=data, x=x_col, y=metric, hue='Experiment', ax=ax, dodge=True, size=3, **exp_palette)
            ax.set_title(title)
            _set_hue_legend(ax, 'Experiment', conf)
        elif mode == 'genotype_by_plate':
            g = sns.catplot(**_catplot_kwargs(data, x_col, metric, 'Experiment', 'PlateCondition', conf, 'violin'))
            g.fig.suptitle(title, y=1.02)
            g.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close('all')
            return True
        elif mode == 'genotype_by_extra':
            g = sns.catplot(**_catplot_kwargs(data, x_col, metric, 'Experiment', 'ExtraVariable', conf, 'violin'))
            g.fig.suptitle(title, y=1.02)
            g.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close('all')
            return True
        elif mode == 'by_plate_condition':
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.violinplot(data=data, x=x_col, y=metric, hue='PlateCondition', ax=ax,
                           **_palette_kwargs(data, 'PlateCondition', conf))
            ax.set_title(title)
            _set_hue_legend(ax, 'PlateCondition', conf)
        elif mode == 'by_extra_variable':
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.violinplot(data=data, x=x_col, y=metric, hue='ExtraVariable', ax=ax,
                           **_palette_kwargs(data, 'ExtraVariable', conf))
            ax.set_title(title)
            _set_hue_legend(ax, 'ExtraVariable', conf)
        elif mode == 'plate_within_genotype':
            g = sns.catplot(**_catplot_kwargs(data, x_col, metric, 'PlateCondition', 'Experiment', conf, 'violin'))
            g.fig.suptitle(title, y=1.02)
            g.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close('all')
            return True
        elif mode == 'extra_within_genotype':
            g = sns.catplot(**_catplot_kwargs(data, x_col, metric, 'ExtraVariable', 'Experiment', conf, 'violin'))
            g.fig.suptitle(title, y=1.02)
            g.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close('all')
            return True
        else:
            return False

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close('all')
        return True
    except Exception:
        plt.close('all')
        return False


def emit_temporal_comparison_plots(conf, data, metric, base_dir, *, module, metric_slug_name,
                                   analysis_type='temporal', metric_label=None):
    """Generate paired {metric}_{mode}.png for every enabled comparison mode."""
    for mode in get_enabled_comparison_modes(conf):
        output_path = comparison_plot_path(base_dir, mode, metric_slug=metric_slug_name)
        plot_comparison_mode(
            conf, data, metric, mode, output_path,
            metric_label=metric_label or metric,
            module=module, metric_slug_name=metric_slug_name,
            analysis_type=analysis_type, register_index=True,
        )


def emit_interval_comparison_plots(conf, data, metric, base_dir, *, module, metric_slug_name,
                                   analysis_type='interval', x_col='Day', metric_label=None):
    for mode in get_enabled_comparison_modes(conf):
        plot_interval_comparison_mode(
            conf, data, metric, mode, base_dir,
            x_col=x_col, module=module, metric_slug_name=metric_slug_name,
            analysis_type=analysis_type, metric_label=metric_label or metric,
        )
        spec = _mode_spec(mode, conf=conf)
        title = _plot_title(metric_label or metric, spec)
        stats_path = comparison_stats_path(base_dir, mode, metric_slug=metric_slug_name)
        plot_path = comparison_plot_path(base_dir, mode, metric_slug=metric_slug_name)
        if os.path.exists(plot_path):
            append_report_index(
                conf, module, metric_slug_name, analysis_type, mode,
                plot_file_path=plot_path,
                stats_file_path=stats_path if os.path.exists(stats_path) else None,
                description=title,
            )


def emit_scalar_comparison_plots(conf, data, metric, base_dir, *, module, metric_slug_name,
                                 analysis_type='scalar', metric_label=None):
    for mode in get_enabled_comparison_modes(conf):
        plot_scalar_comparison_mode(
            conf, data, metric, mode, base_dir,
            module=module, metric_slug_name=metric_slug_name,
            analysis_type=analysis_type, metric_label=metric_label or metric,
        )
        spec = _mode_spec(mode, conf=conf)
        title = _plot_title(metric_label or metric, spec)
        stats_path = comparison_stats_path(base_dir, mode, metric_slug=metric_slug_name)
        plot_path = comparison_plot_path(base_dir, mode, metric_slug=metric_slug_name)
        if os.path.exists(plot_path):
            append_report_index(
                conf, module, metric_slug_name, analysis_type, mode,
                plot_file_path=plot_path,
                stats_file_path=stats_path if os.path.exists(stats_path) else None,
                description=title,
            )
