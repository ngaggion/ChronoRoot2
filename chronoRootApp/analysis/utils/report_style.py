"""Shared axis labels and stable color palettes for ChronoRoot report figures."""

import seaborn as sns

from .fileUtilities import UNSPECIFIED_FACTOR, normalize_factor_value

GENOTYPE_PALETTE = 'tab10'
PLATE_PALETTE = 'Set2'
EXTRA_PALETTE = 'husl'


def _clean_label(conf, key, default):
    label = conf.get(key, default) if conf else default
    label = str(label).strip() if label is not None else ''
    return label or default


def get_genotype_axis_label(conf):
    return _clean_label(conf, 'genotypeAxisLabel', 'Genotype')


def get_plate_axis_label(conf):
    return _clean_label(conf, 'plateConditionAxisLabel', 'Plate condition')


def get_extra_axis_label(conf):
    return _clean_label(conf, 'extraVariableLabel', 'Run')


_FACTOR_COLUMNS = frozenset({'Experiment', 'PlateCondition', 'ExtraVariable', 'Type'})


def axis_label_for_column(col, conf):
    """User-facing axis label for a DataFrame factor column."""
    return legend_title_for_hue(col, conf)


def apply_factor_axis_labels(ax, conf, *, x_col=None, hue_col=None):
    """Set friendly x-axis and hue legend titles on a single matplotlib Axes."""
    if x_col in _FACTOR_COLUMNS:
        ax.set_xlabel(axis_label_for_column(x_col, conf))
    if hue_col:
        leg = ax.get_legend()
        if leg is not None:
            leg.set_title(legend_title_for_hue(hue_col, conf))


def _usable_values(values):
    return sorted(
        [v for v in values if normalize_factor_value(v) != UNSPECIFIED_FACTOR],
        key=str,
    )


def genotype_color_map(values):
    keys = _usable_values(values)
    if not keys:
        return {}
    colors = sns.color_palette(GENOTYPE_PALETTE, n_colors=len(keys))
    return dict(zip(keys, colors))


def plate_color_map(values):
    keys = _usable_values(values)
    if not keys:
        return {}
    colors = sns.color_palette(PLATE_PALETTE, n_colors=len(keys))
    return dict(zip(keys, colors))


def extra_color_map(values):
    keys = _usable_values(values)
    if not keys:
        return {}
    colors = sns.color_palette(EXTRA_PALETTE, n_colors=len(keys))
    return dict(zip(keys, colors))


def genotype_palette_for_data(data, col='Experiment'):
    if col not in data.columns:
        return {}
    return genotype_color_map(data[col].unique())


def legend_title_for_hue(hue_col, conf):
    if hue_col == 'Experiment':
        return get_genotype_axis_label(conf)
    if hue_col == 'PlateCondition':
        return get_plate_axis_label(conf)
    if hue_col == 'ExtraVariable':
        return get_extra_axis_label(conf)
    if hue_col == 'Type':
        return get_genotype_axis_label(conf)
    return hue_col


def palette_for_hue(data, hue_col):
    if hue_col in ('Experiment', 'Type'):
        return genotype_palette_for_data(data, hue_col if hue_col in data.columns else 'Experiment')
    if hue_col == 'PlateCondition':
        if 'PlateCondition' not in data.columns:
            return {}
        return plate_color_map(data['PlateCondition'].unique())
    if hue_col == 'ExtraVariable':
        if 'ExtraVariable' not in data.columns:
            return {}
        return extra_color_map(data['ExtraVariable'].unique())
    return None
