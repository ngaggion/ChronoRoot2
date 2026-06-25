"""Shared axis labels and stable color palettes for ChronoRoot report figures."""

import seaborn as sns

from .fileUtilities import UNSPECIFIED_FACTOR, normalize_factor_value

GENOTYPE_PALETTE = 'tab10'
PLATE_PALETTE = 'Set2'
EXTRA_PALETTE = 'Set3'


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


def get_extra_variable_label(conf):
    """Backward-compatible alias used by stats_utils and report modules."""
    return get_extra_axis_label(conf)


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


def plate_palette_for_data(data, col='PlateCondition'):
    if col not in data.columns:
        return {}
    return plate_color_map(data[col].unique())


def extra_palette_for_data(data, col='ExtraVariable'):
    if col not in data.columns:
        return {}
    return extra_color_map(data[col].unique())


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
        return plate_palette_for_data(data)
    if hue_col == 'ExtraVariable':
        return extra_palette_for_data(data)
    return None
