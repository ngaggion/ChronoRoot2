"""Shared Report/ folder path helpers — metric-first with paired plot/stats naming."""

import csv
import os
import re

from .report_utils import ensure_directory

MODULE_TEMPORAL = 'temporal_parameters'
MODULE_CONVEX = 'convex_hull'
MODULE_ANGLES = 'angles'

OVERVIEW_SLUG = 'overview'
INDIVIDUAL_PLOTS_DIR = 'individual_plots'
REPORT_INDEX_NAME = 'report_index.csv'

TEMPORAL_METRICS = [
    'MainRootLength (mm)',
    'LateralRootsLength (mm)',
    'TotalLength (mm)',
    'NumberOfLateralRoots',
    'DiscreteLateralDensity (LR/cm)',
    'MainOverTotal (%)',
    'HypocotylLength (mm)',
]

CONVEX_METRICS = [
    'Convex Hull Area',
    'Lateral Root Area Density',
    'Total Root Area Density',
    'Convex Hull Aspect Ratio',
    'Convex Hull Height',
    'Convex Hull Width',
]

FOURIER_PARENT_METRICS = {
    'MR': 'main_root_length',
    'TR': 'total_root_length',
}

ANGLE_METRICS = {
    'mean_emergence_angle': 'Mean emergence angle',
    'first_lr_tip_angle': 'First LR tip',
}

_REPORT_INDEX_ROWS = []


def metric_slug(display_name: str) -> str:
    """Convert a display metric name to a filesystem-safe folder slug."""
    text = str(display_name).strip().lower()
    text = text.replace('%', ' percent')
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'[\s-]+', '_', text).strip('_')
    return text or 'metric'


def report_root(conf) -> str:
    return os.path.join(conf['MainFolder'], 'Report')


def data_dir(conf) -> str:
    path = os.path.join(report_root(conf), 'data')
    ensure_directory(path)
    return path


def data_file(conf, filename: str) -> str:
    return os.path.join(data_dir(conf), filename)


def module_dir(conf, module: str) -> str:
    path = os.path.join(report_root(conf), module)
    ensure_directory(path)
    return path


def metric_dir(conf, module: str, slug: str) -> str:
    path = os.path.join(module_dir(conf, module), slug)
    ensure_directory(path)
    return path


def analysis_dir(conf, module: str, metric_slug_name: str, *subpath: str) -> str:
    """Metric folder or nested sub-analysis path (e.g. fpca/pc1, growth_speed)."""
    path = os.path.join(metric_dir(conf, module, metric_slug_name), *subpath)
    ensure_directory(path)
    return path


def comparison_stats_path(base_dir: str, mode: str, metric_slug: str = '') -> str:
    ensure_directory(base_dir)
    stem = f'{metric_slug}_{mode}' if metric_slug else mode
    return os.path.join(base_dir, f'{stem}_stats.txt')


def comparison_plot_path(base_dir: str, mode: str, metric_slug: str = '') -> str:
    ensure_directory(base_dir)
    stem = f'{metric_slug}_{mode}' if metric_slug else mode
    return os.path.join(base_dir, f'{stem}.png')


def stats_file(conf, module: str, slug: str, comparison_mode: str, *subpath: str) -> str:
    base = analysis_dir(conf, module, slug, *subpath) if subpath else metric_dir(conf, module, slug)
    return comparison_stats_path(base, comparison_mode, metric_slug=slug)


def plot_file(conf, module: str, slug: str, filename: str, *subpath: str) -> str:
    base = analysis_dir(conf, module, slug, *subpath) if subpath else metric_dir(conf, module, slug)
    ensure_directory(base)
    return os.path.join(base, filename)


def table_file(conf, module: str, slug: str, filename: str, *subpath: str) -> str:
    base = analysis_dir(conf, module, slug, *subpath) if subpath else metric_dir(conf, module, slug)
    ensure_directory(base)
    return os.path.join(base, filename)


def overview_dir(conf, module: str) -> str:
    return metric_dir(conf, module, OVERVIEW_SLUG)


def overview_plot(conf, module: str, filename: str) -> str:
    return os.path.join(overview_dir(conf, module), filename)


def overview_table(conf, module: str, filename: str) -> str:
    return os.path.join(overview_dir(conf, module), filename)


def angle_overlays_dir(conf, experiment_slug: str) -> str:
    path = os.path.join(module_dir(conf, MODULE_ANGLES), f'overlays_{experiment_slug}')
    ensure_directory(path)
    return path


def individual_plots_dir(conf, experiment_slug: str) -> str:
    path = os.path.join(report_root(conf), INDIVIDUAL_PLOTS_DIR, experiment_slug)
    ensure_directory(path)
    return path


def temporal_metric_slug(metric_column: str) -> str:
    mapping = {
        'MainRootLength (mm)': 'main_root_length',
        'LateralRootsLength (mm)': 'lateral_root_length',
        'TotalLength (mm)': 'total_root_length',
        'NumberOfLateralRoots': 'number_of_lateral_roots',
        'DiscreteLateralDensity (LR/cm)': 'lateral_root_density',
        'MainOverTotal (%)': 'main_over_total',
        'HypocotylLength (mm)': 'hypocotyl_length',
    }
    return mapping.get(metric_column, metric_slug(metric_column))


def rel_report_path(conf, absolute_path: str) -> str:
    return os.path.relpath(absolute_path, report_root(conf))


def reset_report_index():
    global _REPORT_INDEX_ROWS
    _REPORT_INDEX_ROWS = []


def append_report_index(conf, module, metric_slug_name, analysis_type, comparison_mode,
                        plot_file_path=None, stats_file_path=None, table_file_path=None, description=''):
    row = {
        'module': module,
        'metric_slug': metric_slug_name,
        'analysis_type': analysis_type,
        'comparison_mode': comparison_mode or '',
        'plot_file': rel_report_path(conf, plot_file_path) if plot_file_path else '',
        'stats_file': rel_report_path(conf, stats_file_path) if stats_file_path else '',
        'table_file': rel_report_path(conf, table_file_path) if table_file_path else '',
        'description': description,
    }
    key = (module, metric_slug_name, analysis_type, comparison_mode or '')
    for i, existing in enumerate(_REPORT_INDEX_ROWS):
        if (existing['module'], existing['metric_slug'], existing['analysis_type'],
                existing['comparison_mode']) == key:
            for field in ('plot_file', 'stats_file', 'table_file', 'description'):
                if row[field]:
                    existing[field] = row[field]
            return
    _REPORT_INDEX_ROWS.append(row)


def write_report_index(conf):
    path = os.path.join(report_root(conf), REPORT_INDEX_NAME)
    fieldnames = [
        'module', 'metric_slug', 'analysis_type', 'comparison_mode',
        'plot_file', 'stats_file', 'table_file', 'description',
    ]
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(_REPORT_INDEX_ROWS)
    return path
