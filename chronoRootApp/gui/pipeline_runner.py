"""Launch ChronoRoot analysis pipeline scripts as subprocesses."""

import os
import subprocess

PROJECT_CONFIG_NAME = "project_config.json"


def project_config_path(project_dir):
  return os.path.join(project_dir, PROJECT_CONFIG_NAME)


def run_analysis(project_dir):
  subprocess.Popen([
      "python", "1_analysis.py",
      "--config", project_config_path(project_dir),
  ])


def run_analysis_restart(metadata_path):
  subprocess.Popen([
      "python", "1_analysis.py",
      "--config", metadata_path,
      "--restart",
  ])


def run_analysis_rerun(metadata_path):
  subprocess.Popen([
      "python", "1_analysis.py",
      "--config", metadata_path,
      "--rerun",
  ])


def run_postprocess(project_dir):
  subprocess.Popen([
      "python", "2_postprocess.py",
      "--config", project_config_path(project_dir),
  ])


def run_report(project_dir):
  subprocess.Popen([
      "python", "3_generateReport.py",
      "--config", project_config_path(project_dir),
  ])


def run_calibration_helper(video_dir):
  return subprocess.Popen(
      ["python", "calibration_helper.py", "--video-dir", video_dir],
      stdout=subprocess.PIPE,
      stderr=subprocess.PIPE,
  )
