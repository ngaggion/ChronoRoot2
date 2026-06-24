#!/usr/bin/env python3
"""CLI wrapper for the PyQt screening preview viewer."""

import argparse
import sys

from PyQt5.QtWidgets import QApplication

import plant_viewer


def main():
    parser = argparse.ArgumentParser(description='Preview video sequence from images')
    parser.add_argument('--video-dir', required=True,
                        help='Directory containing the image sequence')
    parser.add_argument('--segmentation-dir', required=True,
                        help='Directory containing the segmentation images')
    parser.add_argument('--time-delta', type=float, default=15.0,
                        help='Time in minutes between frames (default: 15)')
    args = parser.parse_args()

    try:
        images, seg_files, conf = plant_viewer.load_screening_sequence(
            args.video_dir, args.segmentation_dir, args.time_delta
        )
        app = QApplication(sys.argv)
        window = plant_viewer.ChronoViewWindow(images, seg_files, None, conf)
        window.show()
        sys.exit(app.exec_())
    except Exception as e:
        print(f"Error during preview: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
