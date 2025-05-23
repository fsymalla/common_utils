#!/usr/bin/env python

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TABLEAU_COLORS
import os


def load_data(filename):
    """
    Load data from a file containing a (n,2) array.

    Args:
        filename (str): Path to the file

    Returns:
        numpy.ndarray: The loaded data
    """
    try:
        data = np.loadtxt(filename)
        # Check if data has correct shape (n,2)
        if data.ndim == 1 and len(data) == 2:
            # Handle single row case
            data = data.reshape(1, 2)
        elif data.ndim != 2 or data.shape[1] != 2:
            raise ValueError(f"Data in {filename} should have shape (n,2), but has shape {data.shape}")
        return data
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return None


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Plot (n,2) data from file(s)')
    parser.add_argument('files', nargs='+', help='Input files containing (n,2) data')
    parser.add_argument('--save', '-s', help='Save plot to specified file instead of showing it')
    parser.add_argument('--title', '-t', default='', help='Plot title')
    parser.add_argument('--xlabel', '-x', default='X', help='X-axis label')
    parser.add_argument('--ylabel', '-y', default='Y', help='Y-axis label')
    parser.add_argument('--ylim', nargs='*',help='Y-axis limits in min max')
    parser.add_argument('--legend', '-l', action='store_true', help='Show legend')
    parser.add_argument('--figsize', nargs=2, type=float, default=[8, 6],
                        help='Figure size (width height) in inches')
    parser.add_argument('--dpi', type=int, default=100, help='DPI for saved figure')
    parser.add_argument('--no-grid', action='store_true', help='Turn off grid')

    args = parser.parse_args()

    # Create figure
    plt.figure(figsize=tuple(args.figsize))

    # Get list of colors
    colors = list(TABLEAU_COLORS.values())

    # Track if we've successfully loaded at least one file
    any_data_loaded = False

    # Plot each file with a different color
    for i, filename in enumerate(args.files):
        # Load data
        data = load_data(filename)
        if data is None:
            continue

        # Use modulo to cycle through colors if we have more files than colors
        color = colors[i % len(colors)]

        # Create label from filename (basename without extension)
        label = os.path.splitext(os.path.basename(filename))[0]

        # Plot data
        plt.plot(data[:, 0], data[:, 1], color=color, label=label)
        any_data_loaded = True

    if not any_data_loaded:
        print("No valid data files were loaded. Exiting.")
        return

    # Set labels and title
    plt.xlabel(args.xlabel)
    plt.ylabel(args.ylabel)
    if args.title:
        plt.title(args.title)

    # Show grid if not disabled
    if not args.no_grid:
        plt.grid(True, linestyle='--', alpha=0.7)

    # Show legend if requested and we have more than one file
    if args.legend or len(args.files) > 1:
        plt.legend()

    if args.ylim:
        plt.ylim(float(args.ylim[0]), float(args.ylim[1]))

    # Save or show
    if args.save:
        plt.savefig(args.save, dpi=args.dpi, bbox_inches='tight')
        print(f"Plot saved to {args.save}")
    else:
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
