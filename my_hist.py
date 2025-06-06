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
        elif data.ndim == 1:
            data = data.reshape(len(data), 1)
        elif data.ndim != 2:
            breakpoint()
            raise ValueError(f"Data in {filename} should have shape (n,2), but has shape {data.shape}")
        return data
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return None


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Plot histogram of column(s) data from file(s)')
    parser.add_argument('files', nargs='+', help='Input files containing (n,2) data')
    parser.add_argument('--save', '-s', help='Save plot to specified file instead of showing it')
    parser.add_argument('--columns', nargs='*', help='list of columns to be plotted',default=[0])
    parser.add_argument('--title', '-t', default='', help='Plot title')
    parser.add_argument('--xlabel', '-x', default='X', help='X-axis label')
    parser.add_argument('--ylabel', '-y', default='Y', help='Y-axis label')
    parser.add_argument('--xrange', nargs='*',help='min and max of histogram')
    parser.add_argument('--bins', type=int,help='number of bins',default=20)
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
        for col in args.columns:
            color = colors[i % len(colors)]

            # Create label from filename (basename without extension)
            label = os.path.splitext(os.path.basename(filename))[0]

            # Plot data
            if args.xrange is None:
                plt.hist(data[:,col], color=color, label=label,bins=args.bins,alpha=0.6)
            else:
                plt.hist(data[:,col], color=color, label=label,bins=args.bins,range=(args.xrange[0],args.xrange[1]),alpha=0.7)
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

    # Save or show
    if args.save:
        plt.savefig(args.save, dpi=args.dpi, bbox_inches='tight')
        print(f"Plot saved to {args.save}")
    else:
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
