import argparse
import logging
import sys

from chrompert import __version__


def _configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="[%(levelname)s] %(message)s",
        stream=sys.stderr,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="chrompert",
        description="Predict chromatin accessibility changes from ChIP-seq features.",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose debug logging"
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # 1. frip-filter subcommand
    frip_parser = subparsers.add_parser(
        "frip-filter",
        aliases=["frip_filter"],
        help="Take ENCODE TSV and download BED files with highest FRiP scores",
    )
    frip_parser.add_argument(
        "-i", "--input", required=True, help="Path to input ENCODE TSV file"
    )
    frip_parser.add_argument(
        "-m",
        "--min_peaks",
        default=0,
        type=int,
        help="Minimum number of peaks to be considered a valid, high quality file",
    )
    frip_parser.add_argument(
        "-p",
        "--parallel",
        default=8,
        type=int,
        help="Number of parallel threads to use for pipeline (default: 8)",
    )
    frip_parser.add_argument(
        "--data_dir",
        default="data",
        help="Base data directory for bed and log folders (default: data)",
    )

    # 2. feature-analysis subcommand
    fa_parser = subparsers.add_parser(
        "feature-analysis",
        aliases=["feature_analysis"],
        help="Calculate ML feature importances from ATAC-seq peaks and ChIP-seq BED features",
    )
    fa_parser.add_argument(
        "-i", "--input_matrix", required=True, help="Input differential peak matrix TSV"
    )
    fa_parser.add_argument(
        "-o",
        "--output_dir",
        default="data/output",
        help="Output directory for the feature importances and plots (default: data/output)",
    )
    fa_parser.add_argument(
        "-b",
        "--bed_dir",
        default="data/bed",
        help="Directory containing input ChIP-seq BED files (default: data/bed)",
    )
    fa_parser.add_argument(
        "-fc",
        "--fold_change",
        default="log2",
        help="Fold change column name in input matrix (default: log2)",
    )
    fa_parser.add_argument(
        "-d",
        "--diff_peaks",
        default="change",
        help="Differential peak accessibility column name in input matrix (default: change)",
    )
    fa_parser.add_argument(
        "-x",
        "--exclude_features",
        nargs="*",
        default=[],
        help="List of features to exclude from the feature matrix (without .bed extension)",
    )
    fa_parser.add_argument(
        "-p",
        "--permutation",
        action="store_true",
        help="Calculate permutation importances for the Random Forest model",
    )
    fa_parser.add_argument(
        "-j",
        "--threads",
        type=int,
        default=1,
        help="Number of parallel threads to use (default: 1)",
    )
    fa_parser.add_argument(
        "--save_matrix",
        action="store_true",
        help="Optionally save the binary peak-by-feature matrix CSV to output_dir",
    )

    # 3. reset-bed subcommand
    reset_parser = subparsers.add_parser(
        "reset-bed",
        aliases=["reset_bed"],
        help="Clear downloaded BED and log files",
    )
    reset_parser.add_argument(
        "--data_dir",
        default="data",
        help="Base data directory containing bed and log folders (default: data)",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    _configure_logging(args.verbose)

    try:
        if args.command in ("frip-filter", "frip_filter"):
            from chrompert.frip_filter import run_frip_filter

            run_frip_filter(
                input_file=args.input,
                min_peaks=args.min_peaks,
                parallel=args.parallel,
                data_dir=args.data_dir,
            )
        elif args.command in ("feature-analysis", "feature_analysis"):
            from chrompert.feature_analysis import run_feature_analysis

            run_feature_analysis(
                input_matrix=args.input_matrix,
                output_dir=args.output_dir,
                bed_dir=args.bed_dir,
                fold_change=args.fold_change,
                diff_peaks=args.diff_peaks,
                exclude_features=args.exclude_features,
                permutation=args.permutation,
                threads=args.threads,
                save_matrix=args.save_matrix,
            )
        elif args.command in ("reset-bed", "reset_bed"):
            from chrompert.reset_bed import run_reset_bed

            run_reset_bed(data_dir=args.data_dir)
    except Exception as exc:
        if args.verbose:
            raise
        logging.getLogger(__name__).error("%s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
