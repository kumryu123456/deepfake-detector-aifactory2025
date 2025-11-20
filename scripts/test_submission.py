"""Submission validation and fixing script.

This script validates submission.csv format and fixes common issues.

Usage:
    python scripts/test_submission.py --input submission.csv
    python scripts/test_submission.py --input submission.csv --fix --output submission_fixed.csv

Example:
    # Validate only
    python scripts/test_submission.py --input submission.csv

    # Validate and fix
    python scripts/test_submission.py \\
        --input submission.csv \\
        --fix \\
        --output submission_fixed.csv
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd


def validate_submission_format(df: pd.DataFrame, verbose: bool = True) -> dict:
    """Validate submission DataFrame format.

    Args:
        df: Submission DataFrame
        verbose: Print detailed validation messages

    Returns:
        Dictionary with validation results:
            - valid: Overall validity
            - errors: List of error messages
            - warnings: List of warning messages
    """
    errors = []
    warnings = []

    # Check columns
    expected_columns = ["filename", "label"]
    if list(df.columns) != expected_columns:
        errors.append(
            f"Invalid columns: {df.columns.tolist()}. "
            f"Expected: {expected_columns}"
        )

    # Check for null values
    null_counts = df.isnull().sum()
    if null_counts.any():
        for col in null_counts[null_counts > 0].index:
            errors.append(f"Column '{col}' has {null_counts[col]} null values")

    # Check label values
    if "label" in df.columns:
        unique_labels = df["label"].unique()
        invalid_labels = [l for l in unique_labels if l not in [0, 1]]

        if invalid_labels:
            errors.append(
                f"Invalid label values: {invalid_labels}. "
                f"Labels must be 0 (Real) or 1 (Fake)"
            )

        # Check label data type
        if not pd.api.types.is_integer_dtype(df["label"]):
            warnings.append(
                f"Label column has non-integer dtype: {df['label'].dtype}. "
                f"Should be int"
            )

    # Check filenames
    if "filename" in df.columns:
        # Check for missing extensions
        no_extension = df[~df["filename"].str.contains(".")]
        if len(no_extension) > 0:
            errors.append(
                f"{len(no_extension)} filenames missing extensions. "
                f"Examples: {no_extension['filename'].head(3).tolist()}"
            )

        # Check for duplicates
        duplicates = df[df["filename"].duplicated()]
        if len(duplicates) > 0:
            errors.append(
                f"{len(duplicates)} duplicate filenames found. "
                f"Examples: {duplicates['filename'].head(3).tolist()}"
            )

        # Check for case sensitivity issues
        lower_filenames = df["filename"].str.lower()
        if len(lower_filenames.unique()) != len(df["filename"].unique()):
            warnings.append(
                "Found filenames that differ only in case. "
                "This may cause issues on case-insensitive filesystems."
            )

    # Check number of rows
    if len(df) == 0:
        errors.append("Submission is empty (0 rows)")
    elif len(df) < 10:
        warnings.append(f"Submission has very few rows: {len(df)}")

    # Print results
    if verbose:
        print("=" * 80)
        print("SUBMISSION VALIDATION RESULTS")
        print("=" * 80)

        if errors:
            print("\n❌ ERRORS:")
            for i, error in enumerate(errors, 1):
                print(f"  {i}. {error}")

        if warnings:
            print("\n⚠️  WARNINGS:")
            for i, warning in enumerate(warnings, 1):
                print(f"  {i}. {warning}")

        if not errors and not warnings:
            print("\n✅ All validation checks passed!")

        print("\nSummary:")
        print(f"  Total rows: {len(df)}")
        if "label" in df.columns:
            print(f"  Real (0): {sum(df['label'] == 0)}")
            print(f"  Fake (1): {sum(df['label'] == 1)}")
        print("=" * 80)

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
    }


def fix_submission(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Attempt to fix common submission format issues.

    Args:
        df: Submission DataFrame
        verbose: Print fix messages

    Returns:
        Fixed DataFrame
    """
    df_fixed = df.copy()
    fixes_applied = []

    # Fix column names
    if list(df_fixed.columns) != ["filename", "label"]:
        # Try to rename columns
        if len(df_fixed.columns) == 2:
            df_fixed.columns = ["filename", "label"]
            fixes_applied.append("Renamed columns to ['filename', 'label']")
        else:
            # Can't fix automatically
            if verbose:
                print("Cannot fix: Wrong number of columns")
            return df_fixed

    # Fix null values in labels (default to 0 = Real)
    if df_fixed["label"].isnull().any():
        num_null = df_fixed["label"].isnull().sum()
        df_fixed["label"] = df_fixed["label"].fillna(0)
        fixes_applied.append(f"Filled {num_null} null labels with 0 (Real)")

    # Fix null values in filenames (drop rows)
    if df_fixed["filename"].isnull().any():
        num_null = df_fixed["filename"].isnull().sum()
        df_fixed = df_fixed.dropna(subset=["filename"])
        fixes_applied.append(f"Dropped {num_null} rows with null filenames")

    # Fix label data type
    try:
        df_fixed["label"] = df_fixed["label"].astype(int)
        if not pd.api.types.is_integer_dtype(df.get("label", [])):
            fixes_applied.append("Converted labels to integer type")
    except Exception as e:
        if verbose:
            print(f"Cannot convert labels to int: {e}")

    # Fix invalid label values (clip to [0, 1])
    invalid_labels_mask = ~df_fixed["label"].isin([0, 1])
    if invalid_labels_mask.any():
        num_invalid = invalid_labels_mask.sum()
        df_fixed.loc[invalid_labels_mask, "label"] = df_fixed.loc[invalid_labels_mask, "label"].clip(0, 1)
        fixes_applied.append(f"Clipped {num_invalid} invalid labels to [0, 1]")

    # Remove duplicate filenames (keep first)
    if df_fixed["filename"].duplicated().any():
        num_dupes = df_fixed["filename"].duplicated().sum()
        df_fixed = df_fixed.drop_duplicates(subset=["filename"], keep="first")
        fixes_applied.append(f"Removed {num_dupes} duplicate filenames")

    # Print fixes
    if verbose and fixes_applied:
        print("\n🔧 Applied fixes:")
        for i, fix in enumerate(fixes_applied, 1):
            print(f"  {i}. {fix}")

    return df_fixed


def validate_and_fix_submission(
    input_path: str,
    output_path: str = None,
    auto_fix: bool = False,
) -> bool:
    """Validate submission CSV and optionally fix issues.

    Args:
        input_path: Path to input submission.csv
        output_path: Path to save fixed submission (if auto_fix=True)
        auto_fix: Whether to attempt automatic fixes

    Returns:
        True if validation passed (after fixes if enabled)
    """
    input_path = Path(input_path)

    if not input_path.exists():
        print(f"❌ File not found: {input_path}")
        return False

    # Load submission
    try:
        df = pd.read_csv(input_path)
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        return False

    # Validate
    result = validate_submission_format(df, verbose=True)

    if not result["valid"]:
        if auto_fix:
            print("\n🔧 Attempting to fix issues...")
            df_fixed = fix_submission(df, verbose=True)

            # Re-validate
            result_fixed = validate_submission_format(df_fixed, verbose=False)

            if result_fixed["valid"]:
                # Save fixed version
                if output_path:
                    output_path = Path(output_path)
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    df_fixed.to_csv(output_path, index=False)
                    print(f"\n✅ Fixed submission saved to: {output_path}")
                else:
                    # Overwrite original
                    df_fixed.to_csv(input_path, index=False)
                    print(f"\n✅ Fixed submission saved (overwritten original)")

                return True
            else:
                print("\n❌ Could not fix all issues automatically")
                print("Remaining errors:")
                for error in result_fixed["errors"]:
                    print(f"  - {error}")
                return False
        else:
            print("\n💡 Tip: Use --fix to attempt automatic fixes")
            return False

    return True


def create_submission_with_validation(
    predictions: dict,
    output_path: str,
    validate: bool = True,
) -> bool:
    """Create submission.csv with automatic validation.

    Args:
        predictions: Dictionary mapping filename → label
        output_path: Path to save submission.csv
        validate: Whether to validate format

    Returns:
        True if creation succeeded and validation passed

    Example:
        >>> predictions = {"image1.jpg": 0, "video1.mp4": 1, ...}
        >>> create_submission_with_validation(predictions, "submission.csv")
    """
    # Create DataFrame
    df = pd.DataFrame([
        {"filename": filename, "label": label}
        for filename, label in predictions.items()
    ])

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Created submission: {output_path}")

    # Validate if requested
    if validate:
        result = validate_submission_format(df, verbose=True)
        return result["valid"]

    return True


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Validate and fix submission.csv format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input submission.csv",
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Attempt to automatically fix issues",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save fixed submission (if --fix enabled). If not specified, overwrites input.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress detailed output",
    )

    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()

    success = validate_and_fix_submission(
        input_path=args.input,
        output_path=args.output,
        auto_fix=args.fix,
    )

    if success:
        return 0
    else:
        return 1


if __name__ == "__main__":
    exit(main())
