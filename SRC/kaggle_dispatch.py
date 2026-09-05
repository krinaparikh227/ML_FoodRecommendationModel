"""
kaggle_dispatch.py

Provides command-line orchestration for dispatching compute-intensive training
and evaluation workloads (such as GRU4Rec sequential recommendation and DeBERTa
faithfulness verification) to Kaggle Free Tier cloud NVIDIA T4 GPUs.

Workflow stages supported:
1. Preparation: Creates an isolated staging directory containing the code and kernel-metadata.json.
2. Dispatch: Pushes the kernel to Kaggle via the Kaggle CLI.
3. Polling: Monitors remote execution status until completion.
4. Retrieval: Downloads output tables, model checkpoints, and logs to the local RESULTS directory.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


DEFAULT_USERNAME = "erkushshah"
DEFAULT_WORKSPACE = Path(__file__).resolve().parent.parent
STAGING_BASE_DIR = DEFAULT_WORKSPACE / ".kaggle_staging"
OUTPUTS_BASE_DIR = DEFAULT_WORKSPACE / "RESULTS" / "kaggle_outputs"


def get_authenticated_username():
    """
    Retrieves the authenticated Kaggle username from environment variables
    or defaults to the established project account.

    Returns:
        str: Active Kaggle username.
    """
    return os.environ.get("KAGGLE_USERNAME", DEFAULT_USERNAME)


def ensure_clean_staging_directory(staging_path):
    """
    Prepares a clean staging directory for Kaggle kernel packaging.

    Args:
        staging_path (Path): Target staging directory path.
    """
    if staging_path.exists():
        shutil.rmtree(staging_path)
    staging_path.mkdir(parents=True, exist_ok=True)


def build_kernel_metadata(
    username,
    kernel_slug,
    title,
    code_filename,
    is_notebook,
    enable_gpu=True,
    enable_internet=True,
    dataset_sources=None,
):
    """
    Constructs the dictionary representing kernel-metadata.json conforming
    to Kaggle CLI specifications.

    Args:
        username (str): Kaggle account username.
        kernel_slug (str): Unique URL-safe identifier for the kernel.
        title (str): Display title for the Kaggle notebook/script.
        code_filename (str): Name of the script or notebook file inside staging.
        is_notebook (bool): True if notebook, False if Python script.
        enable_gpu (bool): Whether to allocate free NVIDIA T4 GPU resources.
        enable_internet (bool): Whether outbound internet access is enabled.
        dataset_sources (list): Optional list of Kaggle dataset identifiers.

    Returns:
        dict: Serialized metadata structure.
    """
    return {
        "id": f"{username}/{kernel_slug}",
        "title": title,
        "code_file": code_filename,
        "language": "python",
        "kernel_type": "notebook" if is_notebook else "script",
        "is_private": "true",
        "enable_gpu": "true" if enable_gpu else "false",
        "enable_tpu": "false",
        "enable_internet": "true" if enable_internet else "false",
        "dataset_sources": dataset_sources or [],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }


def push_workload(source_file_path, kernel_slug, title, enable_gpu=True, datasets=None):
    """
    Packages a local notebook or Python script into the staging directory,
    writes the associated kernel-metadata.json, and pushes the execution
    bundle to Kaggle.

    Args:
        source_file_path (str or Path): Local path to .ipynb or .py file.
        kernel_slug (str): Unique kernel slug.
        title (str): Human-readable kernel title.
        enable_gpu (bool): Toggle for GPU accelerator allocation.
        datasets (list): Optional Kaggle dataset references.

    Returns:
        int: Subprocess return code from Kaggle CLI.
    """
    source_path = Path(source_file_path).resolve()
    if not source_path.exists():
        print(f"Error: Source file not found at {source_path}")
        return 1

    is_notebook = source_path.suffix.lower() == ".ipynb"
    staging_dir = STAGING_BASE_DIR / kernel_slug
    ensure_clean_staging_directory(staging_dir)

    target_code_name = source_path.name
    destination_file = staging_dir / target_code_name
    shutil.copy2(source_path, destination_file)

    # Sanitize notebook metadata: inject kernelspec if missing to satisfy Papermill
    if is_notebook:
        try:
            with open(destination_file, "r", encoding="utf-8") as nb_file:
                nb_content = json.load(nb_file)
            if "metadata" not in nb_content:
                nb_content["metadata"] = {}
            if "kernelspec" not in nb_content["metadata"]:
                nb_content["metadata"]["kernelspec"] = {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3",
                }
            if "language_info" not in nb_content["metadata"]:
                nb_content["metadata"]["language_info"] = {
                    "name": "python",
                    "version": "3.10.0",
                }
            with open(destination_file, "w", encoding="utf-8") as nb_file:
                json.dump(nb_content, nb_file, indent=2)
        except Exception as err:
            print(f"Notice: Notebook metadata sanitization skipped: {err}")

    # Stage local interim artifacts if present for relative path resolution
    interim_candidates = DEFAULT_WORKSPACE / "DATA" / "interim_claims"
    if interim_candidates.exists():
        staging_interim = staging_dir / "data" / "interim"
        staging_interim.mkdir(parents=True, exist_ok=True)
        for item in interim_candidates.glob("*.parquet"):
            shutil.copy2(item, staging_interim / item.name)

    # Attach default GroundedNutriRec benchmark dataset if none specified
    default_dataset = "erkushshah/groundednutrirec-benchmark"
    effective_datasets = list(datasets or [])
    if default_dataset not in effective_datasets:
        effective_datasets.append(default_dataset)

    username = get_authenticated_username()
    metadata = build_kernel_metadata(
        username=username,
        kernel_slug=kernel_slug,
        title=title,
        code_filename=target_code_name,
        is_notebook=is_notebook,
        enable_gpu=enable_gpu,
        enable_internet=True,
        dataset_sources=effective_datasets,
    )

    metadata_path = staging_dir / "kernel-metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as file_handle:
        json.dump(metadata, file_handle, indent=2)

    print(f"Prepared kernel package in: {staging_dir}")
    print(f"Kernel Ref: {username}/{kernel_slug}")
    print(f"Hardware Allocation: {'NVIDIA T4 GPU' if enable_gpu else 'Standard CPU'}")
    print(f"Attached Datasets: {effective_datasets}")
    print("Pushing to Kaggle...")

    push_cmd = ["kaggle", "kernels", "push", "-p", str(staging_dir)]
    if enable_gpu:
        push_cmd.extend(["--accelerator", "NvidiaTeslaT4"])

    result = subprocess.run(
        push_cmd,
        capture_output=True,
        text=True,
        shell=True,
    )

    if result.stdout:
        print(result.stdout.strip())
    if result.stderr:
        print(result.stderr.strip())

    return result.returncode


def check_status(kernel_slug):
    """
    Queries Kaggle API for the current execution status of a kernel.

    Args:
        kernel_slug (str): Unique kernel slug.

    Returns:
        int: Subprocess return code.
    """
    username = get_authenticated_username()
    kernel_ref = f"{username}/{kernel_slug}"
    print(f"Checking status for: {kernel_ref}")

    result = subprocess.run(
        ["kaggle", "kernels", "status", kernel_ref],
        capture_output=True,
        text=True,
        shell=True,
    )

    if result.stdout:
        print(result.stdout.strip())
    if result.stderr:
        print(result.stderr.strip())

    return result.returncode


def pull_outputs(kernel_slug, output_destination=None):
    """
    Downloads remote artifacts, figures, checkpoints, and execution logs
    from a completed Kaggle kernel to a local destination directory.

    Args:
        kernel_slug (str): Unique kernel slug.
        output_destination (Path or str, optional): Local directory for files.

    Returns:
        int: Subprocess return code.
    """
    username = get_authenticated_username()
    kernel_ref = f"{username}/{kernel_slug}"

    if output_destination:
        dest_dir = Path(output_destination).resolve()
    else:
        dest_dir = OUTPUTS_BASE_DIR / kernel_slug

    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"Pulling output artifacts for {kernel_ref} into: {dest_dir}")

    result = subprocess.run(
        ["kaggle", "kernels", "output", kernel_ref, "-p", str(dest_dir)],
        capture_output=True,
        text=True,
        shell=True,
    )

    if result.stdout:
        print(result.stdout.strip())
    if result.stderr:
        print(result.stderr.strip())
TARGET_OUTPUTS_BRANCH = "Kaggle-Outputs"


def compute_sha256(file_path):
    """
    Computes cryptographic SHA-256 checksum for artifact provenance.

    Args:
        file_path (Path): Path to the target file.

    Returns:
        str: Hexadecimal SHA-256 digest string.
    """
    import hashlib

    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def generate_experiment_manifest(target_path, kernel_slug):
    """
    Generates a structured experiment_manifest.json labeling every artifact
    with semantic type, file size, storage layer (LFS vs Git tree), and SHA-256 hash.

    Args:
        target_path (Path): Directory containing outputs.
        kernel_slug (str): Workload identifier.

    Returns:
        Path: Path to created manifest file.
    """
    import datetime

    manifest_entries = []
    lfs_threshold_bytes = 45 * 1024 * 1024
    lfs_extensions = {".parquet", ".pt", ".pth", ".bin", ".h5", ".onnx", ".npz", ".pkl"}

    for item in sorted(target_path.rglob("*")):
        if item.is_file() and item.name != "experiment_manifest.json":
            file_size = item.stat().st_size
            rel_path = str(item.relative_to(DEFAULT_WORKSPACE)).replace("\\", "/")
            suffix = item.suffix.lower()

            # Categorize artifact
            if suffix in {".csv", ".tsv", ".json"} and "log" not in item.name.lower():
                category = "metric_table"
            elif suffix in {".png", ".jpg", ".pdf", ".svg"}:
                category = "publication_figure"
            elif suffix in {".pt", ".pth", ".bin", ".h5", ".onnx"}:
                category = "neural_model_checkpoint"
            elif suffix in {".parquet", ".npz", ".pkl"}:
                category = "data_tensor_or_cache"
            else:
                category = "execution_log"

            is_lfs = (file_size >= lfs_threshold_bytes) or (suffix in lfs_extensions)

            manifest_entries.append(
                {
                    "file_name": item.name,
                    "relative_path": rel_path,
                    "size_bytes": file_size,
                    "size_mb": round(file_size / (1024 * 1024), 3),
                    "semantic_category": category,
                    "storage_layer": "Git LFS" if is_lfs else "Standard Git Blob",
                    "sha256": compute_sha256(item),
                }
            )

    manifest_doc = {
        "kernel_slug": kernel_slug,
        "recorded_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "hardware_environment": "Kaggle Cloud Dual NVIDIA T4 GPUs",
        "target_branch": TARGET_OUTPUTS_BRANCH,
        "total_files": len(manifest_entries),
        "artifacts": manifest_entries,
    }

    manifest_path = target_path / "experiment_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_doc, f, indent=2)

    return manifest_path


def stage_and_push_to_github(target_directory, kernel_slug, branch=None, remote_name="origin"):
    """
    Scans the fetched Kaggle output directory, labels all files in a structured
    manifest, dynamically configures Git LFS for large binaries, stages all assets,
    creates a verified commit, and pushes to the dedicated Kaggle-Outputs branch.

    Args:
        target_directory (Path): Local directory containing downloaded Kaggle artifacts.
        kernel_slug (str): Unique identifier of the executed Kaggle workload.
        branch (str, optional): Target git branch. Defaults to 'Kaggle-Outputs'.
        remote_name (str): Git remote repository name. Defaults to 'origin'.

    Returns:
        bool: True if staging, commit, and push succeed; False otherwise.
    """
    target_path = Path(target_directory).resolve()
    if not target_path.exists():
        print(f"Error: Target directory does not exist: {target_path}")
        return False

    target_branch = branch or TARGET_OUTPUTS_BRANCH

    # Generate semantic metadata manifest
    print(f"Generating structured experiment manifest for '{kernel_slug}'...")
    generate_experiment_manifest(target_path, kernel_slug)

    # Detect large files and ensure Git LFS is activated
    lfs_threshold_bytes = 45 * 1024 * 1024
    lfs_extensions = {".parquet", ".pt", ".pth", ".bin", ".h5", ".onnx", ".npz", ".pkl"}
    lfs_patterns_to_track = set()

    staged_relative_paths = []

    for file_path in target_path.rglob("*"):
        if file_path.is_file():
            file_size = file_path.stat().st_size
            rel_path = file_path.relative_to(DEFAULT_WORKSPACE)
            suffix = file_path.suffix.lower()

            if file_size >= lfs_threshold_bytes or suffix in lfs_extensions:
                pattern = f"*{suffix}" if suffix else str(rel_path).replace("\\", "/")
                lfs_patterns_to_track.add(pattern)

            staged_relative_paths.append(str(rel_path))

    # Configure Git LFS patterns if detected
    if lfs_patterns_to_track:
        print(f"Active Git LFS tracking configured for large patterns: {list(lfs_patterns_to_track)}")
        for pattern in lfs_patterns_to_track:
            subprocess.run(
                ["git", "lfs", "track", pattern],
                check=True,
                cwd=str(DEFAULT_WORKSPACE),
            )
        subprocess.run(
            ["git", "add", ".gitattributes"],
            check=True,
            cwd=str(DEFAULT_WORKSPACE),
        )

    if not staged_relative_paths:
        print("Notice: No output files found to stage for Git commit.")
        return True

    try:
        # Stage the output files
        for rel_file in staged_relative_paths:
            subprocess.run(
                ["git", "add", rel_file],
                check=True,
                cwd=str(DEFAULT_WORKSPACE),
            )

        # Check if there are staged changes
        status_check = subprocess.run(
            ["git", "diff", "--cached", "--quiet"],
            cwd=str(DEFAULT_WORKSPACE),
        )
        if status_check.returncode == 0:
            print("Notice: No new changes detected in git index. Repository is up to date.")
            return True

        # Commit with structured message
        commit_message = f"chore(kaggle-outputs): archive verified artifacts and figures for {kernel_slug}"
        subprocess.run(
            ["git", "commit", "-m", commit_message],
            check=True,
            cwd=str(DEFAULT_WORKSPACE),
        )
        print(f"Committed changes with message: '{commit_message}'")

        # Push to dedicated remote branch
        print(f"Pushing commit to {remote_name}/{target_branch}...")
        push_result = subprocess.run(
            ["git", "push", remote_name, f"HEAD:{target_branch}"],
            capture_output=True,
            text=True,
            cwd=str(DEFAULT_WORKSPACE),
        )

        if push_result.stdout:
            print(push_result.stdout.strip())
        if push_result.stderr:
            print(push_result.stderr.strip())

        if push_result.returncode == 0:
            print(f"Successfully pushed Kaggle outputs for '{kernel_slug}' to branch '{target_branch}'.")
            return True
        else:
            print(f"Warning: Git push encountered an error with return code {push_result.returncode}")
            return False

    except subprocess.CalledProcessError as err:
        print(f"Git operation failed: {err}")
        return False


def list_remote_kernels():
    """
    Displays the list of kernels associated with the authenticated Kaggle user.

    Returns:
        int: Subprocess return code.
    """
    result = subprocess.run(
        ["kaggle", "kernels", "list", "--mine"],
        capture_output=True,
        text=True,
        shell=True,
    )

    if result.stdout:
        print(result.stdout.strip())
    if result.stderr:
        print(result.stderr.strip())

    return result.returncode


def build_argument_parser():
    """
    Configures command-line argument schema for the dispatcher.

    Returns:
        argparse.ArgumentParser: Populated parser instance.
    """
    parser = argparse.ArgumentParser(
        description="GroundedNutriRec Kaggle GPU Workload Dispatcher"
    )
    subparsers = parser.add_subparsers(dest="action", help="Subcommands")

    # Subcommand: push
    push_parser = subparsers.add_parser("push", help="Push and start kernel on Kaggle")
    push_parser.add_argument(
        "--file", "-f", required=True, help="Path to .ipynb or .py file"
    )
    push_parser.add_argument(
        "--slug", "-s", required=True, help="Unique kernel slug (letters, numbers, hyphens)"
    )
    push_parser.add_argument(
        "--title", "-t", required=True, help="Human-readable title for the kernel"
    )
    push_parser.add_argument(
        "--cpu", action="store_true", help="Force CPU mode instead of default NVIDIA T4 GPU"
    )
    push_parser.add_argument(
        "--dataset", "-d", action="append", help="Kaggle dataset reference to attach"
    )

    # Subcommand: status
    status_parser = subparsers.add_parser("status", help="Check remote execution status")
    status_parser.add_argument(
        "--slug", "-s", required=True, help="Kernel slug to query"
    )

    # Subcommand: pull
    pull_parser = subparsers.add_parser("pull", help="Download output files and metrics")
    pull_parser.add_argument(
        "--slug", "-s", required=True, help="Kernel slug to download from"
    )
    pull_parser.add_argument(
        "--output-dir", "-o", help="Custom destination directory"
    )
    pull_parser.add_argument(
        "--git-push", action="store_true", help="Automatically commit and push fetched outputs to GitHub"
    )
    pull_parser.add_argument(
        "--branch", "-b", help="Target git branch for push (defaults to current branch)"
    )

    # Subcommand: list
    subparsers.add_parser("list", help="List all user kernels on Kaggle")

    return parser


def main():
    """
    Main entry point for command-line execution.
    """
    parser = build_argument_parser()
    args = parser.parse_args()

    if not args.action:
        parser.print_help()
        sys.exit(1)

    if args.action == "push":
        return_code = push_workload(
            source_file_path=args.file,
            kernel_slug=args.slug,
            title=args.title,
            enable_gpu=not args.cpu,
            datasets=args.dataset,
        )
        sys.exit(return_code)

    elif args.action == "status":
        return_code = check_status(kernel_slug=args.slug)
        sys.exit(return_code)

    elif args.action == "pull":
        dest_dir = Path(args.output_dir).resolve() if args.output_dir else (OUTPUTS_BASE_DIR / args.slug)
        return_code = pull_outputs(
            kernel_slug=args.slug,
            output_destination=dest_dir,
        )
        if return_code == 0 and args.git_push:
            stage_and_push_to_github(
                target_directory=dest_dir,
                kernel_slug=args.slug,
                branch=args.branch,
            )
        sys.exit(return_code)

    elif args.action == "list":
        return_code = list_remote_kernels()
        sys.exit(return_code)


if __name__ == "__main__":
    main()

