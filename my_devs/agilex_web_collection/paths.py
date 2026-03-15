from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AgilexWebPaths:
    repo_root: Path
    package_root: Path
    static_dir: Path
    runtime_dir: Path
    jobs_root: Path
    agilex_root: Path
    script_path: Path
    outputs_root: Path
    dataset_root: Path

    @classmethod
    def discover(cls) -> "AgilexWebPaths":
        package_root = Path(__file__).resolve().parent
        repo_root = package_root.parents[1]
        agilex_root = repo_root / "my_devs" / "add_robot" / "agilex"
        outputs_root = agilex_root / "outputs"
        return cls(
            repo_root=repo_root,
            package_root=package_root,
            static_dir=package_root / "static",
            runtime_dir=package_root / "runtime",
            jobs_root=package_root / "runtime" / "agilex_jobs",
            agilex_root=agilex_root,
            script_path=agilex_root / "record.sh",
            outputs_root=outputs_root,
            dataset_root=outputs_root,
        )

    def job_dir(self, job_id: str) -> Path:
        return self.jobs_root / job_id
