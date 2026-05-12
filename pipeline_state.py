"""
pipeline_state.py
-----------------
Manages the pipeline_state.json file that powers the resume system.

Every phase (dataset, training, evaluation) calls this module to:
  - Mark itself as started / completed
  - Record how far training has progressed (last completed epoch)
  - Query what work has already been done (for the --resume flag)

How crash safety works
----------------------
We never write directly to pipeline_state.json. Instead we:
  1. Write the new state to a temporary file (pipeline_state.json.tmp)
  2. Rename the temp file to pipeline_state.json

On Linux and Mac, a file rename is an atomic OS operation — it either
completes fully or doesn't happen at all. So if the process crashes
mid-write, the old valid state file is still intact. The corrupted
temp file is simply ignored on the next run.

State file format
-----------------
{
  "experiment"  : "cifar10_resnet18",
  "last_updated": "2024-05-10 14:23:01",
  "stages": {
    "dataset"   : "done",
    "training"  : {"status": "in_progress", "last_epoch": 12, "total_epochs": 50},
    "evaluation": "pending"
  }
}

Possible status values per stage
---------------------------------
  "pending"     → not started yet
  "in_progress" → started but not finished (training also stores epoch info)
  "done"        → completed successfully

Usage
-----
>>> from pipeline_state import PipelineState
>>> state = PipelineState(state_file="pipeline_state.json", experiment="my_exp")
>>> state.mark_started("dataset")
>>> state.mark_done("dataset")
>>> state.is_done("dataset")
True
>>> state.mark_training_epoch(epoch=12, total_epochs=50)
>>> state.get_resume_epoch()
12
"""

import json
import os
import logging
from datetime import datetime
from pathlib import Path


class PipelineState:
    """
    Reads and writes pipeline_state.json to track completed pipeline stages.

    Parameters
    ----------
    state_file : str or Path
        Path to the state JSON file (e.g. "pipeline_state.json").
    experiment : str
        Name of the current experiment — stored in the file for reference.
    logger     : logging.Logger, optional
        Where to write status messages. Falls back to print() if None.

    Example
    -------
    >>> state = PipelineState("pipeline_state.json", "my_experiment")
    >>> state.mark_started("dataset")
    >>> # ... run dataset phase ...
    >>> state.mark_done("dataset")
    """

    # All recognised stage names
    VALID_STAGES = {"dataset", "training", "evaluation"}

    def __init__(
        self,
        state_file : str,
        experiment : str,
        logger     : logging.Logger = None,
    ):
        self.state_file = Path(state_file)
        self.experiment = experiment
        self.logger     = logger
        self._state     = self._load()

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def is_done(self, stage: str) -> bool:
        """
        Return True if the given stage has already completed successfully.

        Parameters
        ----------
        stage : str  One of "dataset", "training", "evaluation".

        Returns
        -------
        bool  True if stage status is "done", False otherwise.

        Example
        -------
        >>> state.is_done("dataset")
        True
        """
        self._validate_stage(stage)
        status = self._state["stages"].get(stage, "pending")
        if isinstance(status, dict):
            return status.get("status") == "done"
        return status == "done"

    def mark_started(self, stage: str) -> None:
        """
        Mark a stage as in_progress. Call this at the very beginning of
        each phase before any work is done.

        Parameters
        ----------
        stage : str  One of "dataset", "training", "evaluation".
        """
        self._validate_stage(stage)
        self._log(f"Stage '{stage}' started")
        if stage == "training":
            # Preserve epoch info if it already exists (crash recovery)
            existing = self._state["stages"].get("training", {})
            if not isinstance(existing, dict):
                existing = {}
            existing["status"] = "in_progress"
            self._state["stages"]["training"] = existing
        else:
            self._state["stages"][stage] = "in_progress"
        self._save()

    def mark_done(self, stage: str) -> None:
        """
        Mark a stage as completed. Call this only after the stage
        finishes successfully.

        Parameters
        ----------
        stage : str  One of "dataset", "training", "evaluation".
        """
        self._validate_stage(stage)
        self._log(f"Stage '{stage}' completed ✓")
        if stage == "training":
            self._state["stages"]["training"] = {"status": "done"}
        else:
            self._state["stages"][stage] = "done"
        self._save()

    def mark_training_epoch(self, epoch: int, total_epochs: int) -> None:
        """
        Record the last successfully completed training epoch.

        Call this after EACH epoch finishes (after the checkpoint is saved).
        This is what allows training to resume from the right epoch after
        a crash.

        Parameters
        ----------
        epoch        : int  The epoch number just completed (1-indexed).
        total_epochs : int  Total number of epochs planned for this run.

        Example
        -------
        >>> state.mark_training_epoch(epoch=12, total_epochs=50)
        # pipeline_state.json now shows: last_epoch=12, total_epochs=50
        """
        training_info = {
            "status"      : "in_progress",
            "last_epoch"  : epoch,
            "total_epochs": total_epochs,
        }
        self._state["stages"]["training"] = training_info
        self._save()

    def get_resume_epoch(self) -> int:
        """
        Return the epoch number to resume training FROM (i.e. last_epoch + 1).

        Returns 1 if no training progress has been recorded yet.

        Returns
        -------
        int  The next epoch to run. e.g. if last_epoch=12, returns 13.

        Example
        -------
        >>> state.get_resume_epoch()
        13
        """
        training = self._state["stages"].get("training", {})
        if isinstance(training, dict) and "last_epoch" in training:
            return training["last_epoch"] + 1
        return 1

    def reset(self) -> None:
        """
        Reset the state file to a clean slate — all stages set to "pending".

        Called when a fresh run is requested (no --resume flag). This
        ensures a clean state even if a previous run's state file exists.
        """
        self._log("Resetting pipeline state for a fresh run")
        self._state = self._fresh_state()
        self._save()

    def print_status(self) -> None:
        """
        Print the current state of all stages to the logger or console.
        Useful at the start of a resumed run to show what will be skipped.

        Example output
        --------------
        [Resume] dataset    → DONE (skipping)
        [Resume] training   → IN PROGRESS (resuming from epoch 12/50)
        [Resume] evaluation → PENDING
        """
        lines = ["Pipeline state:"]
        for stage in ["dataset", "training", "evaluation"]:
            status = self._state["stages"].get(stage, "pending")
            if isinstance(status, dict):
                s = status.get("status", "pending")
                if s == "done":
                    lines.append(f"  {stage:<12} → DONE")
                elif s == "in_progress":
                    ep    = status.get("last_epoch",   0)
                    total = status.get("total_epochs", "?")
                    lines.append(
                        f"  {stage:<12} → IN PROGRESS "
                        f"(resuming from epoch {ep}/{total})"
                    )
                else:
                    lines.append(f"  {stage:<12} → PENDING")
            else:
                lines.append(f"  {stage:<12} → {str(status).upper()}")
        self._log("\n".join(lines))

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    def _fresh_state(self) -> dict:
        """Return a brand-new state dict with all stages set to pending."""
        return {
            "experiment"  : self.experiment,
            "last_updated": self._now(),
            "stages": {
                "dataset"   : "pending",
                "training"  : "pending",
                "evaluation": "pending",
            },
        }

    def _load(self) -> dict:
        """
        Load state from disk if the file exists, otherwise return a fresh state.
        Handles corrupted files gracefully by falling back to a fresh state.
        """
        if not self.state_file.exists():
            return self._fresh_state()
        try:
            with open(self.state_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, KeyError):
            self._log("WARNING: state file was corrupted — starting fresh")
            return self._fresh_state()

    def _save(self) -> None:
        """
        Write the current state to disk atomically.

        Steps: write to .tmp → rename to real file.
        A crash during the write leaves the old file intact.
        """
        self._state["last_updated"] = self._now()
        tmp_path = self.state_file.with_suffix(".json.tmp")

        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(self._state, f, indent=2)

        # Atomic rename — on POSIX systems this is guaranteed atomic
        os.replace(tmp_path, self.state_file)

    def _validate_stage(self, stage: str) -> None:
        """Raise ValueError if stage is not one of the recognised names."""
        if stage not in self.VALID_STAGES:
            raise ValueError(
                f"Unknown stage '{stage}'. "
                f"Valid stages: {sorted(self.VALID_STAGES)}"
            )

    def _log(self, msg: str) -> None:
        """Write a message to the logger or fall back to print()."""
        if self.logger:
            self.logger.info(msg)
        else:
            print(msg)

    @staticmethod
    def _now() -> str:
        """Return the current datetime as a readable string."""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")