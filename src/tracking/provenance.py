"""Provenance tracking: SQLite trial database and Git branch/PR management."""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# =============================================================================
# SQLite Trial Database
# =============================================================================


class TrialDatabase:
    """SQLite database for trial provenance and history."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._conn = None
        self._init_db()

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def _init_db(self) -> None:
        """Create tables if they don't exist."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS trials (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trial_id INTEGER NOT NULL,
                timestamp TEXT NOT NULL,
                config_hash TEXT,
                augmentations_json TEXT,
                tuning_json TEXT,
                metrics_json TEXT,
                primary_metric REAL,
                decision TEXT NOT NULL,
                reason TEXT,
                diff_json TEXT,
                branch_name TEXT,
                pr_url TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_trial_id ON trials(trial_id);
            CREATE INDEX IF NOT EXISTS idx_decision ON trials(decision);
            CREATE INDEX IF NOT EXISTS idx_config_hash ON trials(config_hash);
        """)
        self.conn.commit()

    def log_trial(self, record: dict[str, Any]) -> int:
        """Log a trial record to the database.

        Returns:
            The database row ID.
        """
        cursor = self.conn.execute(
            """
            INSERT INTO trials (
                trial_id, timestamp, config_hash,
                augmentations_json, tuning_json, metrics_json,
                primary_metric, decision, reason, diff_json,
                branch_name, pr_url
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.get("trial_id"),
                record.get("timestamp", datetime.now().isoformat()),
                record.get("config_hash"),
                json.dumps(record.get("augmentations", {})),
                json.dumps(record.get("tuning", {})),
                json.dumps(record.get("metrics", {})),
                record.get("primary_metric"),
                record.get("decision", "unknown"),
                record.get("reason"),
                json.dumps(record.get("diff", [])),
                record.get("branch_name"),
                record.get("pr_url"),
            ),
        )
        self.conn.commit()
        return cursor.lastrowid

    def get_history(self, n: int = 10) -> list[dict]:
        """Get the last N trials."""
        rows = self.conn.execute(
            "SELECT * FROM trials ORDER BY trial_id DESC LIMIT ?", (n,)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_best(self, direction: str = "minimize") -> dict | None:
        """Get the best accepted trial."""
        order = "ASC" if direction == "minimize" else "DESC"
        row = self.conn.execute(
            f"""
            SELECT * FROM trials
            WHERE decision = 'accepted' AND primary_metric IS NOT NULL
            ORDER BY primary_metric {order}
            LIMIT 1
            """
        ).fetchone()
        return dict(row) if row else None

    def get_accepted(self) -> list[dict]:
        """Get all accepted trials."""
        rows = self.conn.execute(
            "SELECT * FROM trials WHERE decision = 'accepted' ORDER BY trial_id"
        ).fetchall()
        return [dict(r) for r in rows]

    def get_all(self) -> list[dict]:
        """Get all trials."""
        rows = self.conn.execute(
            "SELECT * FROM trials ORDER BY trial_id"
        ).fetchall()
        return [dict(r) for r in rows]

    def get_stats(self) -> dict[str, Any]:
        """Get summary statistics."""
        total = self.conn.execute("SELECT COUNT(*) FROM trials").fetchone()[0]
        accepted = self.conn.execute(
            "SELECT COUNT(*) FROM trials WHERE decision = 'accepted'"
        ).fetchone()[0]
        return {
            "total_trials": total,
            "accepted": accepted,
            "rejected": total - accepted,
            "acceptance_rate": accepted / total if total > 0 else 0,
        }

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None


# =============================================================================
# Git Branch/PR Manager
# =============================================================================


class GitManager:
    """Manages git branches and commits for accepted trial provenance."""

    def __init__(self, repo_path: str):
        try:
            from git import Repo

            self.repo = Repo(repo_path)
        except ImportError:
            raise ImportError("gitpython required. Install with: pip install gitpython")
        except Exception as e:
            logger.warning(f"Git repo not found at {repo_path}: {e}")
            self.repo = None

    def create_trial_branch(self, trial_id: int) -> str:
        """Create a new branch for an accepted trial."""
        if not self.repo:
            return ""

        branch_name = f"agent/trial-{trial_id}"
        try:
            # Store current branch to return to
            original_branch = self.repo.active_branch.name

            # Create and checkout new branch
            self.repo.git.checkout("-b", branch_name)
            logger.info(f"Created branch: {branch_name}")
            return branch_name
        except Exception as e:
            logger.warning(f"Failed to create branch: {e}")
            return ""

    def commit_config_changes(
        self,
        trial_id: int,
        diff_lines: list[str],
    ) -> str | None:
        """Commit config changes to the current branch."""
        if not self.repo:
            return None

        try:
            # Stage config files
            self.repo.index.add([
                "config/augmentations.yaml",
                "config/tuning.yaml",
            ])

            # Build commit message
            diff_str = "\n".join(f"  {line}" for line in diff_lines) if diff_lines else "  (no diff available)"
            message = (
                f"agent: trial {trial_id} — accepted config change\n\n"
                f"Changes:\n{diff_str}\n\n"
                f"This commit was automatically created by the Auto-Augment Agent."
            )

            commit = self.repo.index.commit(message)

            # Return to main branch
            try:
                self.repo.git.checkout("main")
            except Exception:
                try:
                    self.repo.git.checkout("master")
                except Exception:
                    pass

            logger.info(f"Committed config changes: {commit.hexsha[:8]}")
            return commit.hexsha
        except Exception as e:
            logger.warning(f"Failed to commit: {e}")
            return None

    def create_pr_description(
        self,
        trial_id: int,
        metrics: dict,
        diff_lines: list[str],
        reason: str,
    ) -> str:
        """Generate a PR description for an accepted trial."""
        diff_str = "\n".join(f"  {line}" for line in diff_lines)
        metrics_rows = "\n".join(
            f"| {k} | {v:.4f} |" for k, v in metrics.items()
        )

        return f"""## 🧬 Auto-Augment Agent — Trial {trial_id}

### Decision
✅ **Accepted** — {reason}

### Config Changes
```
{diff_str}
```

### Metrics
| Metric | Value |
|--------|-------|
{metrics_rows}

---
*This PR was automatically created by the Auto-Augment Agent.*
*Review the config changes and merge if the improvements look correct.*
"""
