#!/usr/bin/env python3
"""Build Orbit Chase and push the static bundle to a Hugging Face Space."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

# Tools
GIT = "git"
NPM = "npm"
HF_CLI = "hf"

# Environment
ENV_HF_USER = "HF_USER"
ENV_HF_SPACE = "HF_SPACE"
ENV_HF_BRANCH = "HF_BRANCH"
ENV_HF_TOKEN = "HF_TOKEN"
ENV_NODE_ENV = "NODE_ENV"
PRODUCTION = "production"

# Space target
DEFAULT_HF_USER = "killvung"
DEFAULT_HF_SPACE = "orbit-chase"
DEFAULT_HF_BRANCH = "main"

# Paths
DIST_DIRNAME = "dist"
DEPLOY_DIRNAME = ".deploy"
STAGING_DIRNAME = "hf-space"
SPACE_README = Path("deploy/huggingface/README.md")
INDEX_HTML = "index.html"
README_MD = "README.md"
GIT_DIRNAME = ".git"
GITATTRIBUTES = ".gitattributes"
KEEP_IN_STAGING = frozenset({GIT_DIRNAME, GITATTRIBUTES})
TRAINED_PLAYER = Path("models/linear-sarsa-20260815-030554-ep8000.json")

# npm
NPM_CI = "ci"
NPM_RUN = "run"
NPM_BUILD = "build"
NPM_DEPLOY_HF = "deploy:hf"

# Git
ORIGIN = "origin"
HEAD = "HEAD"
GIT_IDENTITY_NAME = "Orbit Chase Deploy"
GIT_IDENTITY_EMAIL = "deploy@orbit-chase.local"
ASSETS_DIRNAME = "assets"
MODELS_DIRNAME = "models"
LABEL_GAME_BUNDLE = "game bundle"
LABEL_TRAINED_PLAYER = "trained player"
LABEL_SPACE_README = "Space README"
COMMIT_SUBJECT_WITH_CHANGES = "Publish {changes} to Hugging Face Space"
COMMIT_SUBJECT_FALLBACK = "Publish Orbit Chase static Space"
COMMIT_BODY_HEADER = "Updated Space files:"

# Hugging Face URLs
HF_HOST = "huggingface.co"
HF_SPACES_PATH = "/spaces/"
HF_SPACE_URL_TEMPLATE = f"https://{HF_HOST}{HF_SPACES_PATH}{{user}}/{{space}}"
HF_AUTHED_REMOTE_TEMPLATE = (
    f"https://{{user}}:{{token}}@{HF_HOST}{HF_SPACES_PATH}{{user}}/{{space}}"
)
HF_NEW_SPACE_URL = f"https://{HF_HOST}/new-space"
HF_SPACE_MARKER = f"{HF_HOST}{HF_SPACES_PATH}"

# Operator hints
STAGING_RESET_HINT = f"run: rm -rf {DEPLOY_DIRNAME}/{STAGING_DIRNAME} && npm run {NPM_DEPLOY_HF}"


@dataclass(frozen=True)
class Config:
    root: Path
    user: str
    space: str
    branch: str
    token: str | None

    @property
    def dist(self) -> Path:
        return self.root / DIST_DIRNAME

    @property
    def staging(self) -> Path:
        return self.root / DEPLOY_DIRNAME / STAGING_DIRNAME

    @property
    def space_readme(self) -> Path:
        return self.root / SPACE_README

    @property
    def space_url(self) -> str:
        return HF_SPACE_URL_TEMPLATE.format(user=self.user, space=self.space)

    @property
    def remote_url(self) -> str:
        if not self.token:
            return self.space_url
        return HF_AUTHED_REMOTE_TEMPLATE.format(
            user=self.user,
            token=self.token,
            space=self.space,
        )

    @classmethod
    def load(cls) -> Config:
        return cls(
            root=Path(__file__).resolve().parents[2],
            user=os.environ.get(ENV_HF_USER, DEFAULT_HF_USER),
            space=os.environ.get(ENV_HF_SPACE, DEFAULT_HF_SPACE),
            branch=os.environ.get(ENV_HF_BRANCH, DEFAULT_HF_BRANCH),
            token=os.environ.get(ENV_HF_TOKEN) or _hf_cli_token() or None,
        )


class GitRepo:
    def __init__(self, path: Path) -> None:
        self.path = path

    def __call__(self, *args: str, check: bool = True) -> str:
        result = _run([GIT, "-C", str(self.path), *args], check=check, capture=True)
        return (result.stdout or "").strip()

    def succeeded(self, *args: str) -> bool:
        return _run(
            [GIT, "-C", str(self.path), *args],
            check=False,
            capture=True,
        ).returncode == 0

    def push(self, branch: str) -> bool:
        return _run(
            [GIT, "-C", str(self.path), "push", "-u", ORIGIN, branch],
            check=False,
        ).returncode == 0


class Deployer:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.git = GitRepo(config.staging)

    def run(self) -> None:
        _require(GIT)
        _require(NPM)
        self._build()
        self._prepare_staging()
        self._sync_dist()
        self._publish()

    def _build(self) -> None:
        _log(f"Installing dependencies ({NPM} {NPM_CI})")
        _run([NPM, NPM_CI], cwd=self.config.root)

        _log("Running production build")
        env = os.environ.copy()
        env[ENV_NODE_ENV] = PRODUCTION
        _run([NPM, NPM_RUN, NPM_BUILD], cwd=self.config.root, env=env)

        dist = self.config.dist
        if not (dist / INDEX_HTML).is_file():
            _fail(f"build did not produce {DIST_DIRNAME}/{INDEX_HTML}")
        if not (dist / TRAINED_PLAYER).is_file():
            _fail(f"trained player model missing from {DIST_DIRNAME}/{TRAINED_PLAYER}")

    def _prepare_staging(self) -> None:
        staging = self.config.staging
        staging.parent.mkdir(parents=True, exist_ok=True)

        if (staging / GIT_DIRNAME).is_dir():
            _log("Using Hugging Face staging clone")
        elif not self._clone_space():
            _log("Space repo not found locally; initializing new staging repository")
            if staging.exists():
                shutil.rmtree(staging)
            staging.mkdir(parents=True)
            self.git("init", "-b", self.config.branch)
            self.git("remote", "add", ORIGIN, self.config.remote_url)

        self.git("config", "user.name", GIT_IDENTITY_NAME)
        self.git("config", "user.email", GIT_IDENTITY_EMAIL)
        self.git("remote", "set-url", ORIGIN, self.config.remote_url)
        self._assert_isolated_staging()

    def _clone_space(self) -> bool:
        staging = self.config.staging
        _log(f"Cloning Hugging Face Space into {staging}")
        if staging.exists():
            shutil.rmtree(staging)

        cloned = _git_clone(self.config.remote_url, staging, branch=self.config.branch)
        if not cloned:
            cloned = _git_clone(self.config.remote_url, staging)
        if not cloned:
            return False

        if not self.git.succeeded("checkout", self.config.branch):
            self.git("checkout", "-b", self.config.branch)
        return True

    def _assert_isolated_staging(self) -> None:
        git_dir = self.config.staging / GIT_DIRNAME
        if not git_dir.is_dir():
            _fail(f"staging repo is missing {git_dir} — {STAGING_RESET_HINT}")

        common = Path(self.git("rev-parse", "--git-common-dir"))
        if not common.is_absolute():
            common = self.config.staging / common
        if common.resolve() != git_dir.resolve():
            _fail(f"staging directory is not an isolated git repo — {STAGING_RESET_HINT}")

    def _assert_hf_remote(self) -> None:
        remote = self.git("remote", "get-url", ORIGIN)
        if HF_SPACE_MARKER not in remote:
            _fail(f"origin is not a Hugging Face Space remote: {remote}")

    def _sync_dist(self) -> None:
        _log(f"Syncing production build ({DIST_DIRNAME}/)")
        staging = self.config.staging
        for entry in staging.iterdir():
            if entry.name in KEEP_IN_STAGING:
                continue
            _remove(entry)
        for source in self.config.dist.iterdir():
            if source.name in KEEP_IN_STAGING:
                continue
            _copy(source, staging / source.name)
        shutil.copy2(self.config.space_readme, staging / README_MD)
        self._assert_isolated_staging()

    def _publish(self) -> None:
        self.git("fetch", ORIGIN, self.config.branch, "--quiet", check=False)
        changed = self._commit_if_needed()
        ahead = self._commits_ahead()

        if not changed and ahead == 0:
            _log("No changes to deploy")
            return
        if not changed:
            _log(f"No new file changes; pushing {ahead} pending commit(s)")

        self._assert_hf_remote()
        _log(f"Pushing to {self.config.space_url}")
        if not self.git.push(self.config.branch):
            _fail(_push_help(self.config.space))
        _log("Deploy complete")
        _log(f"Space URL: {self.config.space_url}")

    def _commit_if_needed(self) -> bool:
        if not self.git("status", "--porcelain"):
            return False
        self.git("add", "-A")
        subject, body = _commit_message(self.git("diff", "--cached", "--name-status"))
        self.git("commit", "-m", subject, "-m", body)
        _log(f"Created commit: {subject}")
        return True

    def _commits_ahead(self) -> int:
        count = self.git(
            "rev-list",
            "--count",
            f"{ORIGIN}/{self.config.branch}..{HEAD}",
            check=False,
        )
        try:
            return int(count)
        except ValueError:
            return 0


def _log(message: str) -> None:
    print(f"==> {message}", flush=True)


def _fail(message: str) -> None:
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(1)


def _require(name: str) -> None:
    if shutil.which(name) is None:
        _fail(f"missing required command: {name}")


def _run(
    args: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    check: bool = True,
    capture: bool = False,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=cwd,
        env=env,
        check=check,
        capture_output=capture,
        text=True,
    )


def _git_clone(remote: str, destination: Path, branch: str | None = None) -> bool:
    args = [GIT, "clone"]
    if branch:
        args.extend(["--branch", branch])
    args.extend([remote, str(destination)])
    return _run(args, check=False, capture=True).returncode == 0


def _hf_cli_token() -> str:
    hf = shutil.which(HF_CLI)
    if hf is None:
        return ""
    result = _run([hf, "auth", "token"], check=False, capture=True)
    return (result.stdout or "").strip() if result.returncode == 0 else ""


def _remove(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()


def _copy(source: Path, destination: Path) -> None:
    if source.is_dir():
        shutil.copytree(source, destination)
    else:
        shutil.copy2(source, destination)


def _commit_message(name_status: str) -> tuple[str, str]:
    entries = [line for line in name_status.splitlines() if line]
    paths = [line.split("\t", 1)[-1] for line in entries]
    labels = _change_labels(paths)
    subject = (
        COMMIT_SUBJECT_WITH_CHANGES.format(changes=", ".join(labels))
        if labels
        else COMMIT_SUBJECT_FALLBACK
    )
    listed = "\n".join(f"- {line}" for line in entries)
    body = f"{COMMIT_BODY_HEADER}\n{listed}" if listed else COMMIT_BODY_HEADER
    return subject, body


def _change_labels(paths: list[str]) -> list[str]:
    labels: list[str] = []
    if any(path == INDEX_HTML or path.startswith(f"{ASSETS_DIRNAME}/") for path in paths):
        labels.append(LABEL_GAME_BUNDLE)
    if any(path.startswith(f"{MODELS_DIRNAME}/") for path in paths):
        labels.append(LABEL_TRAINED_PLAYER)
    if README_MD in paths:
        labels.append(LABEL_SPACE_README)
    return labels


def _push_help(space: str) -> str:
    return f"""git push failed.

Create the Space first at:
  {HF_NEW_SPACE_URL}

Use SDK: Static, name: {space}
(no build step — this deploy pushes pre-built {DIST_DIRNAME}/ files)

Then authenticate with either:
  export {ENV_HF_TOKEN}=hf_...
  npm run {NPM_DEPLOY_HF}

or:
  huggingface-cli login
  npm run {NPM_DEPLOY_HF}"""


def main() -> None:
    Deployer(Config.load()).run()


if __name__ == "__main__":
    main()
