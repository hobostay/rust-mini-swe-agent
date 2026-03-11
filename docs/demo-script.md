# Demo Script

This is a short terminal demo script for recording or posting about `rust-mini-swe-agent`.

## Goal

Show the full loop quickly:

1. run a task with `mini`
2. show the generated trajectory
3. open the inspector

## Suggested setup

- terminal font size slightly larger than normal
- clean prompt
- `OPENAI_API_KEY` already exported
- run from the repository root

## Demo flow

### 1. Show the CLI surface

```bash
cargo run -- --help
```

Say:

> This project recreates the main mini-swe-agent workflow in Rust, including `mini`, `bench`, and `inspector`.

### 2. Run a minimal task

```bash
cargo run -- mini -m gpt-4.1-mini -t "Inspect the repository, list the top-level files, and then finish immediately"
```

Say:

> The agent keeps a linear message history, calls a single `bash` tool, executes commands in the configured environment, and saves the full trajectory.

### 3. Show the saved trajectory path

```bash
ls ~/.config/rust-mini-swe-agent
```

Optional:

```bash
cat ~/.config/rust-mini-swe-agent/last_run.traj.json | head -n 40
```

### 4. Open the inspector

```bash
cargo run -- inspector ~/.config/rust-mini-swe-agent
```

Inside the TUI:

- press `l` to move to the next step
- press `r` to toggle raw JSON view
- press `?` to open help
- press `e` to open the current step in a pager

Say:

> The inspector lets you replay the agent step by step, inspect rendered messages, and drill into the raw trajectory.

### 5. Show benchmark mode

```bash
cargo run -- bench --help
```

Say:

> The same repository also supports benchmark-style batch runs, dataset loading, and environment switching.

## Short version for a 30-45 second clip

```bash
cargo run -- mini -m gpt-4.1-mini -t "Inspect the repository and finish immediately"
cargo run -- inspector ~/.config/rust-mini-swe-agent
```

## Posting angle

Use one of these positioning lines:

- A Rust reimplementation of mini-swe-agent with benchmark runners and trajectory replay
- A hackable Rust coding agent for CLI tasks, eval runs, and multi-environment execution
- Rebuilt mini-swe-agent in Rust with `mini`, `bench`, `inspector`, and ConTree-compatible execution
