# Contributing

Thanks for contributing to `rust-mini-swe-agent`.

## Before opening a PR

Please run the local checks:

```bash
cargo fmt --check
cargo check
cargo test
```

If your change affects runtime behavior, include a short note about what you verified manually.

## Pull requests

Keep PRs scoped. A smaller PR is easier to review and less likely to introduce regressions.

For behavior changes, please include:

- what changed
- why it changed
- how you validated it
- any remaining limitations or follow-up work

If your change touches one of these areas, call that out explicitly:

- model/provider behavior
- environment behavior
- benchmark loading or output
- trajectory format
- inspector/TUI behavior

## Issues

When filing a bug, include enough detail to reproduce it:

- exact command
- relevant config
- provider/environment class
- stderr or trajectory snippet if available

## Style

- Keep changes pragmatic and minimal.
- Preserve existing CLI/config behavior unless the change explicitly intends to alter it.
- Prefer compatibility with the original `mini-swe-agent` behavior where practical.

## Notes

- CI runs `cargo fmt --check`, `cargo check`, and `cargo test`.
- Some integrations depend on external services or credentials, so not every path can be fully exercised in CI.
