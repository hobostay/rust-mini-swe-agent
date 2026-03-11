# rust-mini-swe-agent

[![CI](https://github.com/hobostay/rust-mini-swe-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/hobostay/rust-mini-swe-agent/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](./LICENSE)

一个尽量对齐 [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) 主使用面的 Rust 版复刻。

它提供了一个完整的本地 CLI agent 工作流：模型调用、`bash` action 执行、benchmark 批量运行、trajectory 保存与回放，以及本地 / 容器 / ConTree 兼容环境。

## 特性

- `mini`、`bench`、`bench-single`、`inspector`、`config`
- OpenAI 兼容 `/chat/completions` 和 `/responses`
- 单一 `bash` tool 调用与 text-based bash code block 回退
- `human` / `confirm` / `yolo` 模式
- 本地、Docker、Singularity、Bubblewrap、ConTree 兼容环境
- trajectory 持久化、TUI inspector、benchmark 输出和状态汇总
- 本地 JSON/JSONL、远程 JSON/JSONL、Hugging Face dataset repo

## 快速开始

先准备 API Key：

```bash
export OPENAI_API_KEY=your_key
```

运行一个最小任务：

```bash
cargo run -- mini -m gpt-4.1-mini -t "Inspect the repository and finish immediately"
```

默认输出 trajectory：

```text
~/.config/rust-mini-swe-agent/last_run.traj.json
```

## 运行 `mini`

直接运行：

```bash
cargo run -- mini -m gpt-4.1-mini -t "Create a hello world file and finish the task"
```

或者从 stdin 读任务：

```bash
printf 'Inspect the repository and then finish immediately' | cargo run -- mini -m gpt-4.1-mini
```

输出的 trajectory 默认保存在：

```text
~/.config/rust-mini-swe-agent/last_run.traj.json
```

## 配置

支持通过 `-c config.yaml` 载入配置，配置结构示例：

```yaml
agent:
  system_template: "You are a helpful assistant that can interact with a computer."
  instance_template: "Please solve this issue: {{task}}"
  step_limit: 0
  cost_limit: 3.0
  output_path: "./run.traj.json"
  mode: "confirm"
  whitelist_actions: ["^pwd$", "^ls"]
  confirm_exit: true
environment:
  environment_class: "local"
  cwd: "."
  timeout_secs: 30
  image: null
model:
  model_name: "gpt-4.1-mini"
  base_url: "https://api.openai.com/v1"
  api_key_env: "OPENAI_API_KEY"
  temperature: 0.2
run:
  task: null
  env_startup_command: null
```

仓库里也附带了现成示例：

```bash
cargo run -- mini -c config.example.yaml -t "Inspect the repository and then finish immediately"
```

如果要跑 `contree`，可以在配置里这样写：

```yaml
environment:
  environment_class: "contree"
  image: "docker://docker.io/swebench/sweb.eval.x86_64.demo:latest"
  cwd: "/testbed"
  interpreter: ["bash", "-c"]
  contree_config:
    base_url: "https://your-contree-endpoint"
    token: "your-contree-token"
```

仓库里也带了现成模板：

```bash
cargo run -- mini -c contree -t "Inspect the repository and finish immediately"
```

也支持和原项目类似的多层配置覆盖：

```bash
cargo run -- mini -c mini -c model.temperature=0.0 -t "Inspect the repo"
```

启动时会自动加载全局配置文件：

```text
~/.config/rust-mini-swe-agent/.env
```

## 批量运行

数据集可以来自三类来源：

- 本地 `json` 或 `jsonl`
- 远程 `http(s)` JSON/JSONL
- Hugging Face dataset repo id，例如 `princeton-nlp/SWE-Bench_Lite`

本地记录至少需要：

```json
{"instance_id":"demo-1","problem_statement":"Fix the bug"}
```

批量运行：

```bash
cargo run -- bench --dataset ./instances.json --output-dir ./runs -w 4
```

直接读取 Hugging Face 数据集：

```bash
cargo run -- bench --dataset princeton-nlp/SWE-Bench_Lite --split dev --output-dir ./runs -w 4
```

单例运行：

```bash
cargo run -- bench-single --dataset ./instances.json --instance demo-1
```

## Inspector

```bash
cargo run -- inspector ./runs
cargo run -- inspector ./runs --step 2
```

## Config

```bash
cargo run -- config setup
cargo run -- config show
cargo run -- config set OPENAI_API_KEY sk-xxx
```

## 当前状态

已实现：

- 多子命令 CLI
- 交互模式
- 多 `-c` 配置和 `key=value` 覆盖
- 自动加载全局 `.env`
- 内置 `mini.yaml`、`default.yaml` 和 `contree.yaml`
- bash tool call 校验
- text-based bash code block 解析回退
- `openrouter` / `openrouter_response`
- `requesty`
- `portkey` / `portkey_response`
- `litellm_response` 风格 `/responses` 兼容
- 远程 `http(s)` 数据集读取
- Hugging Face dataset repo 读取
- `MSWEA_SWEBENCH_DATASET_BASE_URL` / `MSWEA_SWEBENCH_DATASET_DIR` 数据集解析
- `MSWEA_HF_DATASET_CONFIG` 自定义 Hugging Face dataset config
- `cost_tracking` / `multimodal_regex` 配置字段兼容
- 本地和 docker 环境
- `singularity` / `bubblewrap` / `contree` 环境入口
- 更接近原版的 docker 持久容器和 singularity sandbox 生命周期
- 提交哨兵 `COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`
- trajectory 持久化
- benchmark runner
- TUI trajectory inspector

已知差距：

- 不是逐 provider 的原生 SDK 全量复刻，部分 provider 仍以兼容层方式实现
- `contree` 已支持 REST/API 与兼容执行路径，但不是 Python `contree_sdk` 的原生 session 实现
- inspector 是 `ratatui` TUI，不是 Python Textual UI
- 仍有少量环境高级参数和边界行为未与 Python 版逐项对齐
