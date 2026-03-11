# rust-mini-swe-agent

这是一个单独目录里的 Rust 版复刻，目标是尽量对齐原项目的主使用面。

- `mini` 子命令
- `bench` / `bench-single`
- `inspector`
- `config show/set/unset/setup`
- YAML 配置加载
- OpenAI 兼容 `/chat/completions` 模型调用
- 单一 `bash` tool call 解析
- 本地和 docker 环境执行
- `human` / `confirm` / `yolo` 模式
- trajectory 保存与回放

它仍然没有把 Python 版里所有 provider 和所有外部系统都逐个搬齐，但主控制流已经拆成与原项目相似的多模块结构。

## 运行 `mini`

先准备环境变量：

```bash
export OPENAI_API_KEY=your_key
```

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

## 当前范围

已经实现：

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
- 简化版 benchmark runner
- 简化版 trajectory inspector

还没实现：

- Python 版里每个 provider 的专用兼容层
- HF 数据集直连和 SWE-bench 专用镜像辅助逻辑
- Textual 风格全屏 inspector UI
- `contree-sdk` 原生集成和更完整的 bubblewrap/singularity 参数集
