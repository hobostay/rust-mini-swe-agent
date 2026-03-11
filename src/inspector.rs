use std::fs;
use std::io::{self, Stdout, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use crossterm::event::{self, Event, KeyCode};
use crossterm::execute;
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout};
use ratatui::style::{Color, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Clear, Paragraph, Wrap};

use crate::types::{Message, Trajectory};

pub fn collect_trajectory_files(path: &Path) -> Result<Vec<PathBuf>> {
    if path.is_file() {
        return Ok(vec![path.to_path_buf()]);
    }
    if !path.is_dir() {
        bail!("path does not exist: {}", path.display());
    }
    let mut files = Vec::new();
    collect_recursive(path, &mut files)?;
    if files.is_empty() {
        bail!("no .traj.json files found in {}", path.display());
    }
    files.sort();
    Ok(files)
}

fn collect_recursive(dir: &Path, out: &mut Vec<PathBuf>) -> Result<()> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            collect_recursive(&path, out)?;
        } else if path
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.ends_with(".traj.json"))
        {
            out.push(path);
        }
    }
    Ok(())
}

pub fn load_trajectory(path: &Path) -> Result<Trajectory> {
    let raw =
        fs::read_to_string(path).with_context(|| format!("failed to read {}", path.display()))?;
    serde_json::from_str(&raw).with_context(|| format!("invalid trajectory {}", path.display()))
}

pub fn messages_to_steps(messages: &[Message]) -> Vec<Vec<Message>> {
    let mut steps = Vec::new();
    let mut current = Vec::new();
    for message in messages {
        if message.role == "assistant" || !message.actions.is_empty() {
            if !current.is_empty() {
                steps.push(current);
                current = Vec::new();
            }
        }
        current.push(message.clone());
    }
    if !current.is_empty() {
        steps.push(current);
    }
    steps
}

pub fn print_trajectory(path: &Path, step: Option<usize>) -> Result<()> {
    let trajectory = load_trajectory(path)?;
    let steps = messages_to_steps(&trajectory.messages);
    if let Some(step) = step {
        print_step(path, &steps, step)?;
        return Ok(());
    }

    println!(
        "trajectory={} steps={} exit_status={}",
        path.display(),
        steps.len(),
        trajectory.info.exit_status
    );
    for (i, step_messages) in steps.iter().enumerate() {
        println!("\n== step {} ==", i + 1);
        for message in step_messages {
            println!("[{}]\n{}\n", message.role, message.content);
        }
    }
    Ok(())
}

pub fn run_tui(paths: Vec<PathBuf>) -> Result<()> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;
    let result = run_app(&mut terminal, InspectorApp::new(paths)?);
    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;
    result
}

fn run_app(terminal: &mut Terminal<CrosstermBackend<Stdout>>, mut app: InspectorApp) -> Result<()> {
    loop {
        terminal.draw(|frame| app.draw(frame))?;
        if event::poll(Duration::from_millis(100))?
            && let Event::Key(key) = event::read()?
        {
            match key.code {
                KeyCode::Char('q') => return Ok(()),
                KeyCode::Right | KeyCode::Char('l') => app.next_step(),
                KeyCode::Left | KeyCode::Char('h') => app.prev_step(),
                KeyCode::Char('L') => app.next_trajectory()?,
                KeyCode::Char('H') => app.prev_trajectory()?,
                KeyCode::Down | KeyCode::Char('j') => app.scroll(3),
                KeyCode::Up | KeyCode::Char('k') => app.scroll(-3),
                KeyCode::Home | KeyCode::Char('0') => app.first_step(),
                KeyCode::End | KeyCode::Char('$') => app.last_step(),
                KeyCode::Char('?') => app.toggle_help(),
                KeyCode::Char('e') => app.open_current_step_in_pager(terminal)?,
                KeyCode::Char('E') => app.open_current_trajectory_in_pager(terminal)?,
                KeyCode::Char('r') => app.toggle_raw_view(),
                _ => {}
            }
        }
    }
}

struct InspectorApp {
    paths: Vec<PathBuf>,
    trajectories: Vec<Trajectory>,
    steps: Vec<Vec<Vec<Message>>>,
    trajectory_index: usize,
    step_index: usize,
    scroll: u16,
    show_help: bool,
    raw_view: bool,
    status: String,
}

impl InspectorApp {
    fn new(paths: Vec<PathBuf>) -> Result<Self> {
        let trajectories: Vec<Trajectory> = paths
            .iter()
            .map(|path| load_trajectory(path))
            .collect::<Result<_>>()?;
        let steps = trajectories
            .iter()
            .map(|traj| messages_to_steps(&traj.messages))
            .collect();
        Ok(Self {
            paths,
            trajectories,
            steps,
            trajectory_index: 0,
            step_index: 0,
            scroll: 0,
            show_help: false,
            raw_view: false,
            status: String::new(),
        })
    }

    fn draw(&self, frame: &mut ratatui::Frame<'_>) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3),
                Constraint::Min(1),
                Constraint::Length(2),
            ])
            .split(frame.area());

        let title = format!(
            "Trajectory {}/{}  Step {}/{}  status={}  {}",
            self.trajectory_index + 1,
            self.paths.len(),
            self.step_index + 1,
            self.current_steps().len().max(1),
            self.trajectories[self.trajectory_index].info.exit_status,
            self.paths[self.trajectory_index].display()
        );
        let header =
            Paragraph::new(title).block(Block::default().borders(Borders::ALL).title("Inspector"));
        frame.render_widget(header, chunks[0]);

        let lines = self.current_step_lines();
        let body = Paragraph::new(lines)
            .wrap(Wrap { trim: false })
            .scroll((self.scroll, 0))
            .block(Block::default().borders(Borders::ALL).title("Messages"));
        frame.render_widget(body, chunks[1]);

        let help = Paragraph::new(format!(
            "h/l step  H/L trajectory  j/k scroll  0/$ first/last  e/E pager  r raw={}  ? help  q quit{}",
            if self.raw_view { "on" } else { "off" },
            if self.status.is_empty() {
                String::new()
            } else {
                format!("  status={}", self.status)
            }
        ))
            .block(Block::default().borders(Borders::ALL).title("Keys"));
        frame.render_widget(help, chunks[2]);

        if self.show_help {
            let area = centered_rect(70, 60, frame.area());
            frame.render_widget(Clear, area);
            let help = Paragraph::new(vec![
                Line::from("Navigation"),
                Line::from("h/l or left/right: previous/next step"),
                Line::from("H/L: previous/next trajectory"),
                Line::from("j/k or up/down: scroll"),
                Line::from("0/$ or home/end: first/last step"),
                Line::from(""),
                Line::from("Views"),
                Line::from("r: toggle rendered/raw JSON view"),
                Line::from("e: open current step in pager"),
                Line::from("E: open full trajectory in pager"),
                Line::from(""),
                Line::from("Other"),
                Line::from("?: toggle this help"),
                Line::from("q: quit"),
            ])
            .wrap(Wrap { trim: false })
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title("Inspector Help"),
            );
            frame.render_widget(help, area);
        }
    }

    fn current_steps(&self) -> &Vec<Vec<Message>> {
        &self.steps[self.trajectory_index]
    }

    fn current_step_lines(&self) -> Vec<Line<'static>> {
        let step = &self.current_steps()[self.step_index];
        if self.raw_view {
            return serde_json::to_string_pretty(step)
                .unwrap_or_else(|_| "[]".to_string())
                .lines()
                .map(|line| Line::from(line.to_string()))
                .collect();
        }
        let mut lines = Vec::new();
        for message in step {
            lines.push(Line::from(Span::styled(
                format!("[{}]", message.role.to_uppercase()),
                Style::default().fg(Color::Yellow),
            )));
            for line in message.content.lines() {
                lines.push(Line::from(line.to_string()));
            }
            lines.push(Line::from(String::new()));
        }
        lines
    }

    fn next_step(&mut self) {
        if self.step_index + 1 < self.current_steps().len() {
            self.step_index += 1;
            self.scroll = 0;
        }
    }

    fn prev_step(&mut self) {
        if self.step_index > 0 {
            self.step_index -= 1;
            self.scroll = 0;
        }
    }

    fn first_step(&mut self) {
        self.step_index = 0;
        self.scroll = 0;
    }

    fn last_step(&mut self) {
        self.step_index = self.current_steps().len().saturating_sub(1);
        self.scroll = 0;
    }

    fn next_trajectory(&mut self) -> Result<()> {
        if self.trajectory_index + 1 < self.paths.len() {
            self.trajectory_index += 1;
            self.step_index = 0;
            self.scroll = 0;
        }
        Ok(())
    }

    fn prev_trajectory(&mut self) -> Result<()> {
        if self.trajectory_index > 0 {
            self.trajectory_index -= 1;
            self.step_index = 0;
            self.scroll = 0;
        }
        Ok(())
    }

    fn scroll(&mut self, delta: i16) {
        if delta.is_negative() {
            self.scroll = self.scroll.saturating_sub(delta.unsigned_abs());
        } else {
            self.scroll = self.scroll.saturating_add(delta as u16);
        }
    }

    fn toggle_help(&mut self) {
        self.show_help = !self.show_help;
    }

    fn toggle_raw_view(&mut self) {
        self.raw_view = !self.raw_view;
        self.scroll = 0;
    }

    fn open_current_step_in_pager(
        &mut self,
        terminal: &mut Terminal<CrosstermBackend<Stdout>>,
    ) -> Result<()> {
        let step = self.current_steps()[self.step_index].clone();
        let raw = serde_json::to_string_pretty(&step)?;
        self.open_in_pager(terminal, &raw)
    }

    fn open_current_trajectory_in_pager(
        &mut self,
        terminal: &mut Terminal<CrosstermBackend<Stdout>>,
    ) -> Result<()> {
        let raw = fs::read_to_string(&self.paths[self.trajectory_index])?;
        self.open_in_pager(terminal, &raw)
    }

    fn open_in_pager(
        &mut self,
        terminal: &mut Terminal<CrosstermBackend<Stdout>>,
        content: &str,
    ) -> Result<()> {
        disable_raw_mode()?;
        execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
        terminal.show_cursor()?;

        let pager = pager_command();
        let status = match run_pager(&pager, content) {
            Ok(()) => format!("opened in {}", pager[0]),
            Err(error) => format!("pager failed: {error}"),
        };

        enable_raw_mode()?;
        execute!(terminal.backend_mut(), EnterAlternateScreen)?;
        self.status = status;
        Ok(())
    }
}

fn print_step(path: &Path, steps: &[Vec<Message>], step: usize) -> Result<()> {
    if step == 0 || step > steps.len() {
        bail!("step {} out of range for {}", step, path.display());
    }
    println!("trajectory={} step={}", path.display(), step);
    for message in &steps[step - 1] {
        println!("[{}]\n{}\n", message.role, message.content);
    }
    Ok(())
}

fn pager_command() -> Vec<String> {
    if command_exists("jless") {
        return vec!["jless".to_string()];
    }
    if command_exists("less") {
        return vec!["less".to_string(), "-R".to_string()];
    }
    vec!["cat".to_string()]
}

fn run_pager(command: &[String], content: &str) -> Result<()> {
    let (program, args) = command.split_first().context("empty pager command")?;
    let mut child = Command::new(program)
        .args(args)
        .stdin(std::process::Stdio::piped())
        .spawn()
        .with_context(|| format!("failed to launch pager {}", program))?;
    if let Some(stdin) = child.stdin.as_mut() {
        stdin.write_all(content.as_bytes())?;
    }
    let status = child.wait()?;
    if !status.success() {
        bail!("pager exited with status {}", status);
    }
    Ok(())
}

fn command_exists(program: &str) -> bool {
    Command::new("sh")
        .args(["-lc", &format!("command -v {} >/dev/null 2>&1", program)])
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}

fn centered_rect(
    percent_x: u16,
    percent_y: u16,
    area: ratatui::layout::Rect,
) -> ratatui::layout::Rect {
    let vertical = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage((100 - percent_y) / 2),
            Constraint::Percentage(percent_y),
            Constraint::Percentage((100 - percent_y) / 2),
        ])
        .split(area);
    Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage((100 - percent_x) / 2),
            Constraint::Percentage(percent_x),
            Constraint::Percentage((100 - percent_x) / 2),
        ])
        .split(vertical[1])[1]
}
