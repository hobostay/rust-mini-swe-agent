use anyhow::{Context, Result};
use minijinja::{Environment, context};

use crate::types::CommandOutput;

pub fn render_agent_template(
    template: &str,
    task: &str,
    model_name: &str,
    api_calls: usize,
    cost: f64,
) -> Result<String> {
    let env = Environment::new();
    env.render_str(
        template,
        context! {
            task => task,
            model_name => model_name,
            n_model_calls => api_calls,
            model_cost => cost,
        },
    )
    .context("failed to render agent template")
}

pub fn render_observation_template(template: &str, output: &CommandOutput) -> Result<String> {
    let env = Environment::new();
    env.render_str(
        template,
        context! {
            output => context! {
                output => output.output.as_str(),
                returncode => output.returncode,
                exception_info => output.exception_info.as_str(),
            },
        },
    )
    .context("failed to render observation template")
}

pub fn render_format_error_template(
    template: &str,
    error: &str,
    model_name: &str,
    api_calls: usize,
    cost: f64,
) -> Result<String> {
    let env = Environment::new();
    env.render_str(
        template,
        context! {
            error => error,
            model_name => model_name,
            n_model_calls => api_calls,
            model_cost => cost,
        },
    )
    .context("failed to render format error template")
}
