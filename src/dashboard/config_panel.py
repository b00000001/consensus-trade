"""
dashboard/config_panel.py
Agent configuration UI panel for the dashboard.
Allows operators to reassign agents to different models without editing files manually.
"""
import json
import logging
from typing import Optional

import dash
from dash import dcc, html, callback, ctx
from dash.dependencies import Input, Output, State

from src.config_loader import get_backends, save_backends, get_available_models, hot_reload_all


log = logging.getLogger(__name__)

PANEL_ID = "config-panel"
CURRENT_BACKENDS_KEY = "config-panel-current"


def get_current_assignments() -> list[dict]:
    """Return list of {id, role, adapter, model, description} for each backend."""
    backends = get_backends()
    return [
        {
            "id": be.get("id", ""),
            "role": be.get("role", ""),
            "adapter": be.get("adapter", ""),
            "model": be.get("model", ""),
            "description": be.get("description", ""),
            "fanout": be.get("includeInFanout", False),
        }
        for be in backends.get("backends", [])
    ]


def render_agent_row(agent: dict, available_models: list[dict]) -> html.Div:
    """Render a single agent config row with model dropdown."""
    adapter = agent.get("adapter", "")

    # Available models vary by adapter type
    if adapter == "ollama":
        model_options = [{"label": m.get("name", m.get("model", "")), "value": m.get("name", m.get("model", ""))} for m in available_models]
    elif adapter == "minimax":
        model_options = [{"label": "MiniMax-M2.7", "value": "MiniMax-M2.7"}]
    elif adapter == "claude-cli":
        model_options = [
            {"label": "Sonnet 4 (claude-sonnet-4-6)", "value": "claude-sonnet-4-6"},
            {"label": "Opus 4 (claude-opus-4)", "value": "claude-opus-4"},
            {"label": "Haiku 4 (claude-haiku-4)", "value": "claude-haiku-4"},
        ]
    elif adapter == "gemini-cli":
        model_options = [
            {"label": "Gemini 2.0 Flash", "value": "gemini-2.0-flash"},
            {"label": "Gemini 3 Flash", "value": "gemini-3-flash"},
            {"label": "Gemini 3 Ultra", "value": "gemini-3-ultra"},
        ]
    else:
        model_options = [{"label": agent.get("model", ""), "value": agent.get("model", "")}]

    return html.Div(className="config-agent-row", children=[
        html.Div(className="config-agent-id", children=[
            html.Strong(agent.get("id", "")),
            html.Div(agent.get("role", ""), className="config-agent-role"),
        ]),
        html.Div(className="config-agent-adapter", children=adapter),
        html.Div(className="config-agent-model", children=[
            dcc.Dropdown(
                id={"type": "model-dropdown", "index": agent.get("id", "")},
                options=model_options,
                value=agent.get("model", ""),
                clearable=False,
                style={"minWidth": "160px"},
            )
        ]),
        html.Div(className="config-agent-desc", children=agent.get("description", "")),
    ])


def render_config_panel(app: dash.Dash) -> html.Div:
    """
    Build the CONFIG tab content.
    Must be called after app is created and all adapters are importable.
    """
    available_models = get_available_models()

    assignments = get_current_assignments()
    rows = [render_agent_row(a, available_models) for a in assignments]

    synthesis = get_backends().get("synthesis", {})
    judge_id = synthesis.get("judgeBackendId", "claude-judge")
    confidence_threshold = synthesis.get("confidenceThreshold", 0.65)

    return html.Div(id=PANEL_ID, children=[
        html.H2("Agent Configuration", className="config-title"),
        html.P("Change which model each agent uses. Changes are written to config/backends.json and take effect immediately.", className="config-subtitle"),

        # Confidence threshold
        html.Div(className="config-synthesis-section", children=[
            html.H3("Judge Synthesis"),
            html.Label("Confidence threshold (triggers judge escalation below this):"),
            dcc.Slider(
                id="confidence-threshold-slider",
                min=0.0, max=1.0, step=0.05,
                value=confidence_threshold,
                marks={0.0: "0", 0.5: "0.5", 0.65: "0.65", 0.8: "0.8", 1.0: "1.0"},
            ),
            html.Div(id="confidence-threshold-display", children=f"{confidence_threshold:.2f}"),
        ]),

        # Agent rows header
        html.Div(className="config-header-row", children=[
            html.Div("Agent"),
            html.Div("Adapter"),
            html.Div("Model"),
            html.Div("Description"),
        ]),

        # Agent rows
        html.Div(className="config-agent-rows", children=rows),

        # Apply button
        html.Div(className="config-actions", children=[
            html.Button("Apply & Reload", id="apply-config-btn", n_clicks=0, className="config-apply-btn"),
            html.Div(id="config-apply-status", className="config-status"),
        ]),

        # Ollama models section
        html.Div(className="config-ollama-models", children=[
            html.H3("Available Ollama Models"),
            html.Div(id="ollama-models-list", children=[
                html.Div(m.get("name", "")) for m in available_models
            ] if available_models else "No models found (is Ollama running?)"),
        ]),
    ])


def register_config_callbacks(app: dash.Dash):
    """
    Register all config panel callbacks on the app.
    Call after app.layout is set.
    """

    @callback(
        Output("config-apply-status", "children"),
        Input("apply-config-btn", "n_clicks"),
        [State({"type": "model-dropdown", "index": dash.ALL}, "value"),
         State({"type": "model-dropdown", "index": dash.ALL}, "id"),
         State("confidence-threshold-slider", "value")],
        prevent_initial_call=True,
    )
    def apply_config(n_clicks, all_values, all_ids, confidence_threshold):
        if n_clicks == 0:
            return ""

        # Build backend_id -> new_model map
        model_map = {}
        for cid, val in zip(all_ids, all_values):
            agent_id = cid.get("index", "")
            if agent_id and val:
                model_map[agent_id] = val

        # Load current backends
        backends = get_backends()

        # Update model for each changed backend
        for be in backends.get("backends", []):
            bid = be.get("id", "")
            if bid in model_map:
                be["model"] = model_map[bid]

        # Update synthesis settings
        if "synthesis" not in backends:
            backends["synthesis"] = {}
        backends["synthesis"]["confidenceThreshold"] = confidence_threshold

        # Save
        success = save_backends(backends)

        if success:
            hot_reload_all()
            return html.Span("✅ Config applied and reloaded!", className="config-success")
        else:
            return html.Span("❌ Failed to write config", className="config-error")

    @callback(
        Output("confidence-threshold-display", "children"),
        Input("confidence-threshold-slider", "value"),
    )
    def update_threshold_display(val):
        return f"{val:.2f}"