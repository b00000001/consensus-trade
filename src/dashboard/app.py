"""
dashboard/app.py — ConsensusTrade
Enterprise dashboard with hot_path (live P&L, risk, execution) and
cold_path (backtest, analytics). Thread-safe state management.
"""
import logging
import os
import threading
from datetime import datetime
from typing import Optional

import dash
from dash import dcc, html, dash_table, callback
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go

from dashboard.config_panel import render_config_panel, register_config_callbacks

from execution.portfolio import Portfolio
from execution.paper_trader import PaperTrader


log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Thread-safe Dashboard Store
# ═══════════════════════════════════════════════════════════════════════════════

class DashboardStore:
    """Thread-safe shared state between orchestrator and Dash callbacks."""

    def __init__(self):
        self._lock = threading.Lock()
        self.paper_trader: Optional[PaperTrader] = None
        self.portfolio: Optional[Portfolio] = None
        self.execution_feed: list[dict] = []
        self.agent_votes: dict = {}
        self.agent_models: dict = {}
        self.consensus_log: list[dict] = []
        self.risk_status: dict = {
            "circuit_breaker_tripped": False,
            "daily_loss_pct": 0.0,
            "trades_this_hour": 0,
            "consecutive_losses": 0,
        }

    def push_execution(self, event: dict):
        with self._lock:
            self.execution_feed.append({
                "time": datetime.now().strftime("%H:%M:%S"),
                "asset": event.get("asset", ""),
                "signal": event.get("signal", ""),
                "quantity": event.get("quantity", ""),
                "price": event.get("price", ""),
                "status": event.get("status", ""),
            })
            if len(self.execution_feed) > 200:
                self.execution_feed = self.execution_feed[-100:]

    def push_consensus(self, decision: dict, agent_models: dict = None):
        agents = decision.get("agent_ids", [])
        models = agent_models or {}
        agent_str = ", ".join(
            f"{aid} ({models.get(aid, '?')})" for aid in agents
        ) if agents else "—"

        with self._lock:
            self.consensus_log.append({
                "time": datetime.now().strftime("%H:%M:%S"),
                "asset": decision.get("asset", ""),
                "decision": decision.get("signal", ""),
                "confidence": f"{decision.get('confidence', 0):.2f}",
                "reason": decision.get("reason", ""),
                "agents": agent_str,
            })
            if len(self.consensus_log) > 200:
                self.consensus_log = self.consensus_log[-100:]

    def set_agent_votes(self, asset: str, votes: dict, agent_models: dict = None):
        with self._lock:
            self.agent_votes[asset] = votes
            if agent_models:
                self.agent_models.update(agent_models)

    def set_risk_status(self, status: dict):
        with self._lock:
            self.risk_status.update(status)

    def get_execution_feed(self, limit: int = 50) -> list[dict]:
        with self._lock:
            return list(self.execution_feed[-limit:])

    def get_consensus_log(self, limit: int = 50) -> list[dict]:
        with self._lock:
            return list(self.consensus_log[-limit:])

    def get_agent_vote_rows(self) -> list[dict]:
        with self._lock:
            rows = []
            for asset, votes in self.agent_votes.items():
                for agent_id, signal in votes.items():
                    model = self.agent_models.get(agent_id, "—")
                    rows.append({
                        "asset": asset,
                        "agent": agent_id,
                        "signal": signal,
                        "model": model,
                    })
            return rows

    def get_risk_status(self) -> dict:
        with self._lock:
            return dict(self.risk_status)


# Module-level singleton store
_store = DashboardStore()


# ═══════════════════════════════════════════════════════════════════════════════
# Public API (called by orchestrator)
# ═══════════════════════════════════════════════════════════════════════════════

def update_execution_feed_event(event: dict):
    _store.push_execution(event)


def update_consensus_log_event(decision: dict, agent_models: dict = None):
    _store.push_consensus(decision, agent_models)


def update_agent_votes(asset: str, votes: dict, agent_models: dict = None):
    _store.set_agent_votes(asset, votes, agent_models)


def update_risk_status(status: dict):
    _store.set_risk_status(status)


# ═══════════════════════════════════════════════════════════════════════════════
# Table style constants
# ═══════════════════════════════════════════════════════════════════════════════

TABLE_STYLE_HEADER = {
    "backgroundColor": "#1A2230",
    "color": "#4B5C72",
    "fontFamily": "'Inter', sans-serif",
    "fontSize": "10px",
    "fontWeight": "700",
    "textTransform": "uppercase",
    "letterSpacing": "0.1em",
    "padding": "10px 14px",
    "border": "none",
    "borderBottom": "1px solid rgba(100, 130, 180, 0.12)",
}

TABLE_STYLE_CELL = {
    "backgroundColor": "transparent",
    "color": "#E8EDF5",
    "fontFamily": "'JetBrains Mono', monospace",
    "fontSize": "12px",
    "padding": "10px 14px",
    "border": "none",
    "borderBottom": "1px solid rgba(100, 130, 180, 0.08)",
    "whiteSpace": "nowrap",
    "overflow": "hidden",
    "textOverflow": "ellipsis",
    "maxWidth": "250px",
}

TABLE_STYLE_DATA_COND = [
    {"if": {"row_index": "odd"}, "backgroundColor": "rgba(26, 34, 48, 0.3)"},
]


# ═══════════════════════════════════════════════════════════════════════════════
# App factory
# ═══════════════════════════════════════════════════════════════════════════════

def create_dashboard(
    paper_trader: PaperTrader,
    portfolio: Portfolio,
    port: int = 8050,
) -> dash.Dash:
    _store.paper_trader = paper_trader
    _store.portfolio = portfolio

    app = dash.Dash(
        __name__,
        external_stylesheets=[
            "https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@300;400;500;700&display=swap"
        ],
        meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
    )
    app.title = "ConsensusTrade | Command Center"

    # ── Build Layout ─────────────────────────────────────────────────────────
    app.layout = html.Div(style={"minHeight": "100vh"}, children=[

        # ── HEADER HUD ───────────────────────────────────────────────────────
        html.Header(id="command-hud", children=[
            html.Div(className="hud-logo", children=[
                html.Span("CONSENSUS"), "TRADE"
            ]),
            html.Div(className="hud-metrics", children=[
                _hud_metric("Firm Equity", "live-equity"),
                _hud_metric("Session P&L", "live-pnl"),
                _hud_metric("Risk Delta", "live-pnl-pct"),
            ]),
        ]),

        # ── TAB NAVIGATION ──────────────────────────────────────────────────
        dcc.Tabs(className="dash-tabs", children=[

            # ═══ HOT PATH: COMMAND CENTER ═══════════════════════════════════
            dcc.Tab(
                label="COMMAND CENTER",
                className="dash-tab",
                selected_className="dash-tab--selected",
                children=[_build_command_center()],
            ),

            # ═══ COLD PATH: STRATEGY AUDIT ══════════════════════════════════
            dcc.Tab(
                label="STRATEGY AUDIT",
                className="dash-tab",
                selected_className="dash-tab--selected",
                children=[_build_strategy_audit()],
            ),

            # ═══ CONFIG ═════════════════════════════════════════════════════
            dcc.Tab(
                label="⚙ CONFIG",
                className="dash-tab",
                selected_className="dash-tab--selected",
                children=[
                    html.Div(
                        id="config-tab-content",
                        style={"padding": "var(--gap-lg)"},
                        children=render_config_panel(app),
                    ),
                ],
            ),
        ]),

        # ── Refresh Interval ─────────────────────────────────────────────────
        dcc.Interval(id="hot-refresh", interval=3000, n_intervals=0),

        # ── Data Store ───────────────────────────────────────────────────────
        dcc.Store(id="hot-data-store", data={}),
    ])

    # ── Register Callbacks ───────────────────────────────────────────────────
    _register_hot_callbacks(app)
    _register_cold_callbacks(app)
    register_config_callbacks(app)

    return app


# ═══════════════════════════════════════════════════════════════════════════════
# Layout Builders
# ═══════════════════════════════════════════════════════════════════════════════

def _hud_metric(label: str, value_id: str) -> html.Div:
    return html.Div(className="hud-item", children=[
        html.Div(label, className="hud-label"),
        html.Div(id=value_id, className="hud-value"),
    ])


def _build_command_center() -> html.Div:
    """Hot path: live trading dashboard."""
    return html.Div(className="ct-grid ct-grid-2col", children=[

        # ── LEFT COLUMN ──────────────────────────────────────────────────────
        html.Div(children=[

            # Risk Monitor
            html.Div(className="ct-card", children=[
                html.Div("Operational Risk Monitor", className="ct-card-title"),
                html.Div(id="risk-monitor"),
            ]),

            # Execution Feed
            html.Div(className="ct-card", children=[
                html.Div("Direct Execution Feed", className="ct-card-title"),
                dash_table.DataTable(
                    id="execution-feed-table",
                    columns=[
                        {"name": "Time", "id": "time"},
                        {"name": "Asset", "id": "asset"},
                        {"name": "Signal", "id": "signal"},
                        {"name": "Qty", "id": "quantity"},
                        {"name": "Price", "id": "price"},
                        {"name": "Status", "id": "status"},
                    ],
                    data=[],
                    style_as_list_view=True,
                    page_size=8,
                    style_header=TABLE_STYLE_HEADER,
                    style_cell=TABLE_STYLE_CELL,
                    style_data_conditional=TABLE_STYLE_DATA_COND,
                ),
            ]),

        ]),

        # ── RIGHT COLUMN ─────────────────────────────────────────────────────
        html.Div(children=[

            # AI Consensus Log
            html.Div(className="ct-card", children=[
                html.Div("AI Consensus Log", className="ct-card-title"),
                dash_table.DataTable(
                    id="consensus-log-table",
                    columns=[
                        {"name": "Time", "id": "time"},
                        {"name": "Asset", "id": "asset"},
                        {"name": "Decision", "id": "decision"},
                        {"name": "Conf.", "id": "confidence"},
                        {"name": "Logic", "id": "reason"},
                        {"name": "Agents / Models", "id": "agents"},
                    ],
                    data=[],
                    style_as_list_view=True,
                    page_size=10,
                    style_header=TABLE_STYLE_HEADER,
                    style_cell=TABLE_STYLE_CELL,
                    style_data_conditional=TABLE_STYLE_DATA_COND,
                ),
            ]),

            # Agent Distribution
            html.Div(className="ct-card", children=[
                html.Div("Agent Distribution", className="ct-card-title"),
                dash_table.DataTable(
                    id="agent-votes-table",
                    columns=[
                        {"name": "Asset", "id": "asset"},
                        {"name": "Agent", "id": "agent"},
                        {"name": "Signal", "id": "signal"},
                        {"name": "Model", "id": "model"},
                    ],
                    data=[],
                    style_as_list_view=True,
                    style_header=TABLE_STYLE_HEADER,
                    style_cell=TABLE_STYLE_CELL,
                    style_data_conditional=TABLE_STYLE_DATA_COND,
                ),
            ]),

            # Bot Heartbeat
            html.Div(className="ct-card", children=[
                html.Div("Bot Heartbeat Status", className="ct-card-title"),
                dash_table.DataTable(
                    id="heartbeat-table",
                    columns=[
                        {"name": "Agent", "id": "agent_id"},
                        {"name": "Model", "id": "model"},
                        {"name": "Latency", "id": "latency_ms"},
                        {"name": "Status", "id": "status"},
                        {"name": "Last Check", "id": "timestamp"},
                    ],
                    data=[],
                    style_as_list_view=True,
                    style_header=TABLE_STYLE_HEADER,
                    style_cell=TABLE_STYLE_CELL,
                    style_data_conditional=TABLE_STYLE_DATA_COND,
                ),
            ]),
        ]),
    ])


def _build_strategy_audit() -> html.Div:
    """Cold path: backtesting and performance analytics."""
    return html.Div(style={"padding": "var(--gap-lg)"}, children=[

        # Backtest Engine
        html.Div(className="ct-card", children=[
            html.Div("Backtest Engine v2.1", className="ct-card-title"),
            html.Div(style={"display": "flex", "gap": "12px", "alignItems": "center", "flexWrap": "wrap"}, children=[
                dcc.Input(
                    id="bt-symbol", type="text",
                    placeholder="BTC/USDT", value="BTC/USDT",
                    className="ct-input",
                ),
                dcc.Input(
                    id="bt-start", type="text",
                    placeholder="2024-01-01",
                    className="ct-input",
                ),
                dcc.Input(
                    id="bt-end", type="text",
                    placeholder="2024-12-31",
                    className="ct-input",
                ),
                dcc.Dropdown(
                    id="bt-strategies",
                    options=[
                        {"label": "Momentum", "value": "momentum"},
                        {"label": "Mean Reversion", "value": "mean_reversion"},
                        {"label": "Breakout", "value": "breakout"},
                    ],
                    multi=True,
                    value=["momentum"],
                    style={"minWidth": "240px"},
                ),
                html.Button(
                    "INITIATE TEST", id="bt-run-btn", n_clicks=0,
                    className="ct-btn ct-btn-primary",
                ),
            ]),
        ]),

        # Results Table
        html.Div(className="ct-card", children=[
            html.Div("Historical Attribution", className="ct-card-title"),
            dash_table.DataTable(
                id="backtest-results-table",
                columns=[
                    {"name": "Strategy", "id": "strategy"},
                    {"name": "Asset", "id": "asset"},
                    {"name": "Sharpe", "id": "sharpe"},
                    {"name": "Sortino", "id": "sortino"},
                    {"name": "Max DD", "id": "max_dd"},
                    {"name": "Win Rate", "id": "win_rate"},
                    {"name": "Trades", "id": "trade_count"},
                ],
                data=[],
                style_as_list_view=True,
                style_header=TABLE_STYLE_HEADER,
                style_cell=TABLE_STYLE_CELL,
                style_data_conditional=TABLE_STYLE_DATA_COND,
            ),
        ]),

        # Chart
        html.Div(className="ct-card", children=[
            html.Div("Performance Attribution Chart", className="ct-card-title"),
            dcc.Graph(id="strategy-comparison-chart"),
        ]),

        # Performance Reports
        html.Div(className="ct-card", children=[
            html.Div("Phase 9: Performance Reports", className="ct-card-title"),
            html.Div(id="performance-reports-list"),
            html.Div(style={"marginTop": "var(--gap-md)"}, children=[
                html.Button(
                    "Generate Narrative", id="gen-report-btn", n_clicks=0,
                    className="ct-btn",
                ),
            ]),
            html.Div(id="report-status", style={"marginTop": "var(--gap-sm)"}),
        ]),
    ])


# ═══════════════════════════════════════════════════════════════════════════════
# Risk HTML builder
# ═══════════════════════════════════════════════════════════════════════════════

def _build_risk_html() -> list:
    status = _store.get_risk_status()
    is_tripped = status.get("circuit_breaker_tripped", False)

    return [
        # Circuit Breaker
        html.Div(className="risk-row", children=[
            html.Span("Circuit Breaker", className="risk-label"),
            html.Span(
                children=[
                    html.Span("⚠ " if is_tripped else ""),
                    html.Span(
                        "ACTUATED" if is_tripped else "NOMINAL",
                        className="risk-tripped" if is_tripped else "",
                        style={
                            "color": "var(--accent-red)" if is_tripped else "var(--accent-green)",
                            "fontWeight": "700",
                        },
                    ),
                ],
                className="risk-value",
            ),
        ]),

        # Daily Loss
        html.Div(className="risk-row", children=[
            html.Span("Daily Performance Limit", className="risk-label"),
            html.Span(
                f"{status.get('daily_loss_pct', 0):.2%}",
                className="risk-value",
                style={"color": "var(--accent-cyan)"},
            ),
        ]),

        # Trades/hr
        html.Div(className="risk-row", children=[
            html.Span("Tactical Velocity (Trades/hr)", className="risk-label"),
            html.Span(str(status.get("trades_this_hour", 0)), className="risk-value"),
        ]),

        # Consecutive Losses
        html.Div(className="risk-row", children=[
            html.Span("Loss Sequence Threshold", className="risk-label"),
            html.Span(str(status.get("consecutive_losses", 0)), className="risk-value"),
        ]),
    ]


# ═══════════════════════════════════════════════════════════════════════════════
# HOT PATH CALLBACKS
# ═══════════════════════════════════════════════════════════════════════════════

def _register_hot_callbacks(app: dash.Dash):

    @callback(
        Output("live-equity", "children"),
        Output("live-pnl", "children"),
        Output("live-pnl-pct", "children"),
        Output("risk-monitor", "children"),
        Input("hot-refresh", "n_intervals"),
    )
    def refresh_hud(n):
        portfolio = _store.portfolio
        if portfolio is None:
            return "—", "—", "—", []

        equity = portfolio.total_equity()
        pnl = portfolio.total_pnl()
        pnl_pct = portfolio.pnl_pct() * 100

        pnl_color = "var(--accent-green)" if pnl >= 0 else "var(--accent-red)"

        return (
            f"${equity:,.2f}",
            html.Span(f"${pnl:,.2f}", style={"color": pnl_color}),
            html.Span(f"{pnl_pct:+.2f}%", style={"color": pnl_color}),
            _build_risk_html(),
        )

    @callback(
        Output("execution-feed-table", "data"),
        Input("hot-refresh", "n_intervals"),
    )
    def refresh_execution_feed(n):
        return _store.get_execution_feed()

    @callback(
        Output("consensus-log-table", "data"),
        Input("hot-refresh", "n_intervals"),
    )
    def refresh_consensus_log(n):
        return _store.get_consensus_log()

    @callback(
        Output("agent-votes-table", "data"),
        Input("hot-refresh", "n_intervals"),
    )
    def refresh_agent_votes(n):
        return _store.get_agent_vote_rows()

    @callback(
        Output("heartbeat-table", "data"),
        Input("hot-refresh", "n_intervals"),
    )
    def refresh_heartbeat(n):
        try:
            from src.db_models import BotHeartbeat
            latest = BotHeartbeat.latest_per_agent(hours=1)
            rows = []
            for agent_id, beat in latest.items():
                if beat.success:
                    if beat.latency_ms < 2000:
                        status = "🟢 OK"
                    elif beat.latency_ms < 10000:
                        status = "🟡 SLOW"
                    else:
                        status = "🔴 LATENT"
                else:
                    status = "🔴 DOWN"
                rows.append({
                    "agent_id": agent_id,
                    "model": beat.model,
                    "latency_ms": f"{beat.latency_ms:.0f}ms" if beat.success else "—",
                    "status": status,
                    "timestamp": beat.timestamp.strftime("%H:%M:%S"),
                })
            return rows
        except Exception:
            return []


# ═══════════════════════════════════════════════════════════════════════════════
# COLD PATH CALLBACKS
# ═══════════════════════════════════════════════════════════════════════════════

def _register_cold_callbacks(app: dash.Dash):

    @callback(
        Output("backtest-results-table", "data"),
        Output("strategy-comparison-chart", "figure"),
        Input("bt-run-btn", "n_clicks"),
        State("bt-symbol", "value"),
        State("bt-start", "value"),
        State("bt-end", "value"),
        State("bt-strategies", "value"),
    )
    def run_backtest(n_clicks, symbol, start, end, strategies):
        if n_clicks == 0:
            return [], _empty_chart()

        try:
            from backtest.engine import BacktestEngine

            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            output_dir = os.path.join(project_root, "data", "backtests")
            engine = BacktestEngine(output_dir=output_dir)
            results = engine.run(
                symbol=symbol or "BTC/USDT",
                strategies=strategies or ["momentum"],
                start_date=start or "2024-01-01",
                end_date=end or "2024-12-31",
            )
            rows = [
                {
                    "strategy": r.get("strategy_name", ""),
                    "asset": r.get("asset", ""),
                    "sharpe": f"{r.get('sharpe_ratio', 0):.2f}",
                    "sortino": f"{r.get('sortino_ratio', 0):.2f}",
                    "max_dd": f"{r.get('max_drawdown', 0):.2%}",
                    "win_rate": f"{r.get('win_rate', 0):.1%}",
                    "trade_count": r.get("trade_count", 0),
                }
                for r in results
            ]
            fig = _make_comparison_chart(rows)
            return rows, fig
        except Exception as e:
            log.error("Backtest error: %s", e)
            return [], _empty_chart()

    @callback(
        Output("performance-reports-list", "children"),
        Output("report-status", "children"),
        Input("gen-report-btn", "n_clicks"),
        State("bt-strategies", "value"),
    )
    def generate_report(n_clicks, strategies):
        if n_clicks == 0:
            return [], ""

        try:
            from backtest.report import generate_narrative, PerformanceMetrics

            reports = []
            for strategy in (strategies or ["momentum"]):
                metrics = PerformanceMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
                narrative = generate_narrative(metrics, strategy)
                reports.append(
                    html.Div(
                        style={"marginBottom": "8px", "color": "var(--text-secondary)", "fontSize": "13px"},
                        children=[
                            html.Strong(f"{strategy}: ", style={"color": "var(--text-primary)"}),
                            narrative,
                        ],
                    )
                )
            return reports, html.Span(
                "✅ Reports generated",
                style={"color": "var(--accent-green)", "fontSize": "13px"},
            )
        except Exception as e:
            log.error("Report generation error: %s", e)
            return [], html.Span(
                f"❌ Error: {e}",
                style={"color": "var(--accent-red)", "fontSize": "13px"},
            )


# ═══════════════════════════════════════════════════════════════════════════════
# Chart helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _empty_chart() -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#4B5C72",
        font_family="Inter",
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
        margin=dict(l=0, r=0, t=0, b=0),
        height=200,
        annotations=[{
            "text": "Run a backtest to see results",
            "xref": "paper", "yref": "paper",
            "x": 0.5, "y": 0.5,
            "showarrow": False,
            "font": {"size": 14, "color": "#4B5C72"},
        }],
    )
    return fig


def _make_comparison_chart(rows: list[dict]) -> go.Figure:
    if not rows:
        return _empty_chart()

    strategies = [r["strategy"] for r in rows]
    sharpes = [float(r["sharpe"]) for r in rows]

    colors = ["#00C2FF", "#00E68A", "#FFB020", "#8B5CF6", "#FF3B5C"]

    fig = go.Figure(data=[
        go.Bar(
            name="Sharpe Ratio",
            x=strategies,
            y=sharpes,
            marker_color=[colors[i % len(colors)] for i in range(len(strategies))],
            marker_line_color="rgba(255,255,255,0.1)",
            marker_line_width=1,
            opacity=0.9,
        )
    ])
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#E8EDF5",
        font_family="Inter",
        title=None,
        xaxis=dict(
            showgrid=False, zeroline=False,
            tickfont=dict(size=12, color="#7B8DA4"),
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(100, 130, 180, 0.08)",
            zeroline=False,
            tickfont=dict(size=11, color="#7B8DA4"),
        ),
        margin=dict(l=40, r=20, t=20, b=40),
        height=300,
        bargap=0.4,
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# Standalone run
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app = create_dashboard(
        paper_trader=None,
        portfolio=Portfolio(cash=10_000.0),
        port=8050,
    )
    app.run(host="0.0.0.0", port=8050, debug=False)
