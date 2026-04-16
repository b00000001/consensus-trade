"""
Phase 7 — dashboard/app.py
Dash app with hot_path (live P&L, risk, execution feed) and
cold_path (backtest runner, analytics). Strictly separated.
"""
import logging
from datetime import datetime
from typing import Optional

import dash
from dash import dcc, html, dash_table, callback, ctx
from dash.dependencies import Input, Output, State
import plotly.express as px
import plotly.graph_objects as go

from dashboard.config_panel import render_config_panel, register_config_callbacks

from execution.portfolio import Portfolio
from execution.paper_trader import PaperTrader


log = logging.getLogger(__name__)

# Global state (updated by orchestrator callbacks)
_global_paper_trader: Optional[PaperTrader] = None
_global_portfolio: Optional[Portfolio] = None
_execution_feed: list[dict] = []   # recent execution events
_agent_votes: dict = {}             # asset -> {agent_id: signal}
_agent_models: dict = {}           # agent_id -> model name
_consensus_log: list[dict] = []     # consensus decisions
_risk_status: dict = {
    "circuit_breaker_tripped": False,
    "daily_loss_pct": 0.0,
    "trades_this_hour": 0,
    "consecutive_losses": 0,
}


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_dashboard(
    paper_trader: PaperTrader,
    portfolio: Portfolio,
    port: int = 8050,
) -> dash.Dash:
    global _global_paper_trader, _global_portfolio
    _global_paper_trader = paper_trader
    _global_portfolio = portfolio

    app = dash.Dash(
        __name__,
        external_stylesheets=[
            "https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;700&family=Outfit:wght@300;600;800&display=swap"
        ],
        meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}]
    )
    app.title = "ConsensusTrade | CEO Command Center"

    # ---- Hot path layout ----
    app.layout = html.Div([
        # --- COMMAND HUD ---
        html.Header(id="command-hud", children=[
            html.Div(className="hud-logo", children=[
                html.Span("CONSENSUS"), "TRADE"
            ]),
            html.Div(className="hud-metrics", children=[
                html.Div(className="hud-item", children=[
                    html.Div("Firm Equity", className="hud-label"),
                    html.Div(id="live-equity", className="hud-value")
                ]),
                html.Div(className="hud-item", children=[
                    html.Div("Session P&L", className="hud-label"),
                    html.Div(id="live-pnl", className="hud-value")
                ]),
                html.Div(className="hud-item", children=[
                    html.Div("Risk Delta", className="hud-label"),
                    html.Div(id="live-pnl-pct", className="hud-value")
                ]),
            ])
        ]),

        dcc.Tabs(className="dash-tabs", children=[
            # ---- HOT PATH TAB ----
            dcc.Tab(label="COMMAND CENTER", className="dash-tab", selected_className="dash-tab--selected", children=[
                html.Div([
                    html.Div(className="row", style={"display": "flex", "gap": "20px"}, children=[
                        # Left Column: Risk and Feed
                        html.Div(style={"flex": "1"}, children=[
                            html.Div(className="ceo-card", children=[
                                html.Div("Operational Risk Monitor", className="card-title"),
                                html.Div(id="risk-monitor", style={"display": "flex", "flexDirection": "column", "gap": "10px"})
                            ]),
                            html.Div(className="ceo-card", children=[
                                html.Div("Direct Execution Feed", className="card-title"),
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
                                    page_size=10,
                                ),
                            ]),
                        ]),

                        # Right Column: Consensus and Votes
                        html.Div(style={"flex": "2"}, children=[
                            html.Div(className="ceo-card", children=[
                                html.Div("AI Consensus Log", className="card-title"),
                                dash_table.DataTable(
                                    id="consensus-log-table",
                                    columns=[
                                        {"name": "Time", "id": "time"},
                                        {"name": "Asset", "id": "asset"},
                                        {"name": "Decision", "id": "decision"},
                                        {"name": "Conf.", "id": "confidence"},
                                        {"name": "Logic Summary", "id": "reason"},
                                        {"name": "Agents / Models", "id": "agents"},
                                    ],
                                    data=[],
                                    style_as_list_view=True,
                                    page_size=12,
                                ),
                            ]),
                            html.Div(className="ceo-card", children=[
                                html.Div("Agent Distribution", className="card-title"),
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
                                ),
                            ]),
                            html.Div(className="ceo-card", children=[
                                html.Div("Bot Heartbeat Status", className="card-title"),
                                dash_table.DataTable(
                                    id="heartbeat-table",
                                    columns=[
                                        {"name": "Agent", "id": "agent_id"},
                                        {"name": "Model", "id": "model"},
                                        {"name": "Latency (ms)", "id": "latency_ms"},
                                        {"name": "Status", "id": "status"},
                                        {"name": "Last Check", "id": "timestamp"},
                                    ],
                                    data=[],
                                    style_as_list_view=True,
                                ),
                            ]),
                        ]),
                    ]),
                ], style={"padding": "30px"}),

                # Hot path refresh interval
                dcc.Interval(id="hot-refresh", interval=3000, n_intervals=0),
            ]),

            # ---- COLD PATH TAB ----
            dcc.Tab(label="STRATEGY AUDIT", className="dash-tab", selected_className="dash-tab--selected", children=[
                html.Div([
                    html.Div(className="ceo-card", children=[
                        html.Div("Backtest Engine v2.1", className="card-title"),
                        html.Div([
                            dcc.Input(id="bt-symbol", type="text", placeholder="BTC/USDT", value="BTC/USDT", style={"background": "#1C242F", "border": "1px solid var(--border)", "color": "white", "padding": "10px", "borderRadius": "4px"}),
                            dcc.Input(id="bt-start", type="text", placeholder="2024-01-01", style={"background": "#1C242F", "border": "1px solid var(--border)", "color": "white", "padding": "10px", "borderRadius": "4px"}),
                            dcc.Input(id="bt-end", type="text", placeholder="2024-12-31", style={"background": "#1C242F", "border": "1px solid var(--border)", "color": "white", "padding": "10px", "borderRadius": "4px"}),
                            dcc.Dropdown(
                                id="bt-strategies",
                                options=[
                                    {"label": "Momentum", "value": "momentum"},
                                    {"label": "Mean Reversion", "value": "mean_reversion"},
                                    {"label": "Breakout", "value": "breakout"},
                                ],
                                multi=True,
                                value=["momentum"],
                                style={"width": "300px"}
                            ),
                            html.Button("INITIATE TEST", id="bt-run-btn", n_clicks=0, style={"background": "var(--accent-primary)", "color": "var(--bg-base)", "border": "none", "padding": "10px 20px", "borderRadius": "4px", "fontWeight": "800", "cursor": "pointer"}),
                        ], style={"display": "flex", "gap": "15px", "alignItems": "center"}),
                    ]),

                    html.Div(className="ceo-card", children=[
                        html.Div("Historical Attribution", className="card-title"),
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
                        ),
                    ]),

                    html.Div(className="ceo-card", children=[
                        html.Div("Performance Attribution Chart", className="card-title"),
                        dcc.Graph(id="strategy-comparison-chart"),
                    ]),
                    html.Div(className="ceo-card", children=[
                        html.Div("Phase 9: Performance Reports", className="card-title"),
                        html.Div(id="performance-reports-list"),
                        html.Button("Generate Narrative", id="gen-report-btn", n_clicks=0, className="bt-run-btn"),
                        html.Div(id="report-status"),
                    ]),
                ], style={"padding": "30px"}),
            ]),

            # ---- CONFIG TAB ----
            dcc.Tab(label="⚙ CONFIG", className="dash-tab", selected_className="dash-tab--selected", children=[
                html.Div(id="config-tab-content", style={"padding": "30px"}),
            ]),
        ]),

        # Store for cross-callback data
        dcc.Store(id="hot-data-store", data={}),
    ])

    # =========================================================================
    # HOT PATH CALLBACKS
    # =========================================================================

    @callback(
        Output("live-equity", "children"),
        Output("live-pnl", "children"),
        Output("live-pnl-pct", "children"),
        Output("risk-monitor", "children"),
        Input("hot-refresh", "n_intervals"),
    )
    def update_hot_equity(n):
        if _global_portfolio is None:
            return "—", "—", "—", "—"
        equity = _global_portfolio.total_equity()
        pnl = _global_portfolio.total_pnl()
        pnl_pct = _global_portfolio.pnl_pct() * 100
        return (
            f"Equity: ${equity:,.2f}",
            f"P&L: ${pnl:,.2f}",
            f"P&L%: {pnl_pct:.2f}%",
            _build_risk_html(),
        )

    @callback(
        Output("execution-feed-table", "data"),
        Input("hot-refresh", "n_intervals"),
    )
    def update_execution_feed(n):
        return _execution_feed[-50:]  # last 50

    @callback(
        Output("consensus-log-table", "data"),
        Input("hot-refresh", "n_intervals"),
    )
    def update_consensus_log(n):
        rows = _consensus_log[-50:]
        # Attach model names to agent column in each row
        for row in rows:
            row["agents"] = row.get("agents", "—")
        return rows

    @callback(
        Output("agent-votes-table", "data"),
        Input("hot-refresh", "n_intervals"),
    )
    def update_agent_votes(n):
        rows = []
        for asset, votes in _agent_votes.items():
            for agent_id, signal in votes.items():
                model = _agent_models.get(agent_id, "?")
                rows.append({"asset": asset, "agent": agent_id, "signal": signal, "model": model})
        return rows

    @callback(
        Output("heartbeat-table", "data"),
        Input("hot-refresh", "n_intervals"),
    )
    def update_heartbeat(n):
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

    # =========================================================================
    # COLD PATH CALLBACKS
    # =========================================================================

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
        if n_clicks == 0 or _global_paper_trader is None:
            return [], go.Figure()

        # Import backtest lazily to keep hot path fast
        from backtest.engine import BacktestEngine

        engine = BacktestEngine(output_dir="/opt/consensus-trade/data/backtests")
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
                "calmar": f"{r.get('calmar_ratio', 0):.2f}",
                "max_dd": f"{r.get('max_drawdown', 0):.2%}",
                "win_rate": f"{r.get('win_rate', 0):.1%}",
                "trade_count": r.get("trade_count", 0),
            }
            for r in results
        ]

        fig = _make_comparison_chart(rows)
        return rows, fig

    @callback(
        Output("performance-reports-list", "children"),
        Output("report-status", "children"),
        Input("gen-report-btn", "n_clicks"),
        State("bt-strategies", "value"),
    )
    def generate_report(n_clicks, strategies):
        if n_clicks == 0:
            return [], ""

        from backtest.report import compute_metrics, generate_narrative

        # Compute metrics from most recent backtest data
        # For now show narrative for configured strategies
        reports = []
        for strategy in (strategies or ["momentum"]):
            # Placeholder — real implementation reads from backtest results DB
            # Using zero metrics as placeholder until backtest data is populated
            from backtest.report import PerformanceMetrics
            metrics = PerformanceMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0)
            narrative = generate_narrative(metrics, strategy)
            reports.append(html.Div(f"**{strategy}**: {narrative}", style={"marginBottom": "8px"}))
        return reports, "✅ Reports generated"

    # Inject config panel into the CONFIG tab
    from dash import html
    config_tab_content = app.layout.children[2].children[4]  # dcc.Tabs > children[2] = CONFIG tab children
    if hasattr(config_tab_content, 'id') and config_tab_content.id == "config-tab-content":
        config_tab_content.children = render_config_panel(app)

    register_config_callbacks(app)

    return app


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_risk_html() -> list:
    is_tripped = _risk_status.get("circuit_breaker_tripped", False)
    return [
        html.Div(children=[
            html.Span("Circuit Breaker: ", style={"color": "var(--text-secondary)"}),
            html.Span("ACTUATED ⚠️" if is_tripped else "OFFLINE ✅", className="risk-tripped" if is_tripped else ""),
        ], style={"fontSize": "1.1rem", "fontWeight": "700"}),
        html.Div([
            html.Span("Daily Performance Limit: ", style={"color": "var(--text-secondary)"}),
            html.Span(f"{_risk_status.get('daily_loss_pct', 0):.2%}", style={"color": "var(--accent-primary)"}),
        ]),
        html.Div([
            html.Span("Tactical Velocity (Trades/hr): ", style={"color": "var(--text-secondary)"}),
            html.Span(f"{_risk_status.get('trades_this_hour', 0)}", style={"color": "var(--text-primary)"}),
        ]),
        html.Div([
            html.Span("Loss Sequence Threshold: ", style={"color": "var(--text-secondary)"}),
            html.Span(f"{_risk_status.get('consecutive_losses', 0)}", style={"color": "var(--text-primary)"}),
        ]),
    ]


def _make_comparison_chart(rows: list[dict]) -> go.Figure:
    if not rows:
        return go.Figure()
    strategies = [r["strategy"] for r in rows]
    sharpes = [float(r["sharpe"]) for r in rows]
    
    fig = go.Figure(data=[
        go.Bar(
            name="Sharpe Ratio", 
            x=strategies, 
            y=sharpes,
            marker_color="#00D2FF",
            marker_line_color="#F8FAFC",
            marker_line_width=1,
            opacity=0.8
        )
    ])
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color="#F8FAFC",
        font_family="Outfit",
        title={"text": "STRATEGY SHARPE ATTRIBUTION", "font": {"size": 24}},
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=True, gridcolor="rgba(148, 163, 184, 0.1)", zeroline=False),
        margin=dict(l=40, r=40, t=80, b=40),
    )
    return fig


# ---------------------------------------------------------------------------
# Update helpers (called by orchestrator)
# ---------------------------------------------------------------------------

def update_execution_feed_event(event: dict):
    _execution_feed.append({
        "time": datetime.now().strftime("%H:%M:%S"),
        "trade_id": event.get("trade_id", ""),
        "asset": event.get("asset", ""),
        "signal": event.get("signal", ""),
        "quantity": event.get("quantity", ""),
        "price": event.get("price", ""),
        "status": event.get("status", ""),
    })


def update_consensus_log_event(decision: dict, agent_models: dict = None):
    _consensus_log.append({
        "time": datetime.now().strftime("%H:%M:%S"),
        "asset": decision.get("asset", ""),
        "decision": decision.get("signal", ""),
        "confidence": f"{decision.get('confidence', 0):.2f}",
        "reason": decision.get("reason", ""),
        "escalated": "✅" if decision.get("escalated") else "—",
        "agents": ", ".join(f"{aid} ({agent_models.get(aid, '?')})" for aid in decision.get("agent_ids", [])) or "—",
    })


def update_agent_votes(asset: str, votes: dict, agent_models: dict = None):
    _agent_votes[asset] = votes
    _agent_models = agent_models or {}


def update_risk_status(status: dict):
    _risk_status.update(status)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app = create_dashboard(
        paper_trader=None,
        portfolio=Portfolio(cash=10_000.0),
        port=8050,
    )
    app.run(host="0.0.0.0", port=8050, debug=False)
