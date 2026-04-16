"""
Phase 1a — main.py
Entry point for ConsensusTrade.
"""
import argparse
import logging
import os
import sys

# Fix module search path — cd to script dir and add src to path
_BASE = os.path.dirname(os.path.abspath(__file__))
os.chdir(_BASE)
sys.path.insert(0, os.path.join(_BASE, "src"))

from orchestrator import TradingOrchestrator


def main():
    parser = argparse.ArgumentParser(description="ConsensusTrade")
    parser.add_argument("--poll-interval", type=int, default=300,
                        help="Trading cycle interval in seconds (default: 300)")
    parser.add_argument("--starting-cash", type=float, default=10_000.0,
                        help="Starting cash for paper trading (default: 10000)")
    parser.add_argument("--db-path", type=str,
                        default="data\live.db",
                        help="Path to live SQLite database")
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--dashboard-port", type=int, default=8050,
                        help="Dash dashboard port (default: 8050)")
    parser.add_argument("--dashboard-only", action="store_true",
                        help="Run dashboard only (no trading cycle)")
    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("trading.log"),
        ],
    )

    log = logging.getLogger(__name__)
    log.info("Starting ConsensusTrade...")

    orchestrator = TradingOrchestrator(
        db_path=args.db_path,
        poll_interval_seconds=args.poll_interval,
        starting_cash=args.starting_cash,
    )

    if args.dashboard_only:
        log.info("Dashboard mode — starting Dash app on port %d", args.dashboard_port)
        orchestrator.dash_app.run(host="0.0.0.0", port=args.dashboard_port, debug=False)
    else:
        orchestrator.start()
        log.info("Trading loop running. Press Ctrl+C to stop.")
        try:
            import time
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            log.info("Shutdown requested...")
            orchestrator.stop()


if __name__ == "__main__":
    main()

