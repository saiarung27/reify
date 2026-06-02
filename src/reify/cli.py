"""``reify`` command-line entry point.

Subcommands
-----------
* ``reify init`` — create the data directory and emit a default config.
* ``reify serve`` — start the MCP server (stdio).
* ``reify remember <text>`` — ingest a memory.
* ``reify reflect`` — drain the Active Reconsolidation queue.
* ``reify recall <query>`` — quick retrieval against the local store.
* ``reify what-changed`` — list recent transitions.
* ``reify what-do-you-know`` — dump everything about an identity.
* ``reify forget`` — tombstone one memory or a filtered set.
* ``reify export`` — export the store to markdown.
* ``reify eval <suite>`` — run the user-value harness or benchmark suites.
* ``reify version`` — print the installed version.

The implementations of ``serve`` land in v0.1. ``remember``, ``recall``,
``what-changed``, ``what-do-you-know``, and ``export`` are wired to the
real engine in v0.1. ``init`` and ``version`` are functional today.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import typer
from rich.console import Console

from reify import EngineConfig, Reify
from reify.config import get_settings
from reify.storage import SQLiteStore
from reify.version import __version__

app = typer.Typer(
    name="reify",
    help="A model-portable cognitive substrate for AI.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()


def _engine_config(settings: object, *, llm_provider: object | None = None) -> EngineConfig:
    return EngineConfig(
        llm=llm_provider,
        default_ttl_days=settings.default_ttl_days,
        reinforcement_factor=settings.reinforcement_factor,
        decay_sweep_interval_seconds=settings.decay_sweep_interval_seconds,
    )


@app.command()
def version() -> None:
    """Print the installed Reify version."""
    console.print(f"reify {__version__}")


@app.command()
def init() -> None:
    """Create the data directory and emit a default configuration.

    Idempotent; safe to re-run. After this completes, the host's
    ``mcp.json`` should point at ``reify serve``.
    """
    settings = get_settings()
    settings.ensure_dirs()
    console.print(f"[green]Reify initialized[/green] at [bold]{settings.home}[/bold]")
    console.print(f"  database:  {settings.db_path}")
    console.print(f"  cache:     {settings.cache_dir}")
    console.print()
    console.print("Next: configure an LLM provider and run [bold]reify serve[/bold].")


@app.command()
def serve() -> None:
    """Start the MCP server over stdio.

    Exposes remember, recall, whatchanged, whatdoyouknow tools for MCP
    hosts (Claude Desktop, Cursor, Windsurf, etc.).
    """
    from reify.server.mcp import serve_stdio

    asyncio.run(serve_stdio())


@app.command()
def remember(
    text: str = typer.Argument(..., help="Text to remember."),
    identity: str = typer.Option(None, "--identity", "-i", help="Identity ID (optional)."),
    at: str = typer.Option(None, "--at", help="Timestamp for the memory (ISO-8601 format)."),
) -> None:
    """Ingest a memory into the store."""
    settings = get_settings()
    settings.ensure_dirs()

    from datetime import datetime, timezone

    async def _run() -> None:
        store = SQLiteStore(path=settings.db_path)
        
        # Create LLM provider if configured
        llm_provider = None
        try:
            from reify.llm import create_llm_provider
            llm_provider = create_llm_provider(settings.llm_reconsolidate)
        except Exception as e:
            # Graceful degradation - continue without LLM
            console.print(f"[yellow]Warning: LLM provider not configured: {e}[/yellow]")
        
        engine = Reify(storage=store, config=_engine_config(settings, llm_provider=llm_provider))
        async with engine:
            at_dt: datetime | None = None
            if at:
                at_dt = datetime.fromisoformat(at)
                if at_dt.tzinfo is None:
                    at_dt = at_dt.replace(tzinfo=timezone.utc)
            receipt = await engine.remember(
                text,
                source="cli:remember",
                identity_id=identity,
                at=at_dt,
            )
            console.print(f"[green]Committed {len(receipt.committed)} memory(ies)[/green]")
            for m in receipt.committed:
                console.print(f"  - {m.network}: {m.content[:60]}..." if len(m.content) > 60 else f"  - {m.network}: {m.content}")

    asyncio.run(_run())


@app.command()
def reflect() -> None:
    """Drain the Active Reconsolidation queue."""
    settings = get_settings()
    settings.ensure_dirs()

    async def _run() -> None:
        store = SQLiteStore(path=settings.db_path)
        llm_provider = None
        try:
            from reify.llm import create_llm_provider

            llm_provider = create_llm_provider(settings.llm_reconsolidate)
        except Exception as e:
            console.print(f"[yellow]Warning: LLM provider not configured: {e}[/yellow]")

        engine = Reify(storage=store, config=_engine_config(settings, llm_provider=llm_provider))
        async with engine:
            processed = await engine.flush_arw()
            console.print(f"[green]Processed {processed} queued memory(ies)[/green]")

    asyncio.run(_run())


@app.command()
def recall(
    query: str = typer.Argument(..., help="Natural-language query."),
    identity: str = typer.Option(None, "--identity", "-i", help="Scope to identity (optional)."),
    at: str = typer.Option(None, "--at", help="Query as of this timestamp (ISO-8601 format)."),
    limit: int = typer.Option(20, "--limit", "-n", help="Max results to return."),
) -> None:
    """Run a recall against the local store."""
    settings = get_settings()
    settings.ensure_dirs()

    from datetime import datetime, timezone

    async def _run() -> None:
        store = SQLiteStore(path=settings.db_path)
        engine = Reify(storage=store, config=_engine_config(settings))
        async with engine:
            at_dt: datetime | None = None
            if at:
                at_dt = datetime.fromisoformat(at)
                if at_dt.tzinfo is None:
                    at_dt = at_dt.replace(tzinfo=timezone.utc)
            response = await engine.recall(query, identity_id=identity, valid_at=at_dt, limit=limit)
            console.print(f"[bold]Query:[/bold] {response.query.text}")
            console.print(f"[bold]Results:[/bold] {len(response.results)} hit(s)\n")
            for r in response.results:
                m = r.memory
                console.print(f"[cyan]{m.network}[/cyan] (score: {r.score:.2f})")
                console.print(f"  {m.content}")
                if m.valid_until:
                    console.print(f"  [dim]Valid: {m.valid_at} → {m.valid_until}[/dim]")
                else:
                    console.print(f"  [dim]Valid since: {m.valid_at}[/dim]")
                console.print()

    asyncio.run(_run())


@app.command()
def what_changed(
    days: int = typer.Option(30, "--days", "-d", help="Lookback window in days."),
    identity: str = typer.Option(None, "--identity", "-i", help="Scope to identity (optional)."),
    at: str = typer.Option(None, "--at", help="Query as of this timestamp (ISO-8601 format)."),
    limit: int = typer.Option(20, "--limit", "-n", help="Max results to return."),
) -> None:
    """List recent transitions from the Bitemporal Transition Graph."""
    settings = get_settings()
    settings.ensure_dirs()

    from datetime import datetime, timezone

    async def _run() -> None:
        store = SQLiteStore(path=settings.db_path)
        engine = Reify(storage=store, config=_engine_config(settings))
        async with engine:
            at_dt: datetime | None = None
            if at:
                at_dt = datetime.fromisoformat(at)
                if at_dt.tzinfo is None:
                    at_dt = at_dt.replace(tzinfo=timezone.utc)
            transitions = await engine.whatchanged(
                days=days,
                identity_id=identity,
                at=at_dt,
                limit=limit,
            )
            console.print(f"[bold]Transitions[/bold] (last {days} days): {len(transitions)}\n")
            for t in transitions:
                console.print(f"[cyan]{t.kind}[/cyan]")
                console.print(f"  From: {t.from_memory_id}")
                console.print(f"  To:   {t.to_memory_id}")
                console.print(f"  At:   {t.at}")
                if t.summary:
                    console.print(f"  {t.summary}")
                console.print()

    asyncio.run(_run())


@app.command()
def what_do_you_know(
    about: str = typer.Argument(None, help="Optional query to scope results."),
    identity: str = typer.Option(None, "--identity", "-i", help="Scope to identity (optional)."),
    at: str = typer.Option(None, "--at", help="Query as of this timestamp (ISO-8601 format)."),
    limit: int = typer.Option(50, "--limit", "-n", help="Max results to return."),
) -> None:
    """Dump everything Reify currently believes, optionally scoped."""
    settings = get_settings()
    settings.ensure_dirs()

    from datetime import datetime, timezone

    async def _run() -> None:
        store = SQLiteStore(path=settings.db_path)
        engine = Reify(storage=store, config=_engine_config(settings))
        async with engine:
            at_dt: datetime | None = None
            if at:
                at_dt = datetime.fromisoformat(at)
                if at_dt.tzinfo is None:
                    at_dt = at_dt.replace(tzinfo=timezone.utc)
            memories = await engine.whatdoyouknow(
                about,
                identity_id=identity,
                valid_at=at_dt,
                limit=limit,
            )
            console.print(f"[bold]Knowledge[/bold]: {len(memories)} memory(ies)\n")
            for m in memories:
                console.print(f"[cyan]{m.network}[/cyan]")
                console.print(f"  {m.content}")
                if m.valid_until:
                    console.print(f"  [dim]Valid: {m.valid_at} → {m.valid_until}[/dim]")
                else:
                    console.print(f"  [dim]Valid since: {m.valid_at}[/dim]")
                console.print()

    asyncio.run(_run())


@app.command()
def export(
    path: Path = typer.Argument(..., help="Output markdown file path."),
    identity: str = typer.Option(None, "--identity", "-i", help="Scope to identity (optional)."),
) -> None:
    """Export the store to a human-readable markdown file."""
    settings = get_settings()
    settings.ensure_dirs()

    async def _run() -> None:
        store = SQLiteStore(path=settings.db_path)
        engine = Reify(storage=store, config=_engine_config(settings))
        async with engine:
            count = await engine.export_markdown(path, identity_id=identity)
            console.print(f"[green]Exported {count} memories to {path}[/green]")

    asyncio.run(_run())


@app.command()
def forget(
    memory_id: str | None = typer.Argument(None, help="Memory id to tombstone."),
    tag: str | None = typer.Option(None, "--tag", help="Forget all memories carrying this tag."),
    before: str | None = typer.Option(None, "--before", help="Forget memories created before this timestamp."),
    identity: str | None = typer.Option(None, "--identity", "-i", help="Scope bulk forget to an identity."),
) -> None:
    """Tombstone one memory or a filtered set."""
    settings = get_settings()
    settings.ensure_dirs()

    from datetime import datetime, timezone

    if memory_id is None and tag is None and before is None:
        raise typer.BadParameter("Provide a memory id, --tag, or --before.")

    async def _run() -> None:
        store = SQLiteStore(path=settings.db_path)
        engine = Reify(storage=store, config=_engine_config(settings))
        async with engine:
            before_dt: datetime | None = None
            if before:
                before_dt = datetime.fromisoformat(before)
                if before_dt.tzinfo is None:
                    before_dt = before_dt.replace(tzinfo=timezone.utc)
            deleted = await engine.forget(
                memory_id,
                tag=tag,
                before=before_dt,
                identity_id=identity,
            )
            console.print(f"[green]Forgot {deleted} memory(ies)[/green]")

    asyncio.run(_run())


@app.command(name="eval")
def evaluate(
    suite: str = typer.Argument(
        ...,
        help="Suite: user-value | longmemeval-fixture | longmemeval | multisession-synthesis | reify-native | locomo | beam",
    ),
    dataset: Path | None = typer.Option(
        None,
        "--dataset",
        help="Optional JSON or JSONL dataset path for benchmark suites.",
    ),
    answer_llm: str | None = typer.Option(
        None,
        "--answer-llm",
        help="Provider:model spec used to generate benchmark hypotheses.",
    ),
    judge_llm: str | None = typer.Option(
        None,
        "--judge-llm",
        help="Provider:model spec used to judge benchmark hypotheses.",
    ),
    scale: int = typer.Option(
        1,
        "--scale",
        help="Scale multiplier for BEAM dataset (repeats cases N times).",
    ),
) -> None:
    """Run the user-value harness or a benchmark suite against the local engine."""
    normalized = suite.strip().lower()
    if normalized == "user-value":
        from reify.eval import run_user_value_suite

        report = asyncio.run(run_user_value_suite())
        console.print("[bold]reify eval user-value[/bold]")
        console.print(f"cases: {report.total_cases}")
        console.print(f"useful recalls: {report.useful_recalls}/{report.total_cases}")
        console.print(f"recall-without-LLM rate: {report.recall_without_llm_rate:.1%}")
        console.print(f"30-day recall accuracy: {report.recall_accuracy_30d:.1%}")
        console.print(f"time-to-useful-recall p50: {report.time_to_useful_recall_p50_ms:.2f} ms")
        console.print(f"time-to-useful-recall p95: {report.time_to_useful_recall_p95_ms:.2f} ms")
        console.print(f"cost per useful recall: ${report.cost_per_useful_recall_usd:.6f}")
        console.print(f"LLM invocations during recall: {report.llm_invocations}")
        console.print(f"embed invocations during recall: {report.embed_invocations}")
        for case_result in report.case_results:
            status = "PASS" if case_result.useful else "FAIL"
            console.print(
                f"  {status} {case_result.case.name}: "
                f"{case_result.latency_ms:.2f} ms, "
                f"results={case_result.result_count}, "
                f"top_hit={case_result.top_hit!r}"
            )
        if not report.passed:
            raise typer.Exit(code=1)
        return

    if normalized == "longmemeval-fixture":
        from reify.eval import run_longmemeval_suite

        report = asyncio.run(run_longmemeval_suite(dataset_label="bundled fixture"))
        console.print("[bold]reify eval longmemeval-fixture[/bold]")
        console.print(f"dataset: {report.dataset_label}")
        console.print(f"cases: {len(report.case_results)}")
        console.print(f"overall accuracy: {report.overall_accuracy:.1%}")
        for category, accuracy in sorted(report.per_category_accuracy.items()):
            console.print(f"category {category}: {accuracy:.1%}")
        for case_result in report.case_results:
            status = "PASS" if case_result.correct else "FAIL"
            console.print(
                f"  {status} {case_result.case.name}: "
                f"results={case_result.result_count}, "
                f"top_hit={case_result.top_hit!r}"
            )
        if not report.passed_target:
            raise typer.Exit(code=1)
        return

    if normalized == "longmemeval":
        from reify.eval import run_longmemeval_benchmark
        from reify.llm import create_llm_provider

        if dataset is None:
            console.print(
                "[red]reify eval longmemeval[/red] requires [bold]--dataset[/bold]. "
                "Use [bold]reify eval longmemeval-fixture[/bold] for the bundled scaffold cases."
            )
            raise typer.Exit(code=2)
        settings = get_settings()
        answer_provider = None
        judge_provider = None
        answer_model = None
        judge_model = None

        answer_spec = answer_llm or settings.llm_reconsolidate
        judge_spec = judge_llm or settings.llm_reconsolidate
        try:
            answer_provider = create_llm_provider(answer_spec)
            answer_model = answer_spec.partition(":")[2]
        except Exception as exc:
            console.print(f"[yellow]Official answer generation disabled:[/yellow] {exc}")
        try:
            judge_provider = create_llm_provider(judge_spec)
            judge_model = judge_spec.partition(":")[2]
        except Exception as exc:
            console.print(f"[yellow]Official judging disabled:[/yellow] {exc}")

        report = asyncio.run(
            run_longmemeval_benchmark(
                dataset_path=dataset,
                answer_provider=answer_provider,
                answer_model=answer_model,
                judge_provider=judge_provider,
                judge_model=judge_model,
            )
        )
        console.print("[bold]reify eval longmemeval[/bold]")
        console.print(f"dataset: {dataset}")
        console.print("[bold]Reify Retrieval Proxy[/bold]")
        console.print(f"overall accuracy: {report.proxy.overall_accuracy:.1%}")
        for category, accuracy in sorted(report.proxy.per_category_accuracy.items()):
            console.print(f"category {category}: {accuracy:.1%}")
        for case_result in report.proxy.case_results:
            status = "PASS" if case_result.correct else "FAIL"
            console.print(
                f"  {status} {case_result.case.name}: "
                f"results={case_result.result_count}, "
                f"top_hit={case_result.top_hit!r}"
            )
        if report.official is None:
            console.print(
                "[yellow]Official LongMemEval QA score not run.[/yellow] "
                "Configure working `--answer-llm` and `--judge-llm` providers "
                "to mirror the official judge semantics."
            )
            return
        console.print("[bold]Official LongMemEval QA[/bold]")
        console.print(f"answer model: {report.official.answer_model}")
        console.print(f"judge model: {report.official.judge_model}")
        console.print(f"overall accuracy: {report.official.overall_accuracy:.1%}")
        console.print(f"task-averaged accuracy: {report.official.task_averaged_accuracy:.1%}")
        if report.official.abstention_accuracy is not None:
            console.print(f"abstention accuracy: {report.official.abstention_accuracy:.1%}")
        for question_type, accuracy in sorted(report.official.per_question_type_accuracy.items()):
            console.print(f"question type {question_type}: {accuracy:.1%}")
        if not report.proxy.passed_target or not report.official.passed_target:
            raise typer.Exit(code=1)
        return

    if normalized in {"multisession-synthesis", "reify-native"}:
        from reify.eval import run_multisession_synthesis_suite

        invoked_name = "reify-native" if normalized == "reify-native" else "multisession-synthesis"
        if dataset is None:
            console.print(
                f"[red]reify eval {invoked_name}[/red] requires [bold]--dataset[/bold]. "
                "Pass an official LongMemEval JSON file as the input corpus."
            )
            raise typer.Exit(code=2)
        report = asyncio.run(run_multisession_synthesis_suite(dataset_path=dataset))
        console.print(f"[bold]reify eval {invoked_name}[/bold]")
        console.print(f"dataset: {dataset}")
        console.print("[bold]Capability Track Breakdown[/bold]")
        console.print("scoring: current Reify retrieval proxy, grouped by product-aligned tracks")
        console.print(f"overall accuracy: {report.overall_accuracy:.1%}")
        for track, accuracy in sorted(report.per_track_accuracy.items()):
            console.print(f"track {track}: {accuracy:.1%}")
        for case_result in report.case_results:
            status = "PASS" if case_result.correct else "FAIL"
            console.print(
                f"  {status} {case_result.case.name} [{case_result.case.track}]: "
                f"results={case_result.result_count}, "
                f"top_hit={case_result.top_hit!r}"
            )
        if not report.passed_target:
            console.print(
                f"[yellow]target not met[/yellow]: overall and every track must be >= {report.track_target:.0%}"
            )
            raise typer.Exit(code=1)
        return

    if normalized == "locomo":
        from reify.eval import (
            load_locomo_cases,
            run_locomo_suite,
        )

        if dataset is None:
            cases = None
            label = "bundled fixture"
        else:
            cases = load_locomo_cases(dataset)
            label = str(dataset)
        report = asyncio.run(run_locomo_suite(cases=cases, dataset_label=label))
        console.print("[bold]reify eval locomo[/bold]")
        console.print(f"dataset: {report.dataset_label}")
        console.print(f"cases: {len(report.case_results)}")
        console.print(f"overall accuracy: {report.overall_accuracy:.1%}")
        console.print(f"target: {report.target:.1%}")
        for category, accuracy in sorted(report.per_category_accuracy.items()):
            console.print(f"category {category}: {accuracy:.1%}")
        for case_result in report.case_results:
            status = "PASS" if case_result.correct else "FAIL"
            console.print(
                f"  {status} {case_result.case.name} [{case_result.case.category}]: "
                f"results={case_result.result_count}, "
                f"top_hit={case_result.top_hit!r}"
            )
        if not report.passed_target:
            raise typer.Exit(code=1)
        return

    if normalized == "beam":
        from reify.eval.beam import default_beam_fixture_cases, run_beam_suite

        settings = get_settings()
        settings.ensure_dirs()

        async def _run_beam() -> BEAMReport:
            store = SQLiteStore(path=settings.db_path)
            engine = Reify(storage=store, config=_engine_config(settings))
            async with engine:
                return await run_beam_suite(
                    engine=engine,
                    cases=default_beam_fixture_cases() if dataset is None else None,
                    dataset_label=str(dataset) if dataset else "bundled fixture",
                    scale=scale,
                )

        report = asyncio.run(_run_beam())
        console.print("[bold]reify eval beam[/bold]")
        console.print(f"dataset: {report.dataset_label}")
        console.print(f"scale: {report.scale}")
        console.print(f"cases: {report.total_cases}")
        console.print(f"overall accuracy: {report.overall_accuracy:.1%}")
        console.print(f"target: {report.target:.1%}")
        console.print(f"p50 latency: {report.p50_latency_ms:.1f}ms")
        console.print(f"p95 latency: {report.p95_latency_ms:.1f}ms")
        console.print(f"p99 latency: {report.p99_latency_ms:.1f}ms")
        for case_result in report.case_results:
            status = "PASS" if case_result.correct else "FAIL"
            console.print(
                f"  {status} query={case_result.case.query!r}: "
                f"results={case_result.result_count}, "
                f"latency={case_result.latency_ms:.1f}ms, "
                f"top_hit={case_result.top_hit!r}"
            )
        if not report.passed_target:
            raise typer.Exit(code=1)
        return

    console.print(
        f"[yellow]reify eval[/yellow] does not implement suite {suite!r} yet. "
        "See ROADMAP.md (v0.3+)."
    )
    raise typer.Exit(code=2)


@app.command()
def serve_proxy(
    host: str = typer.Option("127.0.0.1", "--host", help="Host to bind to."),
    port: int = typer.Option(8000, "--port", help="Port to bind to."),
    upstream: str = typer.Option("openai:gpt-4o", "--upstream", help="Upstream LLM provider spec."),
    context_limit: int = typer.Option(5, "--context-limit", help="Max memories to inject as context."),
) -> None:
    """Start the OpenAI-compatible HTTP proxy."""
    import uvicorn

    from reify.server.openai_proxy import OpenAIProxy, OpenAIProxyConfig

    settings = get_settings()
    settings.ensure_dirs()

    config = OpenAIProxyConfig(
        upstream_spec=upstream,
        context_limit=context_limit,
    )

    store = SQLiteStore(path=settings.db_path)
    engine = Reify(storage=store, config=_engine_config(settings))
    proxy = OpenAIProxy(engine, config)

    from reify.server.security import assert_safe_bind
    assert_safe_bind(host)

    console.print(f"[green]Starting OpenAI proxy on http://{host}:{port}[/green]")
    console.print(f"Upstream: {upstream}")
    console.print(f"Context limit: {context_limit}")

    uvicorn.run(proxy.app, host=host, port=port)


@app.command()
def serve_rest(
    host: str = typer.Option("127.0.0.1", "--host", help="Host to bind to."),
    port: int = typer.Option(8001, "--port", help="Port to bind to."),
) -> None:
    """Start the REST API server."""
    import uvicorn

    from reify.server.rest_api import create_app

    from reify.server.security import assert_safe_bind
    assert_safe_bind(host)

    app = create_app()
    console.print(f"[green]Starting REST API on http://{host}:{port}[/green]")

    uvicorn.run(app, host=host, port=port)


@app.command()
def serve_mcp_sse(
    host: str = typer.Option("127.0.0.1", "--host", help="Host to bind to."),
    port: int = typer.Option(8002, "--port", help="Port to bind to."),
) -> None:
    """Start the MCP server with HTTP/SSE transport."""
    import uvicorn

    from reify.server.mcp_sse import create_sse_app

    from reify.server.security import assert_safe_bind
    assert_safe_bind(host)

    app = create_sse_app()
    console.print(f"[green]Starting MCP SSE server on http://{host}:{port}[/green]")
    console.print(f"SSE endpoint: http://{host}:{port}/sse")
    console.print(f"Messages endpoint: http://{host}:{port}/messages")

    uvicorn.run(app, host=host, port=port)


@app.command()
def serve_dashboard(
    host: str = typer.Option("127.0.0.1", "--host", help="Host to bind to."),
    port: int = typer.Option(3000, "--port", help="Port to bind to."),
) -> None:
    """Start the web dashboard."""
    import subprocess
    from pathlib import Path

    dashboard_path = Path(__file__).parent.parent.parent / "dashboard"
    if not dashboard_path.exists():
        console.print("[red]Dashboard not found. Please build the dashboard first.[/red]")
        raise typer.Exit(code=1)

    console.print(f"[green]Starting web dashboard on http://{host}:{port}[/green]")
    console.print(f"Dashboard path: {dashboard_path}")

    # Start the dashboard dev server
    subprocess.run(
        ["npm", "run", "dev", "--", "--host", host, "--port", str(port)],
        cwd=dashboard_path,
    )


@app.command()
def identity_link(
    identity_a: str = typer.Argument(..., help="First identity ID"),
    identity_b: str = typer.Argument(..., help="Second identity ID"),
    confidence: float = typer.Option(0.8, "--confidence", help="Bond confidence (0-1)"),
) -> None:
    """Link two identities with a bond."""
    settings = get_settings()
    settings.ensure_dirs()

    store = SQLiteStore(path=settings.db_path)
    engine = Reify(storage=store, config=_engine_config(settings))

    async def _link() -> None:
        async with engine:
            from reify.identity.bond_graph import IdentityBond

            bond = IdentityBond(
                from_identity=identity_a,
                to_identity=identity_b,
                confidence=confidence,
                rationale="explicit",
            )
            engine._identity_bond_graph.add_bond(bond)
            console.print(f"[green]Linked {identity_a} to {identity_b} with confidence {confidence}[/green]")

    asyncio.run(_link())


@app.command()
def identity_unlink(
    identity_a: str = typer.Argument(..., help="First identity ID"),
    identity_b: str = typer.Argument(..., help="Second identity ID"),
) -> None:
    """Unlink two identities."""
    settings = get_settings()
    settings.ensure_dirs()

    store = SQLiteStore(path=settings.db_path)
    engine = Reify(storage=store, config=_engine_config(settings))

    async def _unlink() -> None:
        async with engine:
            engine._identity_bond_graph.remove_bond(identity_a, identity_b)
            console.print(f"[green]Unlinked {identity_a} from {identity_b}[/green]")

    asyncio.run(_unlink())


@app.command()
def identity_list(
    identity_id: str = typer.Argument(..., help="Identity ID to list bonds for"),
) -> None:
    """List all bonds for an identity."""
    settings = get_settings()
    settings.ensure_dirs()

    store = SQLiteStore(path=settings.db_path)
    engine = Reify(storage=store, config=_engine_config(settings))

    async def _list() -> None:
        async with engine:
            bonds = engine._identity_bond_graph.get_bonds(identity_id)
            console.print(f"[bold]Bonds for {identity_id}:[/bold]")
            for bond in bonds:
                console.print(f"  - {bond.to_identity} (confidence: {bond.confidence}, rationale: {bond.rationale})")

    asyncio.run(_list())


def main() -> None:
    """Module-level entry point invoked by the ``reify`` console script."""
    try:
        app()
    except KeyboardInterrupt:  # pragma: no cover - signal path
        sys.exit(130)


if __name__ == "__main__":  # pragma: no cover - script path
    main()
