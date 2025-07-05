import asyncio
import logging
import sys
import os
import time
from datetime import datetime

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from typing import List, Optional, Dict, Any
import iroh

from node import Node
from llm_service import LLMService, LLMMessageType
from system_info import collect_and_store_system_info

console = Console()
app = typer.Typer()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Global state
node: Optional[Node] = None
llm_service: Optional[LLMService] = None
llm_nodes: Dict[str, Any] = {}
main_doc_id: Optional[str] = None


async def message_handler(message: dict):
    """Handles incoming messages from the network."""
    msg_type = message.get("type")
    sender_id = message.get("sender_id")
    payload = message.get("payload", {})

    if msg_type == "text_message":
        console.print(
            f"\n[bold cyan][{datetime.fromtimestamp(message.get('timestamp', time.time())).strftime('%H:%M:%S')}] "
            f"[yellow]{sender_id}:[/yellow] {payload.get('content')}[/bold cyan]"
        )
    elif msg_type == "llm_service_info":
        llm_nodes[sender_id] = payload
        console.print(f"[magenta]LLM service discovered from {sender_id}[/magenta]")
    elif msg_type == "llm_message":
        llm_payload = payload
        llm_type = llm_payload.get("llm_type")
        if llm_type == LLMMessageType.RESPONSE.value:
            console.print(f"\n[green]LLM Response from {sender_id}:[/green] {llm_payload.get('response')}")
        elif llm_type == LLMMessageType.STATUS.value:
            console.print(f"[yellow]LLM Status from {sender_id}: {llm_payload.get('message', llm_payload.get('status'))}[/yellow]")
        
        # Let the LLM service also handle it if it's a query for us
        if llm_service and llm_service.is_running:
            await llm_service.handle_llm_message(message)


async def run_node(bootstrap_ticket: Optional[str] = None):
    global node, llm_service, main_doc_id
    
    iroh.iroh_ffi.uniffi_set_event_loop(asyncio.get_running_loop())

    node = Node()
    await node.start()

    node.register_message_handler(message_handler)
    
    if not bootstrap_ticket:
        ticket, doc_id = await node.create_document()
        if ticket:
            main_doc_id = doc_id
            console.print(f"[bold green]Created main document. Share this ticket:[/bold green] [yellow]{ticket}[/yellow]")
        else:
            console.print("[bold red]Failed to create main document.[/bold red]"); return
    else:
        doc_id = await node.join_document(bootstrap_ticket)
        if doc_id:
            main_doc_id = doc_id
            console.print(f"[green]Joined document with ID: {main_doc_id}[/green]")
        else:
            console.print("[red]Failed to join document.[/red]"); return

    llm_service = LLMService(node)

    console.print(f"[bold green]Node started with ID:[/bold green] [yellow]{await node.iroh_node.net().node_id()}[/yellow]")

    while True:
        display_command_menu()
        cmd = await asyncio.get_event_loop().run_in_executor(None, lambda: typer.prompt("\nEnter command", prompt_suffix=" > "))
        args = cmd.strip().split()

        if not args: continue
        if args[0] == "exit": break
        
        await handle_command(args)

    await node.stop()


def display_command_menu():
    """Display available commands."""
    commands = [
        ("text <message>", "Send a text message to the document"),
        ("peers", "List known peers (neighbors in the document swarm)"),
        ("status", "Show node and network status"),
        ("store <key> <value>", "Store a key-value pair in the document"),
        ("get <key>", "Retrieve a value from the document"),
        ("llm start [model_name]", "Start LLM service on this node"),
        ("llm services", "List known LLM services"),
        ("llm query <prompt>", "Broadcast a query to all LLM services"),
        ("exit", "Exit the program"),
    ]
    table = Table(title="Available Commands")
    table.add_column("Command", style="cyan")
    table.add_column("Description", style="green")
    for command, description in commands:
        table.add_row(command, description)
    console.print(Panel.fit(table, border_style="blue"))


async def handle_command(args: List[str]):
    """Handle user commands."""
    global node, llm_service, main_doc_id
    command = args[0]

    if command == "text":
        if len(args) < 2:
            console.print("[bold red]Usage: text <message>[/bold red]"); return
        message_text = " ".join(args[1:])
        message = {
            "type": "text_message",
            "sender_id": str(await node.iroh_node.net().node_id()),
            "payload": {"content": message_text},
            "timestamp": time.time()
        }
        if main_doc_id:
            success = await node.send_message(main_doc_id, message)
            if success: console.print("[green]Message sent.[/green]")
            else: console.print("[red]Failed to send message.[/red]")
        else:
            console.print("[red]Not part of any document to send messages.[/red]")

    elif command == "peers":
        if node and main_doc_id:
            if main_doc_id in node.neighbors and node.neighbors[main_doc_id]:
                table = Table(title=f"Peers in document {main_doc_id[:10]}...")
                table.add_column("Peer ID", style="cyan")
                for peer_id in node.neighbors[main_doc_id]:
                    table.add_row(str(peer_id))
                console.print(table)
            else:
                console.print("[yellow]No other peers detected in the document swarm.[/yellow]")
        else:
            console.print("[yellow]Not connected to any document.[/yellow]")

    elif command == "status":
        if node and node.iroh_node:
            node_id = await node.iroh_node.net().node_id()
            console.print(f"[bold]My Node ID:[/bold] [yellow]{node_id}[/yellow]")
            if main_doc_id:
                console.print(f"[bold]Main Document ID:[/bold] [yellow]{main_doc_id}[/yellow]")
        else:
            console.print("[yellow]Node not started.[/yellow]")

    elif command == "store":
        if len(args) < 3:
            console.print("[bold red]Usage: store <key> <value>[/bold red]"); return
        key, value = args[1], " ".join(args[2:])
        success = await node.store_value(key, value)
        if success: console.print(f"[green]Stored '{key}' in the document.[/green]")
        else: console.print(f"[red]Failed to store '{key}'.[/red]")

    elif command == "get":
        if len(args) < 2:
            console.print("[bold red]Usage: get <key>[/bold red]"); return
        key = args[1]
        value = await node.retrieve_value(key)
        if value is not None: console.print(f"[green]Value for '{key}': {value}[/green]")
        else: console.print(f"[yellow]No value found for '{key}'.[/yellow]")

    elif command == "llm":
        if len(args) < 2:
            console.print("[bold red]Usage: llm <start|services|query>[/bold red]"); return
        sub_command = args[1]
        if sub_command == "start":
            model_name = args[2] if len(args) > 2 else "HuggingFaceTB/SmolLM-135M"
            console.print(f"[cyan]Starting LLM service with model: {model_name}[/cyan]")
            success = await llm_service.start(model_name)
            if success: console.print("[green]LLM service started.[/green]")
            else: console.print("[red]Failed to start LLM service.[/red]")
        
        elif sub_command == "services":
            table = Table(title="LLM Services")
            table.add_column("Node ID", style="cyan")
            table.add_column("Model Name", style="green")
            for node_id, info in llm_nodes.items():
                table.add_row(node_id, info.get("model_name", "N/A"))
            console.print(table)

        elif sub_command == "query":
            if len(args) < 2:
                console.print("[bold red]Usage: llm query <prompt>[/bold red]"); return
            prompt = " ".join(args[2:])
            query_id = await llm_service.send_query(prompt)
            if query_id:
                console.print(f"[green]Query sent (ID: {query_id}), waiting for responses...[/green]")
            else:
                console.print("[red]Failed to send query.[/red]")

    else:
        console.print(f"[bold red]Unknown command: {command}[/bold red]")


@app.command()
def start(
    bootstrap_ticket: Optional[str] = typer.Option(None, help="Ticket of a document to join.")
):
    """Start the Hypercluster node."""
    console.print(
        Panel.fit(
            "[bold cyan]Hypercluster Node with Iroh[/bold cyan]\n"
            "[green]Starting node...[/green]",
            border_style="blue",
        )
    )
    asyncio.run(run_node(bootstrap_ticket))


if __name__ == "__main__":
    app()
