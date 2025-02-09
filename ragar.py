import json
import typer
from pathlib import Path

from rich import print
from rich.prompt import FloatPrompt, Prompt, Confirm
from rich.panel import Panel
from rich.console import Console, Group

# Initialize Typer app
app = typer.Typer(help="RAG Chat CLI Application (ragar)", no_args_is_help=True)
console = Console()

# Default config file name
DEFAULT_CONFIG_FILE = "ragar.config"


def load_config(config_path: Path) -> dict:
    if config_path.exists():
        with open(config_path, "r") as f:
            return json.load(f)
    else:
        return {}


def save_config(config: dict, config_path: Path):
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)


@app.command()
def chat(
    path: str = typer.Option(
        DEFAULT_CONFIG_FILE, "--path", "-p", help="Path to configuration file"
    ),
    clear: bool = typer.Option(
        False, "--clear", "-c", help="Clears output with each message"
    ),
):
    """
    Run the RAG chat application.
    Loads configuration and then starts chat loop.
    """
    config_path = Path(path)
    config = load_config(config_path)
    if (clear):
        console.clear()

    if not config:
        print("[bold red]Configuration file not found or empty. Please run configuration commands first.[/bold red]")
        raise typer.Exit()
    
    # Here you could load your agent based on config (langchain/ollama integration) 
    # and start an interactive chat loop.
    print(f"[green]Starting chat loop with configuration from {config_path}[/green]")
    chat_loop(config, clear)


def chat_loop(config: dict, clear=False):
    console.print("[bold blue]Entering chat mode and loading configuration. Type 'exit' to quit.[/bold blue]")
    from bot import chat_prompt

    if (clear):
        console.clear()


    while True:
        user_input = Prompt.ask("[red]Type 'exit' to quit.[/red] [bold yellow]You[/bold yellow]")
            
        if user_input.lower() in ["exit", "quit"]:
            print("[bold green]Exiting chat. Goodbye![/bold green]")
            break


        panel_group = Group(
            Panel(user_input, title="[bold yellow]You[/bold yellow]"),
            Panel("I am thinking...", title="[bold green]Agent:[/bold green]")
        )
        
        if (clear):
            console.clear()
        console.print(panel_group)

        (response, tokens) = chat_prompt(user_input)

        panel_group = Group(
            Panel(user_input, title="[bold yellow]You[/bold yellow]"),
            Panel(response, title="[bold green]Agent:[/bold green]", subtitle=f"Tokens used: {tokens}")
        )

        if (clear):
            console.clear()
        console.print(panel_group)




@app.command()
def model(
    path: str = typer.Option(
        DEFAULT_CONFIG_FILE, "--path", "-p", help="Path to configuration file"
    ),
    model: str = typer.Option(
        None, "--model", "-m", help="AI language model name. If not provided, the user will be prompted to select a model. IN PROGRESS"
    ),
    temperature: float = typer.Option(
        0, "--temperature", "-t", help="Temperature for the model, between 0 and 1. Lower values make the model more deterministic and consistent, while higher values make the model more creative and random. IN PROGRESS"
    ),
):
    """
    Configure the model selection.
    """
    
    config_path = Path(path)
    config = load_config(config_path)
    
    # Phase 1: Model Selection
    models = ["openai", "anthropic", "groq", "mistral"]
    print("[bold]Model Selection Phase[/bold]")
    print("Available models: " + ", ".join(models))
    selected_model = Prompt.ask("Select a model", choices=models, default="openai")
    api_key = Prompt.ask("Enter API key", password=True)
    temperature_value = FloatPrompt.ask("Enter temperature", default="0")
    
    config["model"] = {
        "name": selected_model,
        "api_key": api_key,
        "temperature": float(temperature_value)
    }
    
    save_config(config, config_path)
    print(f"[green]Model configuration saved to {config_path}.[/green]")


@app.command()
def memory(
    path: str = typer.Option(
        DEFAULT_CONFIG_FILE, "--path", "-p", help="Path to configuration file"
    ),
    preserve_history: bool = typer.Option(
        False, "--preserve", "-k", help="Should conversation history be preserved. IN PROGRESS"
    ),
    reduction_method: str = typer.Option(
        "truncation", "--reduction", "-r", help="Memory reduction method. IN PROGRESS"
    )
):
    """
    Configure conversation memory settings.
    """
    config_path = Path(path)
    config = load_config(config_path)
    
    print("[bold]Memory Configuration Phase[/bold]")
    preserve_history = Confirm.ask("Should conversation history be preserved?")
    memory_config = {"enabled": preserve_history}
    if preserve_history:
        reduction_method = Prompt.ask("Select memory reduction method", choices=["truncation", "summarization"], default="truncation")
        memory_config["reduction_method"] = reduction_method
    
    config["memory"] = memory_config
    save_config(config, config_path)
    print(f"[green]Memory configuration saved to {config_path}.[/green]")


@app.command()
def rag(
    path: str = typer.Option(
        DEFAULT_CONFIG_FILE, "--path", "-p", help="Path to configuration file"
    ),
    use_context: bool = typer.Option(
        False, "--context", "-c", help="Whether you want to use context. IN PROGRESS"
    ), 
    file_path: str = typer.Option(
        None, "--file", "-f", help="Path to file with context. IN PROGRESS"
    ), 
    tokens_chunk_size: int = typer.Option(
        200, "--file", "-f", help="Chunk size of a single document from file. IN PROGRESS"
    ), 
    overlap: int = typer.Option(
        50, "--file", "-f", help="Tokens overlap for each document. IN PROGRESS"
    ),  
    embedding: str = typer.Option(
        None, "--file", "-f", help="What embeeding model you want to use for processing embeedings in chroma database. IN PROGRESS"
    ),  
    retrieval_k: int = typer.Option(
        3, "--file", "-f", help="Number of retrieval documents to be used in context. IN PROGRESS"
    ), 
):
    """
    Configure context settings for RAG.
    """
    config_path = Path(path)
    config = load_config(config_path)
    
    print("[bold]Context (RAG) Configuration Phase[/bold]")
    use_context = Confirm.ask("Do you want to use context?")
    context_config = {"enabled": use_context}
    
    if use_context:
        file_path = Prompt.ask("Enter the context file path")
        tokens_chunk_size = Prompt.ask("Enter token chunk size", default="512")
        overlap = Prompt.ask("Enter overlap for token chunks", default="50")
        embedding = Prompt.ask("Select embedding algorithm", default=config.get("model", {}).get("name", "openai"))
        retrieval_k = Prompt.ask("Enter the K value for context retrieval", default="5")
        
        context_config.update({
            "file": file_path,
            "tokens_chunk_size": int(tokens_chunk_size),
            "overlap": int(overlap),
            "embedding": embedding,
            "retrieval_k": int(retrieval_k),
        })
    
    config["context"] = context_config
    save_config(config, config_path)
    print(f"[green]Context configuration saved to {config_path}.[/green]")


@app.command()
def debug(
    path: str = typer.Option(
        DEFAULT_CONFIG_FILE, "--path", "-p", help="Path to configuration file"
    ),
    debug: bool = typer.Option(
        False, "--debug", "-d", help="Debugging the prompts in langsmith. IN PROGRESS"
    ),
    project_name: bool = typer.Option(
        "ragar", "--name", "-n", help="Langsmith project name. IN PROGRESS"
    ),
    
):
    """
    Configure debugging settings.
    """
    config_path = Path(path)
    config = load_config(config_path)
    
    print("[bold]Debugging Configuration Phase[/bold]")
    debug_enabled = Confirm.ask("Do you want to enable debugging with Langsmith (or similar)?")
    debug_config = {"enabled": debug_enabled}
    if debug_enabled:
        debug_api_key = Prompt.ask("Enter debug API key", password=True)
        project_name = Prompt.ask("Enter Langsmith project name", default="default")
        debug_config.update({
            "api_key": debug_api_key,
            "project_name": project_name
        })
    
    config["debug"] = debug_config
    save_config(config, config_path)
    print(f"[green]Debug configuration saved to {config_path}.[/green]")


if __name__ == "__main__":
    app()