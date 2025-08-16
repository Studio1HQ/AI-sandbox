import asyncio
import os
from typing import Tuple

# Load environment variables from .env file
from dotenv import load_dotenv
from e2b_code_interpreter import Sandbox
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from browser_agent import downloading_task_for_browser_agent
from sandbox_eda import SandboxEDA

load_dotenv()

console = Console()


def start_eda(
    model_for_eda: str,
    dataset_paths: list[str],
    dataset_file_names: list[str],
    api_key_for_sandbox_and_model: str,
    model_api_base_url: str,
    sandbox_domain: str,
    sandbox_template: str,
    sandbox_timeout: int,
):

    with Sandbox(
        template=sandbox_template,
        api_key=api_key_for_sandbox_and_model,
        domain=sandbox_domain,
        timeout=sandbox_timeout,
    ) as sandbox:

        sandbox_eda = SandboxEDA(
            sandbox, model_api_base_url, api_key_for_sandbox_and_model
        )

        console.print(
            f"[bold cyan]Started Sandbox[/bold cyan] (id: {sandbox.sandbox_id})"
        )

        sandbox_eda.upload_files_to_sandbox(dataset_paths, dataset_file_names)

        sandbox_eda.eda_chat(dataset_file_names, model_for_eda)

        console.print(
            f"\n\n[bold cyan]------ EDA Session Completed for Sandbox (id: {sandbox.sandbox_id}) ------[/]"
        )

    console.print(f"[bold cyan]----- Closed Sandbox (id: {sandbox.sandbox_id})-----[/]")


# MAIN MENU CHOICES
async def choice_download_dataset(
    api_key: str, model_api_base_url: str, model_for_browser_agent: str
) -> None | Tuple[str, list[str]]:

    console.print(
        Panel(
            "[bold green]1.[/bold green] Download default dataset (An-j96/SuperstoreData dataset from HuggingFace) via Browser Use\n"
            "[bold green]2.[/bold green] Give instruction for browser use to locate desired dataset, click download and STOP (to avoid infinite download check)\n"
            "[bold green]3.[/bold green] Back to main menu",
            title="Download Menu",
            border_style="white",
        )
    )

    choice = Prompt.ask(
        "\n[bold yellow]Enter your choice[/bold yellow]", choices=["1", "2", "3"]
    ).strip()

    if choice == "1":
        default_dataset_task = "Go to huggingface and search for An-j96/SuperstoreData then go to the files tab and just download the data.csv, then stop."
        download_path, filenames = await downloading_task_for_browser_agent(
            default_dataset_task, api_key, model_for_browser_agent, model_api_base_url
        )

        if filenames is None:
            return # returns to main menu

        return download_path, filenames

    elif choice == "2":
        dataset_download_task = Prompt.ask(
            "\n[bold yellow]Enter the download instructions[/bold yellow]"
        )
        download_path, filenames = await downloading_task_for_browser_agent(
            dataset_download_task, api_key, model_for_browser_agent, model_api_base_url
        )

        if filenames is None:
            return # returns to main menu

        return download_path, filenames

    elif choice == "3":
        return


def choice_proceed_with_already_downloaded_datasets() -> list[str]:
    console.print(
        Panel(
            "[bold green]1.[/bold green] Use default dataset (assumes was previously downloaded to './Download/data.csv')\n"
            "[bold green]2.[/bold green] Provide path to your desired dataset(s)\n"
            "[bold green]3.[/bold green] Back to main menu",
            title="Proceed with Existing Dataset",
            border_style="white",
        )
    )

    choice = Prompt.ask(
        "\n[bold yellow]Enter choice[/bold yellow]", choices=["1", "2", "3"]
    ).strip()

    if choice == "1":
        paths = ["./Download/data.csv"]
    elif choice == "2":
        paths = Prompt.ask(
            "\n[bold yellow]Enter dataset path (single path) or multiple paths separated by commas (e.g., ./Download/data.csv, ./Download/readme.md)[/bold yellow]"
        ).split(",")
        paths = [path.strip() for path in paths]
    elif choice == "3":
        return

    try:
        for path in paths:
            if not os.path.isfile(path):
                raise FileNotFoundError(f"File does not exist at path: {path}")

    except FileNotFoundError as e:
        console.print(f"[bold red]{e}[/bold red]")
        return  # return to main menu

    return paths


async def main(
    api_key_for_sandbox_and_model: str,
    model_api_base_url: str,
    model_for_browser_agent: str,
    model_for_eda: str,
    sandbox_domain: str,
    sandbox_template: str,
    sandbox_timeout_seconds: int
):

    while True:

        # Welcome Banner
        console.print(
            Panel(
                "[bold white]Welcome To Agentic Exploratory Data Analysis[/bold white]\n\n"
                "[grey]How will you like to proceed:[/grey]\n"
                "[grey]1.[/grey] Download a dataset first.\n"
                "[grey]2.[/grey] Proceed with already downloaded dataset.\n"
                "[grey]3.[/grey] Exit",
                title="MAIN MENU",
                border_style="green",
                width=70,
            )
        )

        choice = Prompt.ask(
            "\n[bold yellow]Enter your choice[/bold yellow]", choices=["1", "2", "3"]
        ).strip()

        if choice == "1":
            result = await choice_download_dataset(
                api_key_for_sandbox_and_model,
                model_api_base_url,
                model_for_browser_agent,
            )
            if result:
                download_path, filenames = result
                DATASET_PATHS = [f"{download_path}/{filename}" for filename in filenames]
                DATASET_FILE_NAMES = filenames
            else:
                continue  # User returned to main menu

        elif choice == "2":
            result = choice_proceed_with_already_downloaded_datasets()
            if result:
                DATASET_PATHS = result
                DATASET_FILE_NAMES = [os.path.basename(path) for path in result]
            else:
                continue  # since user click back to main menu.

        elif choice == "3":
            break

        # Start the EDA session
        start_eda(
            model_for_eda,
            DATASET_PATHS,
            DATASET_FILE_NAMES,
            api_key_for_sandbox_and_model,
            model_api_base_url,
            sandbox_domain,
            sandbox_template,
            sandbox_timeout_seconds
        )


if __name__ == "__main__":
    NOVITA_API_KEY = os.getenv("NOVITA_API_KEY")
    NOVITA_BASE_URL = os.getenv("NOVITA_BASE_URL")
    NOVITA_E2B_DOMAIN = os.getenv("NOVITA_E2B_DOMAIN")
    NOVITA_E2B_TEMPLATE = os.getenv("NOVITA_E2B_TEMPLATE")
    NOVITA_MODEL_FOR_BROWSER_AGENT = "zai-org/glm-4.5v"
    NOVITA_MODEL_FOR_EDA = "qwen/qwen3-235b-a22b-instruct-2507"
    NOVITA_SANDBOX_TIMEOUT_SECONDS = 900 # 900 seconds (15 minutes), sandbox instance will be killed automatically after.

    asyncio.run(
        main(
            NOVITA_API_KEY,
            NOVITA_BASE_URL,
            NOVITA_MODEL_FOR_BROWSER_AGENT,
            NOVITA_MODEL_FOR_EDA,
            NOVITA_E2B_DOMAIN,
            NOVITA_E2B_TEMPLATE,
            NOVITA_SANDBOX_TIMEOUT_SECONDS
        )
    )
