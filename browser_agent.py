from typing import Tuple

from browser_use import Agent, BrowserProfile, BrowserSession, Controller
from browser_use.llm import ChatOpenAI
from pydantic import BaseModel
from rich.console import Console
from rich.panel import Panel

console = Console()


class DownloadedFileNames(BaseModel):
    """Output model for getting the names of the downloaded files"""

    names_of_file_with_extension: list[str]


async def downloading_task_for_browser_agent(
    task: str,
    api_key: str,
    model: str,
    model_api_base_url: str,
    use_vision: bool = True,
    download_dir_path: str = "./Download",
) -> Tuple[str, list[str]]:
    """
    Will perform the user's download task via browser use and return download directory path and the
    downloaded files names.

    Returns:
        Tuple of (download_directory, filenames_with_extension)
    """

    agent = Agent(
        task=task,
        llm=ChatOpenAI(base_url=model_api_base_url, model=model, api_key=api_key),
        use_vision=use_vision,
        vision_detail_level="low",  # available options ['low', 'high', 'auto']; note high detail means more token cost; low should suffice for most tasks.
        browser_session=BrowserSession(
            browser_profile=BrowserProfile(
                downloads_path=download_dir_path, user_data_dir="./browser_user_data"
            )
        ),  # set the download directory path for the browser.
        controller=Controller(
            output_model=DownloadedFileNames
        ),  # Get the agent to output the name of the downloaded file at the end of the task.
    )

    try:
        all_results = await agent.run()
        file_names_with_extension = DownloadedFileNames.model_validate_json(
            all_results.final_result()
        ).names_of_file_with_extension  # parse the final agent result to get the file names.

        if file_names_with_extension:
            console.print(
                Panel(
                    f"[bold green]Downloaded files:[/bold green] {file_names_with_extension} to {download_dir_path}",
                    title="Downloaded Files",
                    border_style="green",
                )
            )
        else:
            raise Exception("No files downloaded")

    except Exception as e:
        file_names_with_extension = None
        console.print(
            Panel(
                f"[bold red]Error:[/bold red] {e}",
                title="Execution Error",
                border_style="red",
            )
        )

    return (download_dir_path, file_names_with_extension)
