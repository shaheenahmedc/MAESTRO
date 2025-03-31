"""SWE-bench: Can Language Models Resolve Real-World GitHub Issues?

Carlos E. Jimenez, John Yang, Alexander Wettig, Shunyu Yao, Kexin Pei, Ofir Press, Karthik Narasimhan
https://arxiv.org/abs/2310.06770
"""

import json
import logging
from importlib.util import find_spec
from pathlib import Path
from typing import Callable, Literal, Optional

from inspect_ai import Task, task
from inspect_ai.dataset import FieldSpec, hf_dataset, MemoryDataset
from inspect_ai.scorer import Scorer
from inspect_ai.solver import (
    Solver,
    basic_agent,
    system_message,
)
from inspect_ai.tool import bash
from inspect_ai.util import SandboxEnvironmentSpec
from platformdirs import user_cache_dir

from inspect_evals.swe_bench.scorers import swe_bench_scorer
from inspect_evals.swe_bench.swe_agent_tools import (
    create_new_file,
    edit_file,
    find_file,
    open_file,
    scroll_down,
    scroll_up,
    search_dir,
    search_file,
)

from inspect_evals.swe_bench.autogen_team.main import consult_multi_agent_team
from inspect_ai.solver import bridge

COMPOSE_FILES_DIR = Path(user_cache_dir("inspect_swebench_eval")) / "compose_files/"
DEFAULT_INPUT_PROMPT = "Please solve the following coding issue:\n\n{issue_text}"

logger = logging.getLogger(__name__)


@task
def swe_bench(
    dataset: str = "MariusHobbhahn/swe-bench-verified-mini",
    # dataset: str = "princeton-nlp/SWE-bench_Verified",
    split: str = "test",
    solver: Solver | None = None,
    max_messages: int = 500,
    input_prompt: str = DEFAULT_INPUT_PROMPT,
    instance_ids: list[str] | None = None,
    scorer: Scorer | list[Scorer] | None = None,
    epochs: int = 1,
    sandbox_type: Literal["docker", "k8s"] = "docker",
    build_docker_images: bool = True,
    docker_image_from_id: Callable[
        [str], str
    ] = lambda instance_id: f"sweb.eval.x86_64.{instance_id}:latest",
    allow_internet: bool = True,
    n_samples: int | None = None,
    sample_seed: int = 42,
) -> Task:
    """Returns a Task, representing an evaluation on SWE-bench.

    Args.
        dataset : str
            The dataset to use. This should  either be the name of a dataset in the HF hub, or a path to a dataset on disk.
        split : str
            The split of the dataset to load.
        solver : Solver
            The solver to use when creating the task. If None, uses the default solver.
        max_messages : int
            The maximum number of messages to allow for each sample. Only
            applies when using the default solver.
        instance_ids : list[str]
            A list of instance_ids to filter the dataset by. If None, all instances are used.
        scorer : Scorer | list[Scorer] | None
            The scorer to use when evaluating swe_bench. If None, uses the default scorer. Mostly commonly, this will be a list of scorers to compare to baselines (see the README for more information).
        epochs : int
            Number of times to repeat each sample.
        sandbox_type : Literal["docker", "k8s"]
            The type of sandbox to use for the task.
        build_docker_images : bool
            Whether to build the docker images. Implies sandbox_type = "docker". For k8s, you are responsible for building the images yourself, using the original swebench library.
        docker_image_from_id : Callable[[str], str]
            Used to transform the swe_bench ID (e.g. astropy__astropy-14182) into a docker container name (e.g. "sweb.eval.x86_64.astropy__astropy-14182:latest"). This is useful if you needed to rebuild the images from the swebench library (e.g. to add tooling) with different names.
            It is also useful as AWS ECR does not allow double underscores in image names, so you can replace them here.
            The default value should be fine if you have built the images using the SWE-Bench library in the normal way.
        allow_internet : bool
            Whether to allow the sandbox to access the internet.

    """
    assert find_spec("swebench"), (
        "To run SWE-bench, please install the optional SWE-bench dependency, by running `pip install inspect-evals[swe_bench]`"
    )

    samples = hf_dataset(
        path=dataset,
        split=split,
        sample_fields=FieldSpec(
            input="problem_statement",
            id="instance_id",
            metadata=[
                "base_commit",
                "patch",
                "PASS_TO_PASS",
                "FAIL_TO_PASS",
                "test_patch",
                "version",
                "repo",
                "environment_setup_commit",
                "hints_text",
                "created_at",
            ],
        ),
    )

    for sample in samples:
        # Turn the saved strings into list objects
        sample.metadata = sample.metadata or {}
        sample.metadata["PASS_TO_PASS"] = json.loads(sample.metadata["PASS_TO_PASS"])
        sample.metadata["FAIL_TO_PASS"] = json.loads(sample.metadata["FAIL_TO_PASS"])

    if instance_ids is not None:
        samples = samples.filter(lambda x: x.id in instance_ids)

    # Filter to n random samples using a fixed seed
    if n_samples is not None and n_samples < len(samples):
        # Shuffle the dataset with the specified seed
        samples.shuffle(seed=sample_seed)

        # Take the first n_samples from the shuffled dataset
        samples = samples[:n_samples]

        print(f"Selected {n_samples} random samples with seed {sample_seed}")
        print(f"Sample IDs: {[sample.id for sample in samples]}")

    if build_docker_images:
        if sandbox_type != "docker":
            raise ValueError(
                "If you want to use k8s, you are responsible for building the images yourself, using the original swebench library."
            )
        # Build the images for the samples - can take a long time
        # (import done inline to defer dependency binding until usage)
        from inspect_evals.swe_bench.build_images import build_images

        build_images(samples, force_rebuild=False)

    for sample in samples:
        sample.metadata = sample.metadata or {}
        sample.input = input_prompt.format(issue_text=sample.input)
        if sandbox_type == "docker":
            sample.sandbox = SandboxEnvironmentSpec(
                type="docker",
                config=get_compose_file(
                    str(sample.id), docker_image_from_id, allow_internet=allow_internet
                ),
            )
        elif sandbox_type == "k8s":
            sample.sandbox = SandboxEnvironmentSpec(
                type="k8s",
                config=get_k8s_config_file(
                    str(sample.id), docker_image_from_id, allow_internet=allow_internet
                ),
            )
        else:
            raise ValueError(f"Unknown sandbox type: {sandbox_type}")

    return Task(
        dataset=samples,
        solver=solver or default_solver(max_messages),
        epochs=epochs,
        scorer=scorer or swe_bench_scorer(),
    )


def default_solver(max_messages: int = 50) -> Solver:
    return basic_agent(
        init=system_message(
            "Please solve the coding task below. Once you are done, use your submit tool."
            + "You will be provided with a partial code base and an issue statement"
            "explaining a problem to resolve."
            + "You have a number of tools at your disposal to help you solve the problem. Use these to explore the codebase and make edits to files. These are: \
                    - bash: to run commands in the terminal \
                    - edit_file: to edit a file \
                    - open_file: to open a file \
                    - search_dir: to search for a file in a directory \
                    - search_file: to search for a search term in a file \
                    - create_new_file: to create a new file \
                    - find_file: to find a file in the current directory \
                    - scroll_down: to scroll down in the terminal \
                    - scroll_up: to scroll up in the terminal \
                    - consult_multi_agent_team: to ask a tool-equipped multi-agent Autogen team for a possible solution from each agent."
            + "Please start your solution by using the consult_multi_agent_team tool to ask the multi-agent team for a set of possible solutions. IMPORTANT: When using the consult_multi_agent_team tool, ALWAYS include the complete original coding issue in your question, followed by any specific questions you have. Format your question like this: \
                ORIGINAL ISSUE: \
                [full original issue here] \
                SPECIFIC QUESTION: \
                [your question here]\
                This ensures the multi-agent team has all necessary context. \
                ONLY USE THE CONSULT_AUTOGEN_CORE_DEBATE_TEAM TOOL ONCE AT THE START OF YOUR SOLUTION. \
                Once you have received a set of possible solutions, use these with the other tools to solve the problem, and make the required edits to the codebase. Once you are done, use your submit tool (with a description of the solution you've implemented) to conclude the task."
        ),
        tools=[
            bash(timeout=180),
            edit_file(),
            open_file(),
            search_dir(),
            search_file(),
            create_new_file(),
            find_file(),
            scroll_down(),
            scroll_up(),
            consult_multi_agent_team(timeout=900),
        ],
        max_messages=max_messages,
    )


# # Single agent baseline solver
# def default_solver(max_messages: int = 50) -> Solver:
#     return basic_agent(
#         init=system_message(
#             "Please solve the coding task below. Once you are done, use your submit tool."
#             + "You will be provided with a partial code base and an issue statement"
#             "explaining a problem to resolve."
#             + "You have a number of tools at your disposal to help you solve the problem. Use these to explore the codebase and make edits to files. These are: \
#                     - bash: to run commands in the terminal \
#                     - edit_file: to edit a file \
#                     - open_file: to open a file \
#                     - search_dir: to search for a file in a directory \
#                     - search_file: to search for a search term in a file \
#                     - create_new_file: to create a new file \
#                     - find_file: to find a file in the current directory \
#                     - scroll_down: to scroll down in the terminal \
#                     - scroll_up: to scroll up in the terminal \
#                 Use these tools to solve the problem, and make the required edits to the codebase. Once you are done, use your submit tool (with a description of the solution you've implemented) to conclude the task."
#         ),
#         tools=[
#             bash(timeout=180),
#             edit_file(),
#             open_file(),
#             search_dir(),
#             search_file(),
#             create_new_file(),
#             find_file(),
#             scroll_down(),
#             scroll_up(),
#         ],
#         max_messages=max_messages,
#     )


def get_compose_file(
    instance_id: str, docker_image_from_id: Callable[[str], str], allow_internet: bool
) -> str:
    image_name = docker_image_from_id(instance_id)

    image_compose_file = COMPOSE_FILES_DIR / f"{image_name}.yaml"

    image_compose_file.parent.mkdir(parents=True, exist_ok=True)

    with image_compose_file.open(mode="w+") as f:
        f.write(
            f"""services:
  default:
    image: {image_name}
    command: "sleep infinity"
    working_dir: /testbed
    x-local: true
    {"network_mode: none" if not allow_internet else ""}
    deploy:
      resources:
        limits:
          cpus: '1'"""
        )

    return str(image_compose_file)


def get_k8s_config_file(
    instance_id: str, docker_image_from_id: Callable[[str], str], allow_internet: bool
) -> str:
    image_name = docker_image_from_id(instance_id)

    image_k8s_file = COMPOSE_FILES_DIR / f"{image_name}-k8s.yaml"

    image_k8s_file.parent.mkdir(parents=True, exist_ok=True)

    with image_k8s_file.open(mode="w+") as f:
        f.write(
            f"""
services:
  default:
    image: {image_name}
    command: ["tail", "-f", "/dev/null"]
    workingDir: /testbed
{'allowDomains: ["*"]' if allow_internet else ""}
"""
        )

    return str(image_k8s_file)
