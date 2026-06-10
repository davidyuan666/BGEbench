import time
import logging
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from bgebench.common.llm_client import LLMClient
from bgebench.common.schemas import Generation, Task
from bgebench.common.storage import save_generations_csv, save_generated_code

logger = logging.getLogger(__name__)


def generate_samples(
    tasks: list[Task],
    llm_client: LLMClient,
    samples_per_task: int = 10,
    prompt_variants: Optional[list[str]] = None,
    output_dir: Path = Path("data"),
    model_name: str = "deepseek-chat",
) -> list[Generation]:
    if prompt_variants is None:
        prompt_variants = ["direct"]

    generated_dir = output_dir / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)

    generations: list[Generation] = []
    total = len(tasks) * samples_per_task * len(prompt_variants)

    with tqdm(total=total, desc="Generating samples") as pbar:
        for task in tasks:
            for variant in prompt_variants:
                from bgebench.common.task_loader import build_generation_prompt

                prompt = build_generation_prompt(task, variant)

                for sample_id in range(samples_per_task):
                    try:
                        result = llm_client.generate(prompt=prompt)
                        code = result["code"]
                        loc = len(code.splitlines())
                        tokens = result["completion_tokens"]

                        gen = Generation(
                            task_id=task.task_id,
                            benchmark=task.benchmark,
                            model=model_name,
                            prompt_variant=variant,
                            sample_id=sample_id,
                            generated_code=code,
                            generated_loc=loc,
                            generated_tokens=tokens,
                            accepted_loc=loc,
                            generation_time_s=result["wall_time_s"],
                        )
                        generations.append(gen)

                        save_generated_code(
                            code, task.task_id, sample_id, generated_dir
                        )

                        pbar.set_postfix(
                            task=task.task_id[:20],
                            loc=loc,
                            variant=variant,
                        )
                    except Exception as e:
                        logger.error(
                            "Generation failed for %s sample %d variant %s: %s",
                            task.task_id, sample_id, variant, e,
                        )

                    pbar.update(1)

    csv_path = output_dir / "results" / "generations.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    save_generations_csv(generations, csv_path)

    logger.info(
        "Growth measurement generation complete: %d samples across %d tasks",
        len(generations), len(tasks),
    )
    return generations
