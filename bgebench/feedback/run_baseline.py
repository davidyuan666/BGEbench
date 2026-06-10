import logging
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from bgebench.common.llm_client import LLMClient
from bgebench.common.schemas import Condition, Generation, Task
from bgebench.common.storage import save_generations_csv, save_generated_code
from bgebench.common.task_loader import build_generation_prompt

logger = logging.getLogger(__name__)


def run_baseline(
    tasks: list[Task],
    llm_client: LLMClient,
    samples_per_task: int = 10,
    prompt_variant: str = "direct",
    output_dir: Path = Path("data"),
    model_name: str = "deepseek-chat",
) -> list[Generation]:
    generated_dir = output_dir / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)

    generations: list[Generation] = []
    total = len(tasks) * samples_per_task

    with tqdm(total=total, desc="Baseline generation") as pbar:
        for task in tasks:
            prompt = build_generation_prompt(task, prompt_variant)
            for sample_id in range(samples_per_task):
                try:
                    result = llm_client.generate(prompt=prompt)
                    code = result["code"]
                    loc = len(code.splitlines())

                    gen = Generation(
                        task_id=task.task_id,
                        benchmark=task.benchmark,
                        model=model_name,
                        prompt_variant=prompt_variant,
                        sample_id=sample_id,
                        generated_code=code,
                        generated_loc=loc,
                        generated_tokens=result["completion_tokens"],
                        accepted_loc=loc,
                        generation_time_s=result["wall_time_s"],
                        condition=Condition.BASELINE,
                    )
                    generations.append(gen)
                    save_generated_code(code, task.task_id, sample_id, generated_dir)

                    pbar.set_postfix(task=task.task_id[:20], loc=loc)
                except Exception as e:
                    logger.error(
                        "Baseline generation failed for %s sample %d: %s",
                        task.task_id, sample_id, e,
                    )
                pbar.update(1)

    csv_path = output_dir / "results" / "baseline.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    save_generations_csv(generations, csv_path)

    logger.info("Baseline generation complete: %d samples", len(generations))
    return generations
