import argparse

from models.common.experiment import (
    run_experiment,
)


TASKS = [
    "a",
    "d",
    "m",
    "e",
    "t",
]


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Train Extra Trees, XGBoost, CatBoost "
            "and the weighted ensemble for ADMET."
        )
    )

    parser.add_argument(
        "--task",

        choices=TASKS + ["all"],

        default="all",

        help=(
            "Run one endpoint or all endpoints. "
            "Default: all."
        ),
    )

    parser.add_argument(
        "--gpu",

        action="store_true",

        help=(
            "Use GPU for XGBoost and CatBoost."
        ),
    )

    parser.add_argument(
        "--stop-on-error",

        action="store_true",

        help=(
            "Stop immediately when one task fails."
        ),
    )

    args = parser.parse_args()

    if args.task == "all":
        selected_tasks = TASKS
    else:
        selected_tasks = [
            args.task
        ]

    completed = []
    failed = []

    for task in selected_tasks:

        try:

            summary = run_experiment(
                task=task,
                use_gpu=args.gpu,
            )

            completed.append(
                summary
            )

        except Exception as error:

            failed.append(
                {
                    "task": task,
                    "error": str(error),
                }
            )

            print(
                f"\nTask {task.upper()} failed: "
                f"{error}"
            )

            if args.stop_on_error:
                raise

    print("\n" + "=" * 70)
    print("RUN SUMMARY")
    print("=" * 70)

    for summary in completed:

        print(
            f"{summary['task'].upper()} "
            f"({summary['endpoint']}): "
            f"{summary['best_validation_solution']}"
        )

    for failure in failed:

        print(
            f"{failure['task'].upper()}: "
            f"FAILED - {failure['error']}"
        )

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()