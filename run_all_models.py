import argparse

from models.common.experiment import (
    run_experiment,
)


# ============================================================
# AVAILABLE ADMET ENDPOINTS
# ============================================================

TASKS = [
    "a",  # Absorption
    "d",  # Distribution
    "m",  # Metabolism
    "e",  # Excretion / Clearance
    "t",  # Toxicity
    "s",  # Solubility
]


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Train Extra Trees, XGBoost, CatBoost, "
            "LightGBM and the 4-model weighted "
            "ensemble for ADMET endpoints."
        )
    )


    # --------------------------------------------------------
    # SELECT TASK
    # --------------------------------------------------------

    parser.add_argument(

        "--task",

        choices=TASKS + ["all"],

        default="all",

        help=(
            "Run one endpoint or all endpoints. "
            "Options: a, d, m, e, t, s, all. "
            "Default: all."
        ),
    )


    # --------------------------------------------------------
    # GPU OPTION
    # --------------------------------------------------------

    parser.add_argument(

        "--gpu",

        action="store_true",

        help=(
            "Use GPU for XGBoost and CatBoost. "
            "Extra Trees and LightGBM run on CPU "
            "with the current configuration."
        ),
    )


    # --------------------------------------------------------
    # ERROR HANDLING
    # --------------------------------------------------------

    parser.add_argument(

        "--stop-on-error",

        action="store_true",

        help=(
            "Stop immediately if one task fails. "
            "Otherwise continue with remaining tasks."
        ),
    )


    args = parser.parse_args()


    # ========================================================
    # SELECT ENDPOINTS
    # ========================================================

    if args.task == "all":

        selected_tasks = TASKS

    else:

        selected_tasks = [
            args.task
        ]


    completed = []

    failed = []


    # ========================================================
    # RUN EACH ENDPOINT
    # ========================================================

    for task in selected_tasks:

        try:

            print(
                "\n"
                + "#" * 70
            )

            print(
                f"STARTING TASK: "
                f"{task.upper()}"
            )

            print(
                "#" * 70
            )


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

                    "error": str(
                        error
                    ),
                }
            )


            print(
                f"\nTask "
                f"{task.upper()} "
                f"failed: "
                f"{error}"
            )


            if args.stop_on_error:

                raise


    # ========================================================
    # FINAL RUN SUMMARY
    # ========================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "RUN SUMMARY"
    )

    print(
        "=" * 70
    )


    # --------------------------------------------------------
    # SUCCESSFUL TASKS
    # --------------------------------------------------------

    if completed:

        print(
            "\nCOMPLETED:"
        )


        for summary in completed:

            print(

                f"{summary['task'].upper()} "

                f"({summary['endpoint']}): "

                f"{summary['best_validation_solution']}"
            )


    # --------------------------------------------------------
    # FAILED TASKS
    # --------------------------------------------------------

    if failed:

        print(
            "\nFAILED:"
        )


        for failure in failed:

            print(

                f"{failure['task'].upper()}: "

                f"{failure['error']}"
            )


    # --------------------------------------------------------
    # EXIT WITH ERROR IF ANY TASK FAILED
    # --------------------------------------------------------

    if failed:

        raise SystemExit(1)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()