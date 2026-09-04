# Make eval_tasks runnable as: python3 -m eval_tasks
# Routes to eval.cli for backward compatibility
from eval.cli import main

if __name__ == "__main__":
    main()
