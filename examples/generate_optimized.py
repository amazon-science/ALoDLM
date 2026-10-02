"""Run the bundled engine after `python -m pip install ./optimized`.

Example:
    python examples/generate_optimized.py --model amazon/ALoDLM-8B \
        --prompt "Explain binary search."
"""

from alodlm_optimized.cli import main


if __name__ == "__main__":
    main()
