"""
Test script for the FinAgent evaluation framework.
"""

import asyncio
import argparse
from lmbase.eval.finagent import FinAgentEvaluator
from lmbase.dataset.registry import get

from dotenv import load_dotenv

load_dotenv()


def test_finagent_evaluator():
    """Test the FinAgent evaluator initialization."""
    print("Testing FinAgent evaluator initialization...")

    try:
        # Initialize evaluator with a mock model (won't actually run without API key)
        evaluator = FinAgentEvaluator(model_name="deepseek/deepseek-chat")
        print("✓ FinAgent evaluator initialized successfully")

        return True

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_async_evaluation(save_dir="./logs"):
    """Test the async evaluation functionality."""
    print("\nTesting async evaluation...")

    try:
        evaluator = FinAgentEvaluator(model_name="deepseek/deepseek-chat")

        # Get a sample from the finagent dataset
        config = {"data_name": "finagent", "data_path": "./EXPERIMENT/data/finagent"}
        dataset = get(config, split="train")

        # Test single sample evaluation
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"✓ Prepared sample for evaluation: {sample['main_id']}")

            # Test that the evaluation method can be called
            result = await evaluator.evaluate_single_sample(sample, save_dir=save_dir)
            print("✓ Async evaluation method can be called")
            print(f"  Result keys: {list(result.keys())}")

        return True

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback

        traceback.print_exc()
        return False


def parse_args():
    parser = argparse.ArgumentParser(description="Test FinAgent Evaluation Framework")
    parser.add_argument(
        "-s",
        "--save-dir",
        type=str,
        default="./logs",
        help="Directory to save evaluation logs and trajectories",
    )
    return parser.parse_args()


if __name__ == "__main__":
    print("Testing FinAgent Evaluation Framework")
    print("=" * 50)

    args = parse_args()

    success1 = test_finagent_evaluator()
    success2 = asyncio.run(test_async_evaluation(save_dir=args.save_dir))

    if success1 and success2:
        print(f"\n✓ All tests passed! The FinAgent evaluation framework is ready.")
        print(f"\nLogs and trajectories are saved to: {args.save_dir}")
    else:
        print("\n✗ Some tests failed!")

    exit(0 if (success1 and success2) else 1)
