"""
Interactive CLI for Liquid AI d1-omni-600M running on Intel Iris Xe
------------------------------------------------------------------
Usage:
  python interactive.py
  python interactive.py --state "My flight was delayed by 4 hours. Can I get a reimbursement?"
"""

import argparse
import json
import sys
from d1_iris_xe import D1OmniIrisXe


def main():
    parser = argparse.ArgumentParser(description="Query d1-omni-600M on Intel Iris Xe")
    parser.add_argument("--device", type=str, default="GPU", choices=["GPU", "CPU"], help="Device to use")
    parser.add_argument("--state", type=str, default=None, help="Input text or JSON state to analyze")
    parser.add_argument("--image", type=str, default=None, help="Path to an image file")
    args = parser.parse_args()

    engine = D1OmniIrisXe(device=args.device)

    if args.state is None and args.image is None:
        print("\n--- Enter interactive mode (Press Ctrl+C to exit) ---")
        while True:
            try:
                user_input = input("\nEnter state / text to evaluate:\n> ")
                if not user_input.strip():
                    continue

                questions = {
                    "sentiment": {
                        "type": "choice",
                        "instructions": "What is the sentiment of the text?",
                        "criteria": {
                            "positive": "Satisfied, pleased, constructive",
                            "neutral": "Factual, informational, neutral",
                            "negative": "Angry, frustrated, disappointed"
                        }
                    },
                    "action_required": {
                        "type": "noul",
                        "instructions": "Does this require an immediate human response or action?"
                    }
                }

                res = engine.system_one(user_input, questions)
                print(json.dumps(res, indent=2))
            except KeyboardInterrupt:
                print("\nExiting.")
                break
    else:
        images = None
        if args.image:
            from PIL import Image
            images = [Image.open(args.image)]

        questions = {
            "summary_classification": {
                "type": "noul",
                "instructions": "Is this content safe, legitimate, and valid?"
            }
        }
        res = engine.system_one(args.state, questions, images=images)
        print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
