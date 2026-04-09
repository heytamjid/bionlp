import os
import time
import vertexai
from vertexai.preview.tuning import sft

# ── Configuration ─────────────────────────────────────────────────────────────
PROJECT_ID  = os.environ["PROJECT_ID"]
REGION      = os.environ["REGION"]
BUCKET_NAME = os.environ["BUCKET_NAME"]

vertexai.init(project=PROJECT_ID, location=REGION)
# ──────────────────────────────────────────────────────────────────────────────

def train():
    print("Launching fine-tuning job for PsyDefDetect (DMRS defense classification)...")

    sft_tuning_job = sft.train(
        source_model="gemini-2.5-flash",
        train_dataset=f"gs://{BUCKET_NAME}/datasets/train/train_gemini.jsonl",
        validation_dataset=f"gs://{BUCKET_NAME}/datasets/val/val_gemini.jsonl",

        # ── Hyperparameters ───────────────────────────────────────────────────
        # Dataset is small (~2k samples) so more epochs compensate.
        # Increase to 5-8 if validation loss is still improving at epoch 3.
        epochs=5,

        # adapter_size=4 is the lightest LoRA rank — good starting point.
        # Bump to 8 or 16 if the model struggles with the 9-way label taxonomy.
        adapter_size=16,

        # Conservative multiplier for a small, domain-specific dataset.
        # Lower to 0.5 if you see training loss oscillating.
        learning_rate_multiplier=1,

        tuned_model_display_name="gemini-2.5-flash-psydef-dmrs",
    )

    print(f"\nJob resource name : {sft_tuning_job.resource_name}")
    print("Waiting for job to complete (typically 30–60 min for this dataset size)...\n")

    while not sft_tuning_job.has_ended:
        time.sleep(60)
        sft_tuning_job.refresh()
        print(f"  [{time.strftime('%H:%M:%S')}] Status: {sft_tuning_job.state.name}")

    if sft_tuning_job.has_succeeded:
        endpoint = sft_tuning_job.tuned_model_endpoint_name
        print(f"\nFine-tuning complete!")
        print(f"Tuned model endpoint: {endpoint}")
        print(
            "\nNext step — run baseline + tuned evaluation:\n"
            f"  python evaluate.py --model gemini-2.5-flash --test test.json --output baseline.json\n"
            f"  python evaluate.py --model {endpoint} --test test.json --output tuned.json --baseline baseline.json"
        )
        return endpoint
    else:
        print(f"\nJob ended with status: {sft_tuning_job.state.name}")
        print("Check the Vertex AI console for error details.")
        return None


if __name__ == "__main__":
    train()