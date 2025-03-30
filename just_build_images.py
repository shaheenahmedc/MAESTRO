import json

from inspect_ai.dataset import FieldSpec, hf_dataset

from inspect_evals.swe_bench.build_images import build_images

# Define the sample fields similar to how swe_bench() does it
samples = hf_dataset(
    path="princeton-nlp/SWE-bench_Verified",
    split="test",
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

# Build the images
build_images(samples, max_workers=4, force_rebuild=False)
