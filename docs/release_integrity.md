# Verifying released controller evidence

The Prediction Boundary release manifest binds 7,228 original files and
40,354,588,319 bytes by relative path, size and SHA-256. Its SHA-256 is
`a93b036f3e4a6b681a00601e15aab9aa0b24c306d9b1ea7cf2456380e38ce374`.

The complete raw dataset is available on [Hugging Face](https://huggingface.co/datasets/noetherlabs/prediction-boundary-v1/tree/73234e5a6a1374568179562aab76f9f6caa9354e),
licensed CC-BY-4.0. Download the pinned revision with the Hugging Face CLI
(`huggingface_hub` 2.2.0 was used for release checks):

```sh
hf download noetherlabs/prediction-boundary-v1 \
  --repo-type dataset --revision 73234e5a6a1374568179562aab76f9f6caa9354e \
  --local-dir runs/prediction-boundary-v1
```

Allow space for 40.35 GB of original data plus download metadata and any derived
reports. Verify the complete download before reconstructing a report:

```sh
.venv/bin/python scripts/verify_prediction_boundary.py \
  --run runs/prediction-boundary-v1 \
  --manifest-sha256 a93b036f3e4a6b681a00601e15aab9aa0b24c306d9b1ea7cf2456380e38ce374
```

Hugging Face downloads should use `local_dir` or `hf download --local-dir` to
produce regular local files. The default Hub cache uses symlinks to external
blobs, which the study's evidence reader rejects. Hub metadata under
`.cache/huggingface/` can remain in the download directory.

The verifier reads archives in 1 MiB blocks and does not load arrays, execute a
controller or write to the download. Missing files, changed sizes or hashes,
duplicate manifest paths, escaping paths and descendant symlinks are errors.
File identities are checked during and after the scan to detect concurrent
changes. These checks address accidental corruption and drift; they do not
provide a trusted snapshot against an adversary who can replace the filesystem.

For an existing acquisition directory, keep the published manifest separate:

```sh
.venv/bin/python scripts/verify_prediction_boundary.py \
  --run runs/controller-study-v1 \
  --manifest downloads/release-manifest.json \
  --manifest-sha256 a93b036f3e4a6b681a00601e15aab9aa0b24c306d9b1ea7cf2456380e38ce374
```

The success record covers only the original files declared in the pinned
manifest. Dataset documentation, Hub download metadata and empty execution
locks are outside that inventory. Byte agreement does not establish scientific
validity. Use the [completed-cohort reporting procedure](controller_study_execution.md#completed-cohort-reporting)
to reconstruct trial metrics, frozen selection and paired comparisons.
