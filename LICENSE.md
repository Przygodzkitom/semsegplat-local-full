# Licensing

## This Project

This project is licensed under the **Apache License 2.0**.

Apache 2.0 is a permissive open-source license that allows you to freely use, modify, and distribute this software (including for commercial purposes) provided that you include the original copyright notice and the license text, and document any significant changes you made to the source code.

Full license text: https://www.apache.org/licenses/LICENSE-2.0

---

## Third-Party Dependencies

### Application Services (Docker)

| Component | Version | License | Notes |
|-----------|---------|---------|-------|
| [Label Studio](https://github.com/HumanSignal/label-studio) | 1.20.0 | Apache 2.0 | Annotation platform, run as a Docker service |
| [MinIO](https://github.com/minio/minio) | RELEASE.2025-09-07T16-13-09Z | GNU AGPL v3.0 | S3-compatible object storage, run as a Docker service |

**Note on MinIO:** The MinIO server is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0). This project uses MinIO exclusively as an external service via Docker — its code is neither linked into nor distributed with this project. As a result, the AGPL copyleft requirements do not extend to this project's source code. Interaction with MinIO is performed through the `boto3`/`botocore` libraries (Apache 2.0).

---

### Python Libraries

| Package | License | Notes |
|---------|---------|-------|
| [streamlit](https://github.com/streamlit/streamlit) | Apache 2.0 | Web UI framework |
| [label-studio-sdk](https://github.com/HumanSignal/label-studio-sdk) | Apache 2.0 | Label Studio Python client |
| [boto3](https://github.com/boto/boto3) | Apache 2.0 | AWS/MinIO S3 client |
| [botocore](https://github.com/boto/botocore) | Apache 2.0 | Core AWS client library |
| [google-cloud-storage](https://github.com/googleapis/python-storage) | Apache 2.0 | Google Cloud Storage client |
| [requests](https://github.com/psf/requests) | Apache 2.0 | HTTP library |
| [opencv-python-headless](https://github.com/opencv/opencv-python) | Apache 2.0 | Computer vision library |
| [PyTorch](https://github.com/pytorch/pytorch) | BSD 3-Clause | Deep learning framework |
| [torchvision](https://github.com/pytorch/vision) | BSD 3-Clause | Computer vision utilities for PyTorch |
| [numpy](https://github.com/numpy/numpy) | BSD 3-Clause | Numerical computing |
| [psutil](https://github.com/giampaolo/psutil) | BSD 3-Clause | System and process utilities |
| [python-dotenv](https://github.com/theskumar/python-dotenv) | BSD 3-Clause | Environment variable management |
| [Pillow](https://github.com/python-pillow/Pillow) | HPND | Image processing (MIT/BSD-compatible) |
| [matplotlib](https://github.com/matplotlib/matplotlib) | PSF / BSD-compatible | Plotting and visualization |
| [albumentations](https://github.com/albumentations-team/albumentations) | MIT | Image augmentation |
| [segmentation-models-pytorch](https://github.com/qubvel/segmentation_models.pytorch) | MIT | Semantic segmentation model library |
| [tqdm](https://github.com/tqdm/tqdm) | MIT / MPL 2.0 | Progress bars |

---

### License Compatibility

All Python dependencies use permissive licenses (Apache 2.0, BSD, MIT, PSF, HPND) that are fully compatible with Apache 2.0. The MinIO server (AGPL v3.0) is used only as an isolated external service and does not affect this project's licensing.
