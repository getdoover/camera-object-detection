#!/bin/sh
#
# Build the cloud processor as a container image and push it to ECR.
#
# This is the counterpart to the usual processor `build.sh`, which produces a
# package.zip. This app can't use that: its dependencies are 288MB unzipped / 113MB
# zipped before 22MB of model weights, against Lambda's 250MB unzipped zip limit -- see
# Dockerfile.processor for the breakdown. So it deploys as a container image instead.
#
# Building is all this script does. Creating and updating the Lambda is doover's job:
# `doover app publish --release` skips the package.zip path for an image processor, and
# the release points the function at lambda_config.Code.ImageUri. The reason a script is
# needed at all is that doover doesn't *build* images for processor apps -- PRO types are
# excluded from `builds_image` in `doover app discover`, so nothing in CI builds this
# Dockerfile.
#
# Defaults match doover_config.json's lambda_config.Code.ImageUri, so a plain
# `./build_ecr.sh` pushes exactly what a release will deploy.
#
# Requires AWS credentials with ECR push rights.
#
# Usage:
#   ./build_ecr.sh                        # build + push :main
#   ./build_ecr.sh --tag v3               # push :v3 (also update the config to match)
#   ECR_REPO=other AWS_REGION=us-east-1 ./build_ecr.sh

set -eu

AWS_REGION="${AWS_REGION:-ap-southeast-2}"
ECR_REPO="${ECR_REPO:-doover/object-detection-processor}"
# arm64 = Graviton: cheaper per ms, and the same architecture the device image is built
# for, so `common` runs on wheels we've already exercised.
ARCH="${ARCH:-arm64}"
TAG="main"

while [ $# -gt 0 ]; do
    case "$1" in
        --tag) TAG="$2"; shift 2 ;;
        --region) AWS_REGION="$2"; shift 2 ;;
        --repo) ECR_REPO="$2"; shift 2 ;;
        --arch) ARCH="$2"; shift 2 ;;
        -h|--help) sed -n '2,25p' "$0"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

case "$ARCH" in
    arm64) DOCKER_PLATFORM="linux/arm64" ;;
    amd64|x86_64) DOCKER_PLATFORM="linux/amd64" ;;
    *) echo "unsupported ARCH: $ARCH (use arm64 or amd64)" >&2; exit 2 ;;
esac

for tool in aws docker; do
    command -v "$tool" >/dev/null || { echo "$tool is required" >&2; exit 1; }
done

ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
REGISTRY="${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
IMAGE_URI="${REGISTRY}/${ECR_REPO}:${TAG}"

echo "account   : ${ACCOUNT_ID}"
echo "region    : ${AWS_REGION}"
echo "image     : ${IMAGE_URI}"
echo "platform  : ${DOCKER_PLATFORM}"

# Idempotent: describe first so a re-run doesn't fail on an existing repo.
if ! aws ecr describe-repositories --region "$AWS_REGION" \
        --repository-names "$ECR_REPO" >/dev/null 2>&1; then
    echo "creating ECR repository ${ECR_REPO}..."
    # Lambda reads the image on cold start, so scanning on push is worth having;
    # AES256 encryption is the default and costs nothing.
    aws ecr create-repository --region "$AWS_REGION" \
        --repository-name "$ECR_REPO" \
        --image-scanning-configuration scanOnPush=true \
        --encryption-configuration encryptionType=AES256 >/dev/null
fi

# Without this, CreateFunction fails with:
#   AccessDeniedException: Lambda does not have permission to access the ECR image.
#
# Lambda pulls an image with a *service-linked* call rather than the function's
# execution role, so the grant has to live on the repository -- and that is true even
# when the function and the repo are in the same account, which is the part that
# surprises people. Applied on every run rather than only at create time, so an
# existing repo made by hand or by another tool gets fixed too.
#
# aws:sourceArn is wildcarded across this account's functions in this region rather
# than naming one: doover mints the function name itself (dv-<type>-<id>-<name>), so we
# don't know it here. It still can't be used by any other account's Lambdas.
echo "ensuring the repository policy lets Lambda pull..."
POLICY_FILE="$(mktemp)"
trap 'rm -f "$POLICY_FILE"' EXIT
cat >"$POLICY_FILE" <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "LambdaECRImageRetrievalPolicy",
      "Effect": "Allow",
      "Principal": { "Service": "lambda.amazonaws.com" },
      "Action": ["ecr:BatchGetImage", "ecr:GetDownloadUrlForLayer"],
      "Condition": {
        "StringLike": {
          "aws:sourceArn": "arn:aws:lambda:${AWS_REGION}:${ACCOUNT_ID}:function:*"
        }
      }
    }
  ]
}
EOF
aws ecr set-repository-policy --region "$AWS_REGION" \
    --repository-name "$ECR_REPO" \
    --policy-text "file://${POLICY_FILE}" >/dev/null

echo "logging docker into ECR..."
aws ecr get-login-password --region "$AWS_REGION" \
    | docker login --username AWS --password-stdin "$REGISTRY"

echo "building..."
# --provenance=false: Lambda rejects an image whose manifest is an OCI index carrying
# attestations, which is what buildx produces by default. It wants a single manifest.
docker build \
    --platform "$DOCKER_PLATFORM" \
    --provenance=false \
    -f Dockerfile.processor \
    -t "$IMAGE_URI" \
    .

echo "pushing..."
docker push "$IMAGE_URI"

DIGEST="$(aws ecr describe-images --region "$AWS_REGION" \
    --repository-name "$ECR_REPO" \
    --image-ids "imageTag=${TAG}" \
    --query 'imageDetails[0].imageDigest' --output text)"

cat <<EOF

pushed  ${IMAGE_URI}
digest  ${DIGEST}

Next: 'doover app publish --release' to point the function at it. Note the release
resolves the *tag*, so re-pushing the same tag does not redeploy on its own -- a release
is what picks up a new image.
EOF
