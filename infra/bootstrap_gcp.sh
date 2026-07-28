#!/bin/sh
set -eu

PROJECT=grounded-motion-dclar
BILLING_ACCOUNT=013510-0B4227-C5FE92
REGION=us-central1
REPOSITORY=grounded-motion
BUCKET=grounded-motion-dclar-canary
CONTROL_SA=grounded-motion-control
WORKER_SA=grounded-motion-worker
DEPLOY_SA=grounded-motion-deploy
POOL=github
PROVIDER=valar05-grounded-motion
REPOSITORY_SLUG=Valar05/grounded-motion-mcp
CANARY_DIR=${1:-}

if [ -z "$CANARY_DIR" ] || [ ! -f "$CANARY_DIR/vanguard-walk-v1.mp4" ] || [ ! -f "$CANARY_DIR/walk-sword-carry-v2-attempt-003.mp4" ]; then
  echo "usage: $0 /absolute/path/to/materialized-canary-dir" >&2
  exit 2
fi

actual_billing="$(gcloud billing projects describe "$PROJECT" --format='value(billingAccountName)' 2>/dev/null || true)"
if [ "$actual_billing" != "billingAccounts/$BILLING_ACCOUNT" ]; then
  gcloud billing projects link "$PROJECT" --billing-account "$BILLING_ACCOUNT"
fi

gcloud services enable \
  artifactregistry.googleapis.com \
  iam.googleapis.com \
  iamcredentials.googleapis.com \
  run.googleapis.com \
  storage.googleapis.com \
  sts.googleapis.com \
  --project "$PROJECT"

if ! gcloud artifacts repositories describe "$REPOSITORY" --project "$PROJECT" --location "$REGION" >/dev/null 2>&1; then
  gcloud artifacts repositories create "$REPOSITORY" \
    --project "$PROJECT" --location "$REGION" \
    --repository-format docker --description "Grounded Motion production images"
fi

if ! gcloud storage buckets describe "gs://$BUCKET" >/dev/null 2>&1; then
  gcloud storage buckets create "gs://$BUCKET" \
    --project "$PROJECT" --location "$REGION" --uniform-bucket-level-access
fi
lifecycle_file="$(mktemp)"
trap 'rm -f "$lifecycle_file"' EXIT HUP INT TERM
printf '%s\n' '{"rule":[{"action":{"type":"Delete"},"condition":{"age":30,"matchesPrefix":["executions/"]}}]}' > "$lifecycle_file"
gcloud storage buckets update "gs://$BUCKET" --lifecycle-file "$lifecycle_file"

for account in "$CONTROL_SA" "$WORKER_SA" "$DEPLOY_SA"; do
  email="$account@$PROJECT.iam.gserviceaccount.com"
  if ! gcloud iam service-accounts describe "$email" --project "$PROJECT" >/dev/null 2>&1; then
    gcloud iam service-accounts create "$account" --project "$PROJECT" --display-name "$account"
  fi
done

CONTROL_EMAIL="$CONTROL_SA@$PROJECT.iam.gserviceaccount.com"
WORKER_EMAIL="$WORKER_SA@$PROJECT.iam.gserviceaccount.com"
DEPLOY_EMAIL="$DEPLOY_SA@$PROJECT.iam.gserviceaccount.com"

for member in "$CONTROL_EMAIL" "$WORKER_EMAIL"; do
  gcloud storage buckets add-iam-policy-binding "gs://$BUCKET" \
    --member "serviceAccount:$member" --role roles/storage.objectAdmin >/dev/null
done
gcloud projects add-iam-policy-binding "$PROJECT" \
  --member "serviceAccount:$CONTROL_EMAIL" --role roles/run.invoker >/dev/null
gcloud iam service-accounts add-iam-policy-binding "$CONTROL_EMAIL" \
  --project "$PROJECT" --member "serviceAccount:$CONTROL_EMAIL" \
  --role roles/iam.serviceAccountTokenCreator >/dev/null

for role in roles/artifactregistry.writer roles/run.admin roles/storage.objectAdmin roles/serviceusage.serviceUsageConsumer; do
  gcloud projects add-iam-policy-binding "$PROJECT" \
    --member "serviceAccount:$DEPLOY_EMAIL" --role "$role" >/dev/null
done
for runtime in "$CONTROL_EMAIL" "$WORKER_EMAIL"; do
  gcloud iam service-accounts add-iam-policy-binding "$runtime" --project "$PROJECT" \
    --member "serviceAccount:$DEPLOY_EMAIL" --role roles/iam.serviceAccountUser >/dev/null
done

if ! gcloud iam workload-identity-pools describe "$POOL" --project "$PROJECT" --location global >/dev/null 2>&1; then
  gcloud iam workload-identity-pools create "$POOL" --project "$PROJECT" --location global \
    --display-name "GitHub Actions"
fi
if ! gcloud iam workload-identity-pools providers describe "$PROVIDER" --project "$PROJECT" --location global --workload-identity-pool "$POOL" >/dev/null 2>&1; then
  gcloud iam workload-identity-pools providers create-oidc "$PROVIDER" \
    --project "$PROJECT" --location global --workload-identity-pool "$POOL" \
    --display-name "Valar05 grounded-motion-mcp" \
    --issuer-uri https://token.actions.githubusercontent.com \
    --attribute-mapping "google.subject=assertion.sub,attribute.repository=assertion.repository,attribute.ref=assertion.ref" \
    --attribute-condition "assertion.repository=='$REPOSITORY_SLUG' && assertion.ref=='refs/heads/main'"
fi
PROJECT_NUMBER="$(gcloud projects describe "$PROJECT" --format='value(projectNumber)')"
gcloud iam service-accounts add-iam-policy-binding "$DEPLOY_EMAIL" --project "$PROJECT" \
  --member "principalSet://iam.googleapis.com/projects/$PROJECT_NUMBER/locations/global/workloadIdentityPools/$POOL/attribute.repository/$REPOSITORY_SLUG" \
  --role roles/iam.workloadIdentityUser >/dev/null

SOURCE_SHA=1884498810950170a631d138addfcfbe9996bb90855d6ae7903329bedc500562
CANDIDATE_SHA=e7c100ce62ba4d6d4549f084065818c502600c487eb3e40e10aa936c4d09cc32
printf '%s  %s\n' "$SOURCE_SHA" "$CANARY_DIR/vanguard-walk-v1.mp4" | sha256sum --check --strict
printf '%s  %s\n' "$CANDIDATE_SHA" "$CANARY_DIR/walk-sword-carry-v2-attempt-003.mp4" | sha256sum --check --strict
gcloud storage cp "$CANARY_DIR/vanguard-walk-v1.mp4" "gs://$BUCKET/canary/inputs/vanguard-walk-v1.mp4"
gcloud storage cp "$CANARY_DIR/walk-sword-carry-v2-attempt-003.mp4" "gs://$BUCKET/canary/inputs/walk-sword-carry-v2-attempt-003.mp4"
verify_dir="$(mktemp -d)"
trap 'rm -f "$lifecycle_file"; rm -rf "$verify_dir"' EXIT HUP INT TERM
gcloud storage cp "gs://$BUCKET/canary/inputs/vanguard-walk-v1.mp4" "$verify_dir/source.mp4"
gcloud storage cp "gs://$BUCKET/canary/inputs/walk-sword-carry-v2-attempt-003.mp4" "$verify_dir/candidate.mp4"
printf '%s  %s\n' "$SOURCE_SHA" "$verify_dir/source.mp4" | sha256sum --check --strict
printf '%s  %s\n' "$CANDIDATE_SHA" "$verify_dir/candidate.mp4" | sha256sum --check --strict

PROVIDER_NAME="projects/$PROJECT_NUMBER/locations/global/workloadIdentityPools/$POOL/providers/$PROVIDER"
gh secret set GCP_WORKLOAD_IDENTITY_PROVIDER --repo "$REPOSITORY_SLUG" --body "$PROVIDER_NAME"
gh secret set GCP_DEPLOY_SERVICE_ACCOUNT --repo "$REPOSITORY_SLUG" --body "$DEPLOY_EMAIL"

printf 'project=%s\nproject_number=%s\nbucket=%s\nprovider=%s\ndeploy_service_account=%s\n' \
  "$PROJECT" "$PROJECT_NUMBER" "$BUCKET" "$PROVIDER_NAME" "$DEPLOY_EMAIL"
