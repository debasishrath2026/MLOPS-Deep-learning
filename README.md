# MLOPS-Deep-learning

MLOps pipeline for deep learning models, using GitHub Actions with AWS.

## CI/CD

- `.github/workflows/aws-connect.yml` authenticates to AWS from GitHub Actions via OIDC (no long-lived access keys). Requires the `AWS_ROLE_ARN` repository secret.
