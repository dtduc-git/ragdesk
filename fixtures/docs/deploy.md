# Deploying and rolling back

Deploys run through Terraform for infrastructure and Helm for application
releases. Every release goes out as a canary first: 10 percent of traffic for
15 minutes, promoted only when the error rate stays flat.

## Rollback

If the canary fails, roll back immediately with `kubectl rollout undo
deployment/<name>`. Database migrations are forward-only, so keep each
migration backward compatible for at least one release.

## Blue-green

For the stateful services we run blue-green instead of canary. Cut over by
switching the Service selector, and keep the previous environment warm for 24
hours before tearing it down.
