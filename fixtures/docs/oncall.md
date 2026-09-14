# On-call and escalation

The on-call engineer acknowledges pages within 5 minutes. Incident severity
levels: SEV1 is a full outage, SEV2 is degraded service, SEV3 is a
single-customer issue.

## Escalation path

1. Page the on-call engineer through PagerDuty.
2. If unacknowledged after 10 minutes, PagerDuty automatically escalates to the
   secondary on-call.
3. For SEV1, the incident commander opens a Slack channel named
   `inc-<date>-<slug>` and pulls in the service owner.

Every SEV1 and SEV2 requires a blameless postmortem within five business days.
The runbook for the failing service lives next to its deployment manifests.
