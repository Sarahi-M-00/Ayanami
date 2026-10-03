"""Seed tool catalog (Phase 6). Definitions only — no execution.

DevOps tools are read-only by default. Security tools that touch a target
require the target to be listed in configs/scope.yaml (default-deny).
"""

from __future__ import annotations

from .registry import Registry, ToolDef, asset_in_scope, no_scope_needed

SANDBOX_READ = "read-only subprocess, no network egress except target asset, timeout 30s"
SANDBOX_LOCAL = "local read-only inspection of Ling's machine, no writes"


def devops_tools() -> list[ToolDef]:
    return [
        ToolDef(
            name="git_status", description="Show git working tree status (read).",
            arguments={"type": "object",
                       "properties": {"repo": {"type": "string"}},
                       "required": ["repo"], "additionalProperties": False},
            risk_tier="read_only", requires_confirmation=False,
            sandbox=SANDBOX_LOCAL, scope_check=no_scope_needed),
        ToolDef(
            name="docker_ps", description="List containers (read).",
            arguments={"type": "object", "properties": {},
                       "additionalProperties": False},
            risk_tier="read_only", requires_confirmation=False,
            sandbox=SANDBOX_LOCAL, scope_check=no_scope_needed),
        ToolDef(
            name="kubectl_get",
            description="kubectl get/describe on a resource (read).",
            arguments={"type": "object",
                       "properties": {"resource": {"type": "string"},
                                      "name": {"type": "string"},
                                      "namespace": {"type": "string"}},
                       "required": ["resource"], "additionalProperties": False},
            risk_tier="read_only", requires_confirmation=False,
            sandbox=SANDBOX_READ, scope_check=no_scope_needed),
        ToolDef(
            name="terraform_plan", description="Show a Terraform plan (read).",
            arguments={"type": "object",
                       "properties": {"dir": {"type": "string"}},
                       "required": ["dir"], "additionalProperties": False},
            risk_tier="read_only", requires_confirmation=False,
            sandbox=SANDBOX_READ, scope_check=no_scope_needed),
        ToolDef(
            name="ci_logs", description="Fetch and parse CI logs (read).",
            arguments={"type": "object",
                       "properties": {"pipeline": {"type": "string"},
                                      "tail": {"type": "integer"}},
                       "required": ["pipeline"], "additionalProperties": False},
            risk_tier="read_only", requires_confirmation=False,
            sandbox=SANDBOX_READ, scope_check=no_scope_needed),
        ToolDef(
            name="systemctl_status", description="systemctl status of a service (read).",
            arguments={"type": "object",
                       "properties": {"service": {"type": "string"}},
                       "required": ["service"], "additionalProperties": False},
            risk_tier="read_only", requires_confirmation=False,
            sandbox=SANDBOX_LOCAL, scope_check=no_scope_needed),
        ToolDef(
            name="log_search", description="Search local logs for a pattern (read).",
            arguments={"type": "object",
                       "properties": {"path": {"type": "string"},
                                      "pattern": {"type": "string"}},
                       "required": ["path", "pattern"], "additionalProperties": False},
            risk_tier="read_only", requires_confirmation=False,
            sandbox=SANDBOX_LOCAL, scope_check=no_scope_needed),
    ]


def security_tools() -> list[ToolDef]:
    target = {"type": "object",
              "properties": {"target": {"type": "string"}},
              "required": ["target"], "additionalProperties": False}
    return [
        ToolDef(
            name="port_scan",
            description="Authorized-scope port/service scan. Target MUST be in scope.",
            arguments=target,
            risk_tier="state_changing", requires_confirmation=True,
            sandbox="isolated network namespace, scope target only, timeout 120s",
            scope_check=asset_in_scope("target")),
        ToolDef(
            name="cve_lookup", description="Look up a CVE identifier (read).",
            arguments={"type": "object",
                       "properties": {"cve_id": {"type": "string"}},
                       "required": ["cve_id"], "additionalProperties": False},
            risk_tier="read_only", requires_confirmation=False,
            sandbox=SANDBOX_READ, scope_check=no_scope_needed),
        ToolDef(
            name="image_scan",
            description="Scan a container image for vulnerabilities, Trivy/Grype class (read).",
            arguments={"type": "object",
                       "properties": {"image": {"type": "string"}},
                       "required": ["image"], "additionalProperties": False},
            risk_tier="read_only", requires_confirmation=False,
            sandbox=SANDBOX_READ, scope_check=no_scope_needed),
        ToolDef(
            name="sast_scan",
            description="Static analysis of a repo, Semgrep class (read).",
            arguments={"type": "object",
                       "properties": {"repo": {"type": "string"},
                                      "ruleset": {"type": "string"}},
                       "required": ["repo"], "additionalProperties": False},
            risk_tier="read_only", requires_confirmation=False,
            sandbox=SANDBOX_READ, scope_check=no_scope_needed),
        ToolDef(
            name="hash_lookup", description="Look up a file hash / IOC (read).",
            arguments={"type": "object",
                       "properties": {"hash": {"type": "string"}},
                       "required": ["hash"], "additionalProperties": False},
            risk_tier="read_only", requires_confirmation=False,
            sandbox=SANDBOX_READ, scope_check=no_scope_needed),
        ToolDef(
            name="log_triage", description="Triage security log lines (read).",
            arguments={"type": "object",
                       "properties": {"path": {"type": "string"},
                                      "window": {"type": "string"}},
                       "required": ["path"], "additionalProperties": False},
            risk_tier="read_only", requires_confirmation=False,
            sandbox=SANDBOX_LOCAL, scope_check=no_scope_needed),
    ]


def build_default_registry() -> Registry:
    reg = Registry()
    for tool in devops_tools() + security_tools():
        reg.register(tool)
    return reg
