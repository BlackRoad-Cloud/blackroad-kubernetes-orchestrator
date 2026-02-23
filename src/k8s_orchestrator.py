"""
BlackRoad Kubernetes Orchestrator
Production-quality Kubernetes manifest generation and cluster management.
"""

from __future__ import annotations
import argparse
import json
import re
import sqlite3
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional
import yaml


DB_PATH = Path.home() / ".blackroad" / "k8s_orchestrator.db"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ResourceRequirements:
    cpu_request: str = "100m"
    memory_request: str = "128Mi"
    cpu_limit: str = "200m"
    memory_limit: str = "256Mi"


@dataclass
class DeploymentSpec:
    name: str
    image: str
    replicas: int = 1
    resources: ResourceRequirements = field(default_factory=ResourceRequirements)
    env_vars: dict[str, str] = field(default_factory=dict)
    port: int = 8080
    namespace: str = "default"
    labels: dict[str, str] = field(default_factory=dict)
    annotations: dict[str, str] = field(default_factory=dict)
    image_pull_policy: str = "IfNotPresent"
    restart_policy: str = "Always"

    def __post_init__(self):
        if not self.labels:
            self.labels = {"app": self.name, "managed-by": "blackroad"}
        if self.replicas < 1:
            raise ValueError("replicas must be >= 1")
        if not self.name.replace("-", "").replace("_", "").isalnum():
            raise ValueError(f"Invalid deployment name: {self.name}")


@dataclass
class KubernetesCluster:
    name: str
    context: str
    api_server: str
    namespace: str = "default"
    version: str = "1.28"
    node_count: int = 3
    region: str = "us-east-1"
    tags: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if not self.api_server.startswith(("https://", "http://")):
            raise ValueError("api_server must start with http(s)://")
        if self.node_count < 1:
            raise ValueError("node_count must be >= 1")

    def to_kubeconfig(self) -> dict:
        return {
            "apiVersion": "v1",
            "kind": "Config",
            "clusters": [{
                "name": self.name,
                "cluster": {"server": self.api_server, "insecure-skip-tls-verify": False}
            }],
            "contexts": [{
                "name": self.context,
                "context": {"cluster": self.name, "namespace": self.namespace}
            }],
            "current-context": self.context,
        }


# ---------------------------------------------------------------------------
# Manifest generators
# ---------------------------------------------------------------------------

def generate_deployment_yaml(spec: DeploymentSpec) -> str:
    """Generate a Kubernetes Deployment manifest from a DeploymentSpec."""
    env_list = [{"name": k, "value": v} for k, v in spec.env_vars.items()]
    manifest = {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {
            "name": spec.name,
            "namespace": spec.namespace,
            "labels": spec.labels,
            "annotations": spec.annotations,
        },
        "spec": {
            "replicas": spec.replicas,
            "selector": {"matchLabels": {"app": spec.name}},
            "template": {
                "metadata": {"labels": spec.labels},
                "spec": {
                    "restartPolicy": spec.restart_policy,
                    "containers": [{
                        "name": spec.name,
                        "image": spec.image,
                        "imagePullPolicy": spec.image_pull_policy,
                        "ports": [{"containerPort": spec.port}],
                        "env": env_list,
                        "resources": {
                            "requests": {
                                "cpu": spec.resources.cpu_request,
                                "memory": spec.resources.memory_request,
                            },
                            "limits": {
                                "cpu": spec.resources.cpu_limit,
                                "memory": spec.resources.memory_limit,
                            },
                        },
                        "livenessProbe": {
                            "httpGet": {"path": "/healthz", "port": spec.port},
                            "initialDelaySeconds": 15,
                            "periodSeconds": 20,
                        },
                        "readinessProbe": {
                            "httpGet": {"path": "/ready", "port": spec.port},
                            "initialDelaySeconds": 5,
                            "periodSeconds": 10,
                        },
                    }],
                },
            },
        },
    }
    return yaml.dump(manifest, default_flow_style=False, sort_keys=False)


def generate_service_yaml(spec: DeploymentSpec, service_type: str = "ClusterIP") -> str:
    """Generate a Kubernetes Service manifest."""
    valid_types = {"ClusterIP", "NodePort", "LoadBalancer", "ExternalName"}
    if service_type not in valid_types:
        raise ValueError(f"service_type must be one of {valid_types}")

    manifest: dict[str, Any] = {
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": {
            "name": f"{spec.name}-svc",
            "namespace": spec.namespace,
            "labels": spec.labels,
        },
        "spec": {
            "selector": {"app": spec.name},
            "ports": [{"protocol": "TCP", "port": 80, "targetPort": spec.port}],
            "type": service_type,
        },
    }
    if service_type == "NodePort":
        manifest["spec"]["ports"][0]["nodePort"] = 30000 + (spec.port % 1000)
    return yaml.dump(manifest, default_flow_style=False, sort_keys=False)


def generate_hpa_yaml(
    deployment_name: str,
    min_replicas: int,
    max_replicas: int,
    cpu_threshold: int = 70,
    namespace: str = "default",
) -> str:
    """Generate a HorizontalPodAutoscaler manifest."""
    if min_replicas < 1:
        raise ValueError("min_replicas must be >= 1")
    if max_replicas < min_replicas:
        raise ValueError("max_replicas must be >= min_replicas")
    if not (1 <= cpu_threshold <= 100):
        raise ValueError("cpu_threshold must be between 1 and 100")

    manifest = {
        "apiVersion": "autoscaling/v2",
        "kind": "HorizontalPodAutoscaler",
        "metadata": {"name": f"{deployment_name}-hpa", "namespace": namespace},
        "spec": {
            "scaleTargetRef": {
                "apiVersion": "apps/v1",
                "kind": "Deployment",
                "name": deployment_name,
            },
            "minReplicas": min_replicas,
            "maxReplicas": max_replicas,
            "metrics": [{
                "type": "Resource",
                "resource": {
                    "name": "cpu",
                    "target": {
                        "type": "Utilization",
                        "averageUtilization": cpu_threshold,
                    },
                },
            }],
        },
    }
    return yaml.dump(manifest, default_flow_style=False, sort_keys=False)


def generate_configmap_yaml(name: str, data: dict[str, str], namespace: str = "default") -> str:
    manifest = {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {"name": name, "namespace": namespace},
        "data": data,
    }
    return yaml.dump(manifest, default_flow_style=False, sort_keys=False)


def generate_ingress_yaml(
    name: str,
    host: str,
    service_name: str,
    service_port: int = 80,
    namespace: str = "default",
    tls: bool = True,
) -> str:
    manifest: dict[str, Any] = {
        "apiVersion": "networking.k8s.io/v1",
        "kind": "Ingress",
        "metadata": {
            "name": name,
            "namespace": namespace,
            "annotations": {"nginx.ingress.kubernetes.io/rewrite-target": "/"},
        },
        "spec": {
            "ingressClassName": "nginx",
            "rules": [{
                "host": host,
                "http": {
                    "paths": [{
                        "path": "/",
                        "pathType": "Prefix",
                        "backend": {
                            "service": {
                                "name": service_name,
                                "port": {"number": service_port},
                            }
                        },
                    }]
                },
            }],
        },
    }
    if tls:
        manifest["spec"]["tls"] = [{"hosts": [host], "secretName": f"{name}-tls"}]
    return yaml.dump(manifest, default_flow_style=False, sort_keys=False)


# ---------------------------------------------------------------------------
# Resource helpers
# ---------------------------------------------------------------------------

def _parse_cpu_millis(cpu_str: str) -> float:
    """Convert CPU string (e.g. '500m', '1', '2.5') to millicores float."""
    if cpu_str.endswith("m"):
        return float(cpu_str[:-1])
    return float(cpu_str) * 1000


def _parse_mem_mib(mem_str: str) -> float:
    """Convert memory string to MiB float."""
    units = {"Ki": 1 / 1024, "Mi": 1.0, "Gi": 1024.0, "Ti": 1024 * 1024.0,
             "K": 1 / 1024, "M": 1.0, "G": 1024.0}
    for suffix, factor in units.items():
        if mem_str.endswith(suffix):
            return float(mem_str[: -len(suffix)]) * factor
    return float(mem_str) / (1024 * 1024)


def _format_cpu(millis: float) -> str:
    if millis >= 1000:
        return f"{millis / 1000:.1f}"
    return f"{int(millis)}m"


def _format_mem(mib: float) -> str:
    if mib >= 1024:
        return f"{mib / 1024:.1f}Gi"
    return f"{int(mib)}Mi"


def calculate_resource_limits(
    cpu_req: str, mem_req: str, multiplier: float = 2.0
) -> ResourceRequirements:
    """Calculate resource limits as a multiple of requests."""
    if multiplier < 1.0:
        raise ValueError("multiplier must be >= 1.0")
    cpu_millis = _parse_cpu_millis(cpu_req)
    mem_mib = _parse_mem_mib(mem_req)
    return ResourceRequirements(
        cpu_request=cpu_req,
        memory_request=mem_req,
        cpu_limit=_format_cpu(cpu_millis * multiplier),
        memory_limit=_format_mem(mem_mib * multiplier),
    )


# ---------------------------------------------------------------------------
# Manifest validation
# ---------------------------------------------------------------------------

REQUIRED_FIELDS: dict[str, list[str]] = {
    "Deployment": ["metadata.name", "spec.replicas", "spec.template.spec.containers"],
    "Service": ["metadata.name", "spec.ports", "spec.selector"],
    "HorizontalPodAutoscaler": ["metadata.name", "spec.scaleTargetRef", "spec.maxReplicas"],
    "ConfigMap": ["metadata.name", "data"],
    "Ingress": ["metadata.name", "spec.rules"],
}


def _get_nested(obj: Any, path: str) -> Any:
    parts = path.split(".")
    for p in parts:
        if not isinstance(obj, dict) or p not in obj:
            return None
        obj = obj[p]
    return obj


def validate_manifest(yaml_str: str) -> tuple[bool, list[str]]:
    """Validate a Kubernetes manifest YAML string. Returns (valid, errors)."""
    errors: list[str] = []
    try:
        docs = list(yaml.safe_load_all(yaml_str))
    except yaml.YAMLError as e:
        return False, [f"YAML parse error: {e}"]

    for doc in docs:
        if doc is None:
            continue
        kind = doc.get("kind", "Unknown")
        api_version = doc.get("apiVersion", "")
        if not api_version:
            errors.append(f"Missing apiVersion in {kind}")
        if kind == "Unknown":
            errors.append("Missing kind field")
        required = REQUIRED_FIELDS.get(kind, ["metadata.name"])
        for field_path in required:
            if _get_nested(doc, field_path) is None:
                errors.append(f"{kind}: missing required field '{field_path}'")

    return len(errors) == 0, errors


# ---------------------------------------------------------------------------
# Helm helpers
# ---------------------------------------------------------------------------

def helm_values_to_yaml(values: dict) -> str:
    """Serialize a values dict to Helm-compatible YAML string."""
    return yaml.dump(values, default_flow_style=False, sort_keys=False)


def merge_helm_values(base: dict, override: dict) -> dict:
    """Deep-merge two Helm value dicts (override wins on conflicts)."""
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = merge_helm_values(result[k], v)
        else:
            result[k] = v
    return result


# ---------------------------------------------------------------------------
# SQLite persistence
# ---------------------------------------------------------------------------

def _init_db(path: Path = DB_PATH) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS deployments (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            name        TEXT NOT NULL,
            namespace   TEXT NOT NULL DEFAULT 'default',
            image       TEXT NOT NULL,
            replicas    INTEGER NOT NULL DEFAULT 1,
            created_at  REAL NOT NULL,
            updated_at  REAL NOT NULL,
            manifest    TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS clusters (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            name        TEXT NOT NULL UNIQUE,
            api_server  TEXT NOT NULL,
            context     TEXT NOT NULL,
            version     TEXT,
            node_count  INTEGER,
            region      TEXT,
            created_at  REAL NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS manifests (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            kind        TEXT NOT NULL,
            name        TEXT NOT NULL,
            namespace   TEXT NOT NULL DEFAULT 'default',
            content     TEXT NOT NULL,
            valid       INTEGER NOT NULL DEFAULT 1,
            created_at  REAL NOT NULL
        )
    """)
    conn.commit()
    return conn


def save_deployment(spec: DeploymentSpec, manifest: str, db: sqlite3.Connection) -> int:
    now = time.time()
    cur = db.execute(
        "INSERT INTO deployments (name,namespace,image,replicas,created_at,updated_at,manifest) "
        "VALUES (?,?,?,?,?,?,?)",
        (spec.name, spec.namespace, spec.image, spec.replicas, now, now, manifest),
    )
    db.commit()
    return cur.lastrowid


def list_deployments(namespace: Optional[str] = None, db: sqlite3.Connection = None) -> list[dict]:
    if db is None:
        db = _init_db()
    if namespace:
        rows = db.execute(
            "SELECT * FROM deployments WHERE namespace=? ORDER BY created_at DESC", (namespace,)
        ).fetchall()
    else:
        rows = db.execute("SELECT * FROM deployments ORDER BY created_at DESC").fetchall()
    cols = [d[0] for d in db.execute("SELECT * FROM deployments LIMIT 0").description]
    return [dict(zip(cols, row)) for row in rows]


def save_cluster(cluster: KubernetesCluster, db: sqlite3.Connection) -> int:
    now = time.time()
    cur = db.execute(
        "INSERT OR REPLACE INTO clusters (name,api_server,context,version,node_count,region,created_at) "
        "VALUES (?,?,?,?,?,?,?)",
        (cluster.name, cluster.api_server, cluster.context,
         cluster.version, cluster.node_count, cluster.region, now),
    )
    db.commit()
    return cur.lastrowid


def save_manifest(kind: str, name: str, content: str, namespace: str = "default",
                  db: sqlite3.Connection = None) -> int:
    if db is None:
        db = _init_db()
    valid, _ = validate_manifest(content)
    now = time.time()
    cur = db.execute(
        "INSERT INTO manifests (kind,name,namespace,content,valid,created_at) VALUES (?,?,?,?,?,?)",
        (kind, name, namespace, content, int(valid), now),
    )
    db.commit()
    return cur.lastrowid


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cmd_generate(args: argparse.Namespace) -> None:
    resources = ResourceRequirements() if not args.cpu else calculate_resource_limits(
        args.cpu, args.memory, args.multiplier
    )
    spec = DeploymentSpec(
        name=args.name,
        image=args.image,
        replicas=args.replicas,
        resources=resources,
        port=args.port,
        namespace=args.namespace,
    )
    deployment = generate_deployment_yaml(spec)
    service = generate_service_yaml(spec, args.service_type)
    hpa = generate_hpa_yaml(spec.name, args.min_replicas, args.max_replicas, args.cpu_threshold, args.namespace)

    output = f"---\n{deployment}---\n{service}---\n{hpa}"
    if args.output:
        Path(args.output).write_text(output)
        print(f"Manifests written to {args.output}")
    else:
        print(output)

    if args.save:
        db = _init_db()
        rid = save_deployment(spec, deployment, db)
        print(f"Saved deployment with id={rid}")


def _cmd_validate(args: argparse.Namespace) -> None:
    content = Path(args.file).read_text()
    valid, errors = validate_manifest(content)
    if valid:
        print("✅ Manifest is valid")
    else:
        print("❌ Manifest has errors:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)


def _cmd_list(args: argparse.Namespace) -> None:
    db = _init_db()
    rows = list_deployments(args.namespace, db)
    if not rows:
        print("No deployments found.")
        return
    print(f"{'ID':<5} {'NAME':<20} {'NAMESPACE':<15} {'IMAGE':<30} {'REPLICAS':<8}")
    print("-" * 80)
    for r in rows:
        print(f"{r['id']:<5} {r['name']:<20} {r['namespace']:<15} {r['image']:<30} {r['replicas']:<8}")


def _cmd_helm(args: argparse.Namespace) -> None:
    with open(args.values_file) as f:
        values = yaml.safe_load(f)
    print(helm_values_to_yaml(values))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="BlackRoad Kubernetes Orchestrator")
    sub = parser.add_subparsers(dest="command")

    gen = sub.add_parser("generate", help="Generate Kubernetes manifests")
    gen.add_argument("--name", required=True)
    gen.add_argument("--image", required=True)
    gen.add_argument("--replicas", type=int, default=1)
    gen.add_argument("--port", type=int, default=8080)
    gen.add_argument("--namespace", default="default")
    gen.add_argument("--service-type", default="ClusterIP")
    gen.add_argument("--min-replicas", type=int, default=1)
    gen.add_argument("--max-replicas", type=int, default=5)
    gen.add_argument("--cpu-threshold", type=int, default=70)
    gen.add_argument("--cpu", default=None)
    gen.add_argument("--memory", default=None)
    gen.add_argument("--multiplier", type=float, default=2.0)
    gen.add_argument("--output", "-o", default=None)
    gen.add_argument("--save", action="store_true")

    val = sub.add_parser("validate", help="Validate a manifest file")
    val.add_argument("file")

    lst = sub.add_parser("list", help="List saved deployments")
    lst.add_argument("--namespace", default=None)

    helm = sub.add_parser("helm", help="Convert values file to Helm YAML")
    helm.add_argument("values_file")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    dispatch = {
        "generate": _cmd_generate,
        "validate": _cmd_validate,
        "list": _cmd_list,
        "helm": _cmd_helm,
    }
    if args.command in dispatch:
        dispatch[args.command](args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
