# BlackRoad Kubernetes Orchestrator

> Production-quality Kubernetes manifest generation, cluster management, and HPA configuration.

## Features

- `KubernetesCluster` dataclass with kubeconfig generation
- `DeploymentSpec` with resources, env vars, liveness/readiness probes
- Deployment, Service, HPA, ConfigMap, Ingress manifest generation
- Resource limit calculation (CPU/Memory with multiplier)
- YAML manifest validation with field-level error reporting
- Helm values serialization and deep-merge
- SQLite persistence for deployments and clusters
- CLI: `generate`, `validate`, `list`, `helm`

## Usage

```bash
pip install pyyaml

# Generate manifests
python src/k8s_orchestrator.py generate \
  --name my-api \
  --image nginx:latest \
  --replicas 3 \
  --port 8080 \
  --cpu 200m --memory 256Mi \
  --min-replicas 2 --max-replicas 10 \
  --service-type LoadBalancer \
  --save

# Validate a manifest
python src/k8s_orchestrator.py validate manifest.yaml

# List saved deployments
python src/k8s_orchestrator.py list --namespace production
```

## Tests

```bash
pytest tests/ -v --cov=src
```

## Architecture

```
DeploymentSpec ──► generate_deployment_yaml()  ──► Deployment manifest
                ──► generate_service_yaml()    ──► Service manifest
                ──► generate_hpa_yaml()        ──► HPA manifest

calculate_resource_limits(cpu, mem, 2.0) ──► ResourceRequirements
validate_manifest(yaml_str)              ──► (bool, list[str])
helm_values_to_yaml(values)             ──► YAML string
```
