"""Tests for BlackRoad Kubernetes Orchestrator."""
import pytest
import yaml
from k8s_orchestrator import (
    DeploymentSpec, KubernetesCluster, ResourceRequirements,
    generate_deployment_yaml, generate_service_yaml, generate_hpa_yaml,
    generate_configmap_yaml, generate_ingress_yaml,
    calculate_resource_limits, validate_manifest, helm_values_to_yaml,
    merge_helm_values, _init_db, save_deployment, list_deployments,
)


def make_spec(**kwargs):
    defaults = dict(name="test-app", image="nginx:latest", replicas=2, port=8080)
    defaults.update(kwargs)
    return DeploymentSpec(**defaults)


class TestDeploymentSpec:
    def test_basic_creation(self):
        spec = make_spec()
        assert spec.name == "test-app"
        assert spec.replicas == 2

    def test_default_labels(self):
        spec = make_spec()
        assert spec.labels["app"] == "test-app"
        assert spec.labels["managed-by"] == "blackroad"

    def test_invalid_replicas(self):
        with pytest.raises(ValueError, match="replicas"):
            make_spec(replicas=0)

    def test_full_ref(self):
        spec = make_spec(name="my-service")
        assert spec.name == "my-service"


class TestKubernetesCluster:
    def test_valid_cluster(self):
        cluster = KubernetesCluster(
            name="prod", context="prod-ctx",
            api_server="https://k8s.example.com"
        )
        assert cluster.name == "prod"

    def test_invalid_api_server(self):
        with pytest.raises(ValueError, match="api_server"):
            KubernetesCluster(name="x", context="x", api_server="k8s.example.com")

    def test_invalid_node_count(self):
        with pytest.raises(ValueError):
            KubernetesCluster(name="x", context="x",
                              api_server="https://k8s.example.com", node_count=0)

    def test_kubeconfig_structure(self):
        cluster = KubernetesCluster("c1", "ctx1", "https://api.example.com")
        kc = cluster.to_kubeconfig()
        assert kc["apiVersion"] == "v1"
        assert kc["kind"] == "Config"
        assert len(kc["clusters"]) == 1


class TestGenerateDeploymentYaml:
    def test_produces_valid_yaml(self):
        spec = make_spec()
        result = generate_deployment_yaml(spec)
        doc = yaml.safe_load(result)
        assert doc["kind"] == "Deployment"
        assert doc["metadata"]["name"] == "test-app"

    def test_replicas_in_output(self):
        spec = make_spec(replicas=3)
        doc = yaml.safe_load(generate_deployment_yaml(spec))
        assert doc["spec"]["replicas"] == 3

    def test_env_vars_included(self):
        spec = make_spec(env_vars={"PORT": "8080", "ENV": "prod"})
        doc = yaml.safe_load(generate_deployment_yaml(spec))
        container = doc["spec"]["template"]["spec"]["containers"][0]
        env_names = [e["name"] for e in container["env"]]
        assert "PORT" in env_names
        assert "ENV" in env_names

    def test_resource_limits(self):
        spec = make_spec()
        doc = yaml.safe_load(generate_deployment_yaml(spec))
        container = doc["spec"]["template"]["spec"]["containers"][0]
        assert "resources" in container
        assert "limits" in container["resources"]

    def test_namespace(self):
        spec = make_spec(namespace="production")
        doc = yaml.safe_load(generate_deployment_yaml(spec))
        assert doc["metadata"]["namespace"] == "production"


class TestGenerateServiceYaml:
    def test_clusterip_default(self):
        spec = make_spec()
        doc = yaml.safe_load(generate_service_yaml(spec))
        assert doc["kind"] == "Service"
        assert doc["spec"]["type"] == "ClusterIP"

    def test_loadbalancer_type(self):
        spec = make_spec()
        doc = yaml.safe_load(generate_service_yaml(spec, "LoadBalancer"))
        assert doc["spec"]["type"] == "LoadBalancer"

    def test_invalid_service_type(self):
        spec = make_spec()
        with pytest.raises(ValueError):
            generate_service_yaml(spec, "InvalidType")

    def test_service_name(self):
        spec = make_spec(name="my-api")
        doc = yaml.safe_load(generate_service_yaml(spec))
        assert doc["metadata"]["name"] == "my-api-svc"


class TestGenerateHPAYaml:
    def test_basic_hpa(self):
        result = generate_hpa_yaml("test-app", 1, 10, 70)
        doc = yaml.safe_load(result)
        assert doc["kind"] == "HorizontalPodAutoscaler"
        assert doc["spec"]["minReplicas"] == 1
        assert doc["spec"]["maxReplicas"] == 10

    def test_invalid_min_replicas(self):
        with pytest.raises(ValueError):
            generate_hpa_yaml("app", 0, 5)

    def test_invalid_max_less_than_min(self):
        with pytest.raises(ValueError):
            generate_hpa_yaml("app", 5, 3)

    def test_invalid_cpu_threshold(self):
        with pytest.raises(ValueError):
            generate_hpa_yaml("app", 1, 5, cpu_threshold=0)

    def test_cpu_threshold_in_output(self):
        doc = yaml.safe_load(generate_hpa_yaml("app", 1, 5, cpu_threshold=80))
        metrics = doc["spec"]["metrics"][0]
        assert metrics["resource"]["target"]["averageUtilization"] == 80


class TestCalculateResourceLimits:
    def test_double_multiplier(self):
        res = calculate_resource_limits("100m", "128Mi", 2.0)
        assert res.cpu_request == "100m"
        assert res.cpu_limit == "200m"
        assert res.memory_limit == "256Mi"

    def test_three_x_multiplier(self):
        res = calculate_resource_limits("500m", "512Mi", 3.0)
        assert res.cpu_limit == "1500m"

    def test_cpu_in_cores(self):
        res = calculate_resource_limits("1", "1Gi", 2.0)
        assert "2" in res.cpu_limit or "2000" in res.cpu_limit

    def test_invalid_multiplier(self):
        with pytest.raises(ValueError):
            calculate_resource_limits("100m", "128Mi", 0.5)


class TestValidateManifest:
    def test_valid_deployment(self):
        spec = make_spec()
        yaml_str = generate_deployment_yaml(spec)
        valid, errors = validate_manifest(yaml_str)
        assert valid, errors

    def test_invalid_yaml(self):
        valid, errors = validate_manifest("this: is: not: valid: yaml: [")
        assert not valid
        assert len(errors) > 0

    def test_missing_kind(self):
        yaml_str = "apiVersion: v1\nmetadata:\n  name: test\n"
        valid, errors = validate_manifest(yaml_str)
        assert not valid

    def test_multi_document(self):
        spec = make_spec()
        combined = generate_deployment_yaml(spec) + "---\n" + generate_service_yaml(spec)
        valid, errors = validate_manifest(combined)
        assert valid, errors


class TestHelmHelpers:
    def test_basic_values(self):
        values = {"replicaCount": 2, "image": {"tag": "latest"}}
        result = helm_values_to_yaml(values)
        doc = yaml.safe_load(result)
        assert doc["replicaCount"] == 2

    def test_merge_values(self):
        base = {"a": 1, "b": {"c": 2, "d": 3}}
        override = {"b": {"c": 99}, "e": 5}
        merged = merge_helm_values(base, override)
        assert merged["a"] == 1
        assert merged["b"]["c"] == 99
        assert merged["b"]["d"] == 3
        assert merged["e"] == 5


class TestSQLitePersistence:
    def test_save_and_list(self, tmp_path):
        db = _init_db(tmp_path / "test.db")
        spec = make_spec()
        manifest = generate_deployment_yaml(spec)
        rid = save_deployment(spec, manifest, db)
        assert rid > 0
        rows = list_deployments(db=db)
        assert len(rows) == 1
        assert rows[0]["name"] == "test-app"

    def test_namespace_filter(self, tmp_path):
        db = _init_db(tmp_path / "test.db")
        spec1 = make_spec(name="app1", namespace="ns1")
        spec2 = make_spec(name="app2", namespace="ns2")
        save_deployment(spec1, generate_deployment_yaml(spec1), db)
        save_deployment(spec2, generate_deployment_yaml(spec2), db)
        rows = list_deployments("ns1", db)
        assert len(rows) == 1
        assert rows[0]["name"] == "app1"
