from dataclasses import dataclass
import yaml
from typing import Optional, Union
from azure.identity import AuthenticationRecord, InteractiveBrowserCredential, TokenCachePersistenceOptions
from azure.ai.ml import command, Input, MLClient, Output
from azure.ai.ml.constants import AssetTypes
from enum import Enum
import sys
from pathlib import Path
import os


AccessType = Enum("AccessType", ["READ", "WRITE"])


@dataclass
class AmlConfig:
    """Configuration for AML resources."""
    subscription_id: str
    resource_group_name: str
    workspace_name: str
    compute_name: str
    instance_type: str
    gpu_count_per_node: int

    @classmethod
    def from_yaml(cls, path: str) -> 'AmlConfig':
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)


@dataclass
class BlobContainerInfo:
    """Info about a user defined blob container."""
    subscription_id: str
    resource_group: str
    storage_account: str
    name: str


@dataclass
class ModelAssetInfo:
    """Info about a workspace model asset."""
    name: str
    version: str


@dataclass
class DataAssetInfo:
    """Info about a workspace data asset."""
    name: str
    version: str


@dataclass
class BlobFolderInfo:
    """Info about a folder on the blob storage."""
    container: BlobContainerInfo
    path: str


@dataclass
class AzureRole:
    """Info about an Azure role."""
    role_definition_id: str
    role_name: str



def has_git(image: str) -> bool:
    """Returns if a given Docker image has Git installed."""
    # Currently only one Docker image of interest does not have Git installed.
    return image != "mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04"


def fatal_error(message: str) -> None:
    """Prints an error message and quits."""
    print(f"ERROR: {message}")
    exit(-1)


def fatal_assert(condition: bool, message: str) -> None:
    """Prints an error message if the condition is not satisfied and quits."""
    if not condition:
        fatal_error(message)


class NamedInputInfo:
    """Info about an asset used as named input.

    Provide BlobFolderInfo for input from blob storage.
    Provide ModelAssetInfo for input from workspace's default datastore.

    TODO: This is a regular class instead of a dataclass because
    jsonarparse cannot parse lists of dataclasses if those dataclasses
    have optional fields that are other dataclasses (see
    https://github.com/omni-us/jsonargparse/issues/524).
    """
    def __init__(
            self,
            name: str,
            model_asset: Optional[ModelAssetInfo] = None,
            data_asset: Optional[DataAssetInfo] = None,
            blob_folder: Optional[BlobFolderInfo] = None) -> None:
        """Initializer."""
        self._name = name
        self._model_asset = model_asset
        self._data_asset = data_asset
        self._blob_folder = blob_folder

    @property
    def name(self) -> str:
        """Name of the named input."""
        return self._name

    @property
    def model_asset(self) -> Optional[ModelAssetInfo]:
        """Model asset information."""
        return self._model_asset

    @property
    def data_asset(self) -> Optional[DataAssetInfo]:
        """Data asset information."""
        return self._data_asset

    @property
    def blob_folder(self) -> Optional[BlobFolderInfo]:
        """Blob folder information."""
        return self._blob_folder


class NamedOutputInfo:
    """Info about an asset used as named output.

    Provide BlobFolderInfo for output to blob storage.
    Omit it for output to the root of the workspace's default datastore.

    TODO: This is a regular class instead of a dataclass because
    jsonarparse cannot parse lists of dataclasses if those dataclasses
    have optional fields that are other dataclasses (see
    https://github.com/omni-us/jsonargparse/issues/524).
    """
    def __init__(self, name: str, blob_folder: Optional[BlobFolderInfo]) -> None:
        """Initializer."""
        self._name = name
        self._blob_folder = blob_folder

    @property
    def name(self) -> str:
        """Name of the named output."""
        return self._name

    @property
    def blob_folder(self) -> Optional[BlobFolderInfo]:
        """Blob folder information."""
        return self._blob_folder


def get_credential() -> InteractiveBrowserCredential:
    """Retrieves a credential that can be used in Azure.

    The cached credential is silently retrieved (if available), otherwise the user is prompted to perform
    interactive authentication and the new credential is cached for future use.
    """
    # lib_name is used as a key to isolate cached credentials used by this library
    # from credentials used by any other clients of azure.identity.
    # The following files are used on Windows:
    #   - %LOCALAPPDATA%\aml_training\auth_record.json
    #     (non-secret authentication information)
    #   - %LOCALAPPDATA%\.IdentityService\aml_training.cache.nocae
    #     (encrypted credential cache)
    lib_name = "aml_training"

    if sys.platform.startswith("win"):
        auth_record_root_path = Path(os.environ["LOCALAPPDATA"])
    else:
        auth_record_root_path = Path.home()
    auth_record_path = auth_record_root_path / lib_name / "auth_record.json"
    cache_options = TokenCachePersistenceOptions(name=f"{lib_name}.cache", allow_unencrypted_storage=True)
    if auth_record_path.exists():
        # If auth record file exists try silent authentication.
        # This can fall back to interactive in some cases (e.g. token expired).
        with open(auth_record_path, "r") as f:
            record_json = f.read()
        deserialized_record = AuthenticationRecord.deserialize(record_json)
        credential = InteractiveBrowserCredential(
            authentication_record=deserialized_record,
            cache_persistence_options=cache_options)
    else:
        # Perform interactive authentication through browser and remember
        # authentication record for next time.
        auth_record_path.parent.mkdir(exist_ok=True)
        credential = InteractiveBrowserCredential(
            cache_persistence_options=cache_options)
        record_json = credential.authenticate().serialize()
        with open(auth_record_path, "w") as f:
            f.write(record_json)
    return credential


def validate_access(
        credential: InteractiveBrowserCredential,
        ml_client: MLClient,
        compute_name: str,
        blob_container_info: BlobContainerInfo,
        access_type: AccessType) -> None:
    """Verifies that the cluster has access to a blob container."""
    # Get role assignments for the storage account.
    authorization_client = AuthorizationManagementClient(credential, blob_container_info.subscription_id)
    role_assignments = authorization_client.role_assignments.list_for_resource(
        resource_group_name=blob_container_info.resource_group,
        resource_provider_namespace="Microsoft.Storage",
        resource_type="storageAccounts",
        resource_name=blob_container_info.storage_account)
    role_assignments = list(role_assignments)

    # Get service principal ID associated with the compute cluster.
    compute = ml_client.compute.get(compute_name)
    user_assigned_identities = compute.identity.user_assigned_identities
    fatal_assert(len(user_assigned_identities) > 0, "The cluster does not have a user-assigned managed identity.")
    if len(user_assigned_identities) > 1:
        warnings.warn("Multiple user-assigned managed identities found, using the first one.")
    managed_identity_name = user_assigned_identities[0].resource_id.split("/")[-1]
    principal_id = user_assigned_identities[0].principal_id

    # Check if the service principal is assigned appropriate role.
    role_definition_id_prefix = \
        f"/subscriptions/{blob_container_info.subscription_id}" \
        "/providers/Microsoft.Authorization/roleDefinitions/"
    reader_role = AzureRole(
        role_definition_id=role_definition_id_prefix + "2a2b9908-6ea1-4ae2-8e65-a410df84e7d1",
        role_name="Storage Blob Data Reader")
    contributor_role = AzureRole(
        role_definition_id=role_definition_id_prefix + "ba92f5b4-2d11-453d-a403-e96b0029c9fe",
        role_name="Storage Blob Data Contributor")
    if access_type == AccessType.READ:
        min_role = reader_role
        allowed_roles = [reader_role, contributor_role]
    else:
        assert access_type == AccessType.WRITE
        min_role = contributor_role
        allowed_roles = [contributor_role,]
    try:
        has_access = any([
            ra.role_definition_id == r.role_definition_id
            and ra.principal_id == principal_id
            for r in allowed_roles
            for ra in role_assignments])
    except ResourceNotFoundError:
        fatal_error(
            f"Storage account {blob_container_info.storage_account} not found in resource group"
            f" {blob_container_info.resource_group} of subscription {blob_container_info.subscription_id}.")

    fatal_assert(
        has_access,
        f"The compute cluster {compute_name} doesn't have access "
        f"to storage account {blob_container_info.storage_account}.\n"
        f"Please assign {min_role.role_name} role to {managed_identity_name} identity.")


def register_datastore_if_not_exists(
        datastore_name: str,
        ml_client: MLClient,
        blob_container: BlobContainerInfo) -> None:
    """Registers a datastore backed by a given blob container, if one does not exist."""
    try:
        datastore = ml_client.datastores.get(datastore_name)
    except ResourceNotFoundError:
        # Register new datastore.
        datastore = AzureBlobDatastore(
            name=datastore_name, account_name=blob_container.storage_account, container_name=blob_container.name)
        ml_client.create_or_update(datastore)
        print(f"Registered new datastore '{datastore.name}'.")


def get_input_datastore_uri(datastore_name: str, datastore_path: str) -> str:
    """Returns AML URI for an output datastore location."""
    fatal_assert(Path(datastore_path) != Path(""), "Datastore path must not be empty.")
    return f"azureml://datastores/{datastore_name}/paths/{datastore_path}"


def get_output_datastore_uri(datastore_name: str, experiment_name: str, datastore_path: str) -> str:
    """Returns AML URI for an input datastore location."""
    job_name = "${{name}}"  # Resolved on the server side
    fatal_assert(Path(datastore_path) != Path(""), "Datastore path must not be empty.")
    return f"azureml://datastores/{datastore_name}/paths/{experiment_name}/{job_name}/{datastore_path}"


def get_workspace_asset_uri(asset_info: Union[ModelAssetInfo, DataAssetInfo]) -> str:
    """Returns AML URI for a workspace model or data asset."""
    return f"azureml:{asset_info.name}:{asset_info.version}"


def get_docker_image(cuda_version: Optional[str]) -> str:
    """Returns the Docker image containing the given CUDA version."""
    cpu_image = "mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04"
    gpu_image_map = {
        "10.0": "mcr.microsoft.com/azureml/openmpi3.1.2-cuda10.0-cudnn7-ubuntu18.04",
        "10.1": "mcr.microsoft.com/azureml/openmpi3.1.2-cuda10.1-cudnn7-ubuntu18.04",
        "11.2": "mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.2-cudnn8-ubuntu20.04"
        }
    if cuda_version in gpu_image_map:
        return gpu_image_map[cuda_version]
    elif cuda_version is None:
        return cpu_image
    else:
        choices_str = ", ".join(f"'{k}'" for k in gpu_image_map.keys())
        fatal_error(f"Unknown CUDA version: '{cuda_version}'. The valid choices are: {choices_str}.")
        raise AssertionError()  # Satisfy linter


def update_storage_account_region(blob_container: BlobContainerInfo, aml_config: AmlConfig) -> None:
    """Infers the Azure region of the storage account.

    Modifies the storage account name by replacing "??" with two-character Azure region code inferred from the AML
    workspace name. Does nothing if the storage account name doesn't contain "??", or if the workspace name doesn't
    conform to the standard naming convention (<prefix>-<name>-<region>-ws).
    """
    if "??" in blob_container.storage_account:
        match = re.match(r"\w+-\w+-(\w+)-ws", aml_config.workspace_name)
        if match:
            region_short = match.group(1)
            blob_container.storage_account = blob_container.storage_account.replace("??", region_short)
        else:
            fatal_error("Storage account name contains '??' but the Azure region couldn't be inferred from the AML "
                        "workspace name.")
