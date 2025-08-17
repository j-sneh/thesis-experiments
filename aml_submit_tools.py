"""
Master script which sends jobs to Azure
"""
from azure.ai.ml import command, Input, MLClient, Output, MpiDistribution
from azure.ai.ml.entities import Environment, BuildContext
from azure.identity import AuthenticationRecord, InteractiveBrowserCredential, TokenCachePersistenceOptions
from azure.ai.ml.constants import AssetTypes, InputOutputModes
import argparse
import os
import sys
from pathlib import Path
import webbrowser
import time

from commons.aml_utils import AmlConfig, get_credential


def sumbit_to_aml(args):
    filedir = './'

    compute_target = "/subscriptions/5c9e4789-4852-4ffe-8551-d682affcbd74/resourceGroups/manifold2-rg/providers/Microsoft.MachineLearningServices/virtualclusters/as-manifold2-sa-vc" if args.manifold else "a100x4" 
    SUBSCRIPTION = "5c9e4789-4852-4ffe-8551-d682affcbd74"
    RESOURCE_GROUP = "manifold2-rg" if args.manifold else "playground-rg"
    WS_NAME = "as-manifold2-sa-ws" if args.manifold else "as-playground-w3-ws"
    gpu_count_per_node = 8 if args.manifold else 4
    gpu_count = 8 if args.manifold else 4
    gpu_count = args.gpu_count if hasattr(args, 'gpu_count') else gpu_count

    assert gpu_count % gpu_count_per_node == 0, "Total GPU count is not divisible by per-node GPU count."
    assert not (gpu_count >= 4 and gpu_count_per_node == 1), f"Multi-GPU nodes should be used for a job that requires {gpu_count} GPUs."
    node_count = gpu_count // gpu_count_per_node

    vc_config = {
        "instance_type": "Singularity.ND96r_H100_v5",
        "instance_count": node_count,
        "properties": {
            "AISuperComputer": {
                "slaTier": "Premium"
            }
        }
    } if args.manifold else None

    
    access_credential = get_credential()
    
    # Get a handle to thee workspace
    ml_client = MLClient(
        credential=access_credential,
        subscription_id=SUBSCRIPTION,
        resource_group_name=RESOURCE_GROUP,
        workspace_name=WS_NAME,
    )
    
    # python environment
    # conda environment is installed on top of base image
    # Setup Environment
    custom_job_env = Environment(
        name="attack",
        description="Custom environment for TP AML Training",
        build=BuildContext(path=os.path.join(os.getcwd(), "docker/attack"))
    )
    env = ml_client.environments.create_or_update(custom_job_env)

    if args.manifold:
        # User assigned managed identity to be used by Manifold jobs.
        environment_variables = {
        "_AZUREML_SINGULARITY_JOB_UAI": "/subscriptions/5c9e4789-4852-4ffe-8551-d682affcbd74/resourceGroups/shared-rg/providers/Microsoft.ManagedIdentity/userAssignedIdentities/as-shared-id"
        }
    else:
        environment_variables = {
        "AZUREML_COMMON_RUNTIME_USE_SBOM_CAPABILITY": "true",
        "DATASET_MOUNT_FILE_CACHE_PRUNE_THRESHOLD": 0.6
        }

    blob_account_name = "minionagentsa" if args.manifold else "minionagent"
    if args.manifold:
        data_dir = "azureml://subscriptions/5c9e4789-4852-4ffe-8551-d682affcbd74/resourcegroups/manifold2-rg/workspaces/as-manifold2-sa-ws/datastores/vllm_dataset_formatted/paths/gui_cua_trajectory_data/"
        model_dir = "azureml://subscriptions/5c9e4789-4852-4ffe-8551-d682affcbd74/resourcegroups/manifold2-rg/workspaces/as-manifold2-sa-ws/datastores/vllm_checkpoints/paths/phi_vnext_manifold/phi_vnext/silica_mm/"

    # inputs = {
    #     "model_path": Input(type=AssetTypes.URI_FOLDER, path=model_dir, mode=InputOutputModes.RO_MOUNT),
    #     "data_dir": Input(type=AssetTypes.URI_FOLDER, path=data_dir, mode=InputOutputModes.RO_MOUNT),
    # }

    #output is display name + date + time
    output_uri = f"wasbs://localcuacheckpoints@{blob_account_name}.blob.core.windows.net/" + args.display_name + "/" + time.asctime().replace(" ", "") + "/"

    if args.manifold:
        output_uri = "azureml://datastores/localcua_checkpoints/paths/" + "/" + args.display_name + "/" + time.asctime().replace(" ", "") + "/"

    outputs = {
        "output_dir": Output(type=AssetTypes.URI_FOLDER, path=output_uri, mode=InputOutputModes.RW_MOUNT)
    }

    port = 29500
    # not needed for single node
    # #create a hostfile
    # hostfile_path = os.path.join(filedir, "hostfile")
    # #delete the hostfile if it exists
    # if os.path.exists(hostfile_path):
    #     os.remove(hostfile_path)
    # with open(hostfile_path, 'w') as hostfile:
    #     for i in range(node_count):
    #         hostfile.write(f"node-{i} slots={gpu_count_per_node}\n")

    # lora = "" if not args.disable_lora else "--disable_lora"

    command_to_run = f"""
        cd /mnt/azureml/code; \
        mkdir -p pkg/bin; \
        ln -sf /opt/ollama/bin/ollama pkg/bin/ollama; \
        ./setup.sh; \
        source activate attack; \
        python run_multiple_experiments.py --model qwen3:8b  --cluster-id 1 \
    """

    job = command(
        code=filedir,  # location of source code (relative to this script)
        command=command_to_run,
        environment=env,
        experiment_name='tools',
        compute=compute_target,
        environment_variables=environment_variables,
        outputs=outputs,
        distribution={"type": "pytorch", "process_count_per_instance": 4},
        resources=vc_config,
        instance_count=node_count,
        display_name=args.display_name,
    )

    # Run the experiment
    pipeline_job = ml_client.jobs.create_or_update(job)
    print("Job successfully submitted.")
    try:
        webbrowser.open(pipeline_job.studio_url)
        print("AML Studio url: ", pipeline_job.studio_url)
    except:
        print("Web browser failed to open.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('--config', default="distill_uitars/distillm_v2", type=str)
    parser.add_argument('-n','--display_name', default=time.asctime(), type=str)
    parser.add_argument('-m', '--manifold', action='store_true', help="Use Manifold resources")
    parser.add_argument('-g', '--gpu_count', type=int, help="Total number of GPUs to use for the job. Must be divisible by 4 or 8 depending on the compute target.")
    # parser.add_argument('--disable_lora', action='store_true', help="Disable LoRA")
    # parser.add_argument("--dataset_name", type=str, default="linux_split", help="Name for dataset.")
    args = parser.parse_args()

    sumbit_to_aml(args)
