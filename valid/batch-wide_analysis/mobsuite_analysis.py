import os
import docker
import time
from typing import List
from docker.models.containers import Container
from docker.client import DockerClient
from tools import allign_mobsuite_results


def format_dir_name(dir: str) -> str:
    """
    Format directory name by replacing backslashes with forward slashes.

    Args:
        dir (str): The directory path to format.

    Returns:
        str: The formatted directory path.
    """
    return dir.replace("\\", "/")


def run_docker_container(client: DockerClient, image: str, command: str) -> None:
    """
    Run a Docker container with the specified image and command.

    Args:
        client (DockerClient): The Docker client.
        image (str): The Docker image to use.
        command (str): The command to run in the container.
    """
    # Run the Docker container with the command
    container = client.containers.run(image, command=command, detach=True)

    # Wait for the container to finish
    container.wait()

    # Get the logs from the container
    logs = container.logs()
    print(logs.decode())

    # Stop and remove the container
    time.sleep(2)  # Add a delay before removing the container
    container.stop()
    time.sleep(2)  # Add a delay before removing the container
    container.remove()


def run_mobsuite_on_fasta_files(root_dir: str, output_dir: str) -> None:
    """
    Run MOB-suite on FASTA files in the specified directory.

    Args:
        root_dir (str): The root directory containing FASTA files.
        output_dir (str): The directory to store the output.
    """
    # Create a Docker client
    client = docker.from_env()
    # Check if output dir exists; otherwise create it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Walk through all directories and subdirectories
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".fasta"):
                process_fasta_file(client, subdir, file, output_dir)


def process_fasta_file(
    client: DockerClient, subdir: str, file: str, output_dir: str
) -> None:
    """
    Process a single FASTA file using MOB-suite in a Docker container.

    Args:
        client (DockerClient): The Docker client.
        subdir (str): The subdirectory containing the FASTA file.
        file (str): The name of the FASTA file.
        output_dir (str): The directory to store the output.
    """
    print(f"Processing {file} in {subdir}")
    fasta_path = os.path.join(subdir, file)
    if "\\" in fasta_path:
        fasta_path = format_dir_name(fasta_path)
    out_subdir = (
        "/" + subdir.split("\\")[-1]
        if "\\" in subdir
        else output_dir + "/" + subdir.split("/")[-1]
    )

    # Convert to absolute path
    fasta_dir = os.path.abspath(subdir)
    out_dir = os.path.abspath(output_dir)

    print(f"fasta_path: {fasta_dir}")
    print(f"out_subdir: {out_subdir}")

    # Check if analysis has already been run
    if os.path.exists(os.path.join(out_dir, out_subdir, "contig_report.txt")):
        print(f"Skipping {file} in {subdir} as contig_report.txt already exists")
        return

    image = "kbessonov/mob_suite:3.0.3"

    # Run the Docker container
    container = run_mobsuite_container(client, image, fasta_dir, out_dir)

    # Execute commands inside the container
    execute_commands_in_container(container, file, out_subdir)
    print(container.top())


def run_mobsuite_container(
    client: DockerClient, image: str, fasta_dir: str, out_dir: str
) -> Container:
    """
    Run a MOB-suite Docker container.

    Args:
        client (DockerClient): The Docker client.
        image (str): The Docker image to use.
        fasta_dir (str): The directory containing FASTA files.
        out_dir (str): The output directory.

    Returns:
        Container: The running Docker container.
    """
    return client.containers.run(
        image,
        tty=True,
        stdin_open=True,
        remove=True,
        detach=True,
        mounts=[
            docker.types.Mount(target="/data", source=fasta_dir, type="bind"),
            docker.types.Mount(target="/output", source=out_dir, type="bind"),
        ],
    )


def execute_commands_in_container(
    container: Container, file: str, out_subdir: str
) -> None:
    """
    Execute commands inside the Docker container.

    Args:
        container (Container): The Docker container to execute commands in.
        file (str): The name of the FASTA file.
        out_subdir (str): The output subdirectory.
    """
    commands: List[str] = [
        f"echo 'Hello from inside the container {container}!'",
        f"echo 'About to run mob_recon on {file} to {out_subdir}'",
        f"mob_recon -i /data/{file} -o /output{out_subdir} --force",
    ]

    for cmd in commands:
        try:
            exit_code, output = container.exec_run(cmd)
            print(output.decode())
        except Exception as e:
            print(f"Error running command {cmd} in container {container}: {e}")


if __name__ == "__main__":
    # pull the mob_suite image
    client = docker.from_env()
    client.images.pull("kbessonov/mob_suite:3.0.3")
    run_mobsuite_on_fasta_files(
        root_dir="generated_sequences",
        output_dir=os.path.join(os.curdir, "mobsuite_outputs"),
    )

    # optional: align with curated database
    print("Aligning mob_suite results")
    alignment_results_df = allign_mobsuite_results(
        mobsuite_output_dir=os.path.join(os.curdir, "mobsuite_outputs"),
        output_dir=os.path.join(os.curdir, "mobsuite_outputs"),
    )
