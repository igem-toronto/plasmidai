import os
import docker
import time
from tools import *


def format_dir_name(dir: str):
    return dir.replace("\\", "/")


def run_docker_container(client, image, command):

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


def run_mobsuite_on_fasta_files(root_dir, output_dir):
    # Create a Docker client
    client = docker.from_env()
    # Check if output dir exists; otherwise create it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Walk through all directories and subdirectories
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".fasta"):
                print(f"Processing {file} in {subdir}")
                fasta_path = os.path.join(subdir, file)
                if "\\" in fasta_path:
                    fasta_path = format_dir_name(fasta_path)
                out_subdir = "/" + subdir.split("\\")[-1] if "\\" in subdir else output_dir + "/" + \
                                                                                              subdir.split("/")[-1]
                # Convert to absolute path
                fasta_dir = os.path.abspath(subdir)
                out_dir = os.path.abspath(output_dir)

                print(f"fasta_path: {fasta_dir}")
                print(f"out_subdir: {out_subdir}")
                # make sure we havent already run the analysis. only run if the output directory does not contain a contig_report.txt file
                if os.path.exists(out_dir + out_subdir + "/contig_report.txt"):
                    print(f"Skipping {file} in {subdir} as contig_report.txt already exists")
                    continue

                image = "kbessonov/mob_suite:3.0.3"

                # We want to run the equivalent of
                # run -it --rm -v C:/Users/rodcs/Desktop/my-repos/plasmid-ai/validation/generated_sequences/sequences_feb/:/data kbessonov/mob_suite:3.0.3

                # Run the Docker container
                container = client.containers.run(
                    image,
                    tty=True,
                    stdin_open=True,
                    remove=True,
                    detach=True,
                    mounts=[docker.types.Mount(target='/data', source=fasta_dir, type='bind'), docker.types.Mount(target='/output', source=out_dir, type='bind')]
                )

                # Execute commands inside the container
                commands = [
                    f"echo 'Hello from inside the container {container}!'",
                    f"echo 'About to run mob_recon on {file} to {out_subdir}'",
                    f"mob_recon -i /data/{file} -o /output{out_subdir} --force"
                ]

                for cmd in commands:
                    try:
                        exit_code, output = container.exec_run(cmd)
                        print(output.decode())
                    except Exception as e:
                        print(f"Error running command {cmd} in container {container}: {e}")
                print(container.top())



if __name__ == "__main__":

    # pull the mob_suite image
    client = docker.from_env()
    client.images.pull("kbessonov/mob_suite:3.0.3")
    run_mobsuite_on_fasta_files(root_dir="generated_sequences", output_dir=os.curdir + "/mobsuite_outputs")

    # optional: allign with curated database
    print("Alligning mob_suite results")
    alignment_results_df = allign_mobsuite_results(mobsuite_output_dir=os.curdir + "/mobsuite_outputs",
                                                   output_dir=os.curdir + "/mobsuite_outputs")

