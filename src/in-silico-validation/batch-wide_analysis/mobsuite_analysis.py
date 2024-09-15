import os
import docker
import time


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
                fasta_path = os.path.abspath(fasta_path)
                out_subdir = os.path.abspath(output_dir + out_subdir)

                # Define the Docker image to use
                image = "kbessonov/mob_suite:3.0.3"

                # We want to run the equivalent of
                # run -it --rm -v C:/Users/rodcs/Desktop/my-repos/plasmid-ai/validation/generated_sequences/sequences_feb/:/data kbessonov/mob_suite:3.0.3
                # define the volumes to mount
                volumes = {fasta_path: {'bind': '/data', 'mode': 'rw'},
                           out_subdir: {'bind': '/output', 'mode': 'rw'}}
                # Define the command to run inside the Docker container

                # Run the Docker container
                container = client.containers.run(
                    "kbessonov/mob_suite:3.0.3",
                    volumes=volumes,
                    tty=True,
                    stdin_open=True,
                    remove=True,
                    detach=True
                )

                # Execute commands inside the container
                commands = [
                    "echo 'Hello from inside the container!'",
                    "ls ",
                    "sh -c 'cd /data && ls'"
                ]

                for cmd in commands:
                    exit_code, output = container.exec_run(cmd)
                    print(output.decode())


if __name__ == "__main__":
    run_mobsuite_on_fasta_files(root_dir="..\\generated_sequences", output_dir=os.curdir + "/mobsuite_outputs")
