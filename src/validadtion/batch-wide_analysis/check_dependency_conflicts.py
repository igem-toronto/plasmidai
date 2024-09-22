# Description: This script checks for dependency conflicts in the installed packages.
# The use case in the context of this library is that docker and numpy had conflicting dependencies
# and this script was used to check for such conflicts.

import subprocess
import re
from collections import defaultdict
import json
from typing import Dict, List, Tuple, Set


def get_pipdeptree_output() -> str:
    """
    Run pipdeptree command and return its output.

    Returns:
        str: The output of pipdeptree command.
    """
    result = subprocess.run(["pipdeptree"], stdout=subprocess.PIPE)
    return result.stdout.decode("utf-8")


def parse_pipdeptree_output(output: str) -> Dict[str, List[Tuple[str, str]]]:
    """
    Parse the output of pipdeptree command.

    Args:
        output (str): The output of pipdeptree command.

    Returns:
        Dict[str, List[Tuple[str, str]]]: A dictionary where keys are package names and values are lists of tuples
        containing dependency name and version requirement.
    """
    package_dependencies: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    current_package = None

    for line in output.splitlines():
        if line.startswith(" "):
            match = re.match(r"\s+-\s+(\S+)\s+\[required:\s+([^\],]+)", line)
            if match:
                dependency, version_req = match.groups()
                package_dependencies[current_package].append((dependency, version_req))
        else:
            current_package = line.split("==")[0]

    return package_dependencies


def get_installed_packages() -> List[Dict[str, str]]:
    """
    Get a list of installed packages using pip.

    Returns:
        List[Dict[str, str]]: A list of dictionaries containing package information.
    """
    result = subprocess.run(["pip", "list", "--format=json"], stdout=subprocess.PIPE)
    return json.loads(result.stdout.decode("utf-8"))


def get_package_dependencies(package_name: str) -> List[str]:
    """
    Get dependencies of a specific package.

    Args:
        package_name (str): The name of the package.

    Returns:
        List[str]: A list of package dependencies.
    """
    result = subprocess.run(["pip", "show", package_name], stdout=subprocess.PIPE)
    output = result.stdout.decode("utf-8")
    dependencies = []
    for line in output.splitlines():
        if line.startswith("Requires:"):
            dependencies = line.split(": ")[1].split(", ")
            break
    return dependencies


def update_package(package_name: str) -> None:
    """
    Update a specific package using pip.

    Args:
        package_name (str): The name of the package to update.
    """
    subprocess.run(["pip", "install", "--upgrade", package_name])


def find_conflicts(
    package_dependencies: Dict[str, List[Tuple[str, str]]]
) -> List[Tuple[str, Set[str]]]:
    """
    Find conflicts in package dependencies.

    Args:
        package_dependencies (Dict[str, List[Tuple[str, str]]]): A dictionary of package dependencies.

    Returns:
        List[Tuple[str, Set[str]]]: A list of tuples containing the conflicting dependency and its version requirements.
    """
    dependency_versions: Dict[str, Set[str]] = defaultdict(set)
    conflicts = []

    for package, dependencies in package_dependencies.items():
        for dependency, version_req in dependencies:
            dependency_versions[dependency].add(version_req)

    for dependency, version_reqs in dependency_versions.items():
        if len(version_reqs) > 1:
            conflicts.append((dependency, version_reqs))

    return conflicts


def numpy_dependencies() -> None:
    """
    Check for packages that depend on NumPy and update them.
    """
    installed_packages = get_installed_packages()
    numpy_dependents = []

    for package in installed_packages:
        package_name = package["name"]
        dependencies = get_package_dependencies(package_name)
        if "numpy" in dependencies:
            numpy_dependents.append(package_name)

    if numpy_dependents:
        print("Updating packages that depend on NumPy:")
        for package in numpy_dependents:
            print(f"Updating {package}...")
            update_package(package)
        print("All packages that depend on NumPy have been updated.")
    else:
        print("No packages depend on NumPy.")


def main() -> None:
    """
    Main function to check for dependency conflicts and update NumPy dependents.
    """
    numpy_dependencies()
    output = get_pipdeptree_output()
    package_dependencies = parse_pipdeptree_output(output)
    conflicts = find_conflicts(package_dependencies)

    if conflicts:
        print("Dependency conflicts found:")
        for dependency, version_reqs in conflicts:
            print(f"{dependency}: {', '.join(version_reqs)}")
    else:
        print("No dependency conflicts found.")


if __name__ == "__main__":
    main()
