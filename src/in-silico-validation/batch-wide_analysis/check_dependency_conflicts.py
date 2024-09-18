import subprocess
import re
from collections import defaultdict
import json

def get_pipdeptree_output():
    result = subprocess.run(['pipdeptree'], stdout=subprocess.PIPE)
    return result.stdout.decode('utf-8')

def parse_pipdeptree_output(output):
    package_dependencies = defaultdict(list)
    current_package = None

    for line in output.splitlines():
        if line.startswith(' '):
            match = re.match(r'\s+-\s+(\S+)\s+\[required:\s+([^\],]+)', line)
            if match:
                dependency, version_req = match.groups()
                package_dependencies[current_package].append((dependency, version_req))
        else:
            current_package = line.split('==')[0]

    return package_dependencies

def get_installed_packages():
    result = subprocess.run(['pip', 'list', '--format=json'], stdout=subprocess.PIPE)
    return json.loads(result.stdout.decode('utf-8'))

def get_package_dependencies(package_name):
    result = subprocess.run(['pip', 'show', package_name], stdout=subprocess.PIPE)
    output = result.stdout.decode('utf-8')
    dependencies = []
    for line in output.splitlines():
        if line.startswith('Requires:'):
            dependencies = line.split(': ')[1].split(', ')
            break
    return dependencies

def update_package(package_name):
    subprocess.run(['pip', 'install', '--upgrade', package_name])


def find_conflicts(package_dependencies):
    dependency_versions = defaultdict(set)
    conflicts = []

    for package, dependencies in package_dependencies.items():
        for dependency, version_req in dependencies:
            dependency_versions[dependency].add(version_req)

    for dependency, version_reqs in dependency_versions.items():
        if len(version_reqs) > 1:
            conflicts.append((dependency, version_reqs))

    return conflicts

def numpy_dependencies():
    installed_packages = get_installed_packages()
    numpy_dependents = []

    for package in installed_packages:
        package_name = package['name']
        dependencies = get_package_dependencies(package_name)
        if 'numpy' in dependencies:
            numpy_dependents.append(package_name)

    if numpy_dependents:
        print("Updating packages that depend on NumPy:")
        for package in numpy_dependents:
            print(f"Updating {package}...")
            update_package(package)
        print("All packages that depend on NumPy have been updated.")
    else:
        print("No packages depend on NumPy.")

def main():
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
