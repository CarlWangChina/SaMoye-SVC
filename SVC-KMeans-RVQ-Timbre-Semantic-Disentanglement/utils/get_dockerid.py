import subprocess


def get_dockerId():
    p = subprocess.Popen(
        "cat /proc/self/cgroup | grep /docker | head -1 | cut -d/ -f3",
        shell=True,
        stdout=subprocess.PIPE,
    )
    # p = subprocess.call("cat /proc/self/cgroup | grep /docker | head -1 | cut -d/ -f3", shell=True)
    out = p.stdout.read()
    id = str(out, "utf-8")
    return id
