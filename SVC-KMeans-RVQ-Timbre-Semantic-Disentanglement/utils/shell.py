import subprocess
from .logger import get_logger

logger = get_logger(__name__)


class Shell:
    @staticmethod
    def run(
        cmd,
        on_stdout=lambda e: print("" + e),
        on_stderr=lambda e: print("" + e),
    ):
        logger.info(f"run shell command: {cmd}")
        with subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True
        ) as process:
            outputs = []
            errors = []
            while True:
                output = process.stdout.readline().decode()
                error = process.stderr.readline().decode()
                outputs.append(output)
                errors.append(error)
                if output == "" and process.poll() is not None:
                    break
                if output:
                    on_stdout(output.strip())
                if error:
                    on_stderr(error.strip())
            return_code = process.poll()
            return return_code, "".join(outputs), "".join(errors)
